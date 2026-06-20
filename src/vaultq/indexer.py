from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from vaultq.chunker import Chunk, chunk_markdown
from vaultq.enrich import ChunkRow, KnowledgeExtractor, build_extractor, dedupe_knowledge_objects
from vaultq.store import connect, delete_qdrant_points, read_config, upsert_collection_rows

logger = logging.getLogger("vaultq.indexer")

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)


@dataclass
class IndexStats:
    collections: int = 0
    documents_scanned: int = 0
    documents_indexed: int = 0
    documents_skipped: int = 0
    documents_deleted: int = 0
    chunks_written: int = 0
    knowledge_objects_written: int = 0
    qdrant_points_deleted: int = 0


@dataclass
class RefreshPlan:
    new_paths: List[str]
    changed_paths: List[str]
    unchanged_paths: List[str]
    deleted_documents: List[Dict[str, Any]]
    stamp_backfills: List[Dict[str, Any]]


def _file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean_text(value: Any) -> str:
    return str(value or "").replace("\x00", "\uFFFD")


def _json_safe(data: Any) -> Any:
    if isinstance(data, str):
        return _clean_text(data)
    if isinstance(data, dict):
        return {_clean_text(str(key)): _json_safe(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_json_safe(value) for value in data]
    if isinstance(data, tuple):
        return [_json_safe(value) for value in data]
    return data


def _json_dumps(data: Any) -> str:
    return json.dumps(_json_safe(data), ensure_ascii=False, default=str)


def _parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    try:
        data = yaml.safe_load(match.group(1)) or {}
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    return data, text[match.end() :]


def _derive_title(path: Path, frontmatter: Dict[str, Any], body: str) -> str:
    for key in ("title", "name"):
        value = str(frontmatter.get(key) or "").strip()
        if value:
            return value
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
        if stripped:
            return stripped[:120]
    return path.stem


def _matches_excludes(rel_path: str, patterns: Sequence[str]) -> bool:
    normalized = rel_path.replace("\\", "/")
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in patterns)


def _iter_markdown_files(root: Path, pattern: str, exclude_globs: Sequence[str]) -> Iterable[Path]:
    for path in root.glob(pattern):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if _matches_excludes(rel, exclude_globs):
            continue
        yield path


def _document_qdrant_point_ids(cur, document_id: int) -> List[str]:
    cur.execute(
        """
        SELECT qdrant_point_id
        FROM vq_chunks
        WHERE document_id = %s AND qdrant_point_id IS NOT NULL
        UNION
        SELECT qdrant_point_id
        FROM vq_knowledge_objects
        WHERE document_id = %s AND qdrant_point_id IS NOT NULL
        """,
        (document_id, document_id),
    )
    return [str(row["qdrant_point_id"]) for row in cur.fetchall() if row.get("qdrant_point_id")]


def _insert_document(cur, collection_id: int, root: Path, path: Path, force: bool) -> Optional[Dict[str, Any]]:
    raw = _clean_text(path.read_text(encoding="utf-8"))
    frontmatter, body = _parse_frontmatter(raw)
    title = _derive_title(path, frontmatter, body)
    rel_path = path.relative_to(root).as_posix()
    digest = _file_hash(raw)
    stat = path.stat()
    metadata = {
        "frontmatter_keys": sorted(frontmatter.keys()),
        "file_size": stat.st_size,
        "file_mtime_ns": stat.st_mtime_ns,
        "title_source": "frontmatter" if str(frontmatter.get("title") or "").strip() else "content",
    }
    cur.execute(
        """
        SELECT id, file_hash, rel_path, title
        FROM vq_documents
        WHERE collection_id = %s AND rel_path = %s
        """,
        (collection_id, rel_path),
    )
    existing = cur.fetchone()
    if existing and existing["file_hash"] == digest and not force:
        return None

    cur.execute(
        """
        INSERT INTO vq_documents (
            collection_id, rel_path, abs_path, title, file_hash, markdown_text,
            frontmatter_json, metadata_json, status, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, NOW())
        ON CONFLICT (collection_id, rel_path)
        DO UPDATE SET
            abs_path = EXCLUDED.abs_path,
            title = EXCLUDED.title,
            file_hash = EXCLUDED.file_hash,
            markdown_text = EXCLUDED.markdown_text,
            frontmatter_json = EXCLUDED.frontmatter_json,
            metadata_json = EXCLUDED.metadata_json,
            status = EXCLUDED.status,
            updated_at = NOW()
        RETURNING id, rel_path, title, markdown_text
        """,
        (
            collection_id,
            rel_path,
            str(path.resolve()),
            _clean_text(title),
            digest,
            _clean_text(body),
            _json_dumps(frontmatter),
            _json_dumps(metadata),
            "indexed",
        ),
    )
    return cur.fetchone()


def _replace_chunks(cur, document: Dict[str, Any], chunks: Sequence[Chunk]) -> List[ChunkRow]:
    cur.execute("DELETE FROM vq_chunks WHERE document_id = %s AND version = 1", (document["id"],))
    rows: List[ChunkRow] = []
    for chunk in chunks:
        cur.execute(
            """
            INSERT INTO vq_chunks (
                document_id, version, chunk_index, heading_path, section_title,
                text_content, token_count, char_count, start_line, end_line,
                metadata_json, qdrant_point_id, updated_at
            )
            VALUES (%s, 1, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, NULL, NOW())
            RETURNING id, chunk_index, heading_path, text_content, start_line, end_line
            """,
            (
                document["id"],
                chunk.index,
                _clean_text(chunk.heading_path),
                _clean_text(chunk.section_title),
                _clean_text(chunk.text),
                chunk.token_count,
                len(chunk.text),
                chunk.start_line,
                chunk.end_line,
                _json_dumps({"section_title": chunk.section_title, "token_count": chunk.token_count}),
            ),
        )
        row = cur.fetchone()
        rows.append(
            ChunkRow(
                id=int(row["id"]),
                chunk_index=int(row["chunk_index"]),
                heading_path=row.get("heading_path") or "",
                text_content=row["text_content"],
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
            )
        )
    return rows


def _replace_knowledge_objects(cur, document_id: int, objects: Sequence[Dict[str, Any]]) -> int:
    cur.execute("DELETE FROM vq_knowledge_objects WHERE document_id = %s AND version = 1", (document_id,))
    written = 0
    for obj in objects:
        cur.execute(
            """
            INSERT INTO vq_knowledge_objects (
                document_id, version, object_type, title, body_text, json_payload,
                confidence, source_chunk_ids, source_line_start, source_line_end,
                content_hash, qdrant_point_id, updated_at
            )
            VALUES (%s, 1, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, %s, %s, NULL, NOW())
            """,
            (
                document_id,
                _clean_text(obj["object_type"]),
                _clean_text(obj["title"]),
                _clean_text(obj["body_text"]),
                _json_dumps(obj.get("json_payload") or {}),
                float(obj.get("confidence") or 0.0),
                _json_dumps(obj.get("source_chunk_ids") or []),
                obj.get("source_line_start"),
                obj.get("source_line_end"),
                obj["content_hash"],
            ),
        )
        written += 1
    return written


def _remap_existing_knowledge_objects(cur, document_id: int, chunk_rows: Sequence[ChunkRow]) -> int:
    cur.execute(
        """
        SELECT id, source_line_start, source_line_end
        FROM vq_knowledge_objects
        WHERE document_id = %s AND version = 1
        """,
        (document_id,),
    )
    rows = list(cur.fetchall())
    remapped = 0
    for row in rows:
        start_line = int(row.get("source_line_start") or 1)
        end_line = int(row.get("source_line_end") or start_line)
        overlapping = [
            chunk.id
            for chunk in chunk_rows
            if chunk.end_line >= start_line and chunk.start_line <= end_line
        ]
        cur.execute(
            """
            UPDATE vq_knowledge_objects
            SET source_chunk_ids = %s::jsonb,
                qdrant_point_id = NULL,
                updated_at = NOW()
            WHERE id = %s
            """,
            (_json_dumps(overlapping), row["id"]),
        )
        remapped += 1
    return remapped


def _index_markdown_path(
    cur,
    *,
    collection_id: int,
    root: Path,
    path: Path,
    stats: IndexStats,
    force: bool,
    extractor: Optional[KnowledgeExtractor],
) -> None:
    document = _insert_document(cur, collection_id, root, path, force=force)
    if document is None:
        stats.documents_skipped += 1
        return

    existing_point_ids = _document_qdrant_point_ids(cur, int(document["id"]))
    if existing_point_ids:
        stats.qdrant_points_deleted += delete_qdrant_points(existing_point_ids)

    chunks = chunk_markdown(document["markdown_text"])
    chunk_rows = _replace_chunks(cur, document, chunks)
    stats.documents_indexed += 1
    stats.chunks_written += len(chunk_rows)
    cur.execute("UPDATE vq_documents SET status = %s WHERE id = %s", ("indexed", document["id"]))

    if extractor is not None:
        derived_objects: List[Dict[str, Any]] = []
        for chunk_row in chunk_rows:
            derived_objects.extend(extractor.extract_chunk_objects(document, chunk_row))
        segment_summaries = [obj for obj in derived_objects if obj["object_type"] == "segment_summary"]
        document_summary = extractor.build_document_summary(document, segment_summaries)
        if document_summary is not None:
            derived_objects.append(document_summary)
        derived_objects = dedupe_knowledge_objects(derived_objects)
        stats.knowledge_objects_written += _replace_knowledge_objects(cur, document["id"], derived_objects)
        cur.execute("UPDATE vq_documents SET status = %s WHERE id = %s", ("enriched", document["id"]))
    else:
        stats.knowledge_objects_written += _remap_existing_knowledge_objects(cur, document["id"], chunk_rows)


def run_index_path(
    rel_path: str,
    *,
    base_dir: Optional[Path] = None,
    collection_name: Optional[str] = None,
    force: bool = True,
    skip_knowledge: bool = True,
) -> IndexStats:
    return run_index_paths(
        [rel_path],
        base_dir=base_dir,
        collection_name=collection_name,
        force=force,
        skip_knowledge=skip_knowledge,
    )


def run_index_paths(
    rel_paths: Sequence[str],
    *,
    base_dir: Optional[Path] = None,
    collection_name: Optional[str] = None,
    force: bool = True,
    skip_knowledge: bool = True,
) -> IndexStats:
    collection_ids = upsert_collection_rows(base_dir)
    config = read_config(base_dir)
    stats = IndexStats(collections=len(collection_ids))
    extractor: Optional[KnowledgeExtractor] = None
    if not skip_knowledge:
        extractor = build_extractor()

    normalized_rels = sorted({rel_path.strip().replace("\\", "/") for rel_path in rel_paths if rel_path.strip()})
    try:
        with connect() as conn:
            with conn.cursor() as cur:
                for collection in config.get("collections", []):
                    if collection_name and collection["name"] != collection_name:
                        continue
                    root = Path(collection["path"]).expanduser().resolve()
                    exclude_globs = collection.get("exclude_globs", [])
                    for normalized_rel in normalized_rels:
                        if _matches_excludes(normalized_rel, exclude_globs):
                            continue
                        path = (root / normalized_rel).resolve()
                        try:
                            path.relative_to(root)
                        except ValueError:
                            continue
                        if not path.is_file():
                            continue
                        stats.documents_scanned += 1
                        _index_markdown_path(
                            cur,
                            collection_id=collection_ids[collection["name"]],
                            root=root,
                            path=path,
                            stats=stats,
                            force=force,
                            extractor=extractor,
                        )
            conn.commit()
    finally:
        if extractor is not None:
            extractor.client.close()
    return stats


def _stamp_for_path(path: Path) -> Dict[str, int]:
    stat = path.stat()
    return {"file_mtime_ns": int(stat.st_mtime_ns), "file_size": int(stat.st_size)}


def _metadata_int(metadata: Any, key: str) -> Optional[int]:
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _datetime_to_ns(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp() * 1_000_000_000)


def _classify_refresh_paths(
    *,
    root: Path,
    current_paths: Dict[str, Path],
    indexed_rows: Sequence[Dict[str, Any]],
) -> RefreshPlan:
    indexed = {str(row["rel_path"]): row for row in indexed_rows}
    new_paths: List[str] = []
    changed_paths: List[str] = []
    unchanged_paths: List[str] = []
    deleted_documents: List[Dict[str, Any]] = []
    stamp_backfills: List[Dict[str, Any]] = []

    for rel_path in sorted(current_paths):
        path = current_paths[rel_path]
        row = indexed.get(rel_path)
        if row is None:
            new_paths.append(rel_path)
            continue
        stamp = _stamp_for_path(path)
        metadata = row.get("metadata_json") or {}
        stored_size = _metadata_int(metadata, "file_size")
        stored_mtime = _metadata_int(metadata, "file_mtime_ns")
        indexed_at_ns = _datetime_to_ns(row.get("updated_at"))
        if stored_mtime is None and stored_size == stamp["file_size"]:
            stamp_backfills.append({"id": row["id"], "rel_path": rel_path})
            continue
        if stored_mtime is None and stored_size is None and indexed_at_ns is not None:
            # Legacy rows may predate stat metadata. If the file mtime is not
            # newer than the DB row, backfill stamps without reading markdown.
            if stamp["file_mtime_ns"] <= indexed_at_ns + 2_000_000_000:
                stamp_backfills.append({"id": row["id"], "rel_path": rel_path})
                continue
        if stored_size == stamp["file_size"] and stored_mtime == stamp["file_mtime_ns"]:
            unchanged_paths.append(rel_path)
            continue
        changed_paths.append(rel_path)

    for rel_path, row in indexed.items():
        if rel_path not in current_paths:
            deleted_documents.append({"id": row["id"], "rel_path": rel_path})

    return RefreshPlan(
        new_paths=new_paths,
        changed_paths=changed_paths,
        unchanged_paths=unchanged_paths,
        deleted_documents=deleted_documents,
        stamp_backfills=stamp_backfills,
    )


def run_refresh_changed_paths(
    base_dir: Optional[Path] = None,
    collection_name: Optional[str] = None,
    skip_knowledge: bool = True,
    limit: Optional[int] = None,
) -> IndexStats:
    collection_ids = upsert_collection_rows(base_dir)
    config = read_config(base_dir)
    stats = IndexStats(collections=len(collection_ids))
    extractor: Optional[KnowledgeExtractor] = None
    if not skip_knowledge:
        extractor = build_extractor()

    try:
        with connect() as conn:
            with conn.cursor() as cur:
                for collection in config.get("collections", []):
                    if collection_name and collection["name"] != collection_name:
                        continue
                    root = Path(collection["path"]).expanduser().resolve()
                    exclude_globs = collection.get("exclude_globs", [])
                    files = sorted(_iter_markdown_files(root, collection.get("pattern", "**/*.md"), exclude_globs))
                    if limit is not None:
                        files = files[:limit]
                    current_paths = {path.relative_to(root).as_posix(): path for path in files}
                    stats.documents_scanned += len(current_paths)

                    cur.execute(
                        """
                        SELECT id, rel_path, metadata_json, updated_at
                        FROM vq_documents
                        WHERE collection_id = %s
                        """,
                        (collection_ids[collection["name"]],),
                    )
                    plan = _classify_refresh_paths(
                        root=root,
                        current_paths=current_paths,
                        indexed_rows=list(cur.fetchall()),
                    )
                    if limit is not None:
                        # A limited refresh only samples part of the filesystem. Treating
                        # documents outside that sample as deleted would be destructive.
                        plan.deleted_documents.clear()

                    for row in plan.stamp_backfills:
                        path = current_paths[row["rel_path"]]
                        cur.execute(
                            """
                            UPDATE vq_documents
                            SET metadata_json = metadata_json || %s::jsonb,
                                updated_at = NOW()
                            WHERE id = %s
                            """,
                            (_json_dumps(_stamp_for_path(path)), row["id"]),
                        )
                    stats.documents_skipped += len(plan.unchanged_paths) + len(plan.stamp_backfills)

                    for rel_path in plan.new_paths + plan.changed_paths:
                        _index_markdown_path(
                            cur,
                            collection_id=collection_ids[collection["name"]],
                            root=root,
                            path=current_paths[rel_path],
                            stats=stats,
                            force=True,
                            extractor=extractor,
                        )

                    for row in plan.deleted_documents:
                        stale_point_ids = _document_qdrant_point_ids(cur, int(row["id"]))
                        if stale_point_ids:
                            stats.qdrant_points_deleted += delete_qdrant_points(stale_point_ids)
                        cur.execute("DELETE FROM vq_documents WHERE id = %s", (row["id"],))
                        stats.documents_deleted += 1
            conn.commit()
    finally:
        if extractor is not None:
            extractor.client.close()
    return stats


def run_index(
    base_dir: Optional[Path] = None,
    collection_name: Optional[str] = None,
    force: bool = False,
    skip_knowledge: bool = False,
    limit: Optional[int] = None,
) -> IndexStats:
    collection_ids = upsert_collection_rows(base_dir)
    config = read_config(base_dir)
    stats = IndexStats(collections=len(collection_ids))
    extractor: Optional[KnowledgeExtractor] = None
    if not skip_knowledge:
        extractor = build_extractor()

    with connect() as conn:
        with conn.cursor() as cur:
            for collection in config.get("collections", []):
                if collection_name and collection["name"] != collection_name:
                    continue
                root = Path(collection["path"]).expanduser().resolve()
                exclude_globs = collection.get("exclude_globs", [])
                files = sorted(_iter_markdown_files(root, collection.get("pattern", "**/*.md"), exclude_globs))
                if limit is not None:
                    files = files[:limit]
                seen_rel_paths = {path.relative_to(root).as_posix() for path in files}
                stats.documents_scanned += len(files)

                for path in files:
                    _index_markdown_path(
                        cur,
                        collection_id=collection_ids[collection["name"]],
                        root=root,
                        path=path,
                        stats=stats,
                        force=force,
                        extractor=extractor,
                    )

                cur.execute(
                    "SELECT id, rel_path FROM vq_documents WHERE collection_id = %s",
                    (collection_ids[collection["name"]],),
                )
                for row in cur.fetchall():
                    if row["rel_path"] not in seen_rel_paths:
                        stale_point_ids = _document_qdrant_point_ids(cur, int(row["id"]))
                        if stale_point_ids:
                            stats.qdrant_points_deleted += delete_qdrant_points(stale_point_ids)
                        cur.execute("DELETE FROM vq_documents WHERE id = %s", (row["id"],))
                        stats.documents_deleted += 1
        conn.commit()

    if extractor is not None:
        extractor.client.close()
    return stats
