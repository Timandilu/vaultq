from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import re
from dataclasses import dataclass
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


def _file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    raw = path.read_text(encoding="utf-8")
    frontmatter, body = _parse_frontmatter(raw)
    title = _derive_title(path, frontmatter, body)
    rel_path = path.relative_to(root).as_posix()
    digest = _file_hash(raw)
    metadata = {
        "frontmatter_keys": sorted(frontmatter.keys()),
        "file_size": path.stat().st_size,
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
            title,
            digest,
            body,
            json.dumps(frontmatter),
            json.dumps(metadata),
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
                chunk.heading_path,
                chunk.section_title,
                chunk.text,
                chunk.token_count,
                len(chunk.text),
                chunk.start_line,
                chunk.end_line,
                json.dumps({"section_title": chunk.section_title, "token_count": chunk.token_count}),
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
                obj["object_type"],
                obj["title"],
                obj["body_text"],
                json.dumps(obj.get("json_payload") or {}),
                float(obj.get("confidence") or 0.0),
                json.dumps(obj.get("source_chunk_ids") or []),
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
            (json.dumps(overlapping), row["id"]),
        )
        remapped += 1
    return remapped


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
                    document = _insert_document(cur, collection_ids[collection["name"]], root, path, force=force)
                    if document is None:
                        stats.documents_skipped += 1
                        continue

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
