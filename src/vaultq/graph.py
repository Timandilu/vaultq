from __future__ import annotations

import json
import fnmatch
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from vaultq.link_extraction import ExtractedLink, build_basename_index, extract_links, query_basename_index
from vaultq.store import connect, ensure_schema, read_config

MAX_LINK_FIELD_CHARS = 300
MAX_LINK_KEY_BYTES = 1600


def _link_is_indexable(link) -> bool:
    fields = (
        link.target_identifier,
        link.edge_type,
        link.anchor_text,
        link.source_kind,
    )
    if any(len(str(value or "")) > MAX_LINK_FIELD_CHARS for value in fields):
        return False
    key = "\x1f".join(str(value or "") for value in fields)
    return len(key.encode("utf-8")) <= MAX_LINK_KEY_BYTES


def _matches_excludes(rel_path: str, patterns: list[str]) -> bool:
    normalized = rel_path.replace("\\", "/")
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in patterns)


def _identifier_variants(identifier: str) -> list[str]:
    normalized = (identifier or "").strip().replace("\\", "/").strip("/")
    if not normalized:
        return []
    variants = [normalized]
    if normalized.endswith(".md"):
        variants.append(normalized[:-3])
    else:
        variants.append(f"{normalized}.md")
    return list(dict.fromkeys(variants))


def _iter_markdown_paths(root: Path, pattern: str, exclude_globs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        rel_path = path.relative_to(root).as_posix()
        if _matches_excludes(rel_path, exclude_globs):
            continue
        paths.append(path)
    return paths


def _basename_targets(paths: Iterable[Path], root: Path) -> list[str]:
    targets: list[str] = []
    for path in paths:
        rel_path = path.relative_to(root).as_posix()
        targets.append(rel_path[:-3] if rel_path.endswith(".md") else rel_path)
    return targets


def _expanded_link_targets(link: ExtractedLink, basename_index: Dict[str, list[str]]) -> list[tuple[str, str]]:
    if link.source_kind == "wikilink" and "/" not in link.target_identifier:
        resolved = query_basename_index(basename_index, link.target_identifier, limit=10)
        if resolved:
            return [(target, "wikilink_basename") for target in resolved]
    return [(link.target_identifier, link.edge_type)]


def _edge_key(row: Dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("source_rel_path") or ""),
        str(row.get("target_identifier") or ""),
        str(row.get("edge_type") or ""),
    )


def _collapse_edges(rows: Iterable[Dict[str, Any]], *, depth: int) -> list[Dict[str, Any]]:
    collapsed: dict[tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        key = _edge_key(row)
        edge = collapsed.get(key)
        if edge is None:
            edge = dict(row)
            edge["depth"] = depth
            edge["provenance_count"] = 0
            edge["source_kinds"] = []
            edge["anchor_texts"] = []
            collapsed[key] = edge
        edge["provenance_count"] += 1
        source_kind = str(row.get("source_kind") or "").strip()
        if source_kind and source_kind not in edge["source_kinds"]:
            edge["source_kinds"].append(source_kind)
        anchor_text = str(row.get("anchor_text") or "").strip()
        if anchor_text and anchor_text not in edge["anchor_texts"]:
            edge["anchor_texts"].append(anchor_text)
        edge["confidence"] = max(float(edge.get("confidence") or 0), float(row.get("confidence") or 0))
    return sorted(
        collapsed.values(),
        key=lambda row: (-float(row.get("confidence") or 0), str(row.get("target_identifier") or "")),
    )


def _fetch_edges_for_node(
    cur,
    node: str,
    *,
    collection_name: Optional[str],
    direction: str,
    edge_type: Optional[str],
    depth: int,
) -> list[Dict[str, Any]]:
    variants = _identifier_variants(node)
    if not variants:
        return []
    clauses = []
    params: list[Any] = []
    if direction in ("out", "both"):
        clauses.append("source_rel_path = ANY(%s)")
        params.append(variants)
    if direction in ("in", "both"):
        clauses.append("target_identifier = ANY(%s)")
        params.append(variants)
    where = [f"({' OR '.join(clauses)})"]
    if collection_name:
        where.append("collection_name = %s")
        params.append(collection_name)
    if edge_type:
        where.append("edge_type = %s")
        params.append(edge_type)
    cur.execute(
        f"""
        SELECT *
        FROM vq_links
        WHERE {' AND '.join(where)}
        ORDER BY confidence DESC, updated_at DESC
        LIMIT 500
        """,
        params,
    )
    return _collapse_edges([dict(row) for row in cur.fetchall()], depth=depth)


def extract_graph(collection_name: Optional[str] = None) -> Dict[str, Any]:
    ensure_schema()
    config = read_config()
    stats = {
        "collections": 0,
        "documents": 0,
        "links": 0,
        "links_skipped_too_large": 0,
        "basename_resolved": 0,
        "links_deleted_stale_sources": 0,
    }
    with connect() as conn:
        with conn.cursor() as cur:
            for collection in config.get("collections", []):
                name = collection["name"]
                if collection_name and name != collection_name:
                    continue
                stats["collections"] += 1
                root = Path(collection["path"]).expanduser().resolve()
                pattern = collection.get("pattern", "**/*.md")
                exclude_globs = collection.get("exclude_globs", [])
                markdown_paths = _iter_markdown_paths(root, pattern, exclude_globs)
                seen_rel_paths = [path.relative_to(root).as_posix() for path in markdown_paths]
                basename_index = build_basename_index(_basename_targets(markdown_paths, root))
                for path in markdown_paths:
                    rel_path = path.relative_to(root).as_posix()
                    text = path.read_text(encoding="utf-8").replace("\x00", "\uFFFD")
                    links = extract_links(text)
                    stats["documents"] += 1
                    cur.execute(
                        "UPDATE vq_links SET stale = TRUE WHERE collection_name = %s AND source_rel_path = %s",
                        (name, rel_path),
                    )
                    for link in links:
                        if not _link_is_indexable(link):
                            stats["links_skipped_too_large"] += 1
                            continue
                        expanded_targets = _expanded_link_targets(link, basename_index)
                        if len(expanded_targets) > 1 or expanded_targets[0][0] != link.target_identifier:
                            stats["basename_resolved"] += 1
                        for target_identifier, edge_type in expanded_targets:
                            cur.execute(
                                """
                                INSERT INTO vq_links (
                                    collection_name, source_rel_path, target_identifier, edge_type,
                                    anchor_text, source_kind, confidence, metadata_json, stale, updated_at
                                )
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, FALSE, NOW())
                                ON CONFLICT (collection_name, source_rel_path, target_identifier, edge_type, anchor_text, source_kind)
                                DO UPDATE SET confidence = EXCLUDED.confidence,
                                              metadata_json = EXCLUDED.metadata_json,
                                              stale = FALSE,
                                              updated_at = NOW()
                                """,
                                (
                                    name,
                                    rel_path,
                                    target_identifier,
                                    edge_type,
                                    link.anchor_text,
                                    link.source_kind,
                                    link.confidence,
                                    json.dumps({"resolved_from": link.target_identifier} if target_identifier != link.target_identifier else {}),
                                ),
                            )
                            stats["links"] += 1
                    cur.execute(
                        "DELETE FROM vq_links WHERE collection_name = %s AND source_rel_path = %s AND stale = TRUE",
                        (name, rel_path),
                    )
                if seen_rel_paths:
                    cur.execute(
                        """
                        DELETE FROM vq_links
                        WHERE collection_name = %s
                          AND NOT (source_rel_path = ANY(%s))
                        """,
                        (name, seen_rel_paths),
                    )
                    stats["links_deleted_stale_sources"] += max(0, int(cur.rowcount or 0))
        conn.commit()
    return stats


def backlinks(identifier: str, *, collection_name: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    ensure_schema()
    target = (identifier or "").strip().replace("\\", "/")
    if target.endswith(".md"):
        target = target[:-3]
    with connect() as conn:
        with conn.cursor() as cur:
            if collection_name:
                cur.execute(
                    """
                    SELECT * FROM vq_links
                    WHERE collection_name = %s AND target_identifier = %s
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    (collection_name, target, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM vq_links
                    WHERE target_identifier = %s
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    (target, limit),
                )
            rows = [dict(row) for row in cur.fetchall()]
    return {"identifier": identifier, "normalized_identifier": target, "results": rows}


def neighbors(identifier: str, *, collection_name: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    ensure_schema()
    source = (identifier or "").strip().replace("\\", "/")
    with connect() as conn:
        with conn.cursor() as cur:
            if collection_name:
                cur.execute(
                    """
                    SELECT * FROM vq_links
                    WHERE collection_name = %s AND source_rel_path = %s
                    ORDER BY confidence DESC, updated_at DESC
                    LIMIT %s
                    """,
                    (collection_name, source, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM vq_links
                    WHERE source_rel_path = %s
                    ORDER BY confidence DESC, updated_at DESC
                    LIMIT %s
                    """,
                    (source, limit),
                )
            rows = [dict(row) for row in cur.fetchall()]
    return {"identifier": identifier, "results": rows}


def traverse(
    identifier: str,
    *,
    collection_name: Optional[str] = None,
    depth: int = 2,
    direction: str = "out",
    edge_type: Optional[str] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    ensure_schema()
    start = (identifier or "").strip().replace("\\", "/").strip("/")
    if not start:
        raise ValueError("identifier is required")
    direction = (direction or "out").strip().lower()
    if direction not in {"out", "in", "both"}:
        raise ValueError("direction must be one of: out, in, both")
    max_depth = max(1, min(5, int(depth or 1)))
    max_edges = max(1, min(1000, int(limit or 200)))
    edge_type = (edge_type or "").strip() or None
    queue = deque([(start, 0)])
    visited_nodes = {start}
    seen_edges: set[tuple[str, str, str]] = set()
    edges: list[Dict[str, Any]] = []
    with connect() as conn:
        with conn.cursor() as cur:
            while queue and len(edges) < max_edges:
                node, current_depth = queue.popleft()
                if current_depth >= max_depth:
                    continue
                next_depth = current_depth + 1
                for edge in _fetch_edges_for_node(
                    cur,
                    node,
                    collection_name=collection_name,
                    direction=direction,
                    edge_type=edge_type,
                    depth=next_depth,
                ):
                    key = _edge_key(edge)
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    edges.append(edge)
                    if len(edges) >= max_edges:
                        break
                    next_node = (
                        str(edge.get("source_rel_path") or "")
                        if direction == "in"
                        else str(edge.get("target_identifier") or "")
                    )
                    if direction == "both":
                        variants = set(_identifier_variants(node))
                        if str(edge.get("source_rel_path") or "") in variants:
                            next_node = str(edge.get("target_identifier") or "")
                        else:
                            next_node = str(edge.get("source_rel_path") or "")
                    if next_node and next_node not in visited_nodes:
                        visited_nodes.add(next_node)
                        queue.append((next_node, next_depth))
    return {
        "identifier": identifier,
        "normalized_identifier": start,
        "depth": max_depth,
        "direction": direction,
        "edge_type": edge_type,
        "edge_count": len(edges),
        "nodes_seen": sorted(visited_nodes),
        "results": edges,
        "borrowed_pattern": "GBrain-style typed traversal with direction filters and presentation-level edge dedupe while preserving provenance counts.",
    }
