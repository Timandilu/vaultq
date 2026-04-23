from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import httpx

from vaultq.retrieval_models import (
    bm25_vector_for_text,
    dense_vector_name,
    encode_hybrid_query,
    neighbor_window_size,
    rerank_candidate_limit,
    rerank_documents,
    sparse_backend,
    sparse_vector_name,
    use_qdrant_sparse_vectors,
    use_reranker,
    use_sparse,
)
from vaultq.store import connect, get_settings, status_snapshot

RRF_K = 60


def _qdrant_query(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    settings = get_settings()
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{settings.qdrant_url}/collections/{settings.qdrant_collection}/points/query",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json().get("result", {}).get("points", [])


def _semantic_search(query: str, limit: int) -> List[Dict[str, Any]]:
    query_vectors = encode_hybrid_query(query)
    dense_vector = query_vectors.get("dense_vector") or []
    if not dense_vector:
        return []
    return _qdrant_query({"query": dense_vector, "using": dense_vector_name(), "limit": limit, "with_payload": True})


def _keyword_search(query: str, limit: int) -> List[Dict[str, Any]]:
    if not use_sparse():
        return []
    if sparse_backend() == "postgres":
        return _keyword_search_postgres(query, limit)
    return _qdrant_query(
        {"query": bm25_vector_for_text(query), "using": sparse_vector_name(), "limit": limit, "with_payload": True}
    )


def _keyword_search_postgres(query: str, limit: int) -> List[Dict[str, Any]]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH q AS (
                    SELECT plainto_tsquery('english', %s) || plainto_tsquery('simple', %s) AS tsq
                )
                SELECT
                    COALESCE(c.qdrant_point_id, 'chunk:' || c.id::text) AS point_id,
                    ts_rank_cd(
                        setweight(to_tsvector('english', COALESCE(c.text_content, '')), 'A') ||
                        setweight(to_tsvector('simple', COALESCE(c.text_content, '')), 'B'),
                        q.tsq
                    ) AS score,
                    col.name AS collection_name,
                    d.rel_path,
                    d.title,
                    c.chunk_index,
                    c.heading_path,
                    c.section_title,
                    c.start_line,
                    c.end_line,
                    c.text_content
                FROM vq_chunks c
                JOIN vq_documents d ON d.id = c.document_id
                JOIN vq_collections col ON col.id = d.collection_id
                CROSS JOIN q
                WHERE c.version = 1
                  AND q.tsq @@ (
                    setweight(to_tsvector('english', COALESCE(c.text_content, '')), 'A') ||
                    setweight(to_tsvector('simple', COALESCE(c.text_content, '')), 'B')
                  )
                ORDER BY score DESC, c.id ASC
                LIMIT %s
                """,
                (query, query, limit),
            )
            chunk_rows = list(cur.fetchall())
            cur.execute(
                """
                WITH q AS (
                    SELECT plainto_tsquery('english', %s) || plainto_tsquery('simple', %s) AS tsq
                )
                SELECT
                    COALESCE(ko.qdrant_point_id, 'knowledge:' || ko.id::text) AS point_id,
                    ts_rank_cd(
                        setweight(to_tsvector('english', COALESCE(ko.body_text, '')), 'A') ||
                        setweight(to_tsvector('simple', COALESCE(ko.body_text, '')), 'B'),
                        q.tsq
                    ) AS score,
                    col.name AS collection_name,
                    d.rel_path,
                    d.title,
                    ko.object_type,
                    ko.title AS knowledge_title,
                    ko.body_text,
                    ko.source_chunk_ids,
                    ko.source_line_start,
                    ko.source_line_end
                FROM vq_knowledge_objects ko
                JOIN vq_documents d ON d.id = ko.document_id
                JOIN vq_collections col ON col.id = d.collection_id
                CROSS JOIN q
                WHERE ko.version = 1
                  AND q.tsq @@ (
                    setweight(to_tsvector('english', COALESCE(ko.body_text, '')), 'A') ||
                    setweight(to_tsvector('simple', COALESCE(ko.body_text, '')), 'B')
                  )
                ORDER BY score DESC, ko.id ASC
                LIMIT %s
                """,
                (query, query, limit),
            )
            knowledge_rows = list(cur.fetchall())

    points: List[Dict[str, Any]] = []
    for row in chunk_rows:
        points.append(
            {
                "id": row["point_id"],
                "score": float(row["score"] or 0.0),
                "payload": {
                    "record_type": "chunk",
                    "doc_type": "markdown_chunk",
                    "object_type": "markdown_chunk",
                    "collection_name": row["collection_name"] or "",
                    "rel_path": row["rel_path"] or "",
                    "title": row["title"] or "",
                    "chunk_index": row["chunk_index"],
                    "heading_path": row["heading_path"] or "",
                    "section_title": row["section_title"] or "",
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                    "text": row["text_content"] or "",
                    "raw_text": row["text_content"] or "",
                },
            }
        )
    for row in knowledge_rows:
        points.append(
            {
                "id": row["point_id"],
                "score": float(row["score"] or 0.0),
                "payload": {
                    "record_type": "knowledge_object",
                    "doc_type": row["object_type"] or "knowledge_object",
                    "object_type": row["object_type"] or "knowledge_object",
                    "collection_name": row["collection_name"] or "",
                    "rel_path": row["rel_path"] or "",
                    "title": row["title"] or "",
                    "knowledge_title": row["knowledge_title"] or "",
                    "start_line": row["source_line_start"],
                    "end_line": row["source_line_end"],
                    "source_chunk_ids": row["source_chunk_ids"] or [],
                    "text": row["body_text"] or "",
                    "raw_text": row["body_text"] or "",
                },
            }
        )
    points.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return points[:limit]


def _cross_encoder_rerank(query: str, candidate_points: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if not use_reranker() or not candidate_points:
        return list(candidate_points[:limit])
    docs = [
        point.get("payload", {}).get("text")
        or point.get("payload", {}).get("body_text")
        or point.get("payload", {}).get("raw_text")
        or ""
        for point in candidate_points
    ]
    ranked = rerank_documents(query, docs, None)
    by_score = {idx: score for idx, score in ranked}
    ordered = sorted(
        enumerate(candidate_points),
        key=lambda item: by_score.get(item[0], float("-inf")),
        reverse=True,
    )
    results: List[Dict[str, Any]] = []
    for idx, point in ordered[:limit]:
        clone = dict(point)
        clone["score"] = float(by_score.get(idx, clone.get("score", 0.0)))
        results.append(clone)
    return results


def _fuse_ranked_lists(*ranked_lists: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    fused: Dict[str, Dict[str, Any]] = {}
    for points in ranked_lists:
        for rank, point in enumerate(points, start=1):
            point_id = str(point.get("id"))
            if point_id not in fused:
                fused[point_id] = {"id": point.get("id"), "score": 0.0, "payload": dict(point.get("payload", {}))}
            fused[point_id]["score"] += 1.0 / (RRF_K + rank)
            payload = point.get("payload", {})
            if payload:
                fused[point_id]["payload"].update(payload)
    return sorted(fused.values(), key=lambda item: item.get("score", 0.0), reverse=True)[:limit]


def _chunk_window(rel_path: str, chunk_index: int, radius: int) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.chunk_index, c.text_content, c.start_line, c.end_line
                FROM vq_chunks c
                JOIN vq_documents d ON d.id = c.document_id
                WHERE d.rel_path = %s
                  AND c.version = 1
                  AND c.chunk_index BETWEEN %s AND %s
                ORDER BY c.chunk_index
                """,
                (rel_path, chunk_index - radius, chunk_index + radius),
            )
            rows = list(cur.fetchall())
    if not rows:
        return None
    return {
        "window_text": "\n".join(
            (row["text_content"] or "").strip() for row in rows if (row["text_content"] or "").strip()
        ),
        "window_start_chunk_index": min(int(row["chunk_index"]) for row in rows),
        "window_end_chunk_index": max(int(row["chunk_index"]) for row in rows),
        "window_start_line": min(int(row["start_line"] or 1) for row in rows),
        "window_end_line": max(int(row["end_line"] or 1) for row in rows),
    }


def _knowledge_window(payload: Dict[str, Any], radius: int) -> Optional[Dict[str, Any]]:
    rel_path = str(payload.get("rel_path") or "").strip()
    if not rel_path:
        return None
    source_chunk_ids = payload.get("source_chunk_ids") or []
    with connect() as conn:
        with conn.cursor() as cur:
            if source_chunk_ids:
                cur.execute(
                    """
                    WITH base AS (
                        SELECT MIN(chunk_index) AS start_idx, MAX(chunk_index) AS end_idx
                        FROM vq_chunks
                        WHERE id = ANY(%s)
                    )
                    SELECT c.chunk_index, c.text_content, c.start_line, c.end_line
                    FROM base
                    JOIN vq_documents d ON d.rel_path = %s
                    JOIN vq_chunks c ON c.document_id = d.id
                    WHERE c.version = 1
                      AND c.chunk_index BETWEEN COALESCE(base.start_idx, 0) - %s AND COALESCE(base.end_idx, 0) + %s
                    ORDER BY c.chunk_index
                    """,
                    (source_chunk_ids, rel_path, radius, radius),
                )
            else:
                cur.execute(
                    """
                    SELECT c.chunk_index, c.text_content, c.start_line, c.end_line
                    FROM vq_chunks c
                    JOIN vq_documents d ON d.id = c.document_id
                    WHERE d.rel_path = %s
                      AND c.version = 1
                      AND c.end_line >= COALESCE(%s, 1)
                      AND c.start_line <= COALESCE(%s, c.end_line)
                    ORDER BY c.chunk_index
                    """,
                    (rel_path, payload.get("start_line"), payload.get("end_line")),
                )
            rows = list(cur.fetchall())
    if not rows:
        return None
    return {
        "window_text": "\n".join(
            (row["text_content"] or "").strip() for row in rows if (row["text_content"] or "").strip()
        ),
        "window_start_chunk_index": min(int(row["chunk_index"]) for row in rows),
        "window_end_chunk_index": max(int(row["chunk_index"]) for row in rows),
        "window_start_line": min(int(row["start_line"] or 1) for row in rows),
        "window_end_line": max(int(row["end_line"] or 1) for row in rows),
    }


def _attach_neighbor_windows(points: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    radius = neighbor_window_size()
    if radius <= 0:
        return list(points)
    enriched: List[Dict[str, Any]] = []
    for point in points:
        payload = dict(point.get("payload", {}))
        doc_type = payload.get("doc_type", payload.get("object_type", "markdown_chunk"))
        if doc_type == "markdown_chunk" and payload.get("rel_path") and payload.get("chunk_index") is not None:
            window = _chunk_window(str(payload.get("rel_path")), int(payload.get("chunk_index")), radius)
        else:
            window = _knowledge_window(payload, radius)
        if window:
            payload.update(window)
        clone = dict(point)
        clone["payload"] = payload
        enriched.append(clone)
    return enriched


def result_contract(point: Dict[str, Any], include_metadata: bool = True) -> Dict[str, Any]:
    payload = dict(point.get("payload", {}))
    result = {
        "id": str(point.get("id") or ""),
        "score": float(point.get("score") or 0.0),
        "record_type": payload.get("record_type") or "",
        "doc_type": payload.get("doc_type") or payload.get("object_type") or "",
        "collection_name": payload.get("collection_name") or "",
        "rel_path": payload.get("rel_path") or "",
        "title": payload.get("title") or "",
        "heading_path": payload.get("heading_path") or "",
        "knowledge_title": payload.get("knowledge_title") or "",
        "chunk_index": payload.get("chunk_index"),
        "start_line": payload.get("start_line"),
        "end_line": payload.get("end_line"),
        "text": payload.get("window_text") or payload.get("text") or payload.get("raw_text") or "",
    }
    if include_metadata:
        metadata: Dict[str, Any] = {}
        for key in (
            "section_title",
            "segment_summary",
            "document_summary",
            "contexts",
            "window_start_chunk_index",
            "window_end_chunk_index",
            "window_start_line",
            "window_end_line",
        ):
            value = payload.get(key)
            if value not in (None, "", [], {}):
                metadata[key] = value
        result["metadata"] = metadata
    return result


def search(query: str, limit: int = 5, retrieval_mode: str = "hybrid") -> Dict[str, Any]:
    retrieval_mode = (retrieval_mode or "hybrid").strip().lower()
    if not query or not query.strip():
        return {"query": query, "retrieval_mode": retrieval_mode, "results": []}

    started = time.perf_counter()
    if retrieval_mode == "keyword":
        keyword_started = time.perf_counter()
        points = _attach_neighbor_windows(_keyword_search(query, limit))
        return {
            "query": query,
            "retrieval_mode": retrieval_mode,
            "results": [result_contract(point) for point in points],
            "timings": {"keyword_ms": int((time.perf_counter() - keyword_started) * 1000)},
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
    if retrieval_mode == "semantic":
        semantic_started = time.perf_counter()
        points = _attach_neighbor_windows(_semantic_search(query, limit))
        return {
            "query": query,
            "retrieval_mode": retrieval_mode,
            "results": [result_contract(point) for point in points],
            "timings": {"semantic_ms": int((time.perf_counter() - semantic_started) * 1000)},
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }

    timings: Dict[str, Any] = {}
    keyword_started = time.perf_counter()
    keyword_points = _keyword_search(query, max(limit * 4, 20))
    timings["keyword_ms"] = int((time.perf_counter() - keyword_started) * 1000)

    semantic_started = time.perf_counter()
    semantic_points = _semantic_search(query, max(limit * 4, 20))
    timings["semantic_ms"] = int((time.perf_counter() - semantic_started) * 1000)

    fused = _fuse_ranked_lists(keyword_points, semantic_points, limit=max(limit * 6, 30))
    if use_reranker():
        rerank_started = time.perf_counter()
        reranked = _cross_encoder_rerank(query, fused[: rerank_candidate_limit()], limit)
        timings["rerank_ms"] = int((time.perf_counter() - rerank_started) * 1000)
        strategy = "hybrid_rrf_rerank"
    else:
        reranked = fused[:limit]
        strategy = "hybrid_rrf"

    reranked = _attach_neighbor_windows(reranked)
    return {
        "query": query,
        "retrieval_mode": "hybrid",
        "strategy": strategy,
        "results": [result_contract(point) for point in reranked],
        "keyword_candidates": len(keyword_points),
        "semantic_candidates": len(semantic_points),
        "timings": timings,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


def fetch(point_id: str) -> Dict[str, Any]:
    settings = get_settings()
    with httpx.Client(timeout=60.0) as client:
        resp = client.get(f"{settings.qdrant_url}/collections/{settings.qdrant_collection}/points/{point_id}")
        if resp.status_code == 404:
            raise KeyError(f"Point not found: {point_id}")
        resp.raise_for_status()
        data = resp.json().get("result", {})
    point = {"id": data.get("id"), "score": data.get("score"), "payload": data.get("payload", {})}
    enriched = _attach_neighbor_windows([point])
    return result_contract(enriched[0] if enriched else point, include_metadata=True)


def get_document(identifier: str, full: bool = False) -> Dict[str, Any]:
    with connect() as conn:
        with conn.cursor() as cur:
            if identifier.startswith("#") and identifier[1:].isdigit():
                cur.execute("SELECT * FROM vq_documents WHERE id = %s", (int(identifier[1:]),))
            else:
                cur.execute("SELECT * FROM vq_documents WHERE rel_path = %s", (identifier,))
            row = cur.fetchone()
            if row is None:
                raise KeyError(f"Document not found: {identifier}")
            result = dict(row)
            if not full:
                result["markdown_text"] = result["markdown_text"][:1200]
            cur.execute(
                """
                SELECT chunk_index, heading_path, start_line, end_line, text_content
                FROM vq_chunks
                WHERE document_id = %s AND version = 1
                ORDER BY chunk_index
                """,
                (row["id"],),
            )
            result["chunks"] = list(cur.fetchall())
            cur.execute(
                """
                SELECT object_type, title, body_text, source_line_start, source_line_end
                FROM vq_knowledge_objects
                WHERE document_id = %s AND version = 1
                ORDER BY id
                """,
                (row["id"],),
            )
            result["knowledge_objects"] = list(cur.fetchall())
            return result


def health() -> Dict[str, Any]:
    return status_snapshot()
