from __future__ import annotations

import time
import os
from typing import Any, Dict, List, Optional, Sequence

import httpx

from vaultq.graph_signals import (
    apply_graph_signals,
    compute_floor_threshold,
    graph_signal_floor_ratio,
    graph_signals_enabled,
    graph_signals_top_k,
)
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
from vaultq.session_rerank import adaptive_session_rerank, is_agent_session_path
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


def _has_embedded_vectors() -> bool:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM vq_chunks WHERE version = 1 AND qdrant_point_id IS NOT NULL
                    UNION ALL
                    SELECT 1 FROM vq_knowledge_objects WHERE version = 1 AND qdrant_point_id IS NOT NULL
                    LIMIT 1
                ) AS has_vectors
                """
            )
            row = cur.fetchone() or {}
    return bool(row.get("has_vectors"))


def _tag_points(points: Sequence[Dict[str, Any]], lane: str) -> List[Dict[str, Any]]:
    tagged: List[Dict[str, Any]] = []
    for point in points:
        clone = dict(point)
        payload = dict(clone.get("payload", {}))
        lanes = list(payload.get("retrieval_lanes") or [])
        if lane not in lanes:
            lanes.append(lane)
        payload["retrieval_lanes"] = lanes
        clone["payload"] = payload
        tagged.append(clone)
    return tagged


def _title_search_postgres(query: str, limit: int) -> List[Dict[str, Any]]:
    needle = query.strip()
    if not needle:
        return []
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH q AS (
                    SELECT
                        plainto_tsquery('english', %s) || plainto_tsquery('simple', %s) AS tsq,
                        %s::text AS needle
                )
                SELECT
                    'title:' || d.id::text AS point_id,
                    (
                        ts_rank_cd(
                            setweight(to_tsvector('english', COALESCE(d.title, '')), 'A') ||
                            setweight(to_tsvector('simple', COALESCE(d.rel_path, '')), 'B'),
                            q.tsq
                        )
                        + CASE WHEN LOWER(d.title) = LOWER(q.needle) THEN 2.0 ELSE 0.0 END
                        + CASE WHEN d.title ILIKE '%%' || q.needle || '%%' THEN 1.0 ELSE 0.0 END
                        + CASE WHEN d.rel_path ILIKE '%%' || q.needle || '%%' THEN 0.5 ELSE 0.0 END
                    ) AS score,
                    col.name AS collection_name,
                    d.rel_path,
                    d.title,
                    c.chunk_index,
                    c.heading_path,
                    c.section_title,
                    c.start_line,
                    c.end_line,
                    COALESCE(c.text_content, LEFT(d.markdown_text, 2000)) AS text_content
                FROM vq_documents d
                JOIN vq_collections col ON col.id = d.collection_id
                LEFT JOIN LATERAL (
                    SELECT chunk_index, heading_path, section_title, start_line, end_line, text_content
                    FROM vq_chunks
                    WHERE document_id = d.id AND version = 1
                    ORDER BY chunk_index ASC
                    LIMIT 1
                ) c ON TRUE
                CROSS JOIN q
                WHERE d.status = 'indexed'
                  AND (
                    q.tsq @@ (
                        setweight(to_tsvector('english', COALESCE(d.title, '')), 'A') ||
                        setweight(to_tsvector('simple', COALESCE(d.rel_path, '')), 'B')
                    )
                    OR d.title ILIKE '%%' || q.needle || '%%'
                    OR d.rel_path ILIKE '%%' || q.needle || '%%'
                  )
                ORDER BY score DESC, d.id ASC
                LIMIT %s
                """,
                (query, query, needle, limit),
            )
            rows = list(cur.fetchall())
    return [
        {
            "id": row["point_id"],
            "score": float(row["score"] or 0.0),
            "payload": {
                "record_type": "title",
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
        for row in rows
    ]


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
                existing_lanes = list(fused[point_id]["payload"].get("retrieval_lanes") or [])
                for lane in payload.get("retrieval_lanes") or []:
                    if lane not in existing_lanes:
                        existing_lanes.append(lane)
                fused[point_id]["payload"].update(payload)
                if existing_lanes:
                    fused[point_id]["payload"]["retrieval_lanes"] = existing_lanes
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


def _document_diversity_limit() -> int:
    try:
        value = int((os.getenv("RESULT_MAX_PER_DOC") or "2").strip())
    except Exception:
        return 2
    return max(0, min(10, value))


def _session_results_policy() -> str:
    policy = (os.getenv("SESSION_RESULTS_POLICY") or "penalize").strip().lower()
    if policy in {"exclude", "penalize", "include"}:
        return policy
    return "penalize"


def _session_penalty() -> float:
    try:
        value = float((os.getenv("SESSION_RESULTS_PENALTY") or "0.65").strip())
    except Exception:
        return 0.65
    return max(0.05, min(0.95, value))


def _session_max_results(limit: int) -> int:
    try:
        value = int((os.getenv("SESSION_MAX_RESULTS") or "1").strip())
    except Exception:
        value = 1
    return max(0, min(max(1, int(limit or 1)), value))


def _payload_text(payload: Dict[str, Any]) -> str:
    return str(payload.get("window_text") or payload.get("text") or payload.get("body_text") or payload.get("raw_text") or "")


def _apply_session_policy(
    points: Sequence[Dict[str, Any]],
    *,
    query: str,
    limit: int,
    policy: Optional[str] = None,
    session_penalty: Optional[float] = None,
    session_max_results: Optional[int] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    policy = (policy or _session_results_policy()).strip().lower()
    if policy not in {"exclude", "penalize", "include"}:
        policy = "penalize"
    penalty = max(0.05, min(0.95, float(session_penalty if session_penalty is not None else _session_penalty())))
    max_sessions = (
        max(0, min(max(1, int(limit or 1)), int(session_max_results)))
        if session_max_results is not None
        else _session_max_results(limit)
    )
    meta: Dict[str, Any] = {
        "enabled": True,
        "policy": policy,
        "session_penalty": penalty if policy == "penalize" else None,
        "session_max_results": max_sessions if policy == "penalize" else None,
        "seen": 0,
        "penalized": 0,
        "excluded": 0,
        "capped": 0,
    }
    adjusted: List[Dict[str, Any]] = []
    for point in points:
        clone = dict(point)
        payload = dict(clone.get("payload", {}))
        rel_path = str(payload.get("rel_path") or "").strip()
        if not is_agent_session_path(rel_path):
            adjusted.append(clone)
            continue
        meta["seen"] += 1
        payload["source_class"] = "agent_session"
        if policy == "exclude":
            meta["excluded"] += 1
            continue
        if policy == "penalize":
            original_score = float(clone.get("score") or 0.0)
            session_rerank = adaptive_session_rerank(
                query=query,
                rel_path=rel_path,
                title=str(payload.get("title") or rel_path),
                text=_payload_text(payload),
                evidence=payload,
                session_penalty=penalty,
            )
            effective_score = original_score * float(session_rerank["factor"])
            clone["score"] = effective_score
            payload["session_policy"] = "penalize"
            payload["session_original_score"] = original_score
            payload["session_effective_score"] = effective_score
            payload["session_rerank"] = session_rerank
            meta["penalized"] += 1
        else:
            payload["session_policy"] = "include"
        clone["payload"] = payload
        adjusted.append(clone)

    if policy != "penalize":
        return adjusted, meta

    ordered = sorted(
        adjusted,
        key=lambda point: (
            -float(point.get("score") or 0.0),
            str(point.get("payload", {}).get("rel_path") or point.get("id") or ""),
        ),
    )
    selected: List[Dict[str, Any]] = []
    session_count = 0
    for point in ordered:
        if is_agent_session_path(point.get("payload", {}).get("rel_path")):
            if session_count >= max_sessions:
                meta["capped"] += 1
                continue
            session_count += 1
        selected.append(point)
    return selected, meta


def _diversify_by_document(
    points: Sequence[Dict[str, Any]],
    *,
    limit: int,
    max_per_doc: int,
    fill_deferred: bool = True,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pool = list(points)
    if max_per_doc <= 0 or limit <= 1:
        return pool[:limit], {"enabled": False, "max_per_doc": max_per_doc, "deferred": 0}
    selected: List[Dict[str, Any]] = []
    deferred: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for point in pool:
        rel_path = str(point.get("payload", {}).get("rel_path") or point.get("id") or "")
        count = counts.get(rel_path, 0)
        if count < max_per_doc:
            selected.append(point)
            counts[rel_path] = count + 1
        else:
            deferred.append(point)
    if fill_deferred and len(selected) < limit:
        selected.extend(deferred[: limit - len(selected)])
    output = selected[:limit]
    original_ids = [str(point.get("id") or "") for point in pool[:limit]]
    output_ids = [str(point.get("id") or "") for point in output]
    return output, {
        "enabled": True,
        "max_per_doc": max_per_doc,
        "deferred": len(deferred),
        "applied": output_ids != original_ids,
        "strict": not fill_deferred,
    }


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
        "evidence": {
            "collection_name": payload.get("collection_name") or "",
            "rel_path": payload.get("rel_path") or "",
            "start_line": payload.get("window_start_line") or payload.get("start_line"),
            "end_line": payload.get("window_end_line") or payload.get("end_line"),
            "retrieval_lanes": payload.get("retrieval_lanes") or [],
        },
    }
    if include_metadata:
        metadata: Dict[str, Any] = {}
        for key in (
            "section_title",
            "segment_summary",
            "document_summary",
            "contexts",
            "graph_adjacency_hits",
            "graph_adjacency_boost",
            "graph_session_prefix",
            "graph_session_demoted",
            "graph_session_demote",
            "source_class",
            "session_policy",
            "session_original_score",
            "session_effective_score",
            "session_rerank",
            "window_start_chunk_index",
            "window_end_chunk_index",
            "window_start_line",
            "window_end_line",
            "retrieval_lanes",
        ):
            value = payload.get(key)
            if value not in (None, "", [], {}):
                metadata[key] = value
        result["metadata"] = metadata
    return result


def search(query: str, limit: int = 5, retrieval_mode: str = "hybrid") -> Dict[str, Any]:
    retrieval_mode = (retrieval_mode or "hybrid").strip().lower()
    if retrieval_mode == "balanced":
        retrieval_mode = "hybrid"
    if not query or not query.strip():
        return {"query": query, "retrieval_mode": retrieval_mode, "results": []}

    started = time.perf_counter()
    if retrieval_mode == "keyword":
        keyword_started = time.perf_counter()
        keyword_points = _tag_points(_keyword_search(query, max(limit * 3, 10)), "keyword")
        title_started = time.perf_counter()
        title_points = _tag_points(_title_search_postgres(query, max(limit * 2, 10)), "title")
        fused = _fuse_ranked_lists(title_points, keyword_points, limit=limit)
        fused, session_meta = _apply_session_policy(fused, query=query, limit=limit)
        points = _attach_neighbor_windows(fused)
        result = {
            "query": query,
            "retrieval_mode": retrieval_mode,
            "results": [result_contract(point) for point in points],
            "timings": {
                "keyword_ms": int((title_started - keyword_started) * 1000),
                "title_ms": int((time.perf_counter() - title_started) * 1000),
            },
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
        if session_meta.get("seen"):
            result["session_policy"] = session_meta
        return result
    if retrieval_mode == "title":
        title_started = time.perf_counter()
        raw_points = _tag_points(_title_search_postgres(query, limit), "title")
        raw_points, session_meta = _apply_session_policy(raw_points, query=query, limit=limit)
        points = _attach_neighbor_windows(raw_points)
        result = {
            "query": query,
            "retrieval_mode": retrieval_mode,
            "results": [result_contract(point) for point in points],
            "timings": {"title_ms": int((time.perf_counter() - title_started) * 1000)},
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
        if session_meta.get("seen"):
            result["session_policy"] = session_meta
        return result
    if retrieval_mode == "semantic":
        semantic_started = time.perf_counter()
        raw_points = _tag_points(_semantic_search(query, limit), "semantic")
        raw_points, session_meta = _apply_session_policy(raw_points, query=query, limit=limit)
        points = _attach_neighbor_windows(raw_points)
        result = {
            "query": query,
            "retrieval_mode": retrieval_mode,
            "results": [result_contract(point) for point in points],
            "timings": {"semantic_ms": int((time.perf_counter() - semantic_started) * 1000)},
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
        if session_meta.get("seen"):
            result["session_policy"] = session_meta
        return result

    timings: Dict[str, Any] = {}
    warnings: List[Dict[str, str]] = []
    focused = retrieval_mode == "focused"
    candidate_multiplier = 8 if retrieval_mode == "deep" else 3 if focused else 4
    candidate_floor = 50 if retrieval_mode == "deep" else 12 if focused else 20
    graph_enabled = graph_signals_enabled(retrieval_mode)
    graph_top_k = graph_signals_top_k()

    title_started = time.perf_counter()
    title_points = _tag_points(_title_search_postgres(query, max(limit * candidate_multiplier, candidate_floor)), "title")
    timings["title_ms"] = int((time.perf_counter() - title_started) * 1000)

    keyword_started = time.perf_counter()
    keyword_points = _tag_points(_keyword_search(query, max(limit * candidate_multiplier, candidate_floor)), "keyword")
    timings["keyword_ms"] = int((time.perf_counter() - keyword_started) * 1000)

    if retrieval_mode == "fast":
        semantic_points = []
        fused = _fuse_ranked_lists(title_points, keyword_points, limit=max(limit * 4, 20))
        reranked = fused[:limit]
        strategy = "title_keyword_rrf"
    else:
        semantic_started = time.perf_counter()
        try:
            if not _has_embedded_vectors():
                raise RuntimeError("semantic index has no embedded vectors yet")
            semantic_points = _tag_points(_semantic_search(query, max(limit * candidate_multiplier, candidate_floor)), "semantic")
        except Exception as exc:
            semantic_points = []
            warnings.append(
                {
                    "lane": "semantic",
                    "type": exc.__class__.__name__,
                    "message": str(exc)[:500],
                }
            )
        timings["semantic_ms"] = int((time.perf_counter() - semantic_started) * 1000)
        fused_limit = max(limit * 4, candidate_floor) if focused else max(limit * 6, 30)
        fused = _fuse_ranked_lists(title_points, keyword_points, semantic_points, limit=fused_limit)
        rerank_limit = max(rerank_candidate_limit(), limit * 6) if retrieval_mode == "deep" else rerank_candidate_limit()
        if use_reranker():
            rerank_started = time.perf_counter()
            try:
                rerank_output_limit = max(limit, graph_top_k) if graph_enabled else limit
                reranked = _cross_encoder_rerank(query, fused[:rerank_limit], rerank_output_limit)
            except Exception as exc:
                reranked = fused[: max(limit, graph_top_k) if graph_enabled else limit]
                warnings.append(
                    {
                        "lane": "rerank",
                        "type": exc.__class__.__name__,
                        "message": str(exc)[:500],
                    }
                )
            timings["rerank_ms"] = int((time.perf_counter() - rerank_started) * 1000)
            if focused:
                strategy = "focused_title_hybrid_rrf_rerank"
            else:
                strategy = "title_hybrid_rrf_rerank" if retrieval_mode != "deep" else "deep_title_hybrid_rrf_rerank"
        else:
            if focused:
                reranked = fused[: max(limit * 4, candidate_floor)]
            else:
                reranked = fused[: max(limit, graph_top_k) if graph_enabled else limit]
            if focused:
                strategy = "focused_title_hybrid_rrf"
            else:
                strategy = "title_hybrid_rrf" if retrieval_mode != "deep" else "deep_title_hybrid_rrf"

    graph_meta: Optional[Dict[str, Any]] = None
    if graph_enabled:
        graph_started = time.perf_counter()
        floor_ratio = graph_signal_floor_ratio()
        graph_result = apply_graph_signals(
            reranked,
            enabled=True,
            top_k=graph_top_k,
            floor_threshold=compute_floor_threshold(reranked, floor_ratio),
        )
        reranked = graph_result["results"]
        graph_meta = dict(graph_result["meta"])
        graph_meta["floor_ratio"] = floor_ratio
        timings["graph_signals_ms"] = int((time.perf_counter() - graph_started) * 1000)
        if graph_meta.get("errored"):
            warnings.append(
                {
                    "lane": "graph_signals",
                    "type": str(graph_meta.get("error_type") or "GraphSignalsError"),
                    "message": str(graph_meta.get("error_message") or "")[:500],
                }
            )

    reranked, session_meta = _apply_session_policy(
        reranked,
        query=query,
        limit=max(limit, graph_top_k) if graph_enabled else limit,
    )

    reranked, diversity_meta = _diversify_by_document(
        reranked,
        limit=limit,
        max_per_doc=1 if focused else _document_diversity_limit(),
        fill_deferred=not focused,
    )
    if not focused:
        reranked = _attach_neighbor_windows(reranked)
    result = {
        "query": query,
        "retrieval_mode": retrieval_mode,
        "strategy": strategy,
        "results": [result_contract(point) for point in reranked],
        "title_candidates": len(title_points),
        "keyword_candidates": len(keyword_points),
        "semantic_candidates": len(semantic_points),
        "timings": timings,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }
    if diversity_meta.get("enabled"):
        result["diversity"] = diversity_meta
    if graph_meta:
        result["graph_signals"] = graph_meta
    if session_meta.get("seen"):
        result["session_policy"] = session_meta
    if warnings:
        result["warnings"] = warnings
    return result


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
