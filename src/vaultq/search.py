from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import httpx

from vaultq.embed import _embed_dense_texts
from vaultq.retrieval_models import (
    embed_late_query,
    late_candidate_limit,
    late_vector_name,
    neighbor_window_size,
    rerank_candidate_limit,
    rerank_documents,
    use_multivector,
    use_reranker,
)
from vaultq.store import connect, get_settings

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
    embedding = _embed_dense_texts([query])[0]
    return _qdrant_query({"query": embedding, "using": "dense", "limit": limit, "with_payload": True})


def _keyword_search(query: str, limit: int) -> List[Dict[str, Any]]:
    return _qdrant_query(
        {"query": {"text": query, "model": "qdrant/bm25"}, "using": "bm25", "limit": limit, "with_payload": True}
    )


def _late_interaction_rerank(query: str, candidate_points: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if not use_multivector() or not candidate_points:
        return list(candidate_points[:limit])
    candidate_ids = [point.get("id") for point in candidate_points if point.get("id")]
    if not candidate_ids:
        return list(candidate_points[:limit])
    late_query = embed_late_query(query)
    if not late_query:
        return list(candidate_points[:limit])
    return _qdrant_query(
        {
            "filter": {"must": [{"has_id": candidate_ids}]},
            "query": late_query,
            "using": late_vector_name(),
            "limit": min(limit, len(candidate_ids)),
            "with_payload": True,
        }
    )


def _cross_encoder_rerank(query: str, candidate_points: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if not use_reranker() or not candidate_points:
        return list(candidate_points[:limit])
    docs = [point.get("payload", {}).get("text") or point.get("payload", {}).get("body_text") or "" for point in candidate_points]
    ranked = rerank_documents(query, docs, None)
    by_score = {idx: score for idx, score in ranked}
    ordered = sorted(enumerate(candidate_points), key=lambda item: by_score.get(item[0], float("-inf")), reverse=True)
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
        "window_text": "\n".join((row["text_content"] or "").strip() for row in rows if (row["text_content"] or "").strip()),
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
        "window_text": "\n".join((row["text_content"] or "").strip() for row in rows if (row["text_content"] or "").strip()),
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
        window = None
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


def search(query: str, limit: int = 5, retrieval_mode: str = "hybrid") -> Dict[str, Any]:
    retrieval_mode = (retrieval_mode or "hybrid").strip().lower()
    if retrieval_mode == "keyword":
        points = _attach_neighbor_windows(_keyword_search(query, limit))
        return {"query": query, "retrieval_mode": retrieval_mode, "results": points}
    if retrieval_mode == "semantic":
        points = _attach_neighbor_windows(_semantic_search(query, limit))
        return {"query": query, "retrieval_mode": retrieval_mode, "results": points}

    keyword_points = _keyword_search(query, max(limit * 4, 20))
    semantic_points = _semantic_search(query, max(limit * 4, 20))
    fused = _fuse_ranked_lists(keyword_points, semantic_points, limit=max(limit * 8, late_candidate_limit()))
    late_ranked = _late_interaction_rerank(query, fused, limit=min(len(fused), late_candidate_limit()))
    reranked = _cross_encoder_rerank(query, late_ranked[: rerank_candidate_limit()], limit)
    reranked = _attach_neighbor_windows(reranked)
    return {
        "query": query,
        "retrieval_mode": "hybrid",
        "results": reranked,
        "keyword_candidates": len(keyword_points),
        "semantic_candidates": len(semantic_points),
    }


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
