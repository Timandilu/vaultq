from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

import httpx

from vaultq.embedding_provider import resolve_embedding_provider
from vaultq.retrieval_models import embed_late_documents, late_interaction_dim, late_vector_name, use_multivector
from vaultq.store import connect, get_settings


def _snippet(text: str, max_words: int = 24) -> str:
    words = (text or "").strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).strip() + " ..."


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _build_contextualized_chunk_text(row: Dict[str, Any]) -> str:
    contexts = row.get("contexts") or []
    if isinstance(contexts, str):
        try:
            contexts = json.loads(contexts)
        except Exception:
            contexts = []
    lines = [
        f"Collection: {_clean_text(row.get('collection_name'))}",
        f"Relative path: {_clean_text(row.get('rel_path'))}",
        f"Document title: {_clean_text(row.get('title'))}",
    ]
    heading_path = _clean_text(row.get("heading_path"))
    if heading_path:
        lines.append(f"Heading path: {heading_path}")
    section_summary = _clean_text(row.get("segment_summary"))
    if section_summary:
        lines.append(f"Section summary: {section_summary}")
    document_summary = _clean_text(row.get("document_summary"))
    if document_summary:
        lines.append(f"Document summary: {document_summary}")
    if contexts:
        lines.append("Context: " + " | ".join(_clean_text(value) for value in contexts if _clean_text(value)))
    prev_text = _snippet(_clean_text(row.get("prev_text")), max_words=18)
    next_text = _snippet(_clean_text(row.get("next_text")), max_words=18)
    if prev_text:
        lines.append(f"Previous context: {prev_text}")
    if next_text:
        lines.append(f"Next context: {next_text}")
    lines.append(f"Passage: {_clean_text(row.get('text_content'))}")
    return "\n".join(lines).strip()


def _build_contextualized_knowledge_text(row: Dict[str, Any]) -> str:
    lines = [
        f"Collection: {_clean_text(row.get('collection_name'))}",
        f"Relative path: {_clean_text(row.get('rel_path'))}",
        f"Document title: {_clean_text(row.get('title'))}",
        f"Knowledge object type: {_clean_text(row.get('object_type'))}",
        f"Object title: {_clean_text(row.get('knowledge_title'))}",
    ]
    document_summary = _clean_text(row.get("document_summary"))
    if document_summary:
        lines.append(f"Document summary: {document_summary}")
    lines.append(f"Knowledge text: {_clean_text(row.get('body_text'))}")
    return "\n".join(lines).strip()


def _verify_qdrant_schema(client: httpx.Client) -> None:
    settings = get_settings()
    resp = client.get(f"{settings.qdrant_url}/collections/{settings.qdrant_collection}")
    resp.raise_for_status()
    payload = resp.json().get("result", {}).get("config", {}).get("params", {})
    vectors = payload.get("vectors") or {}
    sparse_vectors = payload.get("sparse_vectors") or {}
    if "dense" not in vectors:
        raise RuntimeError("Qdrant collection is missing named dense vectors")
    if int(vectors["dense"].get("size", 0) or 0) != settings.embedding_dim:
        raise RuntimeError("Qdrant dense vector size mismatch")
    if use_multivector():
        late_name = late_vector_name()
        if late_name not in vectors:
            raise RuntimeError(f"Qdrant collection is missing late vector '{late_name}'")
        expected = late_interaction_dim()
        if expected > 0 and int(vectors[late_name].get("size", 0) or 0) != expected:
            raise RuntimeError("Qdrant late vector size mismatch")
    if "bm25" not in sparse_vectors:
        raise RuntimeError("Qdrant collection is missing sparse vector 'bm25'")


def _embed_dense_texts(texts: Sequence[str]) -> List[List[float]]:
    provider = resolve_embedding_provider()
    if not provider.api_key:
        raise RuntimeError("Embedding API key is not configured")
    payload = {"model": provider.model, "input": list(texts)}
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(f"{provider.base_url}/embeddings", headers=provider.headers, json=payload)
        resp.raise_for_status()
        rows = resp.json().get("data", [])
        return [[float(value) for value in row.get("embedding", [])] for row in rows]


def _fetch_pending_chunk_documents(limit: int) -> List[Dict[str, Any]]:
    query = """
        WITH chunk_base AS (
            SELECT
                c.id,
                c.document_id,
                c.chunk_index,
                c.heading_path,
                c.section_title,
                c.text_content,
                c.start_line,
                c.end_line,
                d.rel_path,
                d.title,
                col.name AS collection_name,
                LAG(c.text_content) OVER (PARTITION BY c.document_id ORDER BY c.chunk_index) AS prev_text,
                LEAD(c.text_content) OVER (PARTITION BY c.document_id ORDER BY c.chunk_index) AS next_text
            FROM vq_chunks c
            JOIN vq_documents d ON d.id = c.document_id
            JOIN vq_collections col ON col.id = d.collection_id
            WHERE c.version = 1
              AND c.qdrant_point_id IS NULL
            ORDER BY c.document_id, c.chunk_index
            LIMIT %s
        )
        SELECT
            cb.*,
            seg.title AS segment_title,
            COALESCE(seg.json_payload ->> 'summary', seg.body_text, '') AS segment_summary,
            COALESCE(ds.json_payload ->> 'summary', ds.body_text, '') AS document_summary,
            COALESCE(ctx.contexts, '[]'::jsonb) AS contexts
        FROM chunk_base cb
        LEFT JOIN LATERAL (
            SELECT ko.title, ko.body_text, ko.json_payload
            FROM vq_knowledge_objects ko
            WHERE ko.document_id = cb.document_id
              AND ko.version = 1
              AND ko.object_type = 'segment_summary'
              AND COALESCE(ko.source_line_start, cb.start_line) <= cb.end_line
              AND COALESCE(ko.source_line_end, cb.end_line) >= cb.start_line
            ORDER BY ko.id ASC
            LIMIT 1
        ) seg ON TRUE
        LEFT JOIN LATERAL (
            SELECT ko.body_text, ko.json_payload
            FROM vq_knowledge_objects ko
            WHERE ko.document_id = cb.document_id
              AND ko.version = 1
              AND ko.object_type = 'document_summary'
            ORDER BY ko.id DESC
            LIMIT 1
        ) ds ON TRUE
        LEFT JOIN LATERAL (
            SELECT jsonb_agg(context_text ORDER BY length(path_prefix) DESC, id ASC) AS contexts
            FROM vq_contexts
            WHERE collection_id = (
                SELECT d2.collection_id FROM vq_documents d2 WHERE d2.id = cb.document_id
            )
              AND (path_prefix = '' OR cb.rel_path = path_prefix OR cb.rel_path LIKE path_prefix || '/%%')
        ) ctx ON TRUE
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (limit,))
            return list(cur.fetchall())


def _fetch_pending_knowledge_documents(limit: int) -> List[Dict[str, Any]]:
    query = """
        SELECT
            ko.id,
            ko.document_id,
            ko.object_type,
            ko.title AS knowledge_title,
            ko.body_text,
            ko.json_payload,
            ko.source_chunk_ids,
            ko.source_line_start,
            ko.source_line_end,
            d.rel_path,
            d.title,
            col.name AS collection_name,
            COALESCE(ds.json_payload ->> 'summary', ds.body_text, '') AS document_summary
        FROM vq_knowledge_objects ko
        JOIN vq_documents d ON d.id = ko.document_id
        JOIN vq_collections col ON col.id = d.collection_id
        LEFT JOIN LATERAL (
            SELECT ko2.body_text, ko2.json_payload
            FROM vq_knowledge_objects ko2
            WHERE ko2.document_id = d.id
              AND ko2.version = 1
              AND ko2.object_type = 'document_summary'
            ORDER BY ko2.id DESC
            LIMIT 1
        ) ds ON TRUE
        WHERE ko.version = 1
          AND ko.qdrant_point_id IS NULL
        ORDER BY ko.document_id, ko.id
        LIMIT %s
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (limit,))
            return list(cur.fetchall())


def _update_point_ids(kind: str, ids: Sequence[int], point_ids: Sequence[str]) -> None:
    table = "vq_chunks" if kind == "chunks" else "vq_knowledge_objects"
    with connect() as conn:
        with conn.cursor() as cur:
            for row_id, point_id in zip(ids, point_ids):
                cur.execute(f"UPDATE {table} SET qdrant_point_id = %s, updated_at = NOW() WHERE id = %s", (point_id, row_id))
        conn.commit()


def _upsert_points(kind: str, rows: Sequence[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    settings = get_settings()
    dense_vectors = _embed_dense_texts([row["text"] for row in rows])
    late_vectors = embed_late_documents([row["text"] for row in rows]) if use_multivector() else []
    points = []
    ids: List[int] = []
    point_ids: List[str] = []
    for idx, row in enumerate(rows):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{kind}-{row['db_id']}"))
        vector_payload: Dict[str, Any] = {
            "dense": dense_vectors[idx],
            "bm25": {"text": row["text"], "model": "qdrant/bm25"},
        }
        if use_multivector():
            vector_payload[late_vector_name()] = late_vectors[idx]
        point = {
            "id": point_id,
            "vector": vector_payload,
            "payload": row["payload"],
        }
        points.append(point)
        ids.append(int(row["db_id"]))
        point_ids.append(point_id)
    with httpx.Client(timeout=180.0) as client:
        _verify_qdrant_schema(client)
        resp = client.put(
            f"{settings.qdrant_url}/collections/{settings.qdrant_collection}/points",
            json={"points": points},
        )
        resp.raise_for_status()
    _update_point_ids(kind, ids, point_ids)
    return len(points)


def embed_pending(kinds: str = "both", limit: int = 100) -> Dict[str, int]:
    chunk_rows: List[Dict[str, Any]] = []
    knowledge_rows: List[Dict[str, Any]] = []
    if kinds in {"chunks", "both"}:
        for row in _fetch_pending_chunk_documents(limit):
            raw_text = row.get("text_content") or ""
            chunk_rows.append(
                {
                    "db_id": int(row["id"]),
                    "text": _build_contextualized_chunk_text(row),
                    "payload": {
                        "record_type": "chunk",
                        "doc_type": "markdown_chunk",
                        "object_type": "markdown_chunk",
                        "collection_name": row.get("collection_name") or "",
                        "rel_path": row.get("rel_path") or "",
                        "title": row.get("title") or "",
                        "chunk_index": row.get("chunk_index"),
                        "heading_path": row.get("heading_path") or "",
                        "section_title": row.get("section_title") or "",
                        "start_line": row.get("start_line"),
                        "end_line": row.get("end_line"),
                        "raw_text": raw_text,
                        "text": _build_contextualized_chunk_text(row),
                    },
                }
            )
    if kinds in {"knowledge", "both"}:
        for row in _fetch_pending_knowledge_documents(limit):
            knowledge_rows.append(
                {
                    "db_id": int(row["id"]),
                    "text": _build_contextualized_knowledge_text(row),
                    "payload": {
                        "record_type": "knowledge_object",
                        "doc_type": row.get("object_type") or "knowledge_object",
                        "object_type": row.get("object_type") or "knowledge_object",
                        "collection_name": row.get("collection_name") or "",
                        "rel_path": row.get("rel_path") or "",
                        "title": row.get("title") or "",
                        "knowledge_title": row.get("knowledge_title") or "",
                        "body_text": row.get("body_text") or "",
                        "source_chunk_ids": row.get("source_chunk_ids") or [],
                        "start_line": row.get("source_line_start"),
                        "end_line": row.get("source_line_end"),
                        "text": _build_contextualized_knowledge_text(row),
                    },
                }
            )
    return {
        "chunks_embedded": _upsert_points("chunks", chunk_rows) if chunk_rows else 0,
        "knowledge_embedded": _upsert_points("knowledge", knowledge_rows) if knowledge_rows else 0,
    }
