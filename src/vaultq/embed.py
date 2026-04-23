from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

import httpx

from vaultq.retrieval_models import (
    contextual_request_max_groups,
    dense_vector_name,
    encode_hybrid_document_groups,
    encode_hybrid_documents,
    hybrid_batch_size,
    hybrid_dense_dim,
    sparse_vector_name,
    use_contextualized_chunk_embeddings,
    use_qdrant_sparse_vectors,
    use_reranker,
    use_sparse,
)
from vaultq.store import connect, get_settings


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _qdrant_retries() -> int:
    return max(1, _env_int("QDRANT_RETRIES", 4))


def _qdrant_max_upsert_bytes() -> int:
    return max(1024 * 1024, _env_int("QDRANT_MAX_UPSERT_BYTES", 20 * 1024 * 1024))


def _qdrant_max_points_per_upsert() -> int:
    return max(1, _env_int("QDRANT_MAX_POINTS_PER_UPSERT", 400))


def _contextual_request_max_documents() -> int:
    return max(1, _env_int("CONTEXTUAL_REQUEST_MAX_DOCUMENTS", 384))


def _contextual_request_max_tokens() -> int:
    return max(1024, _env_int("CONTEXTUAL_REQUEST_MAX_TOKENS", 100000))


def _embedding_window_tokens() -> int:
    return max(1024, _env_int("CONTEXTUAL_WINDOW_TOKENS", 28000))


def _embedding_window_chunks() -> int:
    return max(1, _env_int("CONTEXTUAL_WINDOW_MAX_CHUNKS", 96))


def _qdrant_connect_timeout() -> float:
    return _env_float("QDRANT_CONNECT_TIMEOUT", 20.0)


def _qdrant_request_timeout() -> float:
    return _env_float("QDRANT_REQUEST_TIMEOUT", 180.0)


def _snippet(text: str, max_words: int = 24) -> str:
    words = (text or "").strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).strip() + " ..."


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _estimate_tokens(text: str) -> int:
    return max(1, len(_clean_text(text).split()))


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
    segment_summary = _clean_text(row.get("segment_summary"))
    if segment_summary:
        lines.append(f"Section summary: {segment_summary}")
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


def _qdrant_client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(_qdrant_request_timeout(), connect=_qdrant_connect_timeout()),
        limits=httpx.Limits(max_connections=8, max_keepalive_connections=4, keepalive_expiry=60.0),
    )


def _verify_qdrant_schema(client: httpx.Client) -> None:
    settings = get_settings()
    dense_name = dense_vector_name()
    sparse_name = sparse_vector_name()
    resp = client.get(f"{settings.qdrant_url}/collections/{settings.qdrant_collection}")
    resp.raise_for_status()
    payload = resp.json().get("result", {}).get("config", {}).get("params", {})
    vectors = payload.get("vectors") or {}
    sparse_vectors = payload.get("sparse_vectors") or {}
    if dense_name not in vectors:
        raise RuntimeError("Qdrant collection is missing named dense vectors")
    dense_size = int(vectors[dense_name].get("size", 0) or 0)
    if dense_size != hybrid_dense_dim():
        raise RuntimeError(f"Qdrant dense vector size mismatch: expected {hybrid_dense_dim()}, found {dense_size}")
    if use_qdrant_sparse_vectors() and sparse_name not in sparse_vectors:
        raise RuntimeError(f"Qdrant collection is missing sparse vector '{sparse_name}'")


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
                c.metadata_json,
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
            COALESCE(seg.json_payload ->> 'summary', seg.body_text, '') AS segment_summary,
            COALESCE(ds.json_payload ->> 'summary', ds.body_text, '') AS document_summary,
            COALESCE(ctx.contexts, '[]'::jsonb) AS contexts
        FROM chunk_base cb
        LEFT JOIN LATERAL (
            SELECT ko.body_text, ko.json_payload
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
            rows = cur.fetchall()

    documents: List[Dict[str, Any]] = []
    for row in rows:
        metadata = row.get("metadata_json") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}
        raw_text = row.get("text_content") or ""
        contextual_text = _build_contextualized_chunk_text(row)
        embed_text = raw_text if use_contextualized_chunk_embeddings() else contextual_text
        payload = {
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
            "segment_summary": row.get("segment_summary") or "",
            "document_summary": row.get("document_summary") or "",
            "contexts": row.get("contexts") or [],
            "text": raw_text,
            "raw_text": raw_text,
            "contextual_text": contextual_text,
            **metadata,
        }
        documents.append(
            {
                "db_kind": "chunks",
                "db_id": int(row["id"]),
                "point_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"vaultq:chunk:{row['id']}")),
                "text": embed_text,
                "raw_text": raw_text,
                "payload": payload,
                "source_key": f"document:{row['document_id']}",
                "source_order": int(row.get("chunk_index") or 0),
                "token_count": int(metadata.get("token_count") or _estimate_tokens(raw_text)),
            }
        )
    return documents


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
            rows = cur.fetchall()

    documents: List[Dict[str, Any]] = []
    for row in rows:
        source_chunk_ids = row.get("source_chunk_ids") or []
        if isinstance(source_chunk_ids, str):
            try:
                source_chunk_ids = json.loads(source_chunk_ids)
            except Exception:
                source_chunk_ids = []
        raw_text = row.get("body_text") or ""
        contextual_text = _build_contextualized_knowledge_text(row)
        embed_text = raw_text if use_contextualized_chunk_embeddings() else contextual_text
        payload = {
            "record_type": "knowledge_object",
            "doc_type": row.get("object_type") or "knowledge_object",
            "object_type": row.get("object_type") or "knowledge_object",
            "collection_name": row.get("collection_name") or "",
            "rel_path": row.get("rel_path") or "",
            "title": row.get("title") or "",
            "knowledge_title": row.get("knowledge_title") or "",
            "body_text": raw_text,
            "source_chunk_ids": source_chunk_ids,
            "start_line": row.get("source_line_start"),
            "end_line": row.get("source_line_end"),
            "text": raw_text,
            "raw_text": raw_text,
            "contextual_text": contextual_text,
        }
        documents.append(
            {
                "db_kind": "knowledge_objects",
                "db_id": int(row["id"]),
                "point_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"vaultq:knowledge:{row['id']}")),
                "text": embed_text,
                "raw_text": raw_text,
                "payload": payload,
                "source_key": f"document:{row['document_id']}",
                "source_order": int(row["id"]),
                "token_count": _estimate_tokens(raw_text),
            }
        )
    return documents


def _fetch_pending_documents(limit: int, kinds: str = "both") -> List[Dict[str, Any]]:
    normalized = (kinds or "both").strip().lower()
    if normalized not in {"both", "chunks", "knowledge"}:
        raise ValueError("kinds must be one of: both, chunks, knowledge")
    if normalized == "chunks":
        return _fetch_pending_chunk_documents(limit)
    if normalized == "knowledge":
        return _fetch_pending_knowledge_documents(limit)
    chunk_limit = max(1, limit // 2)
    knowledge_limit = max(1, limit - chunk_limit)
    chunk_documents = _fetch_pending_chunk_documents(chunk_limit)
    knowledge_documents = _fetch_pending_knowledge_documents(knowledge_limit)
    remaining = max(0, limit - len(chunk_documents) - len(knowledge_documents))
    if remaining > 0:
        if len(chunk_documents) < chunk_limit:
            knowledge_documents.extend(_fetch_pending_knowledge_documents(knowledge_limit + remaining)[len(knowledge_documents) :])
        elif len(knowledge_documents) < knowledge_limit:
            chunk_documents.extend(_fetch_pending_chunk_documents(chunk_limit + remaining)[len(chunk_documents) :])
    return chunk_documents + knowledge_documents


def _batch_points(documents: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    dense_name = dense_vector_name()
    sparse_name = sparse_vector_name()
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_bytes = len(b'{"points":[]}')
    for doc in documents:
        point = {
            "id": doc["point_id"],
            "payload": doc["payload"],
            "vector": {dense_name: doc["dense_vector"]},
        }
        if use_qdrant_sparse_vectors():
            point["vector"][sparse_name] = doc["sparse_vector"]
        point_bytes = len(json.dumps(point, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        should_flush = current and (
            len(current) >= _qdrant_max_points_per_upsert() or current_bytes + point_bytes > _qdrant_max_upsert_bytes()
        )
        if should_flush:
            batches.append(current)
            current = []
            current_bytes = len(b'{"points":[]}')
        current.append(point)
        current_bytes += point_bytes
    if current:
        batches.append(current)
    return batches


def _embedding_batches(documents: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    if not documents:
        return []
    if not use_contextualized_chunk_embeddings():
        batch_size = hybrid_batch_size()
        return [documents[start : start + batch_size] for start in range(0, len(documents), batch_size)]
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_source: Optional[str] = None
    current_tokens = 0
    token_limit = _embedding_window_tokens()
    chunk_limit = _embedding_window_chunks()
    for doc in documents:
        source_key = str(doc.get("source_key") or "")
        doc_tokens = max(1, int(doc.get("token_count") or _estimate_tokens(doc.get("text") or "")))
        should_flush = bool(
            current
            and (
                source_key != current_source
                or current_tokens + doc_tokens > token_limit
                or len(current) >= chunk_limit
            )
        )
        if should_flush:
            batches.append(current)
            current = []
            current_tokens = 0
            current_source = None
        if not current:
            current_source = source_key
        current.append(doc)
        current_tokens += doc_tokens
    if current:
        batches.append(current)
    return batches


def _update_qdrant_point_ids(documents: List[Dict[str, Any]]) -> None:
    chunk_updates = [(row["point_id"], row["db_id"]) for row in documents if row["db_kind"] == "chunks"]
    knowledge_updates = [(row["point_id"], row["db_id"]) for row in documents if row["db_kind"] == "knowledge_objects"]
    with connect() as conn:
        with conn.cursor() as cur:
            if chunk_updates:
                cur.executemany("UPDATE vq_chunks SET qdrant_point_id = %s WHERE id = %s", chunk_updates)
            if knowledge_updates:
                cur.executemany(
                    "UPDATE vq_knowledge_objects SET qdrant_point_id = %s WHERE id = %s",
                    knowledge_updates,
                )
        conn.commit()


def _adaptive_upsert(client: httpx.Client, points: List[Dict[str, Any]]) -> None:
    settings = get_settings()
    resp = client.put(
        f"{settings.qdrant_url}/collections/{settings.qdrant_collection}/points?wait=true",
        json={"points": points},
    )
    if resp.status_code == 200:
        return
    detail = resp.text.lower()
    if len(points) > 1 and (resp.status_code in {400, 413} or "32 mb" in detail or "payload" in detail):
        middle = max(1, len(points) // 2)
        _adaptive_upsert(client, points[:middle])
        _adaptive_upsert(client, points[middle:])
        return
    raise RuntimeError(f"Qdrant upsert failed: {resp.status_code} {resp.text}")


def embed_pending(kinds: str = "both", limit: int = 100) -> Dict[str, int]:
    documents = _fetch_pending_documents(limit=max(1, limit), kinds=kinds)
    if not documents:
        return {"embedded": 0, "chunks": 0, "knowledge_objects": 0}

    embed_batches = _embedding_batches(documents)

    if use_contextualized_chunk_embeddings():
        request_groups: List[List[Dict[str, Any]]] = []
        request_group_docs = 0
        request_group_tokens = 0
        request_max_docs = _contextual_request_max_documents()
        request_max_groups = contextual_request_max_groups()
        request_max_tokens = _contextual_request_max_tokens()

        def flush_contextual_groups() -> int:
            if not request_groups:
                return 0
            encoded_groups = encode_hybrid_document_groups([[doc["text"] for doc in batch] for batch in request_groups])
            if len(encoded_groups) != len(request_groups):
                raise RuntimeError(
                    f"Hybrid encoder returned {len(encoded_groups)} groups for {len(request_groups)} contextual batches"
                )
            processed = 0
            for batch_index, embed_batch in enumerate(request_groups):
                embeddings = encoded_groups[batch_index]
                if len(embeddings) != len(embed_batch):
                    raise RuntimeError(
                        f"Hybrid encoder returned {len(embeddings)} vectors for {len(embed_batch)} documents"
                    )
                for doc_index, doc in enumerate(embed_batch):
                    encoded = embeddings[doc_index]
                    doc["dense_vector"] = encoded["dense_vector"]
                    if use_qdrant_sparse_vectors():
                        doc["sparse_vector"] = encoded["sparse_vector"]
                processed += len(embed_batch)
            return processed

        encoded_total = 0
        for embed_batch in embed_batches:
            batch_tokens = sum(max(1, int(doc.get("token_count") or 1)) for doc in embed_batch)
            should_flush = bool(
                request_groups
                and (
                    len(request_groups) >= request_max_groups
                    or request_group_docs + len(embed_batch) > request_max_docs
                    or request_group_tokens + batch_tokens > request_max_tokens
                )
            )
            if should_flush:
                encoded_total += flush_contextual_groups()
                request_groups = []
                request_group_docs = 0
                request_group_tokens = 0
            request_groups.append(embed_batch)
            request_group_docs += len(embed_batch)
            request_group_tokens += batch_tokens
        if request_groups:
            encoded_total += flush_contextual_groups()
    else:
        encoded_total = 0
        for embed_batch in embed_batches:
            embeddings = encode_hybrid_documents([doc["text"] for doc in embed_batch])
            if len(embeddings) != len(embed_batch):
                raise RuntimeError(
                    f"Hybrid encoder returned {len(embeddings)} vectors for {len(embed_batch)} documents"
                )
            for idx, doc in enumerate(embed_batch):
                encoded = embeddings[idx]
                doc["dense_vector"] = encoded["dense_vector"]
                if use_qdrant_sparse_vectors():
                    doc["sparse_vector"] = encoded["sparse_vector"]
            encoded_total += len(embed_batch)

    if encoded_total != len(documents):
        raise RuntimeError(f"Hybrid encoder filled {encoded_total} documents for {len(documents)} pending documents")

    totals = {"embedded": 0, "chunks": 0, "knowledge_objects": 0}
    with _qdrant_client() as client:
        _verify_qdrant_schema(client)
        batch_start = 0
        for point_batch in _batch_points(documents):
            document_batch = documents[batch_start : batch_start + len(point_batch)]
            batch_start += len(point_batch)
            last_error: Optional[Exception] = None
            for attempt in range(1, _qdrant_retries() + 1):
                try:
                    _adaptive_upsert(client, point_batch)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt >= _qdrant_retries():
                        break
                    time.sleep(min(2 * attempt, 10))
            if last_error is not None:
                raise RuntimeError(f"Qdrant upsert failed: {last_error}") from last_error
            _update_qdrant_point_ids(document_batch)
            totals["embedded"] += len(document_batch)
            totals["chunks"] += sum(1 for doc in document_batch if doc["db_kind"] == "chunks")
            totals["knowledge_objects"] += sum(1 for doc in document_batch if doc["db_kind"] == "knowledge_objects")
    return totals
