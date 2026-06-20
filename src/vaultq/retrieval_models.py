from __future__ import annotations

import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from vaultq.embedding_provider import resolve_embedding_provider, resolve_rerank_provider

_HTTP_CLIENT: Optional[httpx.Client] = None
_HTTP_LOCK = threading.Lock()


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


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _clean(value: Optional[str]) -> str:
    return (value or "").strip().strip('"').strip("'")


def dense_vector_name() -> str:
    return _clean(os.getenv("DENSE_VECTOR_NAME")) or "dense"


def sparse_vector_name() -> str:
    return _clean(os.getenv("SPARSE_VECTOR_NAME")) or "sparse"


def sparse_backend() -> str:
    backend = _clean(os.getenv("SPARSE_BACKEND")).lower()
    if backend in {"qdrant", "postgres"}:
        return backend
    return "postgres"


def embedding_provider_name() -> str:
    return resolve_embedding_provider().provider


def hybrid_encoder_model_name() -> str:
    return resolve_embedding_provider().model


def reranker_model_name() -> str:
    return resolve_rerank_provider().model


def use_contextualized_chunk_embeddings() -> bool:
    explicit = os.getenv("ENABLE_CONTEXTUAL_EMBEDDINGS")
    if explicit is not None:
        return _env_flag("ENABLE_CONTEXTUAL_EMBEDDINGS", True)
    provider = resolve_embedding_provider()
    return provider.provider in {"atlas", "voyage"} and provider.model.lower().startswith("voyage-context")


def use_sparse() -> bool:
    return _env_flag("ENABLE_SPARSE_RETRIEVAL", True)


def use_qdrant_sparse_vectors() -> bool:
    return use_sparse() and sparse_backend() == "qdrant"


def use_reranker() -> bool:
    return _env_flag("ENABLE_RERANKER", True)


def hybrid_batch_size(default: int = 16) -> int:
    return max(1, _env_int("HYBRID_BATCH_SIZE", _env_int("EMBED_BATCH_SIZE", default)))


def contextual_request_max_groups(default: int = 4) -> int:
    return max(1, _env_int("CONTEXTUAL_REQUEST_MAX_GROUPS", default))


def _inferred_dense_dim(default: int = 1024) -> int:
    model = hybrid_encoder_model_name().lower()
    if model.startswith("voyage-context") or model.startswith("voyage-"):
        return 1024
    if model in {"text-embedding-3-large", "openai/text-embedding-3-large"}:
        return 3072
    if model in {"text-embedding-3-small", "openai/text-embedding-3-small"}:
        return 1536
    return default


def hybrid_dense_dim(default: int = 1024) -> int:
    inferred = _inferred_dense_dim(default)
    return max(8, _env_int("EMBEDDING_DIM", _env_int("DENSE_VECTOR_SIZE", inferred)))


def rerank_candidate_limit(default: int = 25) -> int:
    return max(5, _env_int("RERANK_CANDIDATES", default))


def neighbor_window_size(default: int = 1) -> int:
    return max(0, _env_int("NEIGHBOR_WINDOW_SIZE", default))


def qdrant_sparse_model_name() -> str:
    return _clean(os.getenv("QDRANT_SPARSE_MODEL")) or "qdrant/bm25"


def validate_embedding_configuration() -> None:
    model = hybrid_encoder_model_name().lower()
    dim = hybrid_dense_dim()
    if model.startswith("voyage-context") and dim not in {256, 512, 1024, 2048}:
        raise RuntimeError(
            f"{hybrid_encoder_model_name()} only supports output dimensions 256, 512, 1024, and 2048; got {dim}."
        )


def _qdrant_sparse_options() -> Dict[str, Any]:
    options: Dict[str, Any] = {}
    tokenizer = _clean(os.getenv("QDRANT_BM25_TOKENIZER"))
    language = _clean(os.getenv("QDRANT_BM25_LANGUAGE"))
    ascii_folding = os.getenv("QDRANT_BM25_ASCII_FOLDING")
    if tokenizer:
        options["tokenizer"] = tokenizer
    if language:
        options["language"] = language
    if ascii_folding is not None and ascii_folding.strip():
        options["ascii_folding"] = ascii_folding.strip().lower() in {"1", "true", "yes", "on"}
    return options


def bm25_vector_for_text(text: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "text": text.replace("\x00", "").strip(),
        "model": qdrant_sparse_model_name(),
    }
    options = _qdrant_sparse_options()
    if options:
        payload["options"] = options
    return payload


def _http_timeout() -> float:
    return max(5.0, _env_float("HTTP_TIMEOUT_SECONDS", 120.0))


def _http_max_retries() -> int:
    return max(1, _env_int("HTTP_MAX_RETRIES", 6))


def _retryable_status(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504, 524, 529}


def _get_http_client() -> httpx.Client:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is not None:
        return _HTTP_CLIENT
    with _HTTP_LOCK:
        if _HTTP_CLIENT is None:
            timeout = _http_timeout()
            _HTTP_CLIENT = httpx.Client(
                timeout=httpx.Timeout(timeout, connect=min(timeout, 30.0)),
                limits=httpx.Limits(max_connections=16, max_keepalive_connections=8, keepalive_expiry=60.0),
            )
    return _HTTP_CLIENT


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except Exception:
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _request_json(
    provider,
    *,
    path: str,
    payload: Dict[str, Any],
    purpose: str,
) -> Dict[str, Any]:
    url = f"{provider.base_url.rstrip('/')}{path}"
    headers = provider.headers
    if not headers.get("Authorization"):
        raise RuntimeError(f"No API key configured for {purpose}")
    last_error: Optional[Exception] = None
    for attempt in range(1, _http_max_retries() + 1):
        try:
            response = _get_http_client().post(url, headers=headers, json=payload)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_error = exc
            if attempt >= _http_max_retries():
                break
            time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
            continue
        if response.status_code == 200:
            return response.json()
        message = response.text[:500]
        last_error = RuntimeError(f"{purpose} failed: {response.status_code} {message}")
        if attempt >= _http_max_retries() or not _retryable_status(response.status_code):
            break
        time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{purpose} failed with an unknown error")


def _query_instruction() -> str:
    return (
        _clean(os.getenv("QUERY_EMBED_INSTRUCTION"))
        or "Given a search query, retrieve relevant passages that answer the query"
    )


def _query_embedding_text(query: str) -> str:
    raw = query.replace("\x00", "").strip()
    if not raw:
        return ""
    provider = resolve_embedding_provider()
    if provider.provider in {"atlas", "voyage"}:
        return raw
    if not _env_flag("ENABLE_QUERY_EMBED_INSTRUCTION", True):
        return raw
    return f"Instruct: {_query_instruction()}\nQuery: {raw}"


def _contextualized_embeddings(
    groups: Sequence[Sequence[str]],
    *,
    input_type: Optional[str] = None,
) -> List[List[List[float]]]:
    clean_groups = [[str(text or "") for text in group] for group in groups if group]
    if not clean_groups:
        return []
    validate_embedding_configuration()
    provider = resolve_embedding_provider()
    body: Dict[str, Any] = {
        "model": provider.model,
        "inputs": clean_groups,
        "output_dimension": hybrid_dense_dim(),
    }
    if input_type:
        body["input_type"] = input_type
    data = _request_json(
        provider,
        path="/contextualizedembeddings",
        payload=body,
        purpose="Contextual embedding request",
    )
    rows = data.get("data") or []
    embeddings: List[Optional[List[List[float]]]] = [None] * len(clean_groups)
    for outer_index, row in enumerate(rows):
        group_index = int(row.get("index", outer_index))
        if group_index < 0 or group_index >= len(clean_groups):
            continue
        inner_rows = row.get("data") or []
        group_embeddings: List[Optional[List[float]]] = [None] * len(clean_groups[group_index])
        for inner_index, item in enumerate(inner_rows):
            chunk_index = int(item.get("index", inner_index))
            if chunk_index < 0 or chunk_index >= len(group_embeddings):
                continue
            raw_embedding = item.get("embedding") or []
            group_embeddings[chunk_index] = [_finite_float(value) for value in raw_embedding]
        if any(item is None for item in group_embeddings):
            raise RuntimeError(
                f"Contextual embedding response mismatch for group {group_index}: "
                f"expected {len(group_embeddings)}, received {len(inner_rows)}"
            )
        embeddings[group_index] = [item or [] for item in group_embeddings]
    if any(group is None for group in embeddings):
        raise RuntimeError(
            f"Contextual embedding response group mismatch: expected {len(clean_groups)}, received {len(rows)}"
        )
    return [group or [] for group in embeddings]


def _embed_texts(texts: Sequence[str], *, input_type: Optional[str] = None) -> List[List[float]]:
    if not texts:
        return []
    validate_embedding_configuration()
    provider = resolve_embedding_provider()
    body: Dict[str, Any] = {
        "model": provider.model,
        "input": list(texts),
    }
    if provider.provider in {"atlas", "voyage"} or provider.model.lower().startswith("voyage-"):
        if input_type:
            body["input_type"] = input_type
        body["output_dimension"] = hybrid_dense_dim()
    data = _request_json(
        provider,
        path="/embeddings",
        payload=body,
        purpose="Embedding request",
    )
    rows = data.get("data") or []
    embeddings: List[Optional[List[float]]] = [None] * len(texts)
    for row in rows:
        index = int(row.get("index", -1))
        if index < 0 or index >= len(texts):
            continue
        raw_embedding = row.get("embedding") or []
        embeddings[index] = [_finite_float(value) for value in raw_embedding]
    if any(item is None for item in embeddings):
        raise RuntimeError(
            f"Embedding response count mismatch: expected {len(texts)}, received {len(rows)}"
        )
    return [item or [] for item in embeddings]


def encode_hybrid_document_groups(text_groups: Sequence[Sequence[str]]) -> List[List[Dict[str, Any]]]:
    normalized_groups: List[List[str]] = []
    for texts in text_groups:
        clean_texts = [str(text or "").replace("\x00", "").strip() for text in texts]
        if clean_texts:
            normalized_groups.append(clean_texts)
    if not normalized_groups:
        return []

    if use_contextualized_chunk_embeddings():
        dense_group_vectors = _contextualized_embeddings(normalized_groups, input_type="document")
    else:
        dense_group_vectors: List[List[List[float]]] = []
        for clean_texts in normalized_groups:
            dense_vectors: List[List[float]] = []
            for start in range(0, len(clean_texts), hybrid_batch_size()):
                dense_vectors.extend(
                    _embed_texts(clean_texts[start : start + hybrid_batch_size()], input_type="document")
                )
            dense_group_vectors.append(dense_vectors)

    if len(dense_group_vectors) != len(normalized_groups):
        raise RuntimeError(
            f"Embedding API returned {len(dense_group_vectors)} dense groups for {len(normalized_groups)} text groups."
        )

    outputs: List[List[Dict[str, Any]]] = []
    for group_index, clean_texts in enumerate(normalized_groups):
        dense_vectors = dense_group_vectors[group_index]
        if len(dense_vectors) != len(clean_texts):
            raise RuntimeError(
                f"Embedding API returned {len(dense_vectors)} dense vectors for {len(clean_texts)} texts "
                f"in group {group_index}."
            )
        group_outputs: List[Dict[str, Any]] = []
        for idx, text in enumerate(clean_texts):
            group_outputs.append(
                {
                    "text": text,
                    "dense_vector": dense_vectors[idx],
                    "sparse_vector": bm25_vector_for_text(text) if use_sparse() else {"indices": [], "values": []},
                }
            )
        outputs.append(group_outputs)
    return outputs


def encode_hybrid_documents(texts: Sequence[str]) -> List[Dict[str, Any]]:
    clean_texts = [str(text or "").replace("\x00", "").strip() for text in texts]
    if not clean_texts:
        return []
    return encode_hybrid_document_groups([clean_texts])[0]


def encode_hybrid_query(query: str) -> Dict[str, Any]:
    raw_query = str(query or "").replace("\x00", "").strip()
    if not raw_query:
        return {
            "dense_vector": [],
            "sparse_vector": {"indices": [], "values": []},
        }
    if use_contextualized_chunk_embeddings():
        dense_vector = _contextualized_embeddings([[raw_query]], input_type="query")[0][0]
    else:
        dense_vector = _embed_texts([_query_embedding_text(raw_query)], input_type="query")[0]
    return {
        "dense_vector": dense_vector,
        "sparse_vector": bm25_vector_for_text(raw_query) if use_sparse() else {"indices": [], "values": []},
    }


def rerank_documents(query: str, documents: Sequence[str], top_n: Optional[int] = None) -> List[Tuple[int, float]]:
    if not use_reranker() or not documents:
        return [(idx, 0.0) for idx in range(len(documents))]
    provider = resolve_rerank_provider()
    if not provider.model:
        return [(idx, 0.0) for idx in range(len(documents))]
    payload: Dict[str, Any] = {
        "model": provider.model,
        "query": str(query or ""),
        "documents": list(documents),
    }
    if top_n is not None:
        limit = max(1, min(top_n, len(documents)))
        if provider.provider == "atlas":
            payload["top_k"] = limit
            payload["return_documents"] = False
            payload["truncation"] = True
        else:
            payload["top_n"] = limit
    data = _request_json(
        provider,
        path="/rerank",
        payload=payload,
        purpose="Rerank request",
    )
    results = data.get("results") or data.get("data") or []
    ranked: List[Tuple[int, float]] = []
    for row in results:
        idx = int(row.get("index", -1))
        if 0 <= idx < len(documents):
            ranked.append((idx, _finite_float(row.get("relevance_score"))))
    ranked.sort(key=lambda item: item[1], reverse=True)
    if top_n is not None:
        ranked = ranked[: max(1, top_n)]
    return ranked
