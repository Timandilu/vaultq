from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

_LATE_MODEL = None
_RERANKER = None
_MODEL_LOCK = threading.Lock()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def late_vector_name() -> str:
    return (os.getenv("LATE_VECTOR_NAME") or "late").strip() or "late"


def late_interaction_model_name() -> str:
    return (os.getenv("LATE_INTERACTION_MODEL") or "colbert-ir/colbertv2.0").strip()


def reranker_model_name() -> str:
    return (os.getenv("RERANKER_MODEL") or "Xenova/ms-marco-MiniLM-L-12-v2").strip()


def use_multivector() -> bool:
    return _env_flag("ENABLE_LATE_INTERACTION", True)


def use_reranker() -> bool:
    return _env_flag("ENABLE_RERANKER", True)


def late_candidate_limit(default: int = 60) -> int:
    return max(5, _env_int("LATE_INTERACTION_CANDIDATES", default))


def rerank_candidate_limit(default: int = 25) -> int:
    return max(5, _env_int("RERANK_CANDIDATES", default))


def neighbor_window_size(default: int = 1) -> int:
    return max(0, _env_int("NEIGHBOR_WINDOW_SIZE", default))


def _fastembed_cache_dir() -> str:
    configured = (os.getenv("FASTEMBED_CACHE_DIR") or "").strip()
    if configured:
        return configured
    default_dir = Path.cwd() / ".vaultq" / ".cache" / "fastembed"
    default_dir.mkdir(parents=True, exist_ok=True)
    return str(default_dir)


def _fastembed_threads() -> Optional[int]:
    threads = _env_int("FASTEMBED_THREADS", 0)
    return threads if threads > 0 else None


def _fastembed_providers() -> Optional[Sequence[str]]:
    provider = (os.getenv("FASTEMBED_PROVIDER") or "").strip()
    return [provider] if provider else None


def _ensure_fastembed():
    try:
        from fastembed import LateInteractionTextEmbedding
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except Exception as exc:
        raise RuntimeError(
            "FastEmbed support is required. Install qdrant-client[fastembed]>=1.14.2."
        ) from exc
    return LateInteractionTextEmbedding, TextCrossEncoder


def get_late_interaction_model():
    if not use_multivector():
        return None
    global _LATE_MODEL
    if _LATE_MODEL is not None:
        return _LATE_MODEL
    LateInteractionTextEmbedding, _ = _ensure_fastembed()
    with _MODEL_LOCK:
        if _LATE_MODEL is None:
            _LATE_MODEL = LateInteractionTextEmbedding(
                late_interaction_model_name(),
                cache_dir=_fastembed_cache_dir(),
                threads=_fastembed_threads(),
                providers=_fastembed_providers(),
                lazy_load=False,
            )
    return _LATE_MODEL


def get_reranker():
    if not use_reranker():
        return None
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER
    _, TextCrossEncoder = _ensure_fastembed()
    with _MODEL_LOCK:
        if _RERANKER is None:
            _RERANKER = TextCrossEncoder(
                reranker_model_name(),
                cache_dir=_fastembed_cache_dir(),
                threads=_fastembed_threads(),
                providers=_fastembed_providers(),
                lazy_load=False,
            )
    return _RERANKER


def _matrix_to_list(matrix) -> List[List[float]]:
    if hasattr(matrix, "tolist"):
        payload = matrix.tolist()
    else:
        payload = list(matrix)
    return [[float(value) for value in row] for row in payload]


def late_interaction_dim() -> int:
    if not use_multivector():
        return 0
    model_name = late_interaction_model_name()
    LateInteractionTextEmbedding, _ = _ensure_fastembed()
    supported = LateInteractionTextEmbedding.list_supported_models()
    for row in supported:
        if row.get("model") == model_name:
            try:
                return int(row.get("dim") or 0)
            except Exception:
                return 0
    return 0


def embed_late_documents(texts: Iterable[str]) -> List[List[List[float]]]:
    model = get_late_interaction_model()
    if model is None:
        return []
    return [_matrix_to_list(matrix) for matrix in model.embed(list(texts))]


def embed_late_query(query: str) -> List[List[float]]:
    model = get_late_interaction_model()
    if model is None:
        return []
    try:
        matrix = next(model.query_embed(query))
    except StopIteration:
        return []
    return _matrix_to_list(matrix)


def rerank_documents(query: str, documents: Sequence[str], top_n: Optional[int] = None) -> List[Tuple[int, float]]:
    reranker = get_reranker()
    if reranker is None or not documents:
        return [(idx, 0.0) for idx in range(len(documents))]
    scores = list(reranker.rerank(query=query, documents=list(documents)))
    ranked = [(idx, float(score)) for idx, score in enumerate(scores)]
    ranked.sort(key=lambda item: item[1], reverse=True)
    if top_n is not None:
        ranked = ranked[: max(1, top_n)]
    return ranked
