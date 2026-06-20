from __future__ import annotations

import math
import os
import re
import time
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Sequence

from vaultq.store import connect

ADJACENCY_BOOST = 1.05
SESSION_DEMOTE = 0.95
ADJACENCY_MIN_HITS = 2
DEFAULT_TOP_K = 20

_SESSION_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_SESSION_MARKERS = {"chat", "chats", "session", "sessions", "run", "runs"}


def _clean_env(value: Optional[str]) -> str:
    return (value or "").strip().strip('"').strip("'")


def _env_bool(name: str, default: bool) -> bool:
    value = _clean_env(os.getenv(name)).lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int = 200) -> int:
    try:
        value = int(_clean_env(os.getenv(name)))
    except Exception:
        return default
    return max(minimum, min(maximum, value))


def graph_signals_top_k() -> int:
    return _env_int("GRAPH_SIGNALS_TOP_K", DEFAULT_TOP_K, minimum=3, maximum=100)


def graph_signal_floor_ratio() -> float:
    try:
        value = float(_clean_env(os.getenv("GRAPH_SIGNAL_FLOOR_RATIO")))
    except Exception:
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def graph_signals_enabled(retrieval_mode: str) -> bool:
    default = retrieval_mode in {"hybrid", "deep"}
    return _env_bool("ENABLE_GRAPH_SIGNALS", default)


def _rel_path(point: Dict[str, Any]) -> str:
    payload = point.get("payload") or {}
    return str(payload.get("rel_path") or "").strip().replace("\\", "/").strip("/")


def _score(point: Dict[str, Any]) -> float:
    try:
        value = float(point.get("score") or 0.0)
    except Exception:
        return 0.0
    return value if math.isfinite(value) else 0.0


def _target_variants(identifier: str) -> list[str]:
    normalized = (identifier or "").strip().replace("\\", "/").strip("/")
    if not normalized:
        return []
    variants = [normalized]
    if normalized.endswith(".md"):
        variants.append(normalized[:-3])
    else:
        variants.append(f"{normalized}.md")
    return list(dict.fromkeys(variants))


def session_prefix(identifier: str) -> Optional[str]:
    normalized = (identifier or "").strip().replace("\\", "/").strip("/")
    if normalized.endswith(".md"):
        normalized = normalized[:-3]
    parts = [part for part in normalized.split("/") if part]
    if len(parts) < 2:
        return None
    for index, part in enumerate(parts[:-1]):
        lowered = part.lower()
        if lowered in _SESSION_MARKERS or _SESSION_DATE_RE.match(part):
            return "/".join(parts[: index + 1])
    leaf = parts[-1].lower()
    if _SESSION_DATE_RE.match(leaf):
        return "/".join(parts[:-1])
    return None


def compute_score_distribution(points: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    scores = [_score(point) for point in points if _score(point) > 0]
    if not scores:
        return {"max": 0.0, "min": 0.0, "mean": 0.0}
    return {"max": max(scores), "min": min(scores), "mean": sum(scores) / len(scores)}


def compute_floor_threshold(points: Sequence[Dict[str, Any]], floor_ratio: float) -> Optional[float]:
    ratio = max(0.0, min(1.0, float(floor_ratio or 0.0)))
    if ratio <= 0:
        return None
    distribution = compute_score_distribution(points)
    maximum = distribution["max"]
    if maximum <= 0:
        return None
    return maximum * ratio


def fetch_adjacency_counts(rel_paths: Sequence[str]) -> Dict[str, int]:
    canonical_rel_paths = list(dict.fromkeys(path for path in rel_paths if path))
    if len(canonical_rel_paths) < 2:
        return {}
    source_variant_to_rel: Dict[str, str] = {}
    target_variant_to_rel: Dict[str, str] = {}
    for rel_path in canonical_rel_paths:
        for variant in _target_variants(rel_path):
            source_variant_to_rel[variant] = rel_path
            target_variant_to_rel[variant] = rel_path
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source_rel_path, target_identifier
                FROM vq_links
                WHERE source_rel_path = ANY(%s)
                  AND target_identifier = ANY(%s)
                """,
                (list(source_variant_to_rel), list(target_variant_to_rel)),
            )
            rows = list(cur.fetchall())
    inbound_sources: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source_rel = source_variant_to_rel.get(str(row["source_rel_path"] or ""))
        target_rel = target_variant_to_rel.get(str(row["target_identifier"] or ""))
        if source_rel and target_rel and source_rel != target_rel:
            inbound_sources[target_rel].add(source_rel)
    return {target_rel: len(sources) for target_rel, sources in inbound_sources.items()}


def apply_graph_signals(
    points: Sequence[Dict[str, Any]],
    *,
    enabled: bool = True,
    top_k: int = DEFAULT_TOP_K,
    floor_threshold: Optional[float] = None,
    adjacency_fn: Optional[Callable[[Sequence[str]], Dict[str, int]]] = None,
) -> Dict[str, Any]:
    started = time.perf_counter()
    working = [dict(point) for point in points]
    meta: Dict[str, Any] = {
        "enabled": bool(enabled),
        "top_k": int(top_k),
        "adjacency_boost": ADJACENCY_BOOST,
        "session_demote": SESSION_DEMOTE,
        "adjacency_min_hits": ADJACENCY_MIN_HITS,
        "floor_threshold": floor_threshold,
        "boosted": 0,
        "demoted": 0,
        "errored": False,
        "borrowed_pattern": "gbrain graph-signals adjacency boost and session diversification",
    }
    if not enabled or not working:
        meta["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        return {"results": working, "meta": meta}

    pool = working[: max(1, int(top_k))]
    meta["score_distribution"] = compute_score_distribution(pool)
    rel_paths = [_rel_path(point) for point in pool]
    try:
        adjacency_counts = (adjacency_fn or fetch_adjacency_counts)(rel_paths)
    except Exception as exc:
        meta["errored"] = True
        meta["error_type"] = exc.__class__.__name__
        meta["error_message"] = str(exc)[:300]
        meta["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        return {"results": working, "meta": meta}

    for point in pool:
        rel_path = _rel_path(point)
        hits = int(adjacency_counts.get(rel_path) or 0)
        if hits < ADJACENCY_MIN_HITS:
            continue
        if floor_threshold is not None and _score(point) < floor_threshold:
            continue
        point["score"] = _score(point) * ADJACENCY_BOOST
        payload = dict(point.get("payload", {}))
        payload["graph_adjacency_hits"] = hits
        payload["graph_adjacency_boost"] = ADJACENCY_BOOST
        point["payload"] = payload
        meta["boosted"] += 1

    by_session: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for point in pool:
        prefix = session_prefix(_rel_path(point))
        if prefix:
            by_session[prefix].append(point)
    for prefix, session_points in by_session.items():
        if len(session_points) < 2:
            continue
        ordered = sorted(session_points, key=lambda point: (-_score(point), _rel_path(point)))
        for point in ordered[1:]:
            if floor_threshold is not None and _score(point) < floor_threshold:
                continue
            point["score"] = _score(point) * SESSION_DEMOTE
            payload = dict(point.get("payload", {}))
            payload["graph_session_prefix"] = prefix
            payload["graph_session_demoted"] = True
            payload["graph_session_demote"] = SESSION_DEMOTE
            point["payload"] = payload
            meta["demoted"] += 1

    remaining = working[len(pool) :]
    ranked = sorted(pool, key=lambda point: (-_score(point), _rel_path(point))) + remaining
    meta["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    return {"results": ranked, "meta": meta}
