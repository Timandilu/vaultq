from __future__ import annotations

from typing import Any, Dict, Optional

from vaultq.chunker import CHUNK_MAX_TOKENS, CHUNK_MIN_TOKENS, CHUNK_TARGET_TOKENS
from vaultq.store import connect


def _pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((part / total) * 100.0, 2)


def assess_chunk_lengths(
    stats: Dict[str, Any],
    *,
    min_tokens: int = CHUNK_MIN_TOKENS,
    max_tokens: int = CHUNK_MAX_TOKENS,
) -> Dict[str, Any]:
    chunks = int(stats.get("chunks") or 0)
    over_max = int(stats.get("over_max_chunks") or 0)
    under_min = int(stats.get("under_min_chunks") or 0)
    p95 = float(stats.get("p95_tokens") or 0)
    max_seen = int(stats.get("max_tokens") or 0)
    over_pct = _pct(over_max, chunks)
    under_pct = _pct(under_min, chunks)

    too_long = bool(chunks and (p95 > max_tokens or over_pct > 10.0 or max_seen > max_tokens * 3))
    too_short = bool(chunks and not too_long and under_pct > 45.0)
    overall = "too_long" if too_long else "too_short" if too_short else "healthy"
    return {
        "overall": overall,
        "too_long": too_long,
        "too_short": too_short,
        "over_max_pct": over_pct,
        "under_min_pct": under_pct,
        "thresholds": {
            "min_tokens": min_tokens,
            "target_tokens": CHUNK_TARGET_TOKENS,
            "max_tokens": max_tokens,
        },
        "reason": (
            "p95 or over-max rate exceeds configured chunk bounds"
            if too_long
            else "many chunks are below the configured minimum"
            if too_short
            else "p95 and over-max rate are inside configured chunk bounds"
        ),
    }


def chunk_length_stats(
    *,
    collection_name: Optional[str] = None,
    sample_limit: int = 12,
    min_tokens: int = CHUNK_MIN_TOKENS,
    max_tokens: int = CHUNK_MAX_TOKENS,
) -> Dict[str, Any]:
    sample_limit = max(1, min(100, int(sample_limit or 12)))
    collection_filter = "AND col.name = %s" if collection_name else ""
    params = (collection_name,) if collection_name else ()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    COUNT(*)::int AS chunks,
                    COUNT(DISTINCT d.id)::int AS documents,
                    COALESCE(MIN(c.token_count), 0)::int AS min_tokens,
                    COALESCE(MAX(c.token_count), 0)::int AS max_tokens,
                    COALESCE(ROUND(AVG(c.token_count)::numeric, 2), 0)::float AS avg_tokens,
                    COALESCE(percentile_cont(0.50) WITHIN GROUP (ORDER BY c.token_count), 0)::float AS p50_tokens,
                    COALESCE(percentile_cont(0.75) WITHIN GROUP (ORDER BY c.token_count), 0)::float AS p75_tokens,
                    COALESCE(percentile_cont(0.90) WITHIN GROUP (ORDER BY c.token_count), 0)::float AS p90_tokens,
                    COALESCE(percentile_cont(0.95) WITHIN GROUP (ORDER BY c.token_count), 0)::float AS p95_tokens,
                    COALESCE(percentile_cont(0.99) WITHIN GROUP (ORDER BY c.token_count), 0)::float AS p99_tokens,
                    COALESCE(SUM(c.token_count), 0)::int AS total_tokens,
                    COALESCE(SUM(c.char_count), 0)::int AS total_chars,
                    COUNT(*) FILTER (WHERE c.token_count < %s)::int AS under_min_chunks,
                    COUNT(*) FILTER (WHERE c.token_count > %s)::int AS over_max_chunks,
                    COUNT(*) FILTER (WHERE c.qdrant_point_id IS NOT NULL)::int AS embedded_chunks
                FROM vq_chunks c
                JOIN vq_documents d ON d.id = c.document_id
                JOIN vq_collections col ON col.id = d.collection_id
                WHERE c.version = 1
                  {collection_filter}
                """,
                (min_tokens, max_tokens, *params),
            )
            summary = dict(cur.fetchone() or {})

            cur.execute(
                f"""
                SELECT
                    col.name AS collection_name,
                    d.rel_path,
                    d.title,
                    c.chunk_index,
                    c.token_count,
                    c.char_count,
                    c.start_line,
                    c.end_line,
                    COALESCE(c.heading_path, '') AS heading_path
                FROM vq_chunks c
                JOIN vq_documents d ON d.id = c.document_id
                JOIN vq_collections col ON col.id = d.collection_id
                WHERE c.version = 1
                  {collection_filter}
                ORDER BY c.token_count DESC, d.rel_path ASC, c.chunk_index ASC
                LIMIT %s
                """,
                (*params, sample_limit),
            )
            longest = [dict(row) for row in cur.fetchall()]

            cur.execute(
                f"""
                SELECT
                    CASE
                        WHEN c.token_count < %s THEN 'under_min'
                        WHEN c.token_count <= %s THEN 'min_to_target'
                        WHEN c.token_count <= %s THEN 'target_to_max'
                        WHEN c.token_count <= %s THEN 'max_to_2x'
                        ELSE 'over_2x_max'
                    END AS bucket,
                    COUNT(*)::int AS chunks
                FROM vq_chunks c
                JOIN vq_documents d ON d.id = c.document_id
                JOIN vq_collections col ON col.id = d.collection_id
                WHERE c.version = 1
                  {collection_filter}
                GROUP BY bucket
                ORDER BY MIN(c.token_count)
                """,
                (min_tokens, CHUNK_TARGET_TOKENS, max_tokens, max_tokens * 2, *params),
            )
            buckets = [dict(row) for row in cur.fetchall()]

    summary["embedded_pct"] = _pct(int(summary.get("embedded_chunks") or 0), int(summary.get("chunks") or 0))
    summary["collection_name"] = collection_name or "all"
    return {
        "summary": summary,
        "assessment": assess_chunk_lengths(summary, min_tokens=min_tokens, max_tokens=max_tokens),
        "buckets": buckets,
        "longest_chunks": longest,
    }
