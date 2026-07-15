from __future__ import annotations

from vaultq.chunk_stats import assess_chunk_lengths


def test_chunk_assessment_accepts_small_long_tail() -> None:
    assessment = assess_chunk_lengths(
        {
            "chunks": 1000,
            "p95_tokens": 305,
            "max_tokens": 430,
            "over_max_chunks": 12,
            "under_min_chunks": 120,
        },
        min_tokens=120,
        max_tokens=360,
    )

    assert assessment["too_long"] is False
    assert assessment["overall"] == "healthy"


def test_chunk_assessment_flags_long_chunks_when_p95_or_rate_is_high() -> None:
    assessment = assess_chunk_lengths(
        {
            "chunks": 1000,
            "p95_tokens": 520,
            "max_tokens": 900,
            "over_max_chunks": 180,
            "under_min_chunks": 40,
        },
        min_tokens=120,
        max_tokens=360,
    )

    assert assessment["too_long"] is True
    assert assessment["overall"] == "too_long"
