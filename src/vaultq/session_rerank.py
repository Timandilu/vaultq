from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Dict

SESSION_PREFIX = "88_Agents/sessions/"

STOPWORDS = {
    "about",
    "again",
    "also",
    "and",
    "are",
    "because",
    "been",
    "being",
    "but",
    "can",
    "could",
    "does",
    "from",
    "have",
    "into",
    "like",
    "more",
    "need",
    "notes",
    "only",
    "over",
    "should",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "this",
    "those",
    "use",
    "with",
    "would",
    "your",
}

GENERIC_SESSION_TERMS = {
    "agent",
    "assistant",
    "chatter",
    "continue",
    "generic",
    "housekeeping",
    "log",
    "logs",
    "raw",
    "run",
    "session",
    "status",
    "transcript",
    "unrelated",
    "update",
    "updates",
}

EVIDENCE_SESSION_TERMS = {
    "bug",
    "decision",
    "debug",
    "evidence",
    "failure",
    "fix",
    "handoff",
    "implementation",
    "receipt",
    "result",
    "root",
    "summary",
    "test",
    "tested",
    "trace",
    "verification",
}


def clean_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").strip("/")


def is_agent_session_path(value: Any) -> bool:
    return clean_path(value).startswith(SESSION_PREFIX)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def tokens(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9]{2,}", re.sub(r"[_-]+", " ", str(value or "").lower()))
        if token not in STOPWORDS
    }


def overlap_ratio(query_terms: set[str], candidate_terms: set[str]) -> float:
    if not query_terms or not candidate_terms:
        return 0.0
    return len(query_terms & candidate_terms) / max(3, len(query_terms))


def dated_session_recency_factor(value: str) -> float:
    match = re.search(r"(20\d{2})[-_/](\d{2})[-_/](\d{2})", value)
    if not match:
        return 1.0
    try:
        parsed = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return 1.0
    age_days = (datetime.now().date() - parsed).days
    if age_days < 0:
        return 0.98
    if age_days <= 14:
        return 1.08
    if age_days <= 90:
        return 1.04
    if age_days <= 365:
        return 1.0
    return 0.94


def adaptive_session_rerank(
    *,
    query: str,
    rel_path: str,
    title: str,
    text: str,
    evidence: Dict[str, Any] | None = None,
    session_penalty: float = 0.65,
) -> Dict[str, Any]:
    evidence = dict(evidence or {})
    rel_path = clean_path(rel_path)
    query_terms = tokens(query)
    title_terms = tokens(title)
    path_terms = tokens(rel_path.replace("/", " "))
    body_terms = tokens(text)
    all_terms = title_terms | path_terms | body_terms

    title_overlap = overlap_ratio(query_terms, title_terms)
    body_overlap = overlap_ratio(query_terms, body_terms)
    path_overlap = overlap_ratio(query_terms, path_terms)
    lane_bonus = 0.08 if "keyword" in set(evidence.get("retrieval_lanes") or []) else 0.0
    specificity = clamp((0.45 * body_overlap) + (0.35 * title_overlap) + (0.20 * path_overlap) + lane_bonus, 0.0, 1.0)

    generic_hits = GENERIC_SESSION_TERMS & all_terms
    generic_noise_hits = generic_hits - query_terms
    evidence_hits = EVIDENCE_SESSION_TERMS & all_terms
    genericity = clamp(len(generic_noise_hits) / 8.0, 0.0, 1.0)

    char_count = len(text)
    focus = 1.0
    if char_count and char_count <= 1800:
        focus += 0.06
    elif char_count >= 5000:
        focus -= 0.15
    if evidence.get("chunk_index") is not None or evidence.get("start_line") is not None:
        focus += 0.04
    focus = clamp(focus, 0.75, 1.12)

    evidence_signal = clamp(1.0 + (0.035 * len(evidence_hits)) - (0.04 * len(generic_noise_hits)), 0.78, 1.18)
    recency = dated_session_recency_factor(" ".join([rel_path, title]))
    base = clamp(float(session_penalty or 0.65), 0.05, 0.95)
    if specificity < 0.12:
        specificity_factor = 0.42 + specificity
    else:
        specificity_factor = 0.76 + (0.48 * specificity)

    factor = base * specificity_factor * focus * evidence_signal * recency * (1.0 - (0.35 * genericity))
    if specificity >= 0.60 and genericity <= 0.30:
        factor = max(factor, min(0.95, base + 0.12))
    factor = clamp(factor, 0.10, 0.95)

    reasons = []
    if specificity >= 0.50:
        reasons.append("specific-query-overlap")
    if evidence_hits:
        reasons.append("evidence-language")
    if genericity >= 0.50:
        reasons.append("generic-transcript-language")
    if recency != 1.0:
        reasons.append("date-signal")
    if "keyword" in set(evidence.get("retrieval_lanes") or []):
        reasons.append("semantic-plus-keyword")

    return {
        "mode": "adaptive",
        "factor": round(factor, 6),
        "components": {
            "specificity": round(specificity, 6),
            "genericity": round(genericity, 6),
            "focus": round(focus, 6),
            "evidence_signal": round(evidence_signal, 6),
            "recency": round(recency, 6),
            "base_penalty": round(base, 6),
        },
        "reasons": reasons,
    }
