from __future__ import annotations

import re
from typing import Any, Dict, Iterable

from vaultq.search import search

_STOPWORDS = {
    "about",
    "after",
    "agent",
    "also",
    "and",
    "for",
    "from",
    "idea",
    "into",
    "like",
    "note",
    "notes",
    "that",
    "the",
    "this",
    "with",
    "work",
}


def _terms(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", (text or "").lower())
        if word not in _STOPWORDS
    }


def _connection_reason(idea: str, row: Dict[str, Any]) -> str:
    shared = sorted(_terms(idea) & _terms(f"{row.get('title', '')} {row.get('text', '')}"))
    if shared:
        return "shared terms: " + ", ".join(shared[:8])
    lanes = row.get("evidence", {}).get("retrieval_lanes") or row.get("metadata", {}).get("retrieval_lanes") or []
    if lanes:
        return "retrieved through " + ", ".join(str(lane) for lane in lanes)
    return "retrieval score indicates semantic proximity"


def _dedupe_by_path(rows: Iterable[Dict[str, Any]], limit: int) -> list[Dict[str, Any]]:
    seen: set[str] = set()
    out: list[Dict[str, Any]] = []
    for row in rows:
        rel_path = str(row.get("rel_path") or row.get("evidence", {}).get("rel_path") or "").strip()
        if not rel_path or rel_path in seen:
            continue
        seen.add(rel_path)
        out.append(row)
        if len(out) >= limit:
            break
    return out


def related_work(idea: str, *, limit: int = 6, mode: str = "focused") -> Dict[str, Any]:
    limit = max(1, min(20, int(limit or 6)))
    mode = mode or "focused"
    retrieval = search(idea, limit=limit, retrieval_mode=mode)
    rows = _dedupe_by_path(retrieval.get("results", []), limit)
    connections = []
    for row in rows:
        evidence = dict(row.get("evidence") or {})
        rel_path = str(row.get("rel_path") or evidence.get("rel_path") or "")
        connections.append(
            {
                "id": row.get("id"),
                "score": row.get("score"),
                "collection_name": evidence.get("collection_name") or row.get("collection_name"),
                "rel_path": rel_path,
                "title": row.get("title") or rel_path,
                "heading_path": row.get("heading_path"),
                "connection_reason": _connection_reason(idea, row),
                "evidence": evidence,
                "text": row.get("text", ""),
            }
        )
    suggested_home = None
    if connections:
        best = connections[0]
        suggested_home = {
            "rel_path": best["rel_path"],
            "title": best["title"],
            "reason": "highest-ranked distinct existing note; continue here before creating a new node",
        }
    return {
        "query": idea,
        "mode": mode,
        "strategy": retrieval.get("strategy"),
        "write_default": "do_not_write",
        "connection_count": len(connections),
        "connections": connections,
        "suggested_existing_home": suggested_home,
        "new_note_recommendation": None if connections else "No strong existing home found; create only under the agent workspace with valid properties.",
        "retrieval": {
            "elapsed_ms": retrieval.get("elapsed_ms"),
            "warnings": retrieval.get("warnings", []),
            "diversity": retrieval.get("diversity"),
        },
    }
