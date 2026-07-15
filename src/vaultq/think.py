from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def think(question: str, *, limit: int = 5, save_report: bool = False, root: Optional[Path] = None) -> Dict[str, Any]:
    from vaultq.search import search

    result = search(question, limit=limit, retrieval_mode="hybrid")
    hits = result.get("results", [])
    citations = [
        {
            "collection": hit.get("collection_name"),
            "rel_path": hit.get("rel_path"),
            "chunk_id": hit.get("id"),
            "line_start": hit.get("start_line"),
            "line_end": hit.get("end_line"),
        }
        for hit in hits
    ]
    answer_lines = []
    for hit in hits[:3]:
        title = hit.get("title") or hit.get("rel_path") or "Untitled"
        snippet = (hit.get("text") or "").strip().replace("\n", " ")
        if snippet:
            answer_lines.append(f"- {title}: {snippet[:360]}")
    gaps = [] if hits else ["No relevant indexed evidence was found."]
    payload: Dict[str, Any] = {
        "question": question,
        "answer": "\n".join(answer_lines),
        "citations": citations,
        "conflicts": [],
        "gaps": gaps,
        "suggested_writes": [],
        "retrieval": {"elapsed_ms": result.get("elapsed_ms"), "result_count": len(hits)},
    }
    if save_report:
        if not root:
            raise ValueError("root is required when save_report=True")
        from vaultq.write_ops import capture_note

        report = capture_note(
            root,
            text=f"Question: {question}\n\n{payload['answer'] or 'No evidence found.'}",
            kind="report",
            domain="ai",
            title=f"Think Report - {question[:60]}",
        )
        payload["saved_report"] = report
    return payload
