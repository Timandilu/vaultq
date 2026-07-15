from __future__ import annotations

from vaultq import related


def test_related_work_returns_source_diverse_connection_cards(monkeypatch) -> None:
    def fake_search(query: str, *, limit: int, retrieval_mode: str):
        assert retrieval_mode == "focused"
        assert limit == 4
        return {
            "query": query,
            "retrieval_mode": retrieval_mode,
            "strategy": "focused_title_hybrid_rrf",
            "results": [
                {
                    "id": "a",
                    "score": 0.91,
                    "rel_path": "10_Garden/Offer Loop.md",
                    "title": "Offer Loop",
                    "text": "Revenue loop and agent workflow for offers.",
                    "evidence": {
                        "collection_name": "second_brain",
                        "rel_path": "10_Garden/Offer Loop.md",
                        "start_line": 3,
                        "end_line": 8,
                        "retrieval_lanes": ["semantic"],
                    },
                },
                {
                    "id": "b",
                    "score": 0.7,
                    "rel_path": "10_Garden/Offer Loop.md",
                    "title": "Offer Loop duplicate",
                    "text": "Duplicate chunk.",
                    "evidence": {"collection_name": "second_brain", "rel_path": "10_Garden/Offer Loop.md"},
                },
                {
                    "id": "c",
                    "score": 0.68,
                    "rel_path": "05_Reports/Agent Workflows.md",
                    "title": "Agent Workflows",
                    "text": "Agents retrieve related work before writing new notes.",
                    "evidence": {
                        "collection_name": "second_brain",
                        "rel_path": "05_Reports/Agent Workflows.md",
                        "retrieval_lanes": ["keyword"],
                    },
                },
            ],
        }

    monkeypatch.setattr(related, "search", fake_search)

    result = related.related_work("agent revenue workflow idea", limit=4)

    assert result["query"] == "agent revenue workflow idea"
    assert result["mode"] == "focused"
    assert [item["rel_path"] for item in result["connections"]] == [
        "10_Garden/Offer Loop.md",
        "05_Reports/Agent Workflows.md",
    ]
    assert result["connections"][0]["connection_reason"].startswith("shared terms:")
    assert result["suggested_existing_home"]["rel_path"] == "10_Garden/Offer Loop.md"
    assert result["write_default"] == "do_not_write"
