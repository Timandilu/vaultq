from __future__ import annotations

from vaultq.search import _apply_session_policy, _diversify_by_document, result_contract


def test_result_contract_exposes_agent_evidence_fields() -> None:
    result = result_contract(
        {
            "id": "chunk:1",
            "score": 0.9,
            "payload": {
                "record_type": "chunk",
                "doc_type": "markdown_chunk",
                "collection_name": "second_brain",
                "rel_path": "10_Garden/Idea.md",
                "title": "Idea",
                "start_line": 4,
                "end_line": 8,
                "window_start_line": 3,
                "window_end_line": 12,
                "retrieval_lanes": ["title", "keyword"],
                "text": "body",
            },
        }
    )

    assert result["evidence"] == {
        "collection_name": "second_brain",
        "rel_path": "10_Garden/Idea.md",
        "start_line": 3,
        "end_line": 12,
        "retrieval_lanes": ["title", "keyword"],
    }
    assert result["metadata"]["retrieval_lanes"] == ["title", "keyword"]


def test_hybrid_search_degrades_when_semantic_lane_fails(monkeypatch) -> None:
    from vaultq import search as search_module

    point = {
        "id": "chunk:1",
        "score": 1.0,
        "payload": {
            "record_type": "chunk",
            "doc_type": "markdown_chunk",
            "collection_name": "second_brain",
            "rel_path": "10_Garden/Idea.md",
            "title": "Idea",
            "text": "local evidence",
        },
    }

    monkeypatch.setattr(search_module, "_title_search_postgres", lambda query, limit: [])
    monkeypatch.setattr(search_module, "_keyword_search", lambda query, limit: [point])
    monkeypatch.setattr(search_module, "_has_embedded_vectors", lambda: False)
    monkeypatch.setattr(search_module, "_semantic_search", lambda query, limit: (_ for _ in ()).throw(AssertionError("semantic should be skipped")))
    monkeypatch.setattr(search_module, "use_reranker", lambda: False)
    monkeypatch.setattr(search_module, "_attach_neighbor_windows", lambda points: list(points))

    result = search_module.search("idea", limit=1, retrieval_mode="hybrid")

    assert result["results"][0]["rel_path"] == "10_Garden/Idea.md"
    assert result["semantic_candidates"] == 0
    assert result["warnings"][0]["lane"] == "semantic"
    assert "no embedded vectors" in result["warnings"][0]["message"]


def test_document_diversity_prefers_distinct_files_before_duplicate_chunks() -> None:
    def point(identifier: str, rel_path: str):
        return {"id": identifier, "score": 1.0, "payload": {"rel_path": rel_path}}

    diversified, meta = _diversify_by_document(
        [
            point("a1", "a.md"),
            point("a2", "a.md"),
            point("a3", "a.md"),
            point("b1", "b.md"),
        ],
        limit=3,
        max_per_doc=1,
    )

    assert [point["id"] for point in diversified] == ["a1", "b1", "a2"]
    assert meta["applied"] is True


def test_focused_search_keeps_results_concise_and_diverse(monkeypatch) -> None:
    from vaultq import search as search_module

    keyword_limits: list[int] = []

    def point(identifier: str, rel_path: str):
        return {
            "id": identifier,
            "score": 1.0,
            "payload": {
                "record_type": "chunk",
                "doc_type": "markdown_chunk",
                "collection_name": "second_brain",
                "rel_path": rel_path,
                "title": rel_path,
                "text": f"evidence from {rel_path}",
            },
        }

    monkeypatch.setattr(search_module, "_title_search_postgres", lambda query, limit: [])

    def keyword_search(query: str, limit: int):
        keyword_limits.append(limit)
        return [
            point("a1", "A.md"),
            point("a2", "A.md"),
            point("b1", "B.md"),
        ]

    monkeypatch.setattr(search_module, "_keyword_search", keyword_search)
    monkeypatch.setattr(search_module, "_has_embedded_vectors", lambda: False)
    monkeypatch.setattr(search_module, "use_reranker", lambda: False)
    monkeypatch.setattr(search_module, "graph_signals_enabled", lambda mode: False)
    monkeypatch.setattr(
        search_module,
        "_attach_neighbor_windows",
        lambda points: (_ for _ in ()).throw(AssertionError("focused mode should not expand neighbors")),
    )

    result = search_module.search("idea", limit=2, retrieval_mode="focused")

    assert result["strategy"] == "focused_title_hybrid_rrf"
    assert result["results"][0]["rel_path"] == "A.md"
    assert result["results"][1]["rel_path"] == "B.md"
    assert result["diversity"]["max_per_doc"] == 1
    assert keyword_limits == [12]


def test_focused_search_does_not_pad_with_duplicate_source_chunks(monkeypatch) -> None:
    from vaultq import search as search_module

    def point(identifier: str):
        return {
            "id": identifier,
            "score": 1.0,
            "payload": {
                "record_type": "chunk",
                "doc_type": "markdown_chunk",
                "collection_name": "second_brain",
                "rel_path": "Only.md",
                "title": "Only",
                "text": "evidence",
            },
        }

    monkeypatch.setattr(search_module, "_title_search_postgres", lambda query, limit: [])
    monkeypatch.setattr(search_module, "_keyword_search", lambda query, limit: [point("a1"), point("a2")])
    monkeypatch.setattr(search_module, "_has_embedded_vectors", lambda: False)
    monkeypatch.setattr(search_module, "use_reranker", lambda: False)
    monkeypatch.setattr(search_module, "graph_signals_enabled", lambda mode: False)
    monkeypatch.setattr(search_module, "_attach_neighbor_windows", lambda points: list(points))

    result = search_module.search("idea", limit=3, retrieval_mode="focused")

    assert [row["id"] for row in result["results"]] == ["a1"]
    assert result["diversity"]["strict"] is True


def test_session_policy_adaptively_penalizes_and_caps_session_hits() -> None:
    def point(identifier: str, score: float, rel_path: str, text: str):
        return {
            "id": identifier,
            "score": score,
            "payload": {
                "record_type": "chunk",
                "doc_type": "markdown_chunk",
                "collection_name": "second_brain",
                "rel_path": rel_path,
                "title": rel_path,
                "text": text,
                "retrieval_lanes": ["semantic", "keyword"],
            },
        }

    reranked, meta = _apply_session_policy(
        [
            point(
                "generic-session",
                0.98,
                "88_Agents/sessions/generic-agent-run.md",
                "Long raw transcript with broad agent chatter, status updates, and unrelated housekeeping.",
            ),
            point(
                "specific-session",
                0.74,
                "88_Agents/sessions/offer-agent-implementation.md",
                "Decision and implementation evidence for connecting offer design to agent loops and monetization.",
            ),
            point(
                "stable-note",
                0.70,
                "10_Garden/Business/Offer Systems.md",
                "Stable note about offer design and monetization loops.",
            ),
        ],
        query="offer design agent implementation monetization loops",
        limit=3,
    )

    rel_paths = [row["payload"]["rel_path"] for row in reranked]
    assert "88_Agents/sessions/offer-agent-implementation.md" in rel_paths
    assert "88_Agents/sessions/generic-agent-run.md" not in rel_paths
    assert "10_Garden/Business/Offer Systems.md" in rel_paths
    assert meta["policy"] == "penalize"
    assert meta["session_max_results"] == 1
    assert meta["capped"] == 1

    session_point = next(row for row in reranked if row["payload"]["rel_path"].startswith("88_Agents/sessions/"))
    assert session_point["payload"]["source_class"] == "agent_session"
    assert session_point["payload"]["session_policy"] == "penalize"
    assert session_point["payload"]["session_rerank"]["mode"] == "adaptive"
    assert session_point["payload"]["session_rerank"]["components"]["specificity"] > 0.35
