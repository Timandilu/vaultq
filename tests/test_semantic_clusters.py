from __future__ import annotations

from vaultq import semantic_clusters


def test_semantic_clusters_anchor_multiple_seed_nodes_on_shared_source(monkeypatch) -> None:
    seeds = [
        {
            "rel_path": "14_Agent_Workspace/Ideas/agent-offer-loop.md",
            "title": "Agent Offer Loop",
            "node_type": "concept",
            "domain": "ai",
            "text": "Agents should surface revenue offer ideas before writing duplicate notes.",
        },
        {
            "rel_path": "14_Agent_Workspace/Ideas/agent-content-loop.md",
            "title": "Agent Content Loop",
            "node_type": "concept",
            "domain": "ai",
            "text": "Use agent research to connect publication ideas to offers.",
        },
    ]
    calls: list[str] = []

    def fake_search(query: str, *, limit: int, retrieval_mode: str):
        calls.append(query)
        assert retrieval_mode == "semantic"
        shared = {
            "id": "blog-x",
            "score": 0.86,
            "rel_path": "06_Library/Imported Blog Posts/Offer Systems.md",
            "title": "Offer Systems",
            "text": "Imported blog post about offers, publication loops, and monetization systems.",
            "evidence": {
                "collection_name": "second_brain",
                "rel_path": "06_Library/Imported Blog Posts/Offer Systems.md",
                "start_line": 12,
                "end_line": 20,
            },
        }
        if "Offer Loop" in query:
            return {
                "results": [
                    {"id": "self", "score": 0.99, "rel_path": "14_Agent_Workspace/Ideas/agent-offer-loop.md"},
                    shared,
                    {
                        "id": "session-a",
                        "score": 0.95,
                        "rel_path": "88_Agents/sessions/old-agent-run.md",
                        "title": "Old Agent Run",
                        "text": "Noisy but useful session transcript.",
                    },
                ]
            }
        return {"results": [shared]}

    monkeypatch.setattr(semantic_clusters, "search", fake_search)

    result = semantic_clusters.semantic_cluster_seed_nodes(
        seeds,
        related_limit=4,
        min_score=0.25,
        session_policy="exclude",
    )

    assert result["cluster_algorithm"] == "semantic_seed_search"
    assert any("Agent Offer Loop" in call for call in calls)
    assert result["seed_count"] == 2
    assert result["cluster_count"] == 1
    cluster = result["clusters"][0]
    assert cluster["anchor"]["rel_path"] == "06_Library/Imported Blog Posts/Offer Systems.md"
    assert [node["rel_path"] for node in cluster["seed_nodes"]] == [
        "14_Agent_Workspace/Ideas/agent-content-loop.md",
        "14_Agent_Workspace/Ideas/agent-offer-loop.md",
    ]
    assert [node["source_class"] for node in cluster["related_nodes"]] == ["imported_text"]
    assert cluster["related_nodes"][0]["connection_count"] == 2


def test_semantic_clusters_can_include_session_logs_with_penalty(monkeypatch) -> None:
    seeds = [
        {
            "rel_path": "10_Garden/Idea.md",
            "title": "Idea",
            "node_type": "note",
            "domain": "general",
            "text": "An idea that appeared in an old agent run.",
        }
    ]

    def fake_search(query: str, *, limit: int, retrieval_mode: str):
        return {
            "results": [
                {
                    "id": "session-a",
                    "score": 0.9,
                    "rel_path": "88_Agents/sessions/agent-run.md",
                    "title": "Agent Run",
                    "text": "Old session transcript with relevant implementation detail.",
                },
                {
                    "id": "note-a",
                    "score": 0.7,
                    "rel_path": "10_Garden/Related Note.md",
                    "title": "Related Note",
                    "text": "Stable note.",
                },
            ]
        }

    monkeypatch.setattr(semantic_clusters, "search", fake_search)

    excluded = semantic_clusters.semantic_cluster_seed_nodes(seeds, session_policy="exclude")
    included = semantic_clusters.semantic_cluster_seed_nodes(seeds, session_policy="penalize", session_penalty=0.5)

    assert all(
        node["source_class"] != "agent_session"
        for cluster in excluded["clusters"]
        for node in cluster["related_nodes"]
    )
    session_node = next(
        node
        for cluster in included["clusters"]
        for node in cluster["related_nodes"]
        if node["source_class"] == "agent_session"
    )
    assert session_node["score"] == 0.9
    assert 0.3 <= session_node["effective_score"] < session_node["score"]
    assert session_node["session_policy"] == "penalize"
    assert session_node["session_rerank"]["mode"] == "adaptive"
    assert session_node["session_rerank"]["factor"] < 1.0


def test_penalized_session_logs_are_adaptive_and_capped(monkeypatch) -> None:
    seeds = [
        {
            "rel_path": "10_Garden/Offer Agent.md",
            "title": "Offer Agent",
            "node_type": "note",
            "domain": "business",
            "text": "Connect offer design notes to agent implementation evidence and monetization loops.",
        }
    ]

    def fake_search(query: str, *, limit: int, retrieval_mode: str):
        assert retrieval_mode == "semantic"
        return {
            "results": [
                {
                    "id": "generic-session",
                    "score": 0.98,
                    "rel_path": "88_Agents/sessions/generic-agent-run.md",
                    "title": "Generic Agent Run",
                    "text": "Long transcript with broad agent chatter, status updates, and unrelated housekeeping.",
                    "evidence": {"retrieval_lanes": ["semantic"]},
                },
                {
                    "id": "specific-session",
                    "score": 0.76,
                    "rel_path": "88_Agents/sessions/offer-agent-implementation.md",
                    "title": "Offer Agent Implementation Evidence",
                    "text": (
                        "Decision and implementation evidence for connecting offer design notes "
                        "to agent implementation loops and monetization ideas."
                    ),
                    "evidence": {"retrieval_lanes": ["semantic", "keyword"], "chunk_index": 4},
                },
                {
                    "id": "stable-note",
                    "score": 0.72,
                    "rel_path": "10_Garden/Business/Offer Systems.md",
                    "title": "Offer Systems",
                    "text": "Stable note about offer design and monetization loops.",
                },
            ]
        }

    monkeypatch.setattr(semantic_clusters, "search", fake_search)

    result = semantic_clusters.semantic_cluster_seed_nodes(
        seeds,
        related_limit=3,
        min_score=0.2,
        session_policy="penalize",
        session_penalty=0.65,
    )

    related = result["clusters"][0]["related_nodes"]
    paths = [node["rel_path"] for node in related]
    assert "88_Agents/sessions/offer-agent-implementation.md" in paths
    assert "88_Agents/sessions/generic-agent-run.md" not in paths
    assert "10_Garden/Business/Offer Systems.md" in paths

    session_node = next(node for node in related if node["source_class"] == "agent_session")
    assert session_node["effective_score"] > 0.45
    assert session_node["session_rerank"]["components"]["specificity"] > 0.35
    assert session_node["session_rerank"]["components"]["genericity"] < 0.5
    assert result["session_max_per_seed"] == 1
