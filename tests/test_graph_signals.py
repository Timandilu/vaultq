from __future__ import annotations

from vaultq.graph_signals import apply_graph_signals, compute_floor_threshold, session_prefix


def _point(rel_path: str, score: float):
    return {
        "id": rel_path,
        "score": score,
        "payload": {
            "rel_path": rel_path,
            "text": rel_path,
        },
    }


def test_session_prefix_only_flags_session_like_paths() -> None:
    assert session_prefix("14_Agent_Workspace/Runs/2026-06-03/review.md") == "14_Agent_Workspace/Runs"
    assert session_prefix("05_Reports/Sessions/2026-06-03/chat.md") == "05_Reports/Sessions"
    assert session_prefix("10_Garden/Idea.md") is None


def test_graph_signals_boost_nodes_with_two_inbound_top_result_links() -> None:
    points = [_point("a.md", 1.0), _point("b.md", 0.9), _point("c.md", 0.8)]

    result = apply_graph_signals(points, adjacency_fn=lambda rels: {"a.md": 2})

    boosted = result["results"][0]
    assert boosted["payload"]["graph_adjacency_hits"] == 2
    assert boosted["score"] > 1.0
    assert result["meta"]["boosted"] == 1


def test_graph_signals_do_not_boost_below_floor() -> None:
    points = [_point("a.md", 1.0), _point("b.md", 0.2)]
    floor = compute_floor_threshold(points, 0.5)

    result = apply_graph_signals(points, floor_threshold=floor, adjacency_fn=lambda rels: {"b.md": 3})

    by_rel = {point["payload"]["rel_path"]: point for point in result["results"]}
    assert "graph_adjacency_hits" not in by_rel["b.md"]["payload"]
    assert result["meta"]["boosted"] == 0


def test_graph_signals_demote_duplicate_session_hits() -> None:
    points = [
        _point("14_Agent_Workspace/Runs/2026-06-03/a.md", 1.0),
        _point("14_Agent_Workspace/Runs/2026-06-03/b.md", 0.99),
        _point("10_Garden/Idea.md", 0.5),
    ]

    result = apply_graph_signals(points, adjacency_fn=lambda rels: {})

    by_rel = {point["payload"]["rel_path"]: point for point in result["results"]}
    assert by_rel["14_Agent_Workspace/Runs/2026-06-03/b.md"]["payload"]["graph_session_demoted"] is True
    assert "graph_session_demoted" not in by_rel["10_Garden/Idea.md"]["payload"]
    assert result["meta"]["demoted"] == 1
