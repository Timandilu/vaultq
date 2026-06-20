from __future__ import annotations

from vaultq.graph import _collapse_edges, _expanded_link_targets, _identifier_variants, _link_is_indexable, _matches_excludes
from vaultq.link_extraction import ExtractedLink


def test_graph_skips_oversized_link_targets() -> None:
    normal = ExtractedLink("10_Garden/Idea", "links_to", "Idea", "wikilink", 1.0)
    oversized = ExtractedLink("x" * 1000, "links_to", "Huge", "wikilink", 1.0)

    assert _link_is_indexable(normal) is True
    assert _link_is_indexable(oversized) is False


def test_graph_respects_collection_exclude_globs() -> None:
    assert _matches_excludes(".obsidian/plugins/card.md", [".obsidian/**"]) is True
    assert _matches_excludes("10_Garden/Idea.md", [".obsidian/**"]) is False


def test_graph_identifier_variants_bridge_paths_and_targets() -> None:
    assert _identifier_variants("10_Garden/Idea.md") == ["10_Garden/Idea.md", "10_Garden/Idea"]
    assert _identifier_variants("10_Garden/Idea") == ["10_Garden/Idea", "10_Garden/Idea.md"]


def test_graph_collapse_dedupes_presentation_edges_and_counts_provenance() -> None:
    rows = [
        {
            "source_rel_path": "a.md",
            "target_identifier": "b",
            "edge_type": "links_to",
            "source_kind": "markdown",
            "anchor_text": "B",
            "confidence": 0.7,
        },
        {
            "source_rel_path": "a.md",
            "target_identifier": "b",
            "edge_type": "links_to",
            "source_kind": "frontmatter",
            "anchor_text": "related",
            "confidence": 0.9,
        },
    ]

    collapsed = _collapse_edges(rows, depth=1)

    assert len(collapsed) == 1
    assert collapsed[0]["provenance_count"] == 2
    assert collapsed[0]["confidence"] == 0.9
    assert collapsed[0]["source_kinds"] == ["markdown", "frontmatter"]


def test_graph_expands_simple_wikilinks_through_basename_index() -> None:
    link = ExtractedLink("Idea", "links_to", "Idea", "wikilink", 1.0)

    rows = _expanded_link_targets(link, {"idea": ["10_Garden/Idea"]})

    assert rows == [("10_Garden/Idea", "wikilink_basename")]


def test_graph_keeps_qualified_or_path_links_literal() -> None:
    qualified = ExtractedLink("Idea", "links_to", "Idea", "wikilink:second-brain", 1.0)
    path_link = ExtractedLink("10_Garden/Idea", "links_to", "Idea", "wikilink", 1.0)

    assert _expanded_link_targets(qualified, {"idea": ["10_Garden/Idea"]}) == [("Idea", "links_to")]
    assert _expanded_link_targets(path_link, {"idea": ["10_Garden/Idea"]}) == [("10_Garden/Idea", "links_to")]
