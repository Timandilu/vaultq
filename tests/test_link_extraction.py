from __future__ import annotations

from vaultq.link_extraction import build_basename_index, extract_links, query_basename_index


def test_link_extraction_ignores_fenced_and_inline_code() -> None:
    links = extract_links("`[[Code Link]]`\n\n```md\n[[Block Link]]\n```\n\nSee [[Real Note|real]].")

    targets = [link.target_identifier for link in links]
    assert targets == ["Real Note"]


def test_link_extraction_handles_qualified_wikilinks_and_frontmatter() -> None:
    markdown = """---
source: 03_Reads/Book.md
project: Project VaultQ
related:
  - 10_Garden/Agent Ideas
---

See [[second-brain:10_Garden/Idea.md|Idea]] and [Plan](08_Plans/Plan.md).
"""

    links = extract_links(markdown)
    rows = {(link.target_identifier, link.edge_type, link.source_kind) for link in links}

    assert ("10_Garden/Idea", "links_to", "wikilink:second-brain") in rows
    assert ("08_Plans/Plan", "links_to", "markdown_link") in rows
    assert ("03_Reads/Book", "derived_from", "frontmatter:source") in rows
    assert ("Project VaultQ", "belongs_to_project", "frontmatter:project") in rows
    assert ("10_Garden/Agent Ideas", "related_to", "frontmatter:related") in rows


def test_basename_index_resolves_unqualified_wikilinks_to_paths() -> None:
    index = build_basename_index(["10_Garden/Idea.md", "20_Archive/Idea.md", "10_Garden/Other.md"])

    assert query_basename_index(index, "Idea") == ["10_Garden/Idea", "20_Archive/Idea"]
    assert query_basename_index(index, "Other.md") == ["10_Garden/Other"]
    assert query_basename_index(index, "Missing") == []
