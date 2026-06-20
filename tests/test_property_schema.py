from __future__ import annotations

from pathlib import Path

from vaultq.property_schema import PropertyValidationError, load_property_schema, validate_frontmatter


def _write_protocol(root: Path) -> None:
    protocol = root / "00_System" / "Protocol - Layout and Tagging.md"
    protocol.parent.mkdir(parents=True)
    protocol.write_text(
        """---
type: system
domain: system
created: 2026-06-03
---

# Protocol

```yaml
type: article | concept | note | read | report | plan | project | resource | draft | system | moc | runbook | sop
domain: ai | ecom | content | strategy | personal | trading | sport | system | general
created: YYYY-MM-DD
created_by: optional; required for agent-generated notes
source: optional
project: optional
topic: optional list for retrieval themes
```
""",
        encoding="utf-8",
    )
    obsidian = root / ".obsidian"
    obsidian.mkdir()
    (obsidian / "types.json").write_text(
        '{"types":{"type":"text","domain":"text","created":"date","created_by":"text","source":"text","project":"text","topic":"multitext","tags":"tags","status":"text"}}',
        encoding="utf-8",
    )


def test_agent_frontmatter_requires_vault_oo_fields(tmp_path: Path) -> None:
    _write_protocol(tmp_path)
    schema = load_property_schema(tmp_path)

    assert "concept" in schema.allowed_types
    assert "ai" in schema.allowed_domains

    result = validate_frontmatter(
        {
            "type": "concept",
            "domain": "ai",
            "created": "2026-06-03",
            "created_by": "agent",
            "topic": ["agentic-second-brain"],
        },
        schema=schema,
        agent_generated=True,
    )

    assert result["ok"] is True


def test_agent_frontmatter_rejects_invented_node_type(tmp_path: Path) -> None:
    _write_protocol(tmp_path)
    schema = load_property_schema(tmp_path)

    try:
        validate_frontmatter(
            {
                "type": "agent_question",
                "domain": "ai",
                "created": "2026-06-03",
                "created_by": "agent",
            },
            schema=schema,
            agent_generated=True,
        )
    except PropertyValidationError as exc:
        assert "type" in str(exc)
    else:
        raise AssertionError("expected invented agent type to fail")


def test_human_frontmatter_does_not_require_created_by(tmp_path: Path) -> None:
    _write_protocol(tmp_path)
    schema = load_property_schema(tmp_path)

    result = validate_frontmatter(
        {"type": "note", "domain": "general", "created": "2026-06-03"},
        schema=schema,
        agent_generated=False,
    )

    assert result["ok"] is True


def test_property_mirror_tags_are_rejected(tmp_path: Path) -> None:
    _write_protocol(tmp_path)
    schema = load_property_schema(tmp_path)

    try:
        validate_frontmatter(
            {
                "type": "note",
                "domain": "ai",
                "created": "2026-06-03",
                "created_by": "agent",
                "tags": ["type/note"],
            },
            schema=schema,
            agent_generated=True,
        )
    except PropertyValidationError as exc:
        assert "tag" in str(exc).lower()
    else:
        raise AssertionError("expected property mirror tag to fail")


def test_parse_frontmatter_tolerates_template_yaml(tmp_path: Path) -> None:
    from vaultq.property_schema import parse_frontmatter

    assert parse_frontmatter("---\n{{ card_data }}\n---\n# Template") == {}
