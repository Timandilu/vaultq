from __future__ import annotations

from pathlib import Path

from vaultq.write_ops import capture_note, propose_note, propose_property


def _write_protocol(root: Path) -> None:
    protocol = root / "00_System" / "Protocol - Layout and Tagging.md"
    protocol.parent.mkdir(parents=True)
    protocol.write_text(
        """# Protocol

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
    policy = root / ".vaultq" / "policy.json"
    policy.parent.mkdir()
    policy.write_text('{"write_enabled": true, "default_mode": "ai_workspace"}', encoding="utf-8")


def test_capture_note_writes_agent_frontmatter_and_receipt(tmp_path: Path) -> None:
    _write_protocol(tmp_path)

    result = capture_note(
        tmp_path,
        text="This is an agent idea.",
        kind="idea",
        domain="ai",
        title="Agent Idea",
    )

    note_path = tmp_path / result["path"]
    receipt_path = tmp_path / result["receipt_path"]
    assert result["ok"] is True
    assert note_path.exists()
    assert receipt_path.exists()
    text = note_path.read_text(encoding="utf-8")
    assert "type: concept" in text
    assert "domain: ai" in text
    assert "created_by: agent" in text


def test_property_proposal_uses_valid_protocol_frontmatter(tmp_path: Path) -> None:
    _write_protocol(tmp_path)

    result = propose_property(tmp_path, property_name="review_state", reason="Need structured review queue status.")

    proposal_path = tmp_path / result["path"]
    assert result["ok"] is True
    text = proposal_path.read_text(encoding="utf-8")
    assert "type: plan" in text
    assert "domain: system" in text
    assert "created_by: agent" in text
    assert "review_state" in text


def test_note_proposal_uses_agent_workspace_and_review_contract(tmp_path: Path) -> None:
    _write_protocol(tmp_path)

    result = propose_note(tmp_path, title="Promote Idea", body="Move this idea after review.")

    proposal_path = tmp_path / result["path"]
    assert result["ok"] is True
    assert result["path"].startswith("14_Agent_Workspace/Proposals/")
    text = proposal_path.read_text(encoding="utf-8")
    assert "type: plan" in text
    assert "domain: system" in text
    assert "created_by: agent" in text
    assert "State: proposed" in text
    assert "reversible review sheet" in text
