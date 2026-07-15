from __future__ import annotations

import json
from pathlib import Path

from vaultq import agent_runtime
from vaultq.agent_runtime import (
    prepare_background_agent,
    run_autonomous_maintenance,
    run_autonomous_maintenance_if_changed,
)


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
```
""",
        encoding="utf-8",
    )
    obsidian = root / ".obsidian"
    obsidian.mkdir()
    (obsidian / "types.json").write_text(
        '{"types":{"type":"text","domain":"text","created":"date","created_by":"text"}}',
        encoding="utf-8",
    )
    policy = root / ".vaultq" / "policy.json"
    policy.parent.mkdir()
    policy.write_text('{"write_enabled": true, "default_mode": "ai_workspace"}', encoding="utf-8")


def test_prepare_background_agent_is_disabled_and_reversible(tmp_path: Path) -> None:
    _write_protocol(tmp_path)

    result = prepare_background_agent(tmp_path, write_sheet=True)

    assert result["ok"] is True
    assert result["started"] is False
    assert result["prepared_only"] is True
    config = json.loads((tmp_path / result["runtime_config"]).read_text(encoding="utf-8"))
    assert config["enabled"] is False
    assert config["workspace_root"] == "14_Agent_Workspace"
    assert config["rules"]["background_worker_only_when_mcp_active"] is True
    assert config["rules"]["background_worker_is_state_gated"] is True
    assert config["rules"]["promotion_actions_are_reversible"] is True
    sheet = result["review_sheet"]
    assert sheet["path"].startswith("14_Agent_Workspace/Review Queue/")
    assert (tmp_path / sheet["path"]).exists()


def test_autonomous_maintenance_promotes_workspace_notes_with_ledger(tmp_path: Path) -> None:
    _write_protocol(tmp_path)
    inbox = tmp_path / "14_Agent_Workspace" / "Inbox"
    inbox.mkdir(parents=True)
    source = inbox / "agent-idea.md"
    source.write_text(
        """---
type: concept
domain: ai
created: 2026-06-03
created_by: agent
---

# Agent Idea

An agent-authored idea ready for workspace promotion.
""",
        encoding="utf-8",
    )

    dry_run = run_autonomous_maintenance(tmp_path, dry_run=True)

    assert dry_run["dry_run"] is True
    assert dry_run["actions"][0]["action"] == "promote_workspace_note"
    assert source.exists()

    result = run_autonomous_maintenance(tmp_path, dry_run=False)

    target = tmp_path / "14_Agent_Workspace" / "Ideas" / "agent-idea.md"
    assert result["ok"] is True
    assert result["actions"][0]["source"] == "14_Agent_Workspace/Inbox/agent-idea.md"
    assert result["actions"][0]["target"] == "14_Agent_Workspace/Ideas/agent-idea.md"
    assert not source.exists()
    assert target.exists()
    assert (tmp_path / result["ledger_entries"][0]).exists()
    assert (tmp_path / result["run_report"]["path"]).exists()
    assert result["safety_receipt"]["path"].startswith(".vaultq/receipts/")
    receipt = json.loads((tmp_path / result["safety_receipt"]["path"]).read_text(encoding="utf-8"))
    assert receipt["operation"] == "agent_maintain"
    assert receipt["canonical_writes"] == 0
    assert receipt["rollback_actions"][0]["inverse_action"].startswith("move `14_Agent_Workspace/Ideas/")


def test_autonomous_maintenance_does_not_promote_human_notes(tmp_path: Path) -> None:
    _write_protocol(tmp_path)
    inbox = tmp_path / "14_Agent_Workspace" / "Inbox"
    inbox.mkdir(parents=True)
    source = inbox / "human-note.md"
    source.write_text(
        """---
type: concept
domain: ai
created: 2026-06-03
created_by: human
---

# Human Note
""",
        encoding="utf-8",
    )

    result = run_autonomous_maintenance(tmp_path, dry_run=False)

    assert result["ok"] is True
    assert result["actions"] == []
    assert source.exists()


def test_autonomous_maintenance_writes_idea_clusters_and_human_update(monkeypatch, tmp_path: Path) -> None:
    _write_protocol(tmp_path)
    monkeypatch.setattr(
        agent_runtime,
        "semantic_cluster_seed_nodes",
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError("semantic offline")),
    )
    ideas = tmp_path / "14_Agent_Workspace" / "Ideas"
    ideas.mkdir(parents=True)
    for title, topic in (("Agent Revenue Loop", "monetization"), ("Agent Offer Loop", "monetization")):
        (ideas / f"{title.lower().replace(' ', '-')}.md").write_text(
            f"""---
type: concept
domain: ai
created: 2026-06-03
created_by: agent
topic:
  - {topic}
---

# {title}

An agent-authored idea about {topic}.
""",
            encoding="utf-8",
        )

    result = run_autonomous_maintenance(tmp_path, dry_run=False)

    assert result["ok"] is True
    assert result["idea_clustering"]["cluster_count"] == 1
    assert result["idea_clustering"]["clusters"][0]["label"] == "monetization"
    cluster_path = tmp_path / result["idea_clustering"]["note"]["path"]
    human_update_path = tmp_path / result["human_update"]["path"]
    assert cluster_path.exists()
    assert "Agent Idea Clusters" in cluster_path.read_text(encoding="utf-8")
    assert human_update_path.exists()
    human_update = human_update_path.read_text(encoding="utf-8")
    assert "Latest Agent Workspace Update" in human_update
    assert "Idea clusters" in human_update
    assert "Safety Receipt" in human_update
    assert "Rollback" in human_update


def test_autonomous_maintenance_uses_semantic_clusters_for_related_sources(monkeypatch, tmp_path: Path) -> None:
    _write_protocol(tmp_path)
    ideas = tmp_path / "14_Agent_Workspace" / "Ideas"
    ideas.mkdir(parents=True)
    (ideas / "agent-revenue-loop.md").write_text(
        """---
type: concept
domain: ai
created: 2026-06-03
created_by: agent
---

# Agent Revenue Loop

Connect agent retrieval to offer and revenue ideas before writing new nodes.
""",
        encoding="utf-8",
    )
    captured_seeds: list[dict] = []

    def fake_semantic_cluster_seed_nodes(seeds, **kwargs):
        captured_seeds.extend(seeds)
        return {
            "ok": True,
            "cluster_algorithm": "semantic_seed_search",
            "seed_count": len(seeds),
            "cluster_count": 1,
            "session_policy": "exclude",
            "clusters": [
                {
                    "label": "Imported Offer Systems",
                    "anchor": {
                        "rel_path": "06_Library/Imported Blog Posts/Offer Systems.md",
                        "title": "Offer Systems",
                        "source_class": "imported_text",
                    },
                    "count": 1,
                    "seed_nodes": [
                        {
                            "rel_path": "14_Agent_Workspace/Ideas/agent-revenue-loop.md",
                            "title": "Agent Revenue Loop",
                            "node_type": "concept",
                            "domain": "ai",
                            "source_class": "agent_workspace",
                        }
                    ],
                    "related_nodes": [
                        {
                            "rel_path": "06_Library/Imported Blog Posts/Offer Systems.md",
                            "title": "Offer Systems",
                            "source_class": "imported_text",
                            "effective_score": 0.82,
                            "connection_count": 1,
                            "seed_paths": ["14_Agent_Workspace/Ideas/agent-revenue-loop.md"],
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr(agent_runtime, "semantic_cluster_seed_nodes", fake_semantic_cluster_seed_nodes, raising=False)

    result = run_autonomous_maintenance(tmp_path, dry_run=False)

    assert captured_seeds[0]["rel_path"] == "14_Agent_Workspace/Ideas/agent-revenue-loop.md"
    assert result["idea_clustering"]["cluster_algorithm"] == "semantic_seed_search"
    cluster_path = tmp_path / result["idea_clustering"]["note"]["path"]
    cluster_text = cluster_path.read_text(encoding="utf-8")
    assert "Related Nodes" in cluster_text
    assert "06_Library/Imported Blog Posts/Offer Systems" in cluster_text


def test_autonomous_maintenance_if_changed_is_state_gated(tmp_path: Path) -> None:
    _write_protocol(tmp_path)
    ideas = tmp_path / "14_Agent_Workspace" / "Ideas"
    ideas.mkdir(parents=True)
    source = ideas / "agent-loop.md"
    source.write_text(
        """---
type: concept
domain: ai
created: 2026-06-03
created_by: agent
topic:
  - maintenance
---

# Agent Loop

An agent-authored idea that should trigger one self-maintenance pass.
""",
        encoding="utf-8",
    )
    state_path = tmp_path / ".vaultq" / "self-maintenance-test.json"

    first = run_autonomous_maintenance_if_changed(tmp_path, state_path=state_path)

    assert first["ok"] is True
    assert first["ran"] is True
    assert state_path.exists()
    cluster_dir = tmp_path / "14_Agent_Workspace" / "Idea Clusters"
    first_cluster_count = len(list(cluster_dir.glob("*.md")))

    second = run_autonomous_maintenance_if_changed(tmp_path, state_path=state_path)

    assert second["ok"] is True
    assert second["ran"] is False
    assert second["reason"] == "workspace unchanged"
    assert len(list(cluster_dir.glob("*.md"))) == first_cluster_count

    source.write_text(source.read_text(encoding="utf-8") + "\nNew agent-maintained line.\n", encoding="utf-8")
    third = run_autonomous_maintenance_if_changed(tmp_path, state_path=state_path)

    assert third["ok"] is True
    assert third["ran"] is True
