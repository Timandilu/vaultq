from __future__ import annotations

from pathlib import Path

from vaultq.policy import PolicyDenied, WriteMode, load_policy


def test_ai_workspace_policy_allows_only_workspace(tmp_path: Path) -> None:
    policy_file = tmp_path / ".vaultq" / "policy.json"
    policy_file.parent.mkdir()
    policy_file.write_text('{"write_enabled": true, "default_mode": "ai_workspace"}', encoding="utf-8")
    policy = load_policy(tmp_path)

    allowed = policy.check_write("14_Agent_Workspace/Inbox/test.md")

    assert allowed.mode == WriteMode.AI_WORKSPACE


def test_ai_workspace_policy_denies_canonical_write(tmp_path: Path) -> None:
    policy_file = tmp_path / ".vaultq" / "policy.json"
    policy_file.parent.mkdir()
    policy_file.write_text('{"write_enabled": true, "default_mode": "ai_workspace"}', encoding="utf-8")
    policy = load_policy(tmp_path)

    try:
        policy.check_write("05_Reports/test.md")
    except PolicyDenied as exc:
        assert "14_Agent_Workspace" in str(exc)
    else:
        raise AssertionError("expected canonical write denial")


def test_policy_denies_path_traversal(tmp_path: Path) -> None:
    policy = load_policy(tmp_path)

    try:
        policy.check_write("../outside.md")
    except PolicyDenied as exc:
        assert "outside" in str(exc).lower() or "traversal" in str(exc).lower()
    else:
        raise AssertionError("expected traversal denial")
