from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from vaultq.ai_workspace import AI_WORKSPACE_ROOT, workspace_rel_path
from vaultq.write_ops import capture_note


def inspect_ai_workspace(root: Path) -> Dict[str, Any]:
    root = Path(root).expanduser().resolve()
    workspace = root / AI_WORKSPACE_ROOT
    files = sorted(workspace.rglob("*.md")) if workspace.exists() else []
    proposals = [path for path in files if "Proposals" in path.parts or "Property Proposals" in path.parts]
    reports = [path for path in files if "Reports" in path.parts or "Runs" in path.parts]
    return {
        "root": str(root),
        "ai_workspace": str(workspace),
        "exists": workspace.exists(),
        "markdown_files": len(files),
        "proposal_files": len(proposals),
        "report_files": len(reports),
        "files": [path.relative_to(root).as_posix() for path in files[:200]],
    }


def maintenance_report(root: Path, *, write_report: bool = False) -> Dict[str, Any]:
    payload = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "ai_workspace": inspect_ai_workspace(root),
        "destructive_actions": [],
        "recommendations": [],
    }
    if not payload["ai_workspace"]["exists"]:
        payload["recommendations"].append(f"Create `{workspace_rel_path()}` before enabling agent writes.")
    if write_report:
        text = (
            f"Checked at: {payload['checked_at']}\n\n"
            f"AI workspace files: {payload['ai_workspace']['markdown_files']}\n"
            f"Proposal files: {payload['ai_workspace']['proposal_files']}\n"
            f"Report files: {payload['ai_workspace']['report_files']}\n\n"
            "No destructive actions were performed."
        )
        payload["saved_report"] = capture_note(root, text=text, kind="report", domain="system", title="VaultQ Maintenance Report")
    return payload
