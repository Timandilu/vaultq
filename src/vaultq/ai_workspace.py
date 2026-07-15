from __future__ import annotations

from pathlib import Path


AI_WORKSPACE_ROOT = "14_Agent_Workspace"
AI_WORKSPACE_DIRS = (
    "Inbox",
    "Notes",
    "Research",
    "Ideas",
    "Questions",
    "Proposals",
    "Runs",
    "Reports",
    "Idea Clusters",
    "Human Updates",
    "Review Queue",
    "Promotion Ledger",
    "Property Proposals",
)


def workspace_root(root: Path) -> Path:
    return Path(root).expanduser().resolve() / AI_WORKSPACE_ROOT


def workspace_rel_path(*parts: str) -> str:
    clean = [AI_WORKSPACE_ROOT]
    clean.extend(part.strip("/\\") for part in parts if part and part.strip("/\\"))
    return "/".join(clean)


def ensure_workspace(root: Path) -> dict[str, str]:
    base = workspace_root(root)
    base.mkdir(parents=True, exist_ok=True)
    paths = {"root": str(base)}
    for name in AI_WORKSPACE_DIRS:
        path = base / name
        path.mkdir(parents=True, exist_ok=True)
        paths[name] = str(path)
    return paths
