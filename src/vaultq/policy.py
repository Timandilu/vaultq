from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Optional

from vaultq.ai_workspace import AI_WORKSPACE_ROOT


class PolicyDenied(ValueError):
    pass


class WriteMode(str, Enum):
    READ_ONLY = "read_only"
    AI_WORKSPACE = "ai_workspace"
    PROPOSAL = "proposal"
    TRUSTED_LOCAL = "trusted_local"
    ADMIN = "admin"


@dataclass(frozen=True)
class PolicyDecision:
    mode: WriteMode
    rel_path: str
    abs_path: Path
    reason: str


@dataclass(frozen=True)
class WritePolicy:
    root: Path
    path: Path
    write_enabled: bool = False
    default_mode: WriteMode = WriteMode.READ_ONLY
    allowed_roots: tuple[str, ...] = (AI_WORKSPACE_ROOT,)
    proposal_roots: tuple[str, ...] = ("05_Reports", "08_Plans", "10_Garden")
    deny_roots: tuple[str, ...] = (".git", ".obsidian", ".trash")
    max_write_bytes: int = 100000
    require_source_for_canonical_edit: bool = True
    allow_delete: bool = False
    allow_move: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_path": str(self.path),
            "root": str(self.root),
            "write_enabled": self.write_enabled,
            "default_mode": self.default_mode.value,
            "allowed_roots": list(self.allowed_roots),
            "proposal_roots": list(self.proposal_roots),
            "deny_roots": list(self.deny_roots),
            "max_write_bytes": self.max_write_bytes,
            "require_source_for_canonical_edit": self.require_source_for_canonical_edit,
            "allow_delete": self.allow_delete,
            "allow_move": self.allow_move,
        }

    def normalize_rel_path(self, rel_path: str) -> str:
        candidate = (rel_path or "").replace("\\", "/").strip()
        if not candidate:
            raise PolicyDenied("Write path is required.")
        if Path(candidate).is_absolute():
            raise PolicyDenied("Absolute write paths are not allowed; use a vault-relative path.")
        normalized = PurePosixPath(candidate)
        if any(part in ("", ".", "..") for part in normalized.parts):
            raise PolicyDenied("Path traversal or empty path segments are not allowed.")
        return normalized.as_posix()

    def resolve_abs_path(self, rel_path: str) -> Path:
        normalized = self.normalize_rel_path(rel_path)
        root = self.root.resolve()
        target = (root / normalized).resolve()
        if target != root and root not in target.parents:
            raise PolicyDenied("Resolved write path escapes the collection root.")
        return target

    def _is_under_any(self, rel_path: str, roots: Iterable[str]) -> bool:
        normalized = rel_path.replace("\\", "/").strip("/")
        for root in roots:
            clean = root.replace("\\", "/").strip("/")
            if normalized == clean or normalized.startswith(f"{clean}/"):
                return True
        return False

    def check_write(self, rel_path: str, *, mode: Optional[str] = None, byte_count: int = 0) -> PolicyDecision:
        normalized = self.normalize_rel_path(rel_path)
        abs_path = self.resolve_abs_path(normalized)
        active_mode = WriteMode((mode or self.default_mode.value).strip())
        if not self.write_enabled and active_mode != WriteMode.ADMIN:
            raise PolicyDenied("Write policy is disabled. Run `vq policy init --mode ai_workspace` first.")
        if byte_count and byte_count > self.max_write_bytes:
            raise PolicyDenied(f"Write payload exceeds max_write_bytes={self.max_write_bytes}.")
        if self._is_under_any(normalized, self.deny_roots):
            raise PolicyDenied(f"Writes under denied roots are not allowed: {normalized}")
        if active_mode == WriteMode.READ_ONLY:
            raise PolicyDenied("Write policy is read_only.")
        if active_mode == WriteMode.AI_WORKSPACE:
            if self._is_under_any(normalized, self.allowed_roots):
                return PolicyDecision(active_mode, normalized, abs_path, "Allowed AI workspace write.")
            raise PolicyDenied(f"Only {', '.join(self.allowed_roots)} writes are allowed in ai_workspace mode.")
        if active_mode == WriteMode.PROPOSAL:
            if self._is_under_any(normalized, self.allowed_roots):
                return PolicyDecision(active_mode, normalized, abs_path, "Allowed AI workspace write.")
            if self._is_under_any(normalized, self.proposal_roots):
                raise PolicyDenied("Canonical roots require a proposal note, not a direct write.")
            raise PolicyDenied("Path is outside allowed workspace and proposal roots.")
        if active_mode in (WriteMode.TRUSTED_LOCAL, WriteMode.ADMIN):
            return PolicyDecision(active_mode, normalized, abs_path, "Allowed trusted local write.")
        raise PolicyDenied(f"Unsupported write mode: {active_mode.value}")


def default_policy_path(root: Path) -> Path:
    return root / ".vaultq" / "policy.json"


def load_policy(root: Path, policy_path: Optional[Path] = None) -> WritePolicy:
    root = Path(root).expanduser().resolve()
    explicit = os.getenv("VQ_WRITE_POLICY_FILE", "").strip()
    path = Path(explicit).expanduser().resolve() if explicit else Path(policy_path or default_policy_path(root))
    data: Dict[str, Any] = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    mode_value = str(data.get("default_mode") or ("ai_workspace" if data.get("write_enabled") else "read_only"))
    return WritePolicy(
        root=root,
        path=path,
        write_enabled=bool(data.get("write_enabled", False)),
        default_mode=WriteMode(mode_value),
        allowed_roots=tuple(data.get("allowed_roots") or (AI_WORKSPACE_ROOT,)),
        proposal_roots=tuple(data.get("proposal_roots") or ("05_Reports", "08_Plans", "10_Garden")),
        deny_roots=tuple(data.get("deny_roots") or (".git", ".obsidian", ".trash")),
        max_write_bytes=int(data.get("max_write_bytes") or 100000),
        require_source_for_canonical_edit=bool(data.get("require_source_for_canonical_edit", True)),
        allow_delete=bool(data.get("allow_delete", False)),
        allow_move=bool(data.get("allow_move", False)),
        raw=data,
    )


def init_policy(root: Path, mode: str = "ai_workspace", *, overwrite: bool = False) -> WritePolicy:
    root = Path(root).expanduser().resolve()
    path = default_policy_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return load_policy(root, path)
    data = {
        "write_enabled": mode != WriteMode.READ_ONLY.value,
        "default_mode": mode,
        "allowed_roots": [AI_WORKSPACE_ROOT],
        "proposal_roots": ["05_Reports", "08_Plans", "10_Garden"],
        "deny_roots": [".git", ".obsidian", ".trash"],
        "max_write_bytes": 100000,
        "require_source_for_canonical_edit": True,
        "allow_delete": False,
        "allow_move": False,
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return load_policy(root, path)
