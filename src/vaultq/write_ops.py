from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from vaultq.ai_workspace import ensure_workspace, workspace_rel_path
from vaultq.policy import load_policy
from vaultq.property_schema import frontmatter_to_markdown, load_property_schema, validate_frontmatter
from vaultq.receipts import operation_id, write_receipt


KIND_TO_TYPE = {
    "idea": "concept",
    "note": "note",
    "question": "note",
    "finding": "report",
    "report": "report",
    "draft": "draft",
}
KIND_TO_DIR = {
    "idea": "Ideas",
    "note": "Notes",
    "question": "Questions",
    "finding": "Reports",
    "report": "Reports",
    "draft": "Inbox",
}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._ -]+", "", value or "").strip().lower()
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug[:80].strip("-") or "agent-note"


def _today() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d")


def _file_hash(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_markdown(root: Path, rel_path: str, frontmatter: Dict[str, Any], body: str, *, operation: str) -> Dict[str, Any]:
    root = Path(root).expanduser().resolve()
    policy = load_policy(root)
    markdown = f"---\n{frontmatter_to_markdown(frontmatter)}\n---\n\n{body.strip()}\n"
    decision = policy.check_write(rel_path, byte_count=len(markdown.encode("utf-8")))
    schema = load_property_schema(root)
    validation = validate_frontmatter(frontmatter, schema=schema, agent_generated=frontmatter.get("created_by") == "agent")
    old_hash = _file_hash(decision.abs_path)
    decision.abs_path.parent.mkdir(parents=True, exist_ok=True)
    decision.abs_path.write_text(markdown, encoding="utf-8", newline="\n")
    new_hash = _file_hash(decision.abs_path)
    op_id = operation_id(operation, decision.rel_path)
    receipt_path = write_receipt(
        root,
        {
            "operation_id": op_id,
            "operation": operation,
            "actor": "agent",
            "mode": decision.mode.value,
            "path": decision.rel_path,
            "old_hash": old_hash,
            "new_hash": new_hash,
            "property_validation": validation,
            "diff_summary": {"added_lines": len(markdown.splitlines()), "removed_lines": 0 if old_hash is None else None},
        },
    )
    return {
        "ok": True,
        "operation_id": op_id,
        "mode": decision.mode.value,
        "path": decision.rel_path,
        "receipt_path": receipt_path,
        "indexed": False,
        "next_step": "background indexing/embedding will pick this up; run `vq index --collection <name> --limit 1 && vq embed --limit 20` only if immediate semantic retrieval is required",
    }


def capture_note(
    root: Path,
    *,
    text: str,
    kind: str = "note",
    domain: str = "general",
    title: Optional[str] = None,
    source: Optional[Iterable[str]] = None,
    topic: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    ensure_workspace(root)
    normalized_kind = (kind or "note").strip().lower()
    note_type = KIND_TO_TYPE.get(normalized_kind)
    if not note_type:
        raise ValueError(f"Unsupported capture kind: {kind}")
    title = (title or text.splitlines()[0] if text.strip() else "Agent Note").strip()[:120]
    created = _today()
    rel_path = workspace_rel_path(KIND_TO_DIR.get(normalized_kind, "Notes"), f"{created}-{_slugify(title)}.md")
    frontmatter: Dict[str, Any] = {
        "type": note_type,
        "domain": domain,
        "created": created,
        "created_by": "agent",
    }
    sources = [item for item in (source or []) if str(item).strip()]
    if sources:
        frontmatter["source"] = ", ".join(str(item).strip() for item in sources)
    topics = [str(item).strip() for item in (topic or []) if str(item).strip()]
    if topics:
        frontmatter["topic"] = topics
    body = f"# {title}\n\n{text.strip()}"
    return _write_markdown(Path(root), rel_path, frontmatter, body, operation="capture")


def propose_property(root: Path, *, property_name: str, reason: str, tag: bool = False) -> Dict[str, Any]:
    ensure_workspace(root)
    created = _today()
    clean_name = _slugify(property_name)
    rel_path = workspace_rel_path("Property Proposals", f"{created}-{clean_name}.md")
    frontmatter = {
        "type": "plan",
        "domain": "system",
        "created": created,
        "created_by": "agent",
    }
    target = "tag" if tag else "property"
    body = (
        f"# Property Proposal - {property_name}\n\n"
        f"## Target\n\nPropose new {target}: `{property_name}`\n\n"
        f"## Reason\n\n{reason.strip()}\n\n"
        "## Required Approval Changes\n\n"
        "- Update `00_System/Protocol - Layout and Tagging.md`.\n"
        "- Update `.obsidian/types.json` when this is a property.\n"
        "- Regenerate `.vaultq/property_schema.json` or `.vaultq/tag_rules.json`.\n"
        "- Run `vq property validate` against affected notes.\n"
    )
    return _write_markdown(Path(root), rel_path, frontmatter, body, operation="property_propose")


def propose_note(root: Path, *, title: str, body: str) -> Dict[str, Any]:
    ensure_workspace(root)
    clean_title = (title or "Agent Proposal").strip()
    if not clean_title:
        clean_title = "Agent Proposal"
    created = _today()
    rel_path = workspace_rel_path("Proposals", f"{created}-{_slugify(clean_title)}.md")
    frontmatter = {
        "type": "plan",
        "domain": "system",
        "created": created,
        "created_by": "agent",
    }
    note_body = (
        f"# Proposal - {clean_title}\n\n"
        f"State: proposed\n\n"
        f"{body.strip()}\n\n"
        "## Review Contract\n\n"
        "- This proposal is agent-authored and not a canonical vault change.\n"
        "- Promote only through a reversible review sheet with before/after hashes.\n"
    )
    return _write_markdown(Path(root), rel_path, frontmatter, note_body, operation="note_propose")


def note_put(root: Path, *, rel_path: str, markdown: str, agent_generated: bool = True) -> Dict[str, Any]:
    from vaultq.property_schema import parse_frontmatter

    root = Path(root).expanduser().resolve()
    policy = load_policy(root)
    decision = policy.check_write(rel_path, byte_count=len(markdown.encode("utf-8")))
    frontmatter = parse_frontmatter(markdown)
    validation = validate_frontmatter(frontmatter, root=root, agent_generated=agent_generated)
    old_hash = _file_hash(decision.abs_path)
    decision.abs_path.parent.mkdir(parents=True, exist_ok=True)
    decision.abs_path.write_text(markdown.rstrip() + "\n", encoding="utf-8", newline="\n")
    op_id = operation_id("note_put", decision.rel_path)
    receipt_path = write_receipt(
        root,
        {
            "operation_id": op_id,
            "operation": "note_put",
            "actor": "agent",
            "mode": decision.mode.value,
            "path": decision.rel_path,
            "old_hash": old_hash,
            "new_hash": _file_hash(decision.abs_path),
            "property_validation": validation,
        },
    )
    return {"ok": True, "operation_id": op_id, "mode": decision.mode.value, "path": decision.rel_path, "receipt_path": receipt_path}


def note_append(root: Path, *, rel_path: str, text: str, heading: Optional[str] = None) -> Dict[str, Any]:
    root = Path(root).expanduser().resolve()
    policy = load_policy(root)
    decision = policy.check_write(rel_path, byte_count=len(text.encode("utf-8")))
    if not decision.abs_path.exists():
        raise FileNotFoundError(f"Cannot append to missing note: {rel_path}")
    old_hash = _file_hash(decision.abs_path)
    prefix = f"\n\n## {heading.strip()}\n\n" if heading else "\n\n"
    with decision.abs_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(prefix + text.strip() + "\n")
    op_id = operation_id("note_append", decision.rel_path)
    receipt_path = write_receipt(
        root,
        {
            "operation_id": op_id,
            "operation": "note_append",
            "actor": "agent",
            "mode": decision.mode.value,
            "path": decision.rel_path,
            "old_hash": old_hash,
            "new_hash": _file_hash(decision.abs_path),
        },
    )
    return {"ok": True, "operation_id": op_id, "mode": decision.mode.value, "path": decision.rel_path, "receipt_path": receipt_path}
