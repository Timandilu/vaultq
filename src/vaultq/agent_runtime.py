from __future__ import annotations

import json
import hashlib
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from vaultq.ai_workspace import AI_WORKSPACE_DIRS, ensure_workspace, workspace_rel_path
from vaultq.property_schema import parse_frontmatter
from vaultq.receipts import write_receipt
from vaultq.semantic_clusters import semantic_cluster_seed_nodes
from vaultq.write_ops import note_put


AGENT_RUNTIME_FILE = ".vaultq/agent_runtime.json"
SELF_MAINTENANCE_STATE_FILE = ".vaultq/self_maintenance_state.json"


def _today() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d")


def _now_slug() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d-%H%M%S")


def runtime_config(root: Path) -> Dict[str, Any]:
    root = Path(root).expanduser().resolve()
    return {
        "enabled": False,
        "prepared_at": datetime.now().astimezone().isoformat(),
        "runtime": "vaultq.autonomous_self_maintenance",
        "workspace_root": workspace_rel_path(),
        "review_queue": workspace_rel_path("Review Queue"),
        "promotion_ledger": workspace_rel_path("Promotion Ledger"),
        "mode": "autonomous_first_reversible",
        "start_command": "vq agent maintain --collection <name> --max-actions 20",
        "activation_contract": "run `vq agent maintain --collection <name>` explicitly, or enable the state-gated loop that runs only inside the MCP process",
        "prepared_only": True,
        "rules": {
            "background_worker_only_when_mcp_active": True,
            "background_worker_is_state_gated": True,
            "workspace_self_maintenance_is_autonomous": True,
            "canonical_writes_require_explicit_promotion": True,
            "write_ledger_before_or_during_promote": True,
            "promotion_actions_are_reversible": True,
            "specific_agent_identity_not_required": True,
        },
        "rollback_contract": {
            "before_hash_required": True,
            "after_hash_required": True,
            "inverse_action_required": True,
            "ledger_path": workspace_rel_path("Promotion Ledger"),
        },
    }


def write_runtime_config(root: Path) -> Path:
    root = Path(root).expanduser().resolve()
    path = root / AGENT_RUNTIME_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(runtime_config(root), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _file_hash(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_dir_for_frontmatter(frontmatter: Dict[str, Any]) -> str:
    note_type = str(frontmatter.get("type") or "note").strip().lower()
    return {
        "concept": "Ideas",
        "report": "Reports",
        "plan": "Proposals",
        "draft": "Inbox",
        "read": "Research",
        "resource": "Research",
        "article": "Research",
        "project": "Research",
        "runbook": "Research",
        "sop": "Research",
        "system": "Research",
        "moc": "Research",
        "note": "Notes",
    }.get(note_type, "Notes")


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for index in range(2, 1000):
        candidate = parent / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not allocate unique path near {path}")


def _is_maintenance_area(path: Path) -> bool:
    maintenance_dirs = {
        "Review Queue",
        "Promotion Ledger",
        "Runs",
        "Reports",
        "Property Proposals",
        "Idea Clusters",
        "Human Updates",
    }
    return any(part in maintenance_dirs for part in path.parts)


def _note_title(markdown: str, path: Path) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or path.stem
    return path.stem.replace("-", " ").strip().title() or "Untitled"


def _topic_values(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    return [str(value).strip().lower()] if str(value).strip() else []


_CLUSTER_STOPWORDS = {
    "about",
    "agent",
    "agentic",
    "and",
    "from",
    "idea",
    "into",
    "note",
    "notes",
    "that",
    "the",
    "this",
    "with",
    "workspace",
}


def _keyword_label(title: str, body: str) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", f"{title} {body}".lower())
    counts: dict[str, int] = {}
    for word in words:
        if word in _CLUSTER_STOPWORDS:
            continue
        counts[word] = counts.get(word, 0) + 1
    if not counts:
        return "general"
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _cluster_label(frontmatter: Dict[str, Any], title: str, body: str) -> str:
    topics = _topic_values(frontmatter.get("topic"))
    if topics:
        return topics[0]
    domain = str(frontmatter.get("domain") or "").strip().lower()
    if domain and domain != "general":
        return domain
    return _keyword_label(title, body)


def _agent_idea_candidates(root: Path) -> list[Dict[str, Any]]:
    workspace = root / workspace_rel_path()
    candidates: list[Dict[str, Any]] = []
    for path in sorted(workspace.rglob("*.md")):
        rel_workspace = path.relative_to(workspace)
        if _is_maintenance_area(rel_workspace):
            continue
        try:
            markdown = path.read_text(encoding="utf-8")
        except Exception:
            continue
        frontmatter = parse_frontmatter(markdown)
        if frontmatter.get("created_by") != "agent":
            continue
        note_type = str(frontmatter.get("type") or "").strip().lower()
        if note_type not in {"concept", "note", "plan", "draft", "report"}:
            continue
        title = _note_title(markdown, path)
        body = re.sub(r"^---\n.*?\n---\n?", "", markdown, flags=re.DOTALL)
        rel_path = path.relative_to(root).as_posix()
        candidates.append(
            {
                "rel_path": rel_path,
                "path": rel_path,
                "title": title,
                "type": note_type,
                "node_type": note_type,
                "domain": str(frontmatter.get("domain") or "general"),
                "topics": _topic_values(frontmatter.get("topic")),
                "label": _cluster_label(frontmatter, title, body),
                "text": body[:2400],
            }
        )
    return candidates


def _agent_workspace_files(root: Path) -> list[Path]:
    root = Path(root).expanduser().resolve()
    workspace = root / workspace_rel_path()
    if not workspace.exists():
        return []
    files: list[Path] = []
    for path in sorted(workspace.rglob("*.md")):
        rel_workspace = path.relative_to(workspace)
        if _is_maintenance_area(rel_workspace):
            continue
        try:
            markdown = path.read_text(encoding="utf-8")
        except Exception:
            continue
        frontmatter = parse_frontmatter(markdown)
        if frontmatter.get("created_by") == "agent":
            files.append(path)
    return files


def agent_workspace_signal(root: Path) -> Dict[str, Any]:
    root = Path(root).expanduser().resolve()
    digest = hashlib.sha256()
    paths: list[str] = []
    for path in _agent_workspace_files(root):
        rel_path = path.relative_to(root).as_posix()
        paths.append(rel_path)
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update((_file_hash(path) or "").encode("ascii"))
        digest.update(b"\0")
    return {
        "hash": digest.hexdigest(),
        "agent_note_count": len(paths),
        "paths": paths,
    }


def _keyword_cluster_agent_ideas(root: Path) -> Dict[str, Any]:
    candidates = _agent_idea_candidates(root)
    grouped: dict[str, list[Dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate["label"], []).append(candidate)
    clusters = [
        {
            "label": label,
            "count": len(items),
            "items": sorted(items, key=lambda item: item["path"]),
        }
        for label, items in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    ]
    return {
        "cluster_algorithm": "topic_domain_keyword_fallback",
        "cluster_count": len(clusters),
        "candidate_count": len(candidates),
        "seed_count": len(candidates),
        "clusters": clusters,
    }


def _cluster_agent_ideas(root: Path) -> Dict[str, Any]:
    candidates = _agent_idea_candidates(root)
    if not candidates:
        return {
            "cluster_algorithm": "semantic_seed_search",
            "cluster_count": 0,
            "candidate_count": 0,
            "seed_count": 0,
            "clusters": [],
        }
    try:
        result = semantic_cluster_seed_nodes(
            candidates,
            related_limit=5,
            min_score=0.35,
            session_policy="exclude",
            exclude_path_prefixes=[
                workspace_rel_path("Idea Clusters"),
                workspace_rel_path("Human Updates"),
                workspace_rel_path("Promotion Ledger"),
                workspace_rel_path("Property Proposals"),
                workspace_rel_path("Review Queue"),
                workspace_rel_path("Runs"),
                workspace_rel_path("Reports"),
            ],
        )
        result["candidate_count"] = result.get("seed_count", len(candidates))
        return result
    except Exception as exc:
        fallback = _keyword_cluster_agent_ideas(root)
        fallback["semantic_error"] = {
            "type": exc.__class__.__name__,
            "message": str(exc)[:500],
        }
        return fallback


def _cluster_markdown(*, clustering: Dict[str, Any], created: str) -> str:
    sections: list[str] = []
    for cluster in clustering["clusters"]:
        if "seed_nodes" in cluster:
            seed_rows = []
            for item in cluster.get("seed_nodes") or []:
                seed_rows.append(
                    f"- `[{item.get('node_type', '')}/{item.get('domain', '')}]` [[{item['rel_path'].removesuffix('.md')}|{item['title']}]]"
                )
            related_rows = []
            for item in cluster.get("related_nodes") or []:
                related_rows.append(
                    f"- `[{item.get('source_class', 'vault_note')}]` [[{item['rel_path'].removesuffix('.md')}|{item['title']}]] "
                    f"(score `{float(item.get('effective_score') or 0.0):.3f}`, connections `{item.get('connection_count', 1)}`)"
                )
            anchor = cluster.get("anchor") or {}
            sections.append(
                f"## {cluster['label']}\n\n"
                f"- semantic anchor: [[{str(anchor.get('rel_path') or '').removesuffix('.md')}|{anchor.get('title') or cluster['label']}]]\n"
                f"- anchor source: `{anchor.get('source_class') or 'unknown'}`\n\n"
                "### Seed Nodes\n\n"
                + ("\n".join(seed_rows) or "- _No seed nodes_")
                + "\n\n### Related Nodes\n\n"
                + ("\n".join(related_rows) or "- _No related indexed nodes found above threshold_")
            )
        else:
            rows = []
            for item in cluster["items"]:
                rows.append(
                    f"- `[{item['type']}/{item['domain']}]` [[{item['path'].removesuffix('.md')}|{item['title']}]]"
                )
            sections.append(f"## {cluster['label']}\n\n" + ("\n".join(rows) or "- _No notes_"))
    body = "\n\n".join(sections) or "_No agent-authored idea notes found._"
    semantic_error = clustering.get("semantic_error")
    error_text = ""
    if semantic_error:
        error_text = (
            "\n\n## Semantic Fallback\n\n"
            f"- fallback reason: `{semantic_error.get('type')}`\n"
            f"- message: `{semantic_error.get('message')}`\n"
        )
    return f"""---
type: report
domain: system
created: {created}
created_by: agent
---

# Agent Idea Clusters

This note is generated by VaultQ autonomous self-maintenance. It clusters agent-authored workspace ideas by semantic proximity to indexed vault nodes, so agents can reuse related thinking and connect ideas to imported/source texts without creating random new node types or tags.

- cluster algorithm: `{clustering.get('cluster_algorithm', 'unknown')}`
- candidate notes: `{clustering.get('candidate_count', clustering.get('seed_count', 0))}`
- clusters: `{clustering['cluster_count']}`
- session policy: `{clustering.get('session_policy', 'exclude')}`

{body}
{error_text}
"""


def _human_update_markdown(
    *,
    actions: list[Dict[str, Any]],
    skipped: list[Dict[str, Any]],
    clustering: Dict[str, Any],
    run_report_path: Optional[str],
    cluster_note_path: Optional[str],
    safety_receipt_path: Optional[str],
    rollback_actions: list[Dict[str, Any]],
    created: str,
    dry_run: bool,
) -> str:
    action_lines = [
        f"- moved [[{row['source'].removesuffix('.md')}]] -> [[{row['target'].removesuffix('.md')}]]"
        for row in actions
    ] or ["- no workspace promotions were needed"]
    cluster_lines = [
        f"- `{cluster['label']}`: {cluster['count']} note(s)"
        for cluster in clustering["clusters"][:12]
    ] or ["- no agent idea clusters found"]
    rollback_lines = [
        f"- `{row['inverse_action']}`"
        for row in rollback_actions
    ] or ["- no rollback actions were needed"]
    return f"""---
type: report
domain: system
created: {created}
created_by: agent
---

# Latest Agent Workspace Update

This is the human-facing summary of the latest VaultQ autonomous self-maintenance pass.

- dry_run: `{str(dry_run).lower()}`
- canonical writes: `0`
- promotions: `{len(actions)}`
- skipped notes: `{len(skipped)}`
- Idea clusters: `{clustering['cluster_count']}`
- run report: `{run_report_path or 'not written'}`
- cluster note: `{cluster_note_path or 'not written'}`
- safety receipt: `{safety_receipt_path or 'not written'}`

## Promotion Changes

{chr(10).join(action_lines)}

## Rollback

{chr(10).join(rollback_lines)}

## Idea Clusters

{chr(10).join(cluster_lines)}

## Safety Receipt

Machine-readable receipt: `{safety_receipt_path or 'not written'}`

## Contract

VaultQ may autonomously maintain `14_Agent_Workspace`. Canonical vault folders still require explicit promotion evidence. Every workspace promotion keeps before/after hashes in the promotion ledger.
"""


def _ledger_markdown(*, action: Dict[str, Any], created: str) -> str:
    return f"""---
type: report
domain: system
created: {created}
created_by: agent
---

# Autonomous Promotion Ledger

## Action

- operation: `{action["action"]}`
- source: `{action["source"]}`
- target: `{action["target"]}`
- before_hash: `{action.get("before_hash") or ""}`
- after_hash: `{action.get("after_hash") or ""}`
- inverse_action: `{action["inverse_action"]}`

## Contract

This entry was created by VaultQ autonomous self-maintenance. It records an internal agent-workspace promotion only. It does not approve or mutate canonical vault folders outside `{workspace_rel_path()}`.
"""


def _run_report_markdown(*, actions: list[Dict[str, Any]], skipped: list[Dict[str, Any]], created: str, dry_run: bool) -> str:
    action_rows = "\n".join(
        f"| {row['action']} | `{row['source']}` | `{row['target']}` | `{row.get('inverse_action', '')}` |"
        for row in actions
    ) or "| none | n/a | n/a | n/a |"
    skipped_rows = "\n".join(
        f"| `{row.get('path', '')}` | {row.get('reason', '')} |"
        for row in skipped
    ) or "| none | n/a |"
    return f"""---
type: report
domain: system
created: {created}
created_by: agent
---

# VaultQ Autonomous Maintenance Run

Dry run: `{str(dry_run).lower()}`

## Promotion Actions

| action | source | target | rollback |
| --- | --- | --- | --- |
{action_rows}

## Skipped

| path | reason |
| --- | --- |
{skipped_rows}
"""


def _review_sheet_markdown(root: Path) -> tuple[str, str]:
    ensure_workspace(root)
    date_text = _today()
    rel_path = workspace_rel_path("Review Queue", f"{_now_slug()}-review-promote-sheet.md")
    workspace = Path(root).expanduser().resolve() / workspace_rel_path()
    candidates = sorted(path for path in workspace.rglob("*.md") if "Review Queue" not in path.parts)
    preview_rows = []
    for path in candidates[:100]:
        rel = path.relative_to(root).as_posix()
        preview_rows.append(f"| review | `{rel}` | no canonical target selected | reversible ledger required |")
    rows = "\n".join(preview_rows) or "| none | no workspace notes found | n/a | n/a |"
    markdown = f"""---
type: report
domain: system
created: {date_text}
created_by: agent
---

# VaultQ Review Promote Sheet

Prepared only. No background worker was started and no canonical note was changed.

## Promotion Rules

- Agent workspace notes may be reviewed autonomously.
- Canonical vault writes require a diff sheet plus human approval.
- Every future promote action must record before hash, after hash, inverse action, source path, and target path in `{workspace_rel_path("Promotion Ledger")}`.
- Specific agent identity is not required; `created_by: agent` is enough.

## Candidate Actions

| action | source | proposed target | rollback |
| --- | --- | --- | --- |
{rows}
"""
    return rel_path, markdown


def prepare_background_agent(root: Path, *, write_sheet: bool = True) -> Dict[str, Any]:
    root = Path(root).expanduser().resolve()
    paths = ensure_workspace(root)
    config_path = write_runtime_config(root)
    result: Dict[str, Any] = {
        "ok": True,
        "started": False,
        "prepared_only": True,
        "workspace": paths,
        "runtime_config": config_path.relative_to(root).as_posix(),
        "mode": "autonomous_first_reversible",
        "next_step": "Run `vq agent maintain --collection <name>` for one autonomous self-maintenance pass, or enable the MCP-tied state-gated loop.",
    }
    if write_sheet:
        rel_path, markdown = _review_sheet_markdown(root)
        result["review_sheet"] = note_put(root, rel_path=rel_path, markdown=markdown, agent_generated=True)
    return result


def run_autonomous_maintenance(root: Path, *, dry_run: bool = False, max_actions: int = 20) -> Dict[str, Any]:
    root = Path(root).expanduser().resolve()
    ensure_workspace(root)
    write_runtime_config(root)
    workspace = root / workspace_rel_path()
    created = _today()
    max_actions = max(1, min(200, int(max_actions or 20)))
    actions: list[Dict[str, Any]] = []
    skipped: list[Dict[str, Any]] = []
    ledger_entries: list[str] = []
    cluster_note: Optional[Dict[str, Any]] = None
    human_update: Optional[Dict[str, Any]] = None
    safety_receipt: Optional[Dict[str, Any]] = None
    target_dirs = set(AI_WORKSPACE_DIRS)

    for source_path in sorted(workspace.rglob("*.md")):
        rel_source = source_path.relative_to(root).as_posix()
        if _is_maintenance_area(source_path.relative_to(workspace)):
            continue
        try:
            text = source_path.read_text(encoding="utf-8")
        except Exception as exc:
            skipped.append({"path": rel_source, "reason": f"read failed: {exc.__class__.__name__}"})
            continue
        frontmatter = parse_frontmatter(text)
        if frontmatter.get("created_by") != "agent":
            skipped.append({"path": rel_source, "reason": "not agent-authored"})
            continue
        target_dir = _target_dir_for_frontmatter(frontmatter)
        if target_dir not in target_dirs:
            target_dir = "Notes"
        if source_path.parent.name == target_dir:
            continue
        target_path = _unique_path(workspace / target_dir / source_path.name)
        rel_target = target_path.relative_to(root).as_posix()
        action = {
            "action": "promote_workspace_note",
            "source": rel_source,
            "target": rel_target,
            "note_type": frontmatter.get("type") or "",
            "inverse_action": f"move `{rel_target}` back to `{rel_source}`",
            "before_hash": _file_hash(source_path),
        }
        actions.append(action)
        if len(actions) >= max_actions:
            break

    if not dry_run:
        for action in actions:
            source_path = root / action["source"]
            target_path = root / action["target"]
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_path), str(target_path))
            action["after_hash"] = _file_hash(target_path)
            ledger_rel = workspace_rel_path(
                "Promotion Ledger",
                f"{_now_slug()}-{Path(action['target']).stem}-ledger.md",
            )
            ledger = note_put(
                root,
                rel_path=ledger_rel,
                markdown=_ledger_markdown(action=action, created=created),
                agent_generated=True,
            )
            ledger_entries.append(ledger["path"])
        report_rel = workspace_rel_path("Runs", f"{_now_slug()}-autonomous-maintenance-run.md")
        run_report = note_put(
            root,
            rel_path=report_rel,
            markdown=_run_report_markdown(actions=actions, skipped=skipped, created=created, dry_run=False),
            agent_generated=True,
        )
        clustering = _cluster_agent_ideas(root)
        cluster_rel = workspace_rel_path("Idea Clusters", f"{_now_slug()}-agent-idea-clusters.md")
        cluster_note = note_put(
            root,
            rel_path=cluster_rel,
            markdown=_cluster_markdown(clustering=clustering, created=created),
            agent_generated=True,
        )
        rollback_actions = [
            {
                "source": action["source"],
                "target": action["target"],
                "inverse_action": action["inverse_action"],
                "before_hash": action.get("before_hash"),
                "after_hash": action.get("after_hash"),
            }
            for action in actions
        ]
        safety_receipt_path = write_receipt(
            root,
            {
                "operation": "agent_maintain",
                "mode": "autonomous_first_reversible",
                "workspace_root": workspace_rel_path(),
                "canonical_writes": 0,
                "dry_run": False,
                "actions": actions,
                "skipped_count": len(skipped),
                "ledger_entries": ledger_entries,
                "run_report": run_report["path"],
                "cluster_note": cluster_note["path"],
                "rollback_actions": rollback_actions,
                "maintenance_patterns": {
                    "db_backed_state": True,
                    "honest_receipts": True,
                    "reversible_workspace_promotions": True,
                    "canonical_writes_require_explicit_promotion": True,
                },
            },
        )
        safety_receipt = {
            "path": safety_receipt_path,
            "rollback_actions": rollback_actions,
        }
        human_update = note_put(
            root,
            rel_path=workspace_rel_path("Human Updates", "Latest Agent Workspace Update.md"),
            markdown=_human_update_markdown(
                actions=actions,
                skipped=skipped,
                clustering=clustering,
                run_report_path=run_report["path"],
                cluster_note_path=cluster_note["path"],
                safety_receipt_path=safety_receipt_path,
                rollback_actions=rollback_actions,
                created=created,
                dry_run=False,
            ),
            agent_generated=True,
        )
    else:
        run_report = None
        clustering = _cluster_agent_ideas(root)
        rollback_actions = [
            {
                "source": action["source"],
                "target": action["target"],
                "inverse_action": action["inverse_action"],
                "before_hash": action.get("before_hash"),
                "after_hash": None,
            }
            for action in actions
        ]

    return {
        "ok": True,
        "dry_run": dry_run,
        "mode": "autonomous_first_reversible",
        "actions": actions,
        "skipped": skipped[:100],
        "ledger_entries": ledger_entries,
        "run_report": run_report,
        "safety_receipt": safety_receipt,
        "idea_clustering": {
            **clustering,
            "note": cluster_note,
        },
        "human_update": human_update,
        "workspace_root": workspace_rel_path(),
        "canonical_writes": 0,
    }


def run_autonomous_maintenance_if_changed(
    root: Path,
    *,
    state_path: Optional[Path] = None,
    max_actions: int = 20,
) -> Dict[str, Any]:
    root = Path(root).expanduser().resolve()
    state_path = Path(state_path).expanduser().resolve() if state_path else root / SELF_MAINTENANCE_STATE_FILE
    before_signal = agent_workspace_signal(root)
    rel_state_path = state_path.relative_to(root).as_posix() if state_path.is_relative_to(root) else str(state_path)
    if before_signal["agent_note_count"] <= 0:
        return {
            "ok": True,
            "ran": False,
            "reason": "no agent-authored workspace notes",
            "signal": before_signal,
            "state_path": rel_state_path,
        }

    previous: Dict[str, Any] = {}
    if state_path.exists():
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
    if previous.get("workspace_signal_hash") == before_signal["hash"]:
        return {
            "ok": True,
            "ran": False,
            "reason": "workspace unchanged",
            "signal": before_signal,
            "state_path": rel_state_path,
        }

    result = run_autonomous_maintenance(root, dry_run=False, max_actions=max_actions)
    after_signal = agent_workspace_signal(root)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "updated_at": datetime.now().astimezone().isoformat(),
                "workspace_signal_hash": after_signal["hash"],
                "agent_note_count": after_signal["agent_note_count"],
                "last_run": {
                    "actions": len(result.get("actions", [])),
                    "clusters": result.get("idea_clustering", {}).get("cluster_count", 0),
                    "human_update": result.get("human_update", {}).get("path"),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "ok": True,
        "ran": True,
        "reason": "workspace changed",
        "signal": after_signal,
        "previous_signal": before_signal,
        "state_path": rel_state_path,
        "result": result,
    }
