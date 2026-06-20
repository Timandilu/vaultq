# VaultQ Agent-First Second Brain Platform Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote VaultQ from a retrieval-grade markdown-vault indexer into an agent-first Second Brain platform where agents can safely read, write, link, synthesize, and maintain their own working memory as an integrated part of the user's vault.

**Architecture:** Keep the markdown vault as the source of truth. Add a capability-gated write layer, a top-level AI workspace inside the vault, a vault-native OO property/tag contract exposed to agents, link/idea graph extraction, lightweight synthesis/gap reporting, and maintenance loops on top of the current Postgres + Qdrant retrieval stack. Borrow GBrain's useful agent-brain patterns, but avoid team/company-brain scope, heavy OAuth federation, and relationship-first CRM assumptions.

**Tech Stack:** Python 3.11+, FastMCP, psycopg/Postgres, Qdrant, Textual, HTTP provider adapters, markdown/frontmatter parsing, pytest.

---

## Source Analysis

### Current VaultQ Baseline

Verified local repo at `C:\Users\Timan\Workspace\AI\vaultq`, commit `1865db6c7eb11f0032f86b3bf8900d4ced4f6e1a`.

VaultQ currently has:

- Portable markdown collection registration via `.vaultq/config.json` / `VQ_CONFIG_DIR`.
- Postgres tables for collections, contexts, documents, chunks, and optional knowledge objects.
- Qdrant dense vector storage and optional Qdrant sparse vectors.
- Postgres FTS lexical search as the portable sparse/keyword default.
- RRF fusion, reranking, and neighbor-window expansion in `src/vaultq/search.py`.
- Heading-aware, paragraph-aware, token-aware markdown chunking in `src/vaultq/chunker.py`.
- Optional grounded LLM extraction into knowledge objects in `src/vaultq/enrich.py`.
- Watch mode for polling collection changes and draining embeddings.
- Textual TUI for operator-facing status, pipeline, search, and MCP guidance.
- Read-only MCP tools:
  - `vaultq_status`
  - `vaultq_collection_list`
  - `vaultq_search`
  - `vaultq_query`
  - `vaultq_get_doc`

Current limitation: the agent can retrieve evidence, but cannot autonomously create, update, link, organize, or maintain vault-native notes through VaultQ.

### GBrain Capability Inventory

Verified GBrain checkout at `C:\Users\Timan\AppData\Local\Temp\gbrain-codex-read`, commit `0bfe0d0c7ebda6f2ab706bbddc023b8c8db21647`.

GBrain has the following relevant surfaces:

- Contract-first operation model with read/write/admin/agent scopes in `src/core/operations.ts`.
- Write operations such as `put_page`, `delete_page`, `restore_page`, `add_tag`, `add_link`, `add_timeline_entry`, `extract_facts`, and `think`.
- Read operations such as `search`, `query`, `get_page`, `list_pages`, `get_links`, `get_backlinks`, `traverse_graph`, `find_trajectory`, `find_contradictions`, `find_orphans`, `list_skills`, and schema inspection.
- A shared MCP dispatcher that generates tools from operations and builds a scoped operation context.
- `gbrain think`: gather evidence, optionally include graph/trajectory context, synthesize a cited answer, report gaps, and optionally persist synthesis.
- Link extraction from markdown links, Obsidian wikilinks, typed references, frontmatter-like structures, and source-qualified links.
- A maintenance/dream cycle: lint, backlinks, sync, synthesize, extract links/facts, patterns, consolidation, embedding, orphan checks, schema suggestion, purge.
- Skillpack/resolver layer that makes the agent use brain-first lookup, capture, ingestion, enrichment, citation fixing, reports, schema authoring, and maintenance workflows.
- Minions/job queue for durable agent jobs.
- Schema packs for page-type/path semantics and agent-authored schema evolution.
- Remote/team/OAuth support, source federation, admin dashboard, and multi-user security work.

### Adopt / Adapt / Skip

| GBrain Capability | VaultQ Decision | Reason |
| --- | --- | --- |
| Contract-first operation registry | Adopt | VaultQ needs one source of truth for CLI, MCP, TUI, and future agent jobs. |
| Read/write/admin scopes | Adopt | Needed before any write-capable MCP. Keep simpler than GBrain OAuth. |
| `put_page` write primitive | Adopt | Core unlock for autonomous agents. Must include dry-run, diff, receipt, and namespace guard. |
| Capture/inbox workflow | Adopt | Best first autonomous behavior: save thoughts, source snippets, decisions, and agent findings. |
| Top-level AI workspace | Adopt | The agent needs its own vault-native workspace and nodes without hiding them outside the user's actual Second Brain. |
| Vault-native OO property/tag schema | Adopt | Agents must stop inventing random tags, domains, and node types; allowed values need to be compiled from the vault protocol into machine-readable policy. |
| Link extraction from Obsidian links | Adopt | Directly useful for Second Brain. Bigger priority than CRM-style person/company graph. |
| Typed graph traversal | Adapt | Keep idea/source/project/link graph first; do not over-index person/company relationships. |
| `think` synthesis | Adapt | Useful, but should be lighter: evidence summary, conflicts, gaps, suggested writes. Full answer-gen can come later. |
| Dream/maintenance cycle | Adapt | Add local maintenance jobs, but keep them transparent and reviewable. |
| Skills/resolver files | Adapt | Add VaultQ agent protocols as markdown instructions/config, not a huge skillpack initially. |
| Schema packs | Adapt | Use a Second Brain profile/path semantics layer; avoid complex universal taxonomy at first. |
| Retrieval evals | Adopt | Needed to safely change ranking. Add real-query replay before ranking experiments. |
| Minions durable jobs | Defer | Useful later, but premature before write primitives and maintenance receipts exist. |
| Team/company brain | Skip | User does not need multi-user/team capability. |
| OAuth federation | Skip initially | Keep local-first. Add token scopes only if HTTP MCP is exposed beyond localhost. |
| Voice/email/calendar ingestion | Defer | Useful after core agent-vault writing works. |

## Product Principles

1. **Markdown remains canonical.** Postgres and Qdrant are indexes/caches. Agent-written knowledge must land in markdown files, not only in database rows.
2. **Agent writes must be auditable.** Every write records actor, reason, source evidence, operation id, timestamp, and changed path.
3. **AI workspace is first-class vault content.** Agents get a top-level vault folder for scratch nodes, findings, inbox captures, run logs, and proposed canonical edits.
4. **Canonical vault edits are gated by policy, not blocked forever.** Default to free writes under the AI workspace and proposal space; require stricter checks for project reports, MOCs, existing canonical files, or destructive actions.
5. **Idea graph over CRM graph.** Link ideas, hypotheses, sources, projects, decisions, questions, and plans first. Person/company links are supported but not the center of the product.
6. **Retrieval stays evidence-first.** Synthesis can help, but the core contract remains cited chunks/pages with line/path evidence and fetchable source docs.
7. **Maintenance is visible.** Background jobs must emit reports and receipts. No invisible overnight mutation of important notes.
8. **Local-first by default.** No team brain, hosted server, or external queue required for the first agent-first release.
9. **No random metadata.** Agents may only use approved vault properties, `type` values, `domain` values, and tags. New properties, property values, or tags must be proposed through a property-author workflow before they become allowed.

## Target Capability Sheet

### A. Writable Agent MCP

VaultQ should expose a write-capable MCP surface with explicit capability levels.

Initial tools:

- `vaultq_capture`: create a new markdown note under an allowed capture namespace.
- `vaultq_note_put`: create or replace a markdown note under an allowed path.
- `vaultq_note_append`: append a section/timeline/log entry to an existing note.
- `vaultq_note_patch`: patch a bounded region by heading or line range.
- `vaultq_note_move`: move/rename a note inside allowed roots.
- `vaultq_note_link`: add an Obsidian-style link or markdown link between notes.
- `vaultq_note_propose`: write a proposed edit when direct canonical writes are not allowed.
- `vaultq_write_status`: show policy, allowed roots, pending proposals, and recent write receipts.

Required write response:

```json
{
  "ok": true,
  "operation_id": "vqop_...",
  "mode": "applied",
  "path": "14_Agent_Workspace/Notes/2026-06-02-example.md",
  "diff_summary": {
    "added_lines": 18,
    "removed_lines": 0
  },
  "receipt_path": ".vaultq/receipts/2026-06-02/vqop_....json",
  "indexed": false,
  "next_step": "run vq index --collection second_brain --limit 1 && vq embed --limit 20"
}
```

Failure response must be structured and non-destructive:

```json
{
  "ok": false,
  "error": {
    "type": "PolicyDenied",
    "message": "Writes to 05_Reports require proposal mode.",
    "hint": "Use vaultq_note_propose or configure write_policy.allowed_roots."
  }
}
```

### B. Top-Level AI Workspace

Default Second Brain layout proposal. This should be an ordinary top-level folder inside the vault, not hidden in `.vaultq` and not outside the markdown corpus:

```text
14_Agent_Workspace/
  Inbox/
  Notes/
  Research/
  Ideas/
  Questions/
  Proposals/
  Runs/
  Reports/
  Property Proposals/
```

This folder is the agent's native workspace. It should be indexed like other vault folders, included in graph extraction, and exposed through retrieval. The distinction is policy-level: direct writes are allowed here, while canonical project/report/plan folders still use proposals by default.

Agent-created notes should use the same sparse OO frontmatter as the rest of the vault. The verified source of truth is `00_System/Protocol - Layout and Tagging.md`, with Obsidian property types checked against `.obsidian/types.json`.

Example:

```yaml
---
type: concept
domain: ai
created: 2026-06-02
created_by: agent
source: optional source path or URL
project: optional existing project
topic:
  - agentic-second-brain
---
```

Agent workspace kinds map onto existing vault `type` values. Do not create `agent_note`, `agent_idea`, `agent_question`, `agent_finding`, `agent_proposal`, `agent_run`, `agent_report`, `node_type`, `kind`, or similar parallel taxonomies.

| Agent workspace kind | Vault `type` to use | Notes |
| --- | --- | --- |
| Inbox capture | `draft` or `note` | Use `draft` for genuinely unfinished loose notes; otherwise use `note`. |
| Idea / hypothesis | `concept` | The vault has no `idea` type; ideas become concepts when saved as nodes. |
| Question | `note` | Keep the question in the title/body; do not invent `question` as a type. |
| Evidence-backed finding | `report` or `note` | Use `report` only when the output is durable and human-readable. |
| Proposed canonical edit | `plan` or `draft` | Use proposal folder/path plus receipt for lifecycle, not `status`. |
| Run or maintenance summary | `report` | A run log is an output/report, not `runbook`; `runbook` is only for procedures. |
| Source or reference extract | `read` or `resource` | Use `read` for source material, `resource` for reusable reference. |
| Property sheet / protocol support | `system` | Vault infrastructure, protocol, and generated policy notes. |

### C. Vault OO Property and Tag Contract

VaultQ must expose the existing vault property rules to every agent before it writes. The goal is not rich taxonomy. The goal is to prevent random invented metadata while preserving a minimal, vault-native way to identify agent-generated material.

Authoritative inputs:

- `00_System/Protocol - Layout and Tagging.md`: human-readable OO layout, metadata, type, domain, and tag rules.
- `.obsidian/types.json`: Obsidian property type registry. Current relevant entries include `type: text`, `domain: text`, `created: date`, `created_by: text`, `source: text`, `topic: multitext`, `project: text`, `status: text`, and `tags: tags`.
- `.vaultq/property_schema.json`: generated machine-readable cache for agents and tests, not an independent source of truth.
- `.vaultq/tag_rules.json`: generated machine-readable tag policy derived from the protocol and vault tag inventory.

Create a vault-visible property contract:

```text
14_Agent_Workspace/
  _Agent Property Sheet.md
```

This note should be generated from the protocol and should link back to `00_System/Protocol - Layout and Tagging.md`. It is the short agent-facing view, not a second schema to hand-maintain.

Create a machine-readable version:

```text
.vaultq/property_schema.json
.vaultq/tag_rules.json
```

Enforced base properties for new agent-written notes:

| Property | Required | Allowed Values / Shape | Purpose |
| --- | --- | --- | --- |
| `type` | yes | `article`, `concept`, `note`, `read`, `report`, `plan`, `project`, `resource`, `draft`, `system`, `moc`, `runbook`, `sop` | Controlled vault node kind. Never use `agent_*` types. Never set pipeline-only `local-chat-session` manually. |
| `domain` | yes | `ai`, `ecom`, `content`, `strategy`, `personal`, `trading`, `sport`, `system`, `general` | Primary outcome or application domain. Do not encode ownership words here. |
| `created` | yes | `YYYY-MM-DD` date | Creation date. This must match Obsidian's registered `date` type. |
| `created_by` | agent notes yes; human notes optional | `human`, `agent`, `mixed` | Note authorship/provenance. Agent-created notes must set `created_by: agent`; co-authored notes use `mixed`. |
| `source` | no | text | Source URL, path, or citation pointer when a note is derived from material. |
| `project` | no | text | Existing project name or link target when the note belongs to a project. |
| `topic` | no | multitext/list | Retrieval themes that folder, `type`, and `domain` do not cover. |
| `tags` | no | Obsidian tags passing tag policy | Optional retrieval/filtering only; not a property mirror. |
| `status` | restricted | text | Not part of the normal schema. Only use where the vault protocol already permits pipeline status, such as blog/article workflows. |

Agent provenance rule:

- Base writes identify agent-generated material visibly with `created_by: agent`, by path (`14_Agent_Workspace/...`), and by operation receipts under `.vaultq/receipts/`.
- Do not add `agent`, `operation_id`, `sources`, `node_type`, `category`, `kind`, `area`, `topic_type`, `author`, or similar fields to note frontmatter unless they have first been approved into the vault protocol and `.obsidian/types.json`.
- `created_by` is intentionally coarse. It answers whether a note was written by the human, an agent, or both. Specific agent ids such as `codex` or `vaultq` belong in receipts or the note body unless a later approved property adds them.

Tag rules:

- Agents may not mirror frontmatter as tags.
- Reject property-disguised tags such as `type/*`, `status/*`, and `domain/*`.
- Agents may not create new domain tags like `business/foo`, `ai/bar`, or `project/x` unless that tag already exists in the vault tag inventory or has been approved in `.vaultq/tag_rules.json`.
- If a note needs domain context, prefer links to existing project/MOC/source notes over new tags.
- Use `topic` for retrieval themes that folder, `type`, and `domain` do not cover.
- Tags are optional. Properties and links carry the main structure.

Property author workflow:

- Add `vaultq_property_propose` MCP/CLI operation for new properties, allowed values, type/domain extensions, or tags.
- Proposals write to `14_Agent_Workspace/Property Proposals/`.
- Proposal notes must themselves use valid frontmatter, normally `type: plan`, `domain: system`, and `created: YYYY-MM-DD`.
- Each property proposal should describe the intended patch to `00_System/Protocol - Layout and Tagging.md`, the matching Obsidian type update in `.obsidian/types.json`, and the migration/validation impact.
- Add `vq property list`, `vq property validate`, and `vq property approve`.
- Only approved protocol/type/tag changes become available to write tools.
- `vaultq_note_put`, `vaultq_note_append`, `vaultq_capture`, and `vaultq_think --save-report` must validate frontmatter before writing.

### D. Write Policy and Safety

Add `.vaultq/policy.json` or config-level `write_policy`.

Policy fields:

```json
{
  "write_enabled": false,
  "default_mode": "proposal",
  "allowed_roots": ["14_Agent_Workspace"],
  "proposal_roots": ["05_Reports", "08_Plans", "10_Garden"],
  "deny_roots": [".obsidian", ".trash", ".git"],
  "max_write_bytes": 100000,
  "require_source_for_canonical_edit": true,
  "allow_delete": false,
  "allow_move": false
}
```

Modes:

- `read_only`: current behavior.
- `ai_workspace`: allow writes only inside the top-level AI workspace.
- `proposal`: allow direct AI workspace writes and canonical edit proposals.
- `trusted_local`: allow canonical writes with source/evidence receipt.
- `admin`: allow destructive operations after explicit local CLI invocation.

### E. Idea-First Link Graph

Add a graph layer focused on ideas and vault semantics.

Link sources:

- Obsidian wikilinks: `[[path]]`, `[[path|label]]`.
- Markdown links: `[label](path.md)`.
- Frontmatter fields: `related`, `sources`, `projects`, `supports`, `contradicts`, `questions`, `status`.
- Agent metadata: source docs and target docs from write receipts.
- Explicit tool calls from `vaultq_note_link`.

Initial edge types:

- `mentions`
- `links_to`
- `supports`
- `contradicts`
- `derived_from`
- `answers`
- `questions`
- `belongs_to_project`
- `updates`
- `proposes_change_to`
- `same_idea_as`

New graph tools:

- `vaultq_graph_neighbors(identifier, edge_type?, depth?, limit?)`
- `vaultq_graph_backlinks(identifier, edge_type?, limit?)`
- `vaultq_graph_path(from_identifier, to_identifier, max_depth?)`
- `vaultq_graph_extract(collection?, changed_only?)`
- `vaultq_link_suggestions(identifier, limit?)`

Graph should be stored in Postgres, not Qdrant.

### F. Retrieval Quality Upgrades

Keep the current VaultQ pipeline, then add selected GBrain retrieval improvements.

Add:

- Title exact/substring boost.
- Alias/frontmatter synonym table.
- Per-page max-pool so one note does not dominate with many chunks.
- Source/path boost profile for Second Brain directories.
- Evidence tags: `exact_title`, `alias`, `frontmatter`, `keyword`, `semantic`, `graph`, `knowledge_object`.
- Create-safety: `exists`, `probable`, `unknown`.
- Search diagnostics: explain which retrieval layer found or missed a target.
- Query modes:
  - `fast`: keyword + title/alias + cached vectors.
  - `balanced`: current hybrid + rerank.
  - `deep`: expansion + graph + rerank + larger context.

Do not add complex team/cross-source boosting yet.

### G. Lightweight Think Layer

Add `vq think` / `vaultq_think`, but keep it practical and less overbuilt than GBrain.

Purpose:

- Use VaultQ retrieval and graph context to produce a cited answer.
- Report conflicts and missing evidence.
- Optionally save the result as an agent report or proposal.

Output contract:

```json
{
  "question": "...",
  "answer": "...",
  "citations": [
    {
      "collection": "second_brain",
      "rel_path": "08_Plans/example.md",
      "chunk_id": "chunk:123",
      "line_start": 12,
      "line_end": 28
    }
  ],
  "conflicts": [],
  "gaps": ["No current deployment note was found."],
  "suggested_writes": [
    {
      "type": "note",
      "domain": "ai",
      "created_by": "agent",
      "title": "Verify current deployment status",
      "target_root": "14_Agent_Workspace/Questions"
    }
  ]
}
```

First version can be CLI-only and MCP read-only. Write/persist can be added after write policy is stable.

### H. Maintenance / Dream Cycle

Add explicit, report-generating maintenance commands:

- `vq maintain links`: extract links/backlinks and stale graph rows.
- `vq maintain orphans`: find notes without inbound/outbound links.
- `vq maintain stale`: identify notes likely stale by age/path/status.
- `vq maintain contradictions`: detect likely contradictions between knowledge objects or notes.
- `vq maintain proposals`: list unresolved agent proposals.
- `vq maintain ai-workspace`: summarize agent notes, promote candidates, archive stale scratch notes.
- `vq maintain all --dry-run`: run non-mutating health report.

Default should be dry-run/report-only.

### I. Agent Protocol Files

Add a small VaultQ-native protocol layer inspired by GBrain skills, but avoid a full skillpack initially.

Files:

```text
.vaultq/
  AGENT_PROTOCOL.md
  policy.json
  property_schema.json
  tag_rules.json
  schema_profile.json
  receipts/
  maintenance/
```

`AGENT_PROTOCOL.md` should tell agents:

- Search before writing.
- Prefer the top-level AI workspace unless user asks for canonical edits.
- Cite retrieved files when writing conclusions.
- Use proposals for canonical edits.
- Maintain backlinks and source references.
- Never delete without explicit permission.
- Read `00_System/Protocol - Layout and Tagging.md`, `_Agent Property Sheet.md`, `.vaultq/property_schema.json`, and `.vaultq/tag_rules.json` before writing frontmatter or tags.
- Never invent new properties, node types, domains, or tags. Use `vaultq_property_propose` instead.

### J. Schema Profile for Second Brain

Add a configurable schema/profile layer for path semantics.

Example:

```json
{
  "profiles": {
    "second_brain": {
      "roots": {
        "05_Reports": {"root_kind": "reports", "default_type": "report", "canonical": true, "write_mode": "proposal"},
        "08_Plans": {"root_kind": "plans", "default_type": "plan", "canonical": true, "write_mode": "proposal"},
        "10_Garden": {"root_kind": "garden", "default_type": "concept", "canonical": false, "write_mode": "proposal"},
        "14_Agent_Workspace": {"root_kind": "ai_workspace", "default_type": "note", "canonical": false, "write_mode": "direct"}
      },
      "frontmatter_alias_fields": ["aliases", "source", "project", "topic", "tags"],
      "graph_fields": ["source", "project", "topic"]
    }
  }
}
```

This should influence:

- Write policy.
- Retrieval source boosts.
- Graph extraction.
- Note-type inference.
- Link suggestions.

### K. Eval and Replay

Add retrieval/write regression fixtures before changing ranking heavily.

Artifacts:

```text
tests/fixtures/retrieval_queries.jsonl
tests/fixtures/write_policy_cases.jsonl
tests/fixtures/markdown_links/
tests/fixtures/second_brain_profile/
```

Core eval commands:

- `vq eval retrieval --fixtures tests/fixtures/retrieval_queries.jsonl`
- `vq eval links --fixtures tests/fixtures/markdown_links`
- `vq eval policy --fixtures tests/fixtures/write_policy_cases.jsonl`
- `vq eval replay --log .vaultq/query-log.jsonl`

Minimum metrics:

- Hit@1 / Hit@5 for target notes.
- MRR for named note retrieval.
- Source diversity.
- Title/alias hit rate.
- Policy allow/deny accuracy.
- Link extraction precision on fixture docs.

## Implementation Plan

## Sprint 1: Operation Registry and Policy Foundation

**Goal:** Create a contract-first operation layer and write-policy model without exposing write MCP yet.

**Demo/Validation:**

- `python -m compileall src`
- `python -m vaultq.cli status --json`
- `python -m vaultq.cli policy --json`
- `python -m vaultq.cli property list --json`
- Unit tests for allow/deny decisions.

### Task 1.1: Add Operation Registry

**Files:**

- Create: `src/vaultq/operations.py`
- Modify: `src/vaultq/cli.py`
- Test: `tests/test_operations.py`

- [ ] Define `Operation`, `OperationParam`, `OperationContext`, and `OperationError`.
- [ ] Register current read operations: status, collection_list, search, query, get_doc.
- [ ] Add scope enum: `read`, `write`, `admin`.
- [ ] Add shared param validation.
- [ ] Keep current CLI behavior working through wrappers; do not rewrite all commands yet.

**Acceptance Criteria:**

- Existing CLI commands still work.
- Operation registry can list read operations and scopes.
- Invalid params return structured errors.

### Task 1.2: Add Write Policy Model

**Files:**

- Create: `src/vaultq/policy.py`
- Modify: `src/vaultq/store.py`
- Test: `tests/test_policy.py`

- [ ] Add default policy loader from `.vaultq/policy.json`.
- [ ] Support env override `VQ_WRITE_POLICY_FILE`.
- [ ] Implement path normalization and root containment checks.
- [ ] Implement `read_only`, `ai_workspace`, `proposal`, `trusted_local`, `admin` modes.
- [ ] Deny `.git`, `.obsidian`, `.trash`, config files, and paths outside registered collection roots by default.

**Acceptance Criteria:**

- Direct canonical write is denied by default.
- `14_Agent_Workspace/...` write is allowed when mode is `ai_workspace`.
- Proposal write is allowed in proposal roots.
- Path traversal attempts are denied.

### Task 1.3: Add CLI Policy Inspection

**Files:**

- Modify: `src/vaultq/cli.py`
- Test: `tests/test_cli_policy.py`

- [ ] Add `vq policy --json`.
- [ ] Add `vq policy init --mode ai_workspace`.
- [ ] Print resolved policy path, mode, allowed roots, proposal roots, deny roots.

**Acceptance Criteria:**

- User can see exactly what an agent could write before enabling write MCP.

### Task 1.4: Add Vault OO Property and Tag Schema

**Files:**

- Create: `src/vaultq/property_schema.py`
- Modify: `src/vaultq/cli.py`
- Test: `tests/test_property_schema.py`

- [ ] Compile `.vaultq/property_schema.json` from `00_System/Protocol - Layout and Tagging.md` and `.obsidian/types.json`.
- [ ] Add `.vaultq/tag_rules.json` from the protocol plus existing vault tag inventory.
- [ ] Add `vq property list --json`.
- [ ] Add `vq property validate <path-or-stdin> --json`.
- [ ] Validate that `type`, `domain`, `created`, and `created_by` are present on agent-generated notes.
- [ ] Validate that agent-generated notes use `created_by: agent`.
- [ ] Validate `type` and `domain` values against the protocol allow-lists.
- [ ] Treat `status` as restricted, not required; allow it only for protocol-approved pipeline contexts.
- [ ] Reject unknown frontmatter keys unless `--proposal` is used.
- [ ] Reject tags not present in `.vaultq/tag_rules.json`.

**Acceptance Criteria:**

- Agents can inspect allowed properties before writing.
- Random `node_type`, invalid `domain` values, `category`, `kind`, `author`, or invented tags fail validation.
- Human-authored notes can be validated without being forced to set `created_by`.

## Sprint 2: AI Workspace and Write Receipts

**Goal:** Let VaultQ create auditable AI workspace notes locally.

**Demo/Validation:**

- `vq policy init --mode ai_workspace`
- `vq capture "test thought" --json`
- Confirm markdown file exists in `14_Agent_Workspace/Inbox`.
- Confirm receipt JSON exists.

### Task 2.1: Add AI Workspace Resolver

**Files:**

- Create: `src/vaultq/ai_workspace.py`
- Modify: `src/vaultq/store.py`
- Test: `tests/test_ai_workspace.py`

- [ ] Resolve configured AI workspace root from policy/config.
- [ ] Default to `14_Agent_Workspace` when the collection appears to be an Obsidian/Second Brain vault.
- [ ] Provide helper paths for `Inbox`, `Notes`, `Ideas`, `Questions`, `Proposals`, `Runs`, `Reports`.
- [ ] Provide helper path for `Property Proposals`.
- [ ] Create directories only through explicit write operations.

**Acceptance Criteria:**

- No directories are created by read/status commands.
- Write operation can create missing AI workspace directories.

### Task 2.2: Add Receipts

**Files:**

- Create: `src/vaultq/receipts.py`
- Test: `tests/test_receipts.py`

- [ ] Generate stable operation ids: `vqop_<timestamp>_<short_hash>`.
- [ ] Write JSON receipts under `.vaultq/receipts/YYYY-MM-DD/`.
- [ ] Include actor, operation, mode, collection, paths, source citations, old/new hash, diff summary, property validation result, and tag validation result.
- [ ] Redact secret-shaped values from receipt error messages.

**Acceptance Criteria:**

- Every applied write creates a receipt.
- Receipt does not contain API keys or bearer tokens.

### Task 2.3: Add Capture Command

**Files:**

- Create: `src/vaultq/write_ops.py`
- Modify: `src/vaultq/cli.py`
- Test: `tests/test_capture.py`

- [ ] Add `vq capture <text> --kind idea|note|question|finding --domain <domain> --json`.
- [ ] Map capture kind to approved vault `type` values: idea -> `concept`, question -> `note`, finding -> `report` or `note`, note -> `note`.
- [ ] Write markdown with vault-approved frontmatter: `type`, `domain`, `created`, and `created_by: agent`.
- [ ] Support `--source rel_path` repeated option.
- [ ] Validate properties and tags before writing.
- [ ] Return operation receipt path.
- [ ] Do not auto-index in first implementation; return next-step command.

**Acceptance Criteria:**

- Agent can create its own nodes without touching canonical vault files.

### Task 2.4: Add Property Author Workflow

**Files:**

- Modify: `src/vaultq/property_schema.py`
- Modify: `src/vaultq/write_ops.py`
- Modify: `src/vaultq/cli.py`
- Test: `tests/test_property_author.py`

- [ ] Add `vq property propose --property <name> --reason <text>`.
- [ ] Add `vq property propose-tag --tag <tag> --reason <text>`.
- [ ] Write proposals to `14_Agent_Workspace/Property Proposals/`.
- [ ] Add `vq property approve <proposal-id> --json` for trusted local approval.
- [ ] Never auto-approve new properties or tags from MCP.

**Acceptance Criteria:**

- Agents have a legitimate path to request new metadata.
- Unknown metadata never slips into normal note writes silently.

## Sprint 3: Write-Capable MCP

**Goal:** Expose safe AI workspace writes over MCP.

**Demo/Validation:**

- `vq mcp --transport stdio --no-banner`
- MCP tools list includes read tools plus write tools only when policy enables write.
- `vaultq_capture` creates a note and receipt.

### Task 3.1: Refactor MCP Around Operation Registry

**Files:**

- Modify: `src/vaultq/mcp_server.py`
- Modify: `src/vaultq/operations.py`
- Test: `tests/test_mcp_read_tools.py`

- [ ] Generate MCP tools from operation definitions.
- [ ] Preserve current read-only tool names for compatibility.
- [ ] Add a `write_enabled` flag to `vaultq_status`.

**Acceptance Criteria:**

- Existing MCP clients keep working.
- Tool list is deterministic.

### Task 3.2: Add Write Tools

**Files:**

- Modify: `src/vaultq/mcp_server.py`
- Modify: `src/vaultq/write_ops.py`
- Test: `tests/test_mcp_write_tools.py`

- [ ] Add `vaultq_capture`.
- [ ] Add `vaultq_note_put` for allowed AI workspace paths.
- [ ] Add `vaultq_note_append`.
- [ ] Add `vaultq_note_propose`.
- [ ] Add `vaultq_property_propose`.
- [ ] Return structured success/error payloads.

**Acceptance Criteria:**

- Write tools are unavailable or return policy-denied in `read_only` mode.
- In `ai_workspace` mode, write tools can only write under configured AI workspace root.
- Proposal tool writes under `14_Agent_Workspace/Proposals`.
- Property proposal tool writes under `14_Agent_Workspace/Property Proposals`.

## Sprint 4: Link Graph Foundation

**Goal:** Extract and query idea-first links from markdown.

**Demo/Validation:**

- `vq graph extract --collection second_brain --json`
- `vq graph backlinks "10_Garden/example.md" --json`
- Tests for wikilinks, markdown links, frontmatter links, and agent receipts.

### Task 4.1: Add Graph Schema

**Files:**

- Modify: `src/vaultq/store.py`
- Create: `src/vaultq/graph.py`
- Test: `tests/test_graph_schema.py`

- [ ] Add `vq_links` table with source document, target identifier, edge type, anchor text, source kind, confidence, timestamps.
- [ ] Add uniqueness constraint to prevent duplicate extracted links.
- [ ] Add soft-stale marker for re-extraction.

**Acceptance Criteria:**

- Schema migration is idempotent.

### Task 4.2: Add Link Extraction

**Files:**

- Create: `src/vaultq/link_extraction.py`
- Test: `tests/test_link_extraction.py`

- [ ] Extract Obsidian wikilinks.
- [ ] Extract markdown links to `.md` files and vault-relative paths.
- [ ] Extract configured frontmatter graph fields.
- [ ] Ignore code blocks and external URLs.
- [ ] Normalize anchors and `.md` suffixes.

**Acceptance Criteria:**

- Fixture docs produce expected edge rows with no duplicate edges.

### Task 4.3: Add Graph CLI and MCP Read Tools

**Files:**

- Modify: `src/vaultq/cli.py`
- Modify: `src/vaultq/mcp_server.py`
- Test: `tests/test_graph_cli.py`

- [ ] Add `vq graph extract`.
- [ ] Add `vq graph backlinks`.
- [ ] Add `vq graph neighbors`.
- [ ] Add MCP `vaultq_graph_neighbors`.
- [ ] Add MCP `vaultq_graph_backlinks`.

**Acceptance Criteria:**

- Agent can discover related idea notes without relying only on semantic search.

## Sprint 5: Retrieval Quality Upgrade

**Goal:** Import the highest-value GBrain retrieval ideas without bloating VaultQ.

**Demo/Validation:**

- `vq query "exact existing note title" --json` returns that note at Hit@1.
- `vq query --explain "alias phrase"` shows alias/title evidence.
- Eval fixtures pass.

### Task 5.1: Add Title and Alias Index

**Files:**

- Modify: `src/vaultq/store.py`
- Modify: `src/vaultq/indexer.py`
- Modify: `src/vaultq/search.py`
- Test: `tests/test_title_alias_retrieval.py`

- [ ] Add `vq_document_aliases` table.
- [ ] Extract aliases from frontmatter fields configured in schema profile.
- [ ] Add title exact/substring match lane.
- [ ] Add alias exact/fuzzy lane.

**Acceptance Criteria:**

- Existing note title queries do not lose to semantically related chunks.

### Task 5.2: Add Evidence Contract

**Files:**

- Modify: `src/vaultq/search.py`
- Modify: `src/vaultq/mcp_server.py`
- Test: `tests/test_result_contract.py`

- [ ] Include `evidence` field on every result.
- [ ] Include `create_safety`.
- [ ] Include retrieval contributors: keyword, semantic, title, alias, graph, knowledge.
- [ ] Preserve backward-compatible fields: id, score, rel_path, text.

**Acceptance Criteria:**

- Agents can decide whether to update existing notes or create new agent nodes.

### Task 5.3: Add Search Explain and Diagnose

**Files:**

- Modify: `src/vaultq/search.py`
- Modify: `src/vaultq/cli.py`
- Test: `tests/test_search_explain.py`

- [ ] Add `vq query --explain`.
- [ ] Add `vq search diagnose "<query>" --target "<rel_path>"`.
- [ ] Report which lanes found/missed the target.

**Acceptance Criteria:**

- Retrieval misses become debuggable without reading logs.

## Sprint 6: Lightweight Think and Proposed Writes

**Goal:** Let the agent synthesize evidence and propose vault updates.

**Demo/Validation:**

- `vq think "what should I do with X?" --json`
- `vq think "... " --save-report`
- Report lands in `14_Agent_Workspace/Reports` with citations.

### Task 6.1: Add Think Pipeline

**Files:**

- Create: `src/vaultq/think.py`
- Modify: `src/vaultq/cli.py`
- Test: `tests/test_think.py`

- [ ] Gather via `search(query, retrieval_mode="hybrid")`.
- [ ] Optionally add graph neighbors for top results.
- [ ] Build LLM prompt with source chunks as untrusted evidence.
- [ ] Require JSON output: answer, citations, conflicts, gaps, suggested_writes.
- [ ] Refuse to save if answer is empty or citation resolution fails.

**Acceptance Criteria:**

- Think can answer from retrieved evidence and report gaps.

### Task 6.2: Add MCP Read Tool for Think

**Files:**

- Modify: `src/vaultq/mcp_server.py`
- Test: `tests/test_mcp_think.py`

- [ ] Add `vaultq_think`.
- [ ] Default to no persistence over MCP.
- [ ] Include warnings when LLM provider is missing.

**Acceptance Criteria:**

- Agent can request synthesis, but cannot silently persist it until write policy is explicit.

### Task 6.3: Add Save Report / Proposal Mode

**Files:**

- Modify: `src/vaultq/think.py`
- Modify: `src/vaultq/write_ops.py`
- Test: `tests/test_think_save.py`

- [ ] Add `--save-report`.
- [ ] Add `--propose-writes`.
- [ ] Save cited reports under agent reports.
- [ ] Save suggested canonical edits as proposals, not direct edits.
- [ ] Validate report frontmatter and tags against the property sheet before saving.

**Acceptance Criteria:**

- Agent can move from thinking to auditable proposed action.

## Sprint 7: Maintenance Cycle

**Goal:** Add report-first autonomous maintenance without invisible destructive edits.

**Demo/Validation:**

- `vq maintain all --dry-run --json`
- `vq maintain ai-workspace --write-report`
- Maintenance report appears under `14_Agent_Workspace/Runs`.

### Task 7.1: Add Maintenance Runner

**Files:**

- Create: `src/vaultq/maintenance.py`
- Modify: `src/vaultq/cli.py`
- Test: `tests/test_maintenance.py`

- [ ] Add phase runner: links, orphans, stale, proposals, ai_workspace, properties.
- [ ] Add lock file under `.vaultq/maintenance.lock`.
- [ ] Default to dry-run.
- [ ] Emit JSON and markdown reports.

**Acceptance Criteria:**

- Maintenance can be run manually before any scheduling exists.

### Task 7.2: Add AI Workspace Promotion Report

**Files:**

- Modify: `src/vaultq/maintenance.py`
- Test: `tests/test_ai_workspace_maintenance.py`

- [ ] List agent notes with no canonical follow-up.
- [ ] Suggest candidates for promotion to project/report/idea files.
- [ ] Identify stale scratch notes.
- [ ] Identify property/tag drift and link to the relevant property proposals.
- [ ] Never move/delete automatically.

**Acceptance Criteria:**

- Agent work does not rot silently in `14_Agent_Workspace`.

## Sprint 8: Schema Profile and Query Modes

**Goal:** Make VaultQ understand the user's Second Brain shape without hardcoding it.

**Demo/Validation:**

- `vq profile init second_brain --json`
- `vq query --mode deep "..." --json`
- Path boosts and proposal roots reflect profile config.

### Task 8.1: Add Schema Profile Loader

**Files:**

- Create: `src/vaultq/schema_profile.py`
- Modify: `src/vaultq/store.py`
- Test: `tests/test_schema_profile.py`

- [ ] Load `.vaultq/schema_profile.json`.
- [ ] Support root type, write mode, boost, alias fields, graph fields.
- [ ] Provide default `second_brain` starter profile.

**Acceptance Criteria:**

- Profile changes affect retrieval and policy without code edits.

### Task 8.2: Add Query Modes

**Files:**

- Modify: `src/vaultq/search.py`
- Modify: `src/vaultq/cli.py`
- Test: `tests/test_query_modes.py`

- [ ] Add `--mode fast|balanced|deep`.
- [ ] Fast: title/alias/keyword plus optional cached dense.
- [ ] Balanced: current hybrid + rerank.
- [ ] Deep: larger candidate pool, graph neighbors, optional query expansion.

**Acceptance Criteria:**

- User can choose speed/cost/quality explicitly.

## Sprint 9: Eval and Replay

**Goal:** Protect retrieval and write-policy quality as features expand.

**Demo/Validation:**

- `vq eval retrieval --fixtures tests/fixtures/retrieval_queries.jsonl`
- `vq eval policy --fixtures tests/fixtures/write_policy_cases.jsonl`
- CI/local test command can run without live providers.

### Task 9.1: Add Test Harness and Fixtures

**Files:**

- Create: `tests/fixtures/retrieval_queries.jsonl`
- Create: `tests/fixtures/write_policy_cases.jsonl`
- Create: `tests/fixtures/markdown_links/`
- Create: `src/vaultq/eval.py`
- Modify: `src/vaultq/cli.py`

- [ ] Add fixture schema.
- [ ] Add retrieval metric calculation.
- [ ] Add policy case runner.
- [ ] Add link extraction fixture runner.

**Acceptance Criteria:**

- Ranking changes can be checked before they ship.

### Task 9.2: Add Query Logging and Replay

**Files:**

- Modify: `src/vaultq/search.py`
- Create: `src/vaultq/query_log.py`
- Test: `tests/test_query_log.py`

- [ ] Log query, mode, top results, timings, and evidence to `.vaultq/query-log.jsonl` when enabled.
- [ ] Redact text if `VQ_QUERY_LOG_REDACT_TEXT=1`.
- [ ] Add `vq eval replay`.

**Acceptance Criteria:**

- Real usage can become regression coverage.

## Non-Goals for This Upgrade

- No team/multi-user company brain.
- No broad OAuth federation.
- No public HTTP deployment requirement.
- No autonomous deletion.
- No direct canonical rewrites by default.
- No voice/email/calendar ingestion in the first release.
- No generic CRM relationship modeling as the primary graph goal.
- No hidden background mutation without reports.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Agent pollutes canonical vault | Default writes only under `14_Agent_Workspace`; canonical edits become proposals. |
| Agents invent random metadata | Validate all agent-written frontmatter/tags against `.vaultq/property_schema.json` and `.vaultq/tag_rules.json`; unknown values become property proposals, not note metadata. |
| Agent writes hallucinated claims | Require source citations for findings/reports/proposals; record evidence in receipts. |
| Invisible automation reduces trust | Maintenance defaults to dry-run and writes reports. |
| Search quality regresses after graph/title changes | Add eval fixtures before ranking changes. |
| Policy bypass through path traversal or symlinks | Normalize paths, resolve roots, deny symlink escapes, test malicious cases. |
| Graph becomes noisy | Start with deterministic links/frontmatter; add LLM suggestions only later. |
| Synthesis becomes overbuilt | Keep first `think` layer lightweight and evidence-bound. |
| Agent namespace becomes a junk drawer | Add maintenance reports and promotion workflow. |

## Recommended Build Order

1. Operation registry, write policy, and vault OO property/tag schema.
2. AI workspace and receipts.
3. Write-capable MCP for AI workspace only.
4. Link graph extraction and graph read tools.
5. Retrieval title/alias/evidence upgrades.
6. Lightweight think/proposal layer.
7. Maintenance reports.
8. Schema profile and query modes.
9. Eval/replay hardening.

This order creates the agent-first behavior early while keeping the blast radius low. The first truly useful demo is Sprint 3: Codex can autonomously create its own cited notes under `14_Agent_Workspace` through MCP, with receipts, property validation, tag validation, and no canonical write risk.

## Open Decisions

1. Resolved: use `14_Agent_Workspace` as the top-level agent namespace.
2. Canonical proposal format: one proposal markdown per proposed edit, or JSON patch plus rendered markdown preview?
3. Should `vq watch` auto-index AI workspace writes immediately, or should write tools call a narrow single-file index after writing?
4. Should `vaultq_think` be exposed to MCP before or after write tools?
5. Which Second Brain roots should be canonical/proposal-only by default?
6. Resolved: do not expose a separate agent identity property for now; keep visible note provenance at `created_by: agent`.

## Verification Checklist Before Implementation Starts

- [ ] Confirm the AI workspace path.
- [ ] Confirm whether write-capable MCP should start disabled by default.
- [ ] Confirm whether canonical edits should always go through proposal mode initially.
- [ ] Confirm whether `created_by` remains the only visible authorship property for agent-created notes.
- [ ] Add pytest or another test runner to the project before changing write behavior.
- [ ] Create a disposable fixture vault for all write-policy tests.

