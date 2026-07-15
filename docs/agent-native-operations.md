# VaultQ Agent-Native Operations

VaultQ is the local retrieval and write surface for a markdown vault. The intended operator is an AI agent working through MCP, CLI, or the Textual TUI.

For a practical operator guide, see [VaultQ agent workspace usage guide](vaultq-agent-workspace-how-to-use.md).

The core rule is simple:

- canonical markdown files remain the source of truth
- Postgres stores indexed documents, chunks, contexts, graph edges, and runtime state
- Qdrant stores dense retrieval vectors
- MCP exposes the tool surface to agents
- agent-authored material lives under `14_Agent_Workspace`
- agent-authored notes use `created_by: agent`
- property, type, domain, and tag rules come from the vault protocol and Obsidian property registry

## Runtime Surfaces

VaultQ exposes the same system through three surfaces:

- CLI: `python -m vaultq.cli ...` or installed `vq ...`
- TUI: `vq tui` or just `vq`
- MCP: `vq mcp --transport http --host 127.0.0.1 --port 7073`

Use MCP for agent workflows. Use CLI for repeatable maintenance. Use TUI for operator visibility.

## Agent Workspace

The active agent workspace is:

```text
14_Agent_Workspace/
```

Subfolders:

```text
Inbox/
Notes/
Research/
Ideas/
Questions/
Proposals/
Runs/
Reports/
Idea Clusters/
Human Updates/
Review Queue/
Promotion Ledger/
Property Proposals/
```

Agents should not write to old `88_AI_Workspace` paths. `88_Agents/sessions/**` is excluded from VaultQ retrieval because imported session transcripts can dominate normal agent queries with stale implementation history.

## Property Rules

Agents must call `vaultq_write_status` before writing if they are unsure about metadata rules.

Required for normal agent-generated notes:

```yaml
type: note
domain: system
created: YYYY-MM-DD
created_by: agent
```

Approved `created_by` values:

```text
human
agent
mixed
```

Do not invent `node_type`, `agent_question`, random domains, or property-mirroring tags. Use `vaultq_property_propose` for a requested property or tag that is not already approved.

## MCP Tool Order For Agents

Recommended agent sequence:

1. `vaultq_status`
2. `vaultq_write_status`
3. `vaultq_query`, `vaultq_related_work`, or `vaultq_search`
4. `vaultq_graph_neighbors`, `vaultq_graph_backlinks`, or `vaultq_graph_traverse` when explicit link context matters
5. `vaultq_capture`, `vaultq_note_propose`, or `vaultq_property_propose` for writes
6. `vaultq_agent_maintain` for one autonomous workspace maintenance pass

Agents should cite returned evidence fields:

```json
{
  "collection_name": "second_brain",
  "rel_path": "00_System/Protocol - Layout and Tagging.md",
  "start_line": 93,
  "end_line": 125
}
```

## Autonomous Self-Maintenance

VaultQ prioritizes autonomous self-maintenance inside the agent workspace.

The command is:

```powershell
python -m vaultq.cli agent maintain --collection second_brain --json
```

Dry run:

```powershell
python -m vaultq.cli agent maintain --collection second_brain --dry-run --json
```

What it does:

- scans `14_Agent_Workspace`
- ignores maintenance areas such as `Runs`, `Reports`, `Idea Clusters`, `Human Updates`, `Review Queue`, `Promotion Ledger`, and `Property Proposals`
- only promotes notes with `created_by: agent`
- moves notes into the correct agent-workspace subfolder based on `type`
- writes ledger entries under `14_Agent_Workspace/Promotion Ledger`
- writes a run report under `14_Agent_Workspace/Runs`
- writes a JSON safety receipt under `.vaultq/receipts/`
- writes idea clusters under `14_Agent_Workspace/Idea Clusters`
- updates the stable human-facing note `14_Agent_Workspace/Human Updates/Latest Agent Workspace Update.md`
- never writes to canonical vault folders outside `14_Agent_Workspace`

Promotion targets:

| type | workspace target |
| --- | --- |
| `concept` | `Ideas` |
| `report` | `Reports` |
| `plan` | `Proposals` |
| `draft` | `Inbox` |
| `read`, `resource`, `article`, `project`, `runbook`, `sop`, `system`, `moc` | `Research` |
| `note` or unknown approved type | `Notes` |

This is autonomous first promotion, but bounded to the agent namespace. Canonical promotion should still be an explicit, reversible operation.

## Idea Clustering

The autonomous maintenance pass clusters agent-authored workspace notes by
semantic proximity to indexed corpus nodes.

Default behavior:

- seed nodes come from agent-authored notes in `14_Agent_Workspace`
- each seed is searched against the embedded VaultQ corpus using semantic retrieval
- seeds that hit the same strong external note are grouped under that semantic anchor
- related nodes can be ordinary vault notes, imported blog posts, reference texts, or other indexed markdown
- `88_Agents/sessions/**` remains excluded by default
- when explicitly requested with `session_policy=penalize`, session logs are not a normal pool: they get adaptive query-aware reranking and a per-seed cap
- if semantic retrieval is unavailable, VaultQ falls back to the older topic/domain/keyword grouping and records the fallback reason in the cluster note

The generated cluster report is an agent-facing map for future work. It is not a canonical knowledge graph migration. Its job is to reduce duplicate ideas, connect agent ideas to source material, and keep agents from inventing random node types, domains, or tags.

Direct read-only clustering is also available for arbitrary seed node sets:

```powershell
python -m vaultq.cli clusters semantic --collection second_brain --seed-prefix 14_Agent_Workspace/Ideas --json
```

Explicit session-log evidence:

```powershell
python -m vaultq.cli clusters semantic `
  --collection second_brain `
  --seed-prefix 14_Agent_Workspace/Ideas `
  --session-policy penalize `
  --session-max-per-seed 1 `
  --json
```

MCP equivalent:

```text
vaultq_semantic_clusters
```

## Human-Facing Update Note

Every live maintenance pass updates:

```text
14_Agent_Workspace/Human Updates/Latest Agent Workspace Update.md
```

That note is the compact human-facing change surface. It lists workspace promotions, rollback commands, cluster counts, generated report paths, the safety receipt path, and the autonomy contract. Agents should read it before starting a new long-running vault task.

## Runtime Preparation And MCP-Tied Loop

Prepare the runtime without starting a background worker:

```powershell
python -m vaultq.cli agent prepare --collection second_brain --write-sheet --json
```

This writes:

```text
.vaultq/agent_runtime.json
14_Agent_Workspace/Review Queue/<timestamp>-review-promote-sheet.md
```

The runtime config advertises a real one-shot command:

```text
vq agent maintain --collection <name> --max-actions 20
```

The one-shot command remains the manual control surface. The official
self-improving loop is tied to MCP startup and is state-gated so it does not
write fresh run reports when nothing changed.

`start_vaultq_mcp.bat` enables:

```text
VQ_SELF_MAINTAIN_COLLECTION=second_brain
VQ_SELF_MAINTAIN_INTERVAL_SECONDS=1800
VQ_SELF_MAINTAIN_INITIAL_DELAY_SECONDS=60
VQ_SELF_MAINTAIN_MAX_ACTIONS=20
```

When enabled, the worker:

- starts inside the MCP process
- exits when the MCP process exits
- hashes agent-authored workspace notes outside maintenance folders
- runs one maintenance pass only when that hash changes
- stores state at `.vaultq/self_maintenance_state.json`

The separate background indexing worker handles latent retrieval upkeep:
path-only new-file discovery, daily changed-file refresh, and 5-minute checks
for pending embeddings left behind by foreground writes. It stores its current
state in Postgres and can be inspected through:

```powershell
vq background status --collection second_brain --json
```

or the MCP tool:

```text
vaultq_background_status
```

Portable installs keep the loop disabled by default with
`VQ_SELF_MAINTAIN_INTERVAL_SECONDS=0`.

## TUI

The TUI includes:

- Overview: backend, provider, pending embeddings, agent workspace
- Overview: chunk p50/p95/max and chunk health
- Collections: vault path and excludes
- Pipeline: init, index, embed, watch, doctor
- Agent: prepare runtime, dry-run maintenance, run maintenance, idea cluster and human update outputs
- Search: focused, hybrid, semantic, keyword
- MCP: tool list and launch instructions

Use the Agent tab for operator-safe workspace maintenance. Use the Pipeline tab for corpus health.

## Startup

The AI startup script launches VaultQ MCP:

```text
C:\Users\<user>\Workspace\AI\start_all.bat
```

VaultQ MCP endpoint:

```text
http://127.0.0.1:7073/mcp
```

Codex and Antigravity should point at that endpoint.
