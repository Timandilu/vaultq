# VaultQ Agent Workspace Usage Guide

VaultQ is now the official agent-first interface for the Second Brain vault. It
lets Codex, Antigravity, and other MCP-capable agents query the vault, write
agent-owned notes, follow the allowed property schema, cluster ideas, and keep
the agent workspace tidy without mutating canonical vault folders.

## What To Use It For

Use VaultQ when an agent needs to:

- answer from the Second Brain vault with cited local evidence
- create an agent note, proposal, research note, question, or idea
- check allowed `type`, `domain`, property, and tag rules before writing
- avoid random node types, domains, or tags
- maintain `14_Agent_Workspace` autonomously
- cluster agent-generated ideas for later reuse
- update a human-facing note about what changed
- diagnose retrieval quality, chunk length, embedding coverage, or reranker health

Do not use VaultQ self-maintenance as approval to edit canonical vault folders.
Canonical promotions should stay explicit, reversible, and reviewable.

## Runtime Location

VaultQ repo:

```text
C:\Users\Timan\Workspace\AI\vaultq
```

Second Brain collection:

```text
C:\Users\Timan\Workspace\Second_Brain
```

Agent workspace:

```text
C:\Users\Timan\Workspace\Second_Brain\14_Agent_Workspace
```

HTTP MCP endpoint:

```text
http://127.0.0.1:7073/mcp
```

Start command:

```powershell
C:\Users\Timan\Workspace\AI\vaultq\start_vaultq_mcp.bat
```

The broader AI startup script starts VaultQ MCP through that batch file:

```powershell
C:\Users\Timan\Workspace\AI\start_all.bat
```

## Agent Startup Protocol

Installed local skill:

```text
C:\Users\Timan\.agents\skills\vaultq-agent-workspace\SKILL.md
```

At the start of a non-trivial vault task, an agent should:

1. Check MCP availability with `vaultq_status`.
2. Check write rules with `vaultq_write_status` if any write may happen.
3. Query with `vaultq_query` before broad manual file search. Use `vaultq_related_work` for idea-connection checks before creating a new note.
4. Fetch full source context with `vaultq_get_doc` when a chunk is not enough.
5. Use graph tools only when explicit note-link context matters.
6. Write only through policy-gated agent workspace tools.
7. Run `vaultq_agent_maintain` after meaningful agent writes.
8. Read or update the human-facing note when a task materially changes the workspace.

The stable human-facing note is:

```text
14_Agent_Workspace/Human Updates/Latest Agent Workspace Update.md
```

## MCP Tool Cheat Sheet

Read and retrieval:

```text
vaultq_status
vaultq_collection_list
vaultq_operation_list
vaultq_query
vaultq_related_work
vaultq_semantic_clusters
vaultq_search
vaultq_get_doc
vaultq_chunk_stats
vaultq_background_status
vaultq_graph_neighbors
vaultq_graph_backlinks
vaultq_graph_traverse
vaultq_think
```

Write and maintenance:

```text
vaultq_write_status
vaultq_capture
vaultq_note_put
vaultq_note_append
vaultq_note_propose
vaultq_property_propose
vaultq_maintain
vaultq_agent_prepare
vaultq_agent_maintain
```

Recommended defaults:

| Need | Tool |
| --- | --- |
| "What do we know?" | `vaultq_query` |
| "Does this idea connect to existing work?" | `vaultq_related_work` |
| "How do these notes cluster by shared source ideas?" | `vaultq_semantic_clusters` |
| Exact phrase or path lookup | `vaultq_search` |
| Full source after a hit | `vaultq_get_doc` |
| Retrieval health or chunk lengths | `vaultq_chunk_stats` |
| Background index state | `vaultq_background_status` |
| Allowed write schema | `vaultq_write_status` |
| Quick agent note | `vaultq_capture` |
| Proposal that needs review | `vaultq_note_propose` |
| New property/tag request | `vaultq_property_propose` |
| Workspace cleanup and clustering | `vaultq_agent_maintain` |

## CLI Equivalents

Status:

```powershell
vq status --json
vq doctor --live --json
```

Query:

```powershell
vq query "agent workspace property rules" --mode hybrid --json
vq query "what old work relates to this idea?" --mode focused --json
vq related "agent-native retrieval surface for monetizable ideas" --json
vq clusters semantic --collection second_brain --seed-prefix 14_Agent_Workspace/Ideas --json
vq clusters semantic --collection second_brain --seed-prefix 14_Agent_Workspace/Ideas --session-policy penalize --session-max-per-seed 1 --json
vq get "00_System/Protocol - Layout and Tagging.md" --full --json
```

Write-status:

```powershell
vq write-status --collection second_brain --json
```

Manual self-maintenance:

```powershell
vq agent maintain --collection second_brain --max-actions 20 --json
```

Dry run:

```powershell
vq agent maintain --collection second_brain --dry-run --json
```

Chunk stats:

```powershell
vq chunks stats --collection second_brain --json
```

Background index status:

```powershell
vq background status --collection second_brain --json
```

## Property Rules

Agent-generated notes use a minimal frontmatter sheet:

```yaml
type: note
domain: system
created: YYYY-MM-DD
created_by: agent
```

The important rule is not the exact values above, but the constraint:

- use approved `type` values from the vault protocol
- use approved `domain` values from the vault protocol
- set `created_by: agent` for agent-generated notes
- avoid property sprawl
- propose schema changes instead of inventing them

Approved provenance values are:

```text
human
agent
mixed
```

Common mistake to avoid:

```yaml
node_type: agent_question
domain: random-new-domain
tags:
  - ai-generated-node
```

That shape should be rejected or turned into a property proposal.

## Workspace Folders

VaultQ manages this namespace:

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
  Idea Clusters/
  Human Updates/
  Review Queue/
  Promotion Ledger/
  Property Proposals/
```

Agents should not use `88_AI_Workspace` for new notes.

VaultQ retrieval excludes `88_Agents/sessions/**` by default because imported raw
session transcripts can swamp normal retrieval with stale implementation context.
When an agent explicitly needs prior session evidence, use semantic clusters
with `session_policy=penalize` and `session_max_per_seed=1`. That path applies a
query-aware adaptive rerank, not a flat discount, so precise implementation
evidence can surface while broad transcript noise stays capped.

## Autonomous Self-Maintenance

`vaultq_agent_maintain` is autonomous inside `14_Agent_Workspace`.

It can:

- move agent-authored notes into the right workspace subfolder based on `type`
- write a reversible promotion ledger
- write a run report
- write a machine-readable safety receipt under `.vaultq/receipts/`
- cluster agent-authored ideas by semantic anchors in the indexed corpus
- connect agent ideas to imported/source texts and other indexed notes without hard-coding those source classes
- update the stable human-facing note

It cannot:

- write canonical vault notes outside `14_Agent_Workspace`
- approve schema changes
- delete human-authored notes

## Background Retrieval Upkeep

When VaultQ runs through the MCP startup script, background retrieval upkeep is intentionally low-impact:

- the worker starts from a path-only filesystem baseline and does not force a full startup index
- the idle poll is only a cheap scheduler wakeup, not a full vault scan
- new markdown discovery runs as a path-only scan every 10 minutes, then discovered files are indexed and embedded together
- changed or deleted files trigger a stat-diff incremental chunk refresh at most daily; unchanged files are skipped by stored size/mtime before markdown is read
- pending embeddings left by foreground index/write commands are checked every 5 minutes and drained in capped follow-up batches

For the Second Brain startup script, the current defaults are:

```text
idle scheduler wakeup: 300 seconds
new file path scan: 600 seconds
changed chunk refresh: 86400 seconds
pending embed check: 300 seconds
embed limit: 200
max embed batches per pass: 2
```

The worker writes status into Postgres, so agents can inspect state through
`vaultq_background_status` without reading logs.

Operational rule for agents: pending embeddings are not a normal blocker.
If the MCP background worker is enabled, leave them for the 5-minute latent
worker unless the user explicitly needs immediate semantic retrieval of the
newly written note in the same turn. For same-turn closeout, cite the file
readback plus `vaultq_background_status` instead of waiting on a full drain.

The self-maintenance contract is autonomous first, reversible always.

## MCP-Tied Self-Improving Loop

The permanent loop is intentionally tied to MCP startup.

`start_vaultq_mcp.bat` sets:

```text
VQ_SELF_MAINTAIN_COLLECTION=second_brain
VQ_SELF_MAINTAIN_INTERVAL_SECONDS=1800
VQ_SELF_MAINTAIN_INITIAL_DELAY_SECONDS=60
VQ_SELF_MAINTAIN_MAX_ACTIONS=20
```

Behavior:

- the worker starts inside the MCP server process
- it exits when the MCP server exits
- it waits before the first check
- it hashes agent-authored workspace notes outside maintenance folders
- it skips when the hash is unchanged
- it runs one maintenance pass when agent-authored workspace content changed
- it stores state in `.vaultq/self_maintenance_state.json` inside the vault
- every live run writes a safety receipt and rollback pointers

This avoids an always-on background daemon and avoids endless report spam.

Portable installs keep the loop disabled by default in `.env.example`:

```text
VQ_SELF_MAINTAIN_INTERVAL_SECONDS=0
```

## Retrieval Quality

Current chunk distribution after full corpus embedding:

```text
chunks: 40002
documents with chunks: 2906
avg tokens: 266.81
p50 tokens: 304
p75 tokens: 320
p90 tokens: 320
p95 tokens: 320
p99 tokens: 320
max tokens: 360
over max: 0
embedded: 100%
assessment: healthy
```

These chunks are not too long for the current retrieval stack. The p95 and p99
are close to the configured upper bound by design, and the max stays at the
configured cap.

If retrieval quality drops:

1. Run `vq doctor --live --json`.
2. Run `vq chunks stats --collection second_brain --json`.
3. Compare `vq search` with `vq query`.
4. Check `vaultq_background_status` to confirm the latent worker is enabled.
5. Re-index/embed manually only when immediate semantic retrieval is required.
6. Re-run graph extraction if link context looks stale.

## Agent Prompt Snippet

Use this when instructing another agent:

```text
Use VaultQ first for Second Brain work. Check vaultq_status, then
vaultq_write_status before any note write. Query with vaultq_query and use
vaultq_related_work before creating new idea notes. Cite collection/path
evidence. Write agent material only under 14_Agent_Workspace with created_by:
agent and approved type/domain values. Use vaultq_property_propose for schema
changes. After meaningful writes, run vaultq_agent_maintain so the workspace
self-maintains, writes safety receipts, clusters ideas, and updates the
human-facing note.
```

## Smoke Test

Use this PowerShell request to confirm the HTTP MCP endpoint is alive:

```powershell
$body = @{
  jsonrpc = '2.0'
  id = 1
  method = 'initialize'
  params = @{
    protocolVersion = '2025-03-26'
    capabilities = @{}
    clientInfo = @{ name = 'vaultq-smoke'; version = '0.1' }
  }
} | ConvertTo-Json -Depth 8

Invoke-WebRequest `
  -Uri 'http://127.0.0.1:7073/mcp' `
  -Method Post `
  -Body $body `
  -ContentType 'application/json' `
  -Headers @{ Accept='application/json, text/event-stream' }
```

Expected result:

```text
HTTP 200
Content-Type: text/event-stream
```
