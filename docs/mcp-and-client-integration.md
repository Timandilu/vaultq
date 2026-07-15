# MCP And Client Integration

VaultQ is intended to be used by Codex, Antigravity, and other agents through MCP.

## Endpoint

HTTP MCP endpoint:

```text
http://127.0.0.1:7073/mcp
```

Start command:

```powershell
C:\Users\<user>\Workspace\AI\vaultq\start_vaultq_mcp.bat
```

The AI startup script already includes VaultQ:

```text
C:\Users\<user>\Workspace\AI\start_all.bat
```

## Health Check

Plain browser GET may return HTTP `406` because streamable HTTP MCP expects MCP-compatible headers.

Use an initialize request:

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

Expected:

```text
StatusCode: 200
Content-Type: text/event-stream
```

## Tool Surface

Read tools:

```text
vaultq_status
vaultq_collection_list
vaultq_operation_list
vaultq_chunk_stats
vaultq_background_status
vaultq_search
vaultq_query
vaultq_related_work
vaultq_semantic_clusters
vaultq_get_doc
vaultq_graph_neighbors
vaultq_graph_backlinks
vaultq_graph_traverse
vaultq_think
```

Write and maintenance tools:

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

Agents should prefer:

- `vaultq_query` for high-quality retrieval; use `mode="focused"` when the agent needs concise related-work results instead of broad context windows
- `vaultq_related_work` before creating a new idea note, because it returns distinct existing notes plus a suggested existing home
- `vaultq_semantic_clusters` when the agent needs to cluster seed notes by shared semantic anchors across the indexed corpus; use `session_policy="penalize"` and `session_max_per_seed=1` only when prior session-log evidence is explicitly useful
- `vaultq_chunk_stats` when retrieval quality or context-window failures may be caused by chunk length
- `vaultq_background_status` to check whether the MCP-tied index worker is alive, when it last wrote state, and when the next refresh is due
- `vaultq_write_status` before writing
- `vaultq_capture` for ordinary agent notes
- `vaultq_note_propose` for structured proposals
- `vaultq_property_propose` for schema/tag changes
- `vaultq_agent_maintain` for reversible internal cleanup, idea clustering, and the human-facing workspace update note

The longer agent playbook is [VaultQ agent workspace usage guide](vaultq-agent-workspace-how-to-use.md).

## Low-Impact Background Upkeep

The MCP process can also run a lightweight retrieval-upkeep worker when
`VQ_BACKGROUND_INDEX_COLLECTION` is set.

The intended Second Brain settings are:

```text
VQ_BACKGROUND_INDEX_POLL_SECONDS=300
VQ_BACKGROUND_NEW_FILE_DELAY_SECONDS=600
VQ_BACKGROUND_CHANGED_INDEX_SECONDS=86400
VQ_BACKGROUND_EMBED_LIMIT=100
VQ_BACKGROUND_MAX_EMBED_BATCHES=1
```

This worker starts from a path-only filesystem baseline, so MCP startup does not
trigger a full index. The 5-minute poll is only a cheap scheduler wakeup. New
markdown discovery runs as a path-only scan every 10 minutes, then newly
discovered files are indexed and embedded in a batch. Changed or deleted files
are handled by the daily stat-diff incremental chunk refresh. Unchanged files
are skipped by comparing stored file size and mtime metadata before any
markdown read.
If an embed pass hits the batch cap and pending vectors remain, a later pass
continues draining the queue instead of spinning continuously.
If pending vectors already exist at MCP startup, the worker schedules a bounded
drain on the same low-impact delay instead of embedding immediately at process
start.

Worker state is persisted in Postgres under `vq_runtime_state` and can be read
without inspecting logs:

```powershell
vq background status --collection second_brain --json
```

Equivalent MCP tool:

```text
vaultq_background_status
```

## MCP-Tied Self-Maintenance

`start_vaultq_mcp.bat` enables the self-maintenance loop only inside the MCP
process:

```text
VQ_SELF_MAINTAIN_COLLECTION=second_brain
VQ_SELF_MAINTAIN_INTERVAL_SECONDS=1800
VQ_SELF_MAINTAIN_INITIAL_DELAY_SECONDS=60
VQ_SELF_MAINTAIN_MAX_ACTIONS=20
```

The loop is state-gated. It hashes agent-authored notes in
`14_Agent_Workspace` outside maintenance folders, skips unchanged states, and
runs one `vaultq_agent_maintain` equivalent only after agent workspace content
changes. Each live run writes reversible ledgers, a JSON safety receipt under
`.vaultq/receipts/`, idea clusters, and the stable human-facing update note. It
stops automatically when the MCP server stops.

## Codex

Codex config should contain:

```toml
[mcp_servers.vaultq]
url = "http://127.0.0.1:7073/mcp"
```

Location:

```text
C:\Users\<user>\.codex\config.toml
```

## Antigravity

Antigravity config should contain both `url` and `serverURL` for compatibility:

```json
{
  "vaultq": {
    "url": "http://127.0.0.1:7073/mcp",
    "serverURL": "http://127.0.0.1:7073/mcp"
  }
}
```

Location:

```text
C:\Users\<user>\AppData\Roaming\Antigravity\User\mcp.json
```

## Failure Modes

Port not listening:

```powershell
Test-NetConnection -ComputerName 127.0.0.1 -Port 7073
```

Start manually:

```powershell
Start-Process `
  -FilePath 'C:\Users\<user>\Workspace\AI\vaultq\start_vaultq_mcp.bat' `
  -WorkingDirectory 'C:\Users\<user>\Workspace\AI\vaultq' `
  -WindowStyle Hidden
```

Wrong vector dimension:

```powershell
python -m vaultq.cli init --reset
```

Provider failure:

```powershell
python -m vaultq.cli doctor --live --json
```

Stale retrieval scope:

```powershell
python -m vaultq.cli index --collection second_brain --skip-knowledge --json
python -m vaultq.cli graph extract --collection second_brain
```
