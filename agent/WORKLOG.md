# VaultQ Rebuild Worklog

## 2026-04-23

### Current goal

Rebuild `vaultq` into a portable standalone markdown-vault ingestion and retrieval tool with:

- a Textual TUI as the main operator interface
- a scriptable CLI
- an MCP server entrypoint
- contextual Voyage embeddings
- BM25 + dense hybrid retrieval with reranking
- optional grounded LLM enrichment for knowledge objects

### What I learned

- The original `vaultq` repo already had a usable Postgres + Qdrant baseline, but not the higher-quality retrieval path from the reference projects.
- The strongest parts of `copyvector_mcp` and `markvis_ecom_mcp` are:
  - contextual dense embeddings from raw ordered chunk groups from the same source
  - BM25 sparse retrieval in Qdrant
  - reciprocal-rank fusion
  - reranking
  - neighboring chunk windows on final hits
  - schema verification and adaptive Qdrant upsert batching
- A portable rebuild needed to remove implicit machine assumptions and make runtime state explicit through config/env.
- The live smoke run exposed two concrete issues that were worth fixing before closeout:
  - simple CLI commands were slow because `cli.py` imported FastMCP and Textual eagerly
  - `vq init` was not idempotent when the Qdrant collection already existed
- The credential-backed verification exposed two more real integration issues:
  - the provided key was valid against Atlas-hosted Voyage endpoints, not direct `api.voyageai.com`
  - local stock Qdrant rejected text-BM25 sparse vectors because its inference service was not configured

### Chosen plan

1. Add durable task spec under `agent/SPECS/`.
2. Refactor config/provider/retrieval internals around Voyage contextual embeddings, BM25, rerank, and portable env loading.
3. Add Textual TUI and FastMCP server entrypoints.
4. Rewrite the README into a project-front document with architecture and retrieval graphs.
5. Verify imports, CLI entrypoints, and smoke commands in the repo venv.

### What changed

- Created `agent/SPECS/vaultq-tui-rebuild.md`
- Reworked provider/env resolution around portable OpenAI-compatible embedding, rerank, and LLM settings
- Replaced the embedding and retrieval path with:
  - contextual dense embedding support
  - sparse BM25 vectors
  - RRF fusion
  - reranking
  - neighbor-window expansion
  - adaptive Qdrant upsert splitting
- Reworked indexing to be update-aware and deletion-aware for Qdrant-backed rows
- Added a FastMCP server with `status`, `search`, `fetch`, and `get_document`
- Added a Textual TUI with overview, collections, pipeline, search, and MCP panes
- Updated `.env.example` to the current portable runtime contract
- Added `docker-compose.yml` for a local Postgres + Qdrant stack
- Fixed CLI startup by lazy-loading heavy TUI and MCP dependencies only for commands that need them
- Fixed `vq init` so an existing compatible Qdrant collection is treated as success instead of failure
- Added `vq doctor` for backend and live provider validation
- Added provider mismatch detection so `doctor --live` can recommend Atlas when a key fails on direct Voyage but succeeds on Atlas
- Added first-class direct `voyage` provider support while keeping Atlas as the tested default
- Corrected Voyage-context defaults to `1024` dimensions
- Reworked the lexical lane so Postgres full-text search is the portable default and Qdrant sparse remains opt-in
- Replaced the intermediate README with a full front-page project document and Mermaid architecture graphs
- Added `vq watch` for background polling, change detection, deletion-aware reindexing, and bounded embed draining
- Added TUI controls for starting and stopping watch mode and surfaced watch state in the overview
- Reduced watch cold-start latency by removing heavy imports from the help and baseline-snapshot path
- Reworked the MCP server into an agent-native read-only surface with `vaultq_status`, `vaultq_collection_list`, `vaultq_search`, `vaultq_query`, and `vaultq_get_doc`
- Added structured MCP success/error payloads and redaction of secret-shaped values in tool errors
- Documented Codex and Claude MCP configs, startup commands, a smoke transcript, and troubleshooting in the README

### What I ran

- inspected `Workspace/AGENTS.md`
- inspected `vaultq`, `copyvector_mcp`, and `markvis_ecom_mcp`
- reviewed current `vaultq` source files and reference retrieval/indexing files
- checked official docs and current package versions for FastMCP and Textual
- `pip install -e .`
- `python3 -m compileall src`
- `python -m vaultq.cli --help`
- `python -m vaultq.cli mcp --help`
- isolated imports for `VaultQTui` and `build_mcp()`
- `docker compose up -d`
- disposable-vault smoke test with isolated `VQ_CONFIG_DIR`:
  - `python -m vaultq.cli init`
  - `python -m vaultq.cli collection add ...`
  - `python -m vaultq.cli index --collection smoke --skip-knowledge --json`
  - `python -m vaultq.cli status --json`
- live provider validation with the supplied key:
  - `python -m vaultq.cli doctor --live`
  - direct Voyage path failed with `403 Forbidden`
  - Atlas-hosted Voyage path succeeded for contextual embeddings and rerank
- credential-backed end-to-end test:
  - `python -m vaultq.cli init --reset`
  - `python -m vaultq.cli collection add ...`
  - `python -m vaultq.cli context add ...`
  - `python -m vaultq.cli index --skip-knowledge --json`
  - `python -m vaultq.cli embed --kinds chunks --json`
  - `python -m vaultq.cli query "how do we deploy after migrations and smoke tests?" --json`
- background watch validation:
  - `vq watch --once --json`
  - modified a disposable vault with create/update/delete changes
  - `vq watch --once --json`
  - `vq watch --no-initial-sync --interval 1 --max-loops 5 --json`
  - `vq query "owner follow-up actions after rollback" --json`
- MCP validation:
  - `python -m compileall src`
  - `vq mcp --help`
  - in-process FastMCP client `tools/list`
  - in-process FastMCP client `vaultq_collection_list`
  - local STDIO MCP client smoke against a disposable corpus

### Verification result

- The repo compiles cleanly.
- The editable install resolves as `vaultq 0.2.0`.
- CLI help and MCP help work.
- `vq watch --help` works and no longer pulls the heavy store/indexer path just to print usage.
- TUI import and MCP construction work.
- A disposable markdown vault bootstraps correctly against the bundled Postgres + Qdrant stack.
- Verified state from the smoke run:
  - 1 registered collection
  - 2 stored documents
  - 2 stored chunks
  - 2 pending chunk embeddings
- Live Atlas-backed provider calls succeed with the supplied key:
  - contextual query embedding: `1024` dims
  - contextual document-group embedding: `1024` dims
  - rerank returns ordered relevance scores
- End-to-end retrieval test succeeded:
  - indexed `3` documents and embedded `3` chunk vectors
  - the top result for deployment/migrations/smoke-tests was `ops/release.md`
  - the lexical lane contributed keyword candidates after the stemming-aware Postgres fallback fix
- Watch-mode validation succeeded:
  - initial bootstrap indexed `3` documents and embedded `3` chunk vectors
  - a later sync after one edit, one new note, and one deletion indexed `2`, deleted `1`, and embedded `2`
  - a real `--no-initial-sync` background cycle detected live filesystem changes and again indexed `2`, deleted `1`, and embedded `2`
  - retrieval after the live watch update returned `ops/postmortem.md` for `owner follow-up actions after rollback`
- MCP server validation succeeded:
  - tool listing exposes only read-only VaultQ tools
  - `vaultq_collection_list` works without touching Postgres or Qdrant
  - `vaultq_query` returns the same plausible top result as CLI `vq query`
  - `vaultq_get_doc` fetches the source document referenced by the top MCP result

### Remaining gaps

- Optional LLM knowledge extraction was not live-tested because no LLM provider key was supplied in this session.
