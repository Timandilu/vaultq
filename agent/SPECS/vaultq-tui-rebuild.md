# VaultQ TUI Rebuild Spec

## Intent

Turn `vaultq` into a portable standalone tool that can ingest any markdown vault, optionally extract grounded knowledge objects with a simple LLM step, embed the result with Voyage-style contextual embeddings, retrieve with high accuracy, and expose the corpus through both a TUI and MCP server.

## Scope

- Keep the repo Python-first and locally runnable.
- Preserve Postgres + Qdrant as the storage/indexing backend.
- Make configuration portable and local to the repo or runtime environment.
- Make the primary UX a Textual TUI.
- Keep CLI commands for automation and batch operation.
- Add an MCP server entrypoint for local tool use.
- Align retrieval behavior with the reference projects:
  - contextual dense embeddings
  - BM25 sparse retrieval
  - reciprocal-rank fusion
  - reranking
  - neighbor-window expansion

## Non-goals

- Browser UI
- cloud deployment automation
- project-specific hardcoded vault assumptions
- broad multi-repo orchestration
- preserving the existing late-interaction default path if it conflicts with the reference architecture

## Acceptance criteria

### Functional

- The tool can register any markdown root path as a collection.
- The tool can index markdown files with deletion-aware and update-aware behavior.
- The tool can optionally produce grounded knowledge objects via a configurable chat-completions provider.
- The tool can embed pending chunks and knowledge objects into Qdrant.
- The tool can run hybrid retrieval with dense + sparse fusion, reranking, and neighbor windows.
- The tool can start an MCP server with at least `search`, `fetch`, and `status` tools.
- The default `vq` UX launches a usable TUI.

### Portability

- No hardwired absolute machine paths remain in code or docs.
- `.env.example` reflects the current runtime contract.
- Setup docs work for arbitrary vault locations.

### Documentation

- `README.md` becomes the canonical front page.
- The README includes architecture and retrieval-flow graphs.
- The README explains how indexing, embedding, retrieval, TUI, and MCP fit together.

## Design decisions

### Retrieval architecture

- Dense model default: `voyage-context-3`
- Reranker default: `rerank-2.5`
- Sparse retrieval: Qdrant BM25
- Use raw ordered chunk groups for contextual embedding requests when contextual embeddings are enabled.
- Keep knowledge embeddings simple and grounded; do not over-contextualize them with noisy metadata.

### UI architecture

- Textual TUI is the primary operator surface.
- CLI remains available for scripting.
- MCP is a separate entrypoint built on FastMCP.

### Data flow

1. Register collections and path-prefix contexts.
2. Scan markdown files and upsert canonical documents into Postgres.
3. Chunk markdown with heading-aware, paragraph-aware, token-aware splitting.
4. Optionally extract grounded knowledge objects.
5. Embed pending rows into Qdrant with adaptive batching.
6. Search via hybrid retrieval and rerank.

## Risks

- FastMCP and Textual add new runtime dependencies.
- Contextual embedding batching needs careful grouping and payload sizing.
- Qdrant upserts can fail on large payloads if batch sizing is too naive.
- README diagrams must stay accurate to the code after the refactor.

## Rollback

- Changes stay isolated to this repo.
- The current repo is clean, so any regression can be reviewed directly through git diff.

## Verification commands

Run from the repo root:

```bash
python -m compileall src
python -m vaultq.cli --help
python -m vaultq.cli status --json
python -m vaultq.cli search "test query" --json
python -m vaultq.cli mcp --help
python -m vaultq.cli tui --help
```

Commands that require live services should be treated as smoke checks unless Postgres, Qdrant, and API keys are configured.
