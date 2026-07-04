# Retrieval And Embedding

VaultQ retrieval combines lexical search, dense vectors, reranking, graph signals, document diversity, and optional evidence expansion.

## Active Provider Path

The tested provider path is Atlas-hosted Voyage:

```text
EMBED_PROVIDER=atlas
EMBED_BASE_URL=https://ai.mongodb.com/v1
EMBED_MODEL=voyage-context-3
EMBEDDING_DIM=1024
ENABLE_CONTEXTUAL_EMBEDDINGS=true
ENABLE_RERANKER=true
RERANK_BASE_URL=https://ai.mongodb.com/v1
RERANKER_MODEL=rerank-2.5
```

VaultQ can borrow provider settings from another MCP env file using:

```text
VQ_EMBED_ENV_FILE=C:\Users\Timan\Workspace\AI\copyvector_mcp\copyvector_embed_pipeline\.env
```

Only provider and rerank keys are imported from `VQ_EMBED_ENV_FILE`. Database, Qdrant, and unrelated service settings are not imported.

## Retrieval Stack

Hybrid search runs:

1. title search
2. keyword search through Postgres FTS
3. semantic search through Qdrant dense vectors
4. reciprocal-rank fusion
5. reranking
6. graph signals
7. document diversity
8. neighbor-window expansion
9. result contract with evidence

Focused search is the precise related-work mode. It uses the same title,
keyword, semantic, fusion, and optional rerank lanes, but keeps a smaller
candidate set, enforces one result per source document, and skips
neighbor-window expansion so the answer surface stays tight.
If focused mode does not have enough distinct source documents, it returns fewer
results instead of padding with duplicate chunks from the same file.

`vaultq_related_work` wraps focused retrieval for the agent-first workflow. It
deduplicates by source path, returns connection cards with evidence, and
suggests an existing note home before an agent writes a new idea node. It is a
read path; it does not create or mutate markdown.

`vaultq_semantic_clusters` is the cluster surface. It accepts any seed path
prefix, searches each seed semantically against the indexed corpus, then groups
seeds that share the same high-scoring semantic anchor. This is how agent ideas
can connect to imported blog posts, reference texts, or other markdown without a
hard-coded relationship rule.

Example:

```powershell
python -m vaultq.cli clusters semantic `
  --collection second_brain `
  --seed-prefix 14_Agent_Workspace/Ideas `
  --related-limit 5 `
  --json
```

Session transcripts are not part of the default cluster pool. When they are
explicitly requested with `--session-policy penalize`, VaultQ does not apply one
flat discount. It computes an adaptive session rerank factor from:

- query/seed overlap against the session title, path, and chunk text
- evidence-language such as decisions, implementation, fixes, tests, and receipts
- generic transcript-language such as broad status chatter or raw session logs
- chunk focus, so compact evidence chunks beat huge unfocused transcript spans
- optional date signal from dated session paths or titles
- a per-seed session cap, defaulting to one penalized session result

Example:

```powershell
python -m vaultq.cli clusters semantic `
  --collection second_brain `
  --seed-prefix 14_Agent_Workspace/Ideas `
  --session-policy penalize `
  --session-penalty 0.65 `
  --session-max-per-seed 1 `
  --json
```

If semantic retrieval is unavailable, hybrid search fails open to title/keyword and returns a warning. If graph signals are unavailable, retrieval still returns fused/reranked results.

## Graph Signals

Graph signals are borrowed from the mature GBrain retrieval pattern and adapted for VaultQ.

They do two things:

- apply a small adjacency boost when a result is linked by at least two other top candidates
- demote duplicated session/run-like results so one session does not crowd out all evidence

Environment controls:

```text
ENABLE_GRAPH_SIGNALS=true
GRAPH_SIGNALS_TOP_K=20
GRAPH_SIGNAL_FLOOR_RATIO=0
RESULT_MAX_PER_DOC=2
```

`RESULT_MAX_PER_DOC=2` prevents five adjacent chunks from one document from becoming the whole answer surface.

## Corpus Embedding

Check status:

```powershell
python -m vaultq.cli status --json
```

Embed a controlled batch:

```powershell
python -m vaultq.cli embed --kinds chunks --limit 1000 --json
```

Embed all pending chunks with a loop only for an explicit foreground rebuild or
same-turn semantic retrieval requirement. In normal agent sessions, use
`vq background status --collection second_brain --json` and let the MCP-tied
background worker drain pending embeddings.

```powershell
$env:VQ_CONFIG_DIR='C:\Users\Timan\Workspace\AI\vaultq\.vaultq'
$env:VQ_ENV_FILE='C:\Users\Timan\Workspace\AI\vaultq\.env'
while ($true) {
  $status = python -m vaultq.cli status --json | ConvertFrom-Json
  if ($status.pending_chunk_embeddings -le 0) { break }
  python -m vaultq.cli embed --kinds chunks --limit ([Math]::Min(1000, $status.pending_chunk_embeddings)) --json
}
```

Do not reset Qdrant unless the vector size is wrong or you intentionally want to rebuild all vectors.

## Lightweight Refresh

The background worker favors freshness without scanning or embedding the whole
vault:

1. startup captures a path-only baseline and does not run a full index
2. new-file discovery is a path-only scan on the configured delay
3. new files are indexed in a batch, then pending embeddings are drained within
   the configured batch cap
4. pending embeddings left behind by foreground index/write commands are checked
   on `VQ_BACKGROUND_PENDING_EMBED_SECONDS` and drained in bounded batches
5. changed/deleted refresh runs on the longer interval and compares stored file
   size plus mtime metadata before reading markdown
6. unchanged files are skipped; changed files are re-chunked; deleted documents
   have stale Qdrant point ids removed

Inspect worker state:

```powershell
python -m vaultq.cli background status --collection second_brain --json
```

## Chunk Length Analytics

Use:

```powershell
python -m vaultq.cli chunks stats --collection second_brain --json
```

This reports:

- total chunks and documents
- min, average, p50, p75, p90, p95, p99, and max token counts
- embedded chunk coverage
- bucket counts relative to configured min/target/max
- the longest chunk samples with source path and line range
- a health assessment that flags a corpus as `too_long` when p95, over-max rate, or extreme max length exceed the configured chunk bounds

The assessment is a retrieval-health signal, not a hard failure. A few long chunks are acceptable; a high p95 or large over-max percentage means retrieval will become less precise and contextual embedding batches are more likely to hit provider token limits.

## Contextual Window Guard

Atlas `voyage-context-3` rejects contextualized embedding examples that exceed its model context window. VaultQ guards this in two places:

- each outbound contextual text is bounded before embedding
- each contextual document group is split by an actual outbound text estimate, not only by stored chunk metadata

Operational defaults:

```text
CONTEXTUAL_WINDOW_TOKENS=12000
CONTEXTUAL_WINDOW_MAX_CHUNKS=32
CONTEXTUAL_TEXT_MAX_TOKENS=6000
CONTEXTUAL_REQUEST_MAX_GROUPS=4
CONTEXTUAL_REQUEST_MAX_DOCUMENTS=384
CONTEXTUAL_REQUEST_MAX_TOKENS=40000
```

If a corpus contains very long unheaded notes, lower `CONTEXTUAL_TEXT_MAX_TOKENS` first. If the API rejects the total submitted batch, lower `CONTEXTUAL_REQUEST_MAX_TOKENS` or `CONTEXTUAL_REQUEST_MAX_GROUPS`. Do not disable contextual embeddings unless retrieval quality matters less than full-text vector coverage.

## Qdrant Dimension Safety

VaultQ validates Qdrant vector size before embedding.

Atlas `voyage-context-3` uses `1024` dimensions in this runtime. If an older Qdrant collection was created for a `1536`-dim OpenAI model, embedding will fail with a dimension mismatch instead of corrupting the collection.

Repair:

```powershell
python -m vaultq.cli init --reset
python -m vaultq.cli embed --kinds chunks --limit 1000 --json
```

`init --reset` resets the Qdrant collection and clears stored Qdrant point ids. It does not delete markdown notes or Postgres-indexed documents.

## Session Transcript Exclusion

The Second Brain collection excludes:

```text
88_Agents/sessions/**
```

Reason:

- imported session transcripts are useful archival material
- they are too large and stale for default agent retrieval
- they can dominate queries about the current system

The files remain in the vault. VaultQ just excludes them from default retrieval and graph extraction.

## Session Transcript Re-Inclusion Options

The useful information in `88_Agents/sessions/**` should come back through a
controlled lane, not as ordinary first-class chunks in every query. Practical
options:

1. **Separate collection.** Register session logs as `agent_sessions` and query
   them only through explicit tools or seed prefixes. This is cleanest for
   audits and prior-run lookup.
2. **Adaptive penalized same-collection lane.** Index sessions in the main
   collection but mark `source_class=agent_session`. Use
   `session_policy=penalize`, which adaptively reranks session chunks by
   specificity, evidence-language, generic transcript noise, focus, and recency,
   then caps session results per seed. This is better than a fixed global
   penalty because a precise prior implementation note can surface while a
   broad high-score transcript chunk is suppressed.
3. **Summary-only indexing.** Generate stable summaries or extracted claims from
   sessions and index those instead of raw transcripts. This is probably the
   best long-term option because it preserves information without context spam.
4. **Two-stage fallback.** Keep sessions excluded from normal retrieval and only
   search them when the main corpus has weak results, or when the user asks for
   prior agent work, old implementation attempts, or debugging history.
5. **Time-decayed raw logs.** Index raw sessions but sharply down-rank old runs
   and session-like folders. This is useful for recent operational continuity
   but weaker for long-term knowledge quality.

Recommended next implementation if sessions are reintroduced: start with a
separate `agent_sessions` collection plus a summary-only index. Then use the
explicit `session_policy=penalize` path for semantic clusters and prior-attempt
queries with `session_max_per_seed=1` or `2`. Do not simply remove
`88_Agents/sessions/**` from the default excludes.

## Verification Commands

Backend and provider:

```powershell
python -m vaultq.cli doctor --live --json
```

Graph extraction:

```powershell
python -m vaultq.cli graph extract --collection second_brain
```

Retrieval sample:

```powershell
python -m vaultq.cli query "agent workspace property rules created_by domain type" -n 5 --mode hybrid --json
```

Expected behavior:

- provider checks pass
- query embedding returns a `1024`-dim dense vector
- rerank returns scores
- protocol/property notes rank above stale implementation transcripts
- result metadata includes `14_Agent_Workspace` in the collection context
