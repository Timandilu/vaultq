# VaultQ

`vaultq` is a standalone CLI for turning any markdown vault into a retrieval-focused knowledge base backed by Postgres + Qdrant.

It is designed for local notes, documentation repos, second-brain vaults, operating manuals, and project folders where plain vector search is not enough.

The retrieval stack is:

- structure-aware markdown chunking
- contextualized passages before indexing
- BM25 sparse retrieval
- dense semantic retrieval
- ColBERT-style late-interaction multivector reranking
- cross-encoder reranking
- neighboring chunk windows on the final results
- optional LLM enrichment into reusable knowledge objects

The CLI shape is intentionally close to QMD, but the retrieval path follows the more advanced Markvis stack.

## Why This Exists

Most markdown search tools stop at one of these:

- lexical search only
- single-vector semantic search only
- reranking without context-aware chunk preparation

`vaultq` is built for the case where retrieval quality matters more than minimal infrastructure.

The main design choices are:

- keep markdown as the source of truth
- preserve heading and section structure during chunking
- add collection, path, context, and summary metadata into the embedded passage text
- combine lexical and semantic retrieval instead of betting on one
- use late interaction and reranking where semantic precision matters
- keep LLM enrichment optional so you can bootstrap cheaply and upgrade later

## Feature Set

- Index any markdown directory with glob-based include and exclude rules
- Attach hierarchical context to whole collections or subpaths
- Chunk notes with heading-aware and paragraph-aware splitting
- Store original markdown, chunks, and derived knowledge objects in Postgres
- Store dense, sparse, and multivector representations in Qdrant
- Query with `search`, `vsearch`, or full `query`
- Return neighboring chunk windows instead of isolated snippets
- Rechunk without fully re-running LLM extraction when source text has not materially changed

## Architecture

`vaultq` has four major layers:

1. Ingest

- reads markdown files
- parses YAML frontmatter
- stores canonical source documents in Postgres

2. Chunking

- splits by markdown heading boundaries first
- then paragraphs
- then sentences as fallback
- keeps line ranges so chunk-to-source and knowledge-to-chunk remapping remain possible

3. Enrichment

- optional OpenRouter pass
- creates:
  - `segment_summary`
  - `document_summary`
  - `concept`
  - `principle`
  - `method`
  - `sop`

4. Retrieval

- embeds contextualized chunks and knowledge objects
- indexes:
  - dense vectors
  - BM25 sparse vectors
  - late-interaction multivectors
- executes:
  - BM25 retrieval
  - dense retrieval
  - RRF fusion
  - late-interaction rerank
  - cross-encoder rerank
  - neighboring window expansion

## Requirements

- Python `3.11+`
- Postgres
- Qdrant
- embedding provider compatible with OpenAI-style `/embeddings`
- optional OpenRouter key for enrichment

The default environment assumes a local stack similar to the existing AI workspace:

- Postgres on `127.0.0.1:5440`
- Qdrant on `127.0.0.1:6338`

You can override all of that in `.env`.

## Installation

```bash
git clone https://github.com/Timandilu/vaultq.git
cd vaultq

python -m venv .venv
source .venv/bin/activate
pip install -e .

cp .env.example .env
```

If you already use a shared embedding env one directory above the repo as `.ai-embedding.env`, `vaultq` will load that automatically.

## Quick Start

```bash
vq init

vq collection add ~/notes --name notes
vq context add vaultq://notes "Personal markdown vault"
vq context add vaultq://notes/projects "Project notes and runbooks"

vq index --skip-knowledge
vq embed

vq query "how do I run the deployment process?"
```

That is the cheapest first pass:

- no OpenRouter calls
- chunking only
- retrieval from contextualized markdown chunks

## Full Enrichment Pass

If `OPENROUTER_API_KEY` is set, run:

```bash
vq index
vq embed
```

That adds document-level and section-level abstractions on top of the raw chunks.

The knowledge layer is useful when you want retrieval over:

- reusable principles
- named methods
- SOPs
- concepts mentioned across many notes

## Command Reference

### Initialize

```bash
vq init
vq init --reset
```

`vq init` creates:

- local `.vaultq/config.json`
- Postgres schema
- Qdrant collection with dense + sparse + multivector configuration

Use `--reset` only when you want to recreate the Qdrant collection.

### Collections

```bash
vq collection add /path/to/vault --name notes
vq collection add /path/to/docs --name docs --pattern '**/*.md' --exclude 'archive/**'
vq collection list
```

Each collection is a root directory plus:

- a name
- a glob pattern
- optional exclude globs

### Contexts

```bash
vq context add vaultq://notes "Personal notes"
vq context add vaultq://notes/projects "Project notes and runbooks"
vq context add vaultq://notes/clients/acme "Client-specific notes for Acme"
```

Context is inherited by path prefix. The longest matching prefix wins in practice because all matching contexts are included in chunk contextualization ordered by specificity.

### Index

```bash
vq index
vq index --skip-knowledge
vq index --collection notes
vq index --collection notes --force
```

Behavior:

- reads changed files
- updates stored documents
- rebuilds chunks
- optionally regenerates knowledge objects

`--skip-knowledge` is important when:

- you are bootstrapping a large vault
- you want to test chunking first
- you want to rechunk without paying the LLM cost again

### Embed

```bash
vq embed
vq embed --kinds chunks
vq embed --kinds knowledge
vq embed --limit 200
```

This embeds any pending rows from Postgres into Qdrant.

### Retrieval

```bash
vq search "release checklist"
vq vsearch "how to ship the app"
vq query "how do we handle onboarding emails?"
```

Modes:

- `search`: BM25 only
- `vsearch`: dense semantic only
- `query`: hybrid + rerank, best quality

### Fetch

```bash
vq get notes/path/to/doc.md
vq get #42 --full
```

This returns the stored document row, chunks, and knowledge objects.

## Local Config

`vq init` creates `.vaultq/config.json` in the current working directory.

Example:

```json
{
  "collections": [
    {
      "name": "notes",
      "path": "/home/me/notes",
      "pattern": "**/*.md",
      "exclude_globs": ["archive/**"]
    }
  ],
  "contexts": [
    {
      "target": "vaultq://notes",
      "text": "Personal notes and working documents"
    },
    {
      "target": "vaultq://notes/projects",
      "text": "Project notes and operating procedures"
    }
  ]
}
```

## Data Model

Postgres stores:

- `vq_collections`
- `vq_contexts`
- `vq_documents`
- `vq_chunks`
- `vq_knowledge_objects`

Qdrant stores:

- named dense vectors
- named sparse BM25 vectors
- named late-interaction multivectors

The table names are intentionally namespaced with `vq_` so this repo can share the same Postgres instance with other projects.

## Retrieval Pipeline

### Chunk Preparation

Each chunk is embedded with extra context, not just raw markdown text.

The current contextualized chunk includes:

- collection name
- relative path
- document title
- heading path
- section summary when available
- document summary when available
- inherited path context
- previous and next chunk snippets
- the actual passage text

This is the practical replacement for naive chunk-first indexing.

### Query Execution

`vq query` currently does:

1. BM25 candidate search
2. dense vector candidate search
3. RRF fusion
4. ColBERT late-interaction rerank
5. cross-encoder rerank
6. neighboring chunk window expansion

That is the highest-quality path in the repo right now.

## LLM Enrichment

If `OPENROUTER_API_KEY` is set, `vq index` extracts:

- `segment_summary`
- `document_summary`
- `concept`
- `principle`
- `method`
- `sop`

The prompts are designed to stay grounded in the source chunk text and prefer precision over volume.

### Rechunking Without Re-enrichment

If you run:

```bash
vq index --skip-knowledge
```

existing knowledge objects are remapped to new chunk IDs by line-range overlap.

That means you can iterate on chunking and contextualization without automatically paying the full LLM cost again.

## Environment

See [`.env.example`](.env.example) for all variables.

The main groups are:

- Postgres connection
- Qdrant connection and collection name
- embedding provider and model
- OpenRouter enrichment config
- chunking controls
- late-interaction and reranker controls

Important defaults:

- embedding model: `text-embedding-3-large`
- late interaction model: `colbert-ir/colbertv2.0`
- reranker: `Xenova/ms-marco-MiniLM-L-12-v2`

## Current Limitations

- This is not true token-level late chunking over whole documents.
- It is the practical version: structure-aware chunking plus contextualized passages before indexing.
- The CLI is the primary interface right now. There is no MCP server in this repo yet.
- The enrichment prompts are optimized for procedural and knowledge-heavy notes, not arbitrary creative writing.

## Validation Status

The repo has been smoke-tested locally for:

- package install
- CLI help
- schema initialization
- collection registration
- context registration
- indexing markdown files
- embedding pending chunks
- hybrid retrieval
- document fetch

## Roadmap

- MCP server surface for agent-native usage
- better markdown table and callout handling
- richer knowledge-object dedupe across documents
- partial incremental embed scheduling
- export/import utilities for moving stores between machines
