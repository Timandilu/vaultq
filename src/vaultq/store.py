from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
import psycopg
from psycopg.rows import dict_row

from vaultq.embedding_provider import load_env_layers
from vaultq.retrieval_models import (
    dense_vector_name,
    embedding_provider_name,
    hybrid_dense_dim,
    hybrid_encoder_model_name,
    reranker_model_name,
    sparse_backend,
    sparse_vector_name,
    use_contextualized_chunk_embeddings,
    use_qdrant_sparse_vectors,
    use_sparse,
    validate_embedding_configuration,
)
from vaultq.embedding_provider import resolve_embedding_provider, resolve_rerank_provider

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_env_layers(_PROJECT_ROOT)

CONFIG_DIR_NAME = ".vaultq"
CONFIG_FILE_NAME = "config.json"

SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS vq_collections (
    id                BIGSERIAL PRIMARY KEY,
    name              TEXT NOT NULL UNIQUE,
    root_path         TEXT NOT NULL,
    pattern           TEXT NOT NULL DEFAULT '**/*.md',
    exclude_globs     JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS vq_contexts (
    id                BIGSERIAL PRIMARY KEY,
    collection_id     BIGINT NOT NULL REFERENCES vq_collections(id) ON DELETE CASCADE,
    path_prefix       TEXT NOT NULL DEFAULT '',
    context_text      TEXT NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (collection_id, path_prefix, context_text)
);

CREATE TABLE IF NOT EXISTS vq_documents (
    id                BIGSERIAL PRIMARY KEY,
    collection_id     BIGINT NOT NULL REFERENCES vq_collections(id) ON DELETE CASCADE,
    rel_path          TEXT NOT NULL,
    abs_path          TEXT NOT NULL,
    title             TEXT NOT NULL,
    file_hash         TEXT NOT NULL,
    markdown_text     TEXT NOT NULL,
    frontmatter_json  JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata_json     JSONB NOT NULL DEFAULT '{}'::jsonb,
    status            TEXT NOT NULL DEFAULT 'indexed',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (collection_id, rel_path)
);

CREATE TABLE IF NOT EXISTS vq_chunks (
    id                BIGSERIAL PRIMARY KEY,
    document_id       BIGINT NOT NULL REFERENCES vq_documents(id) ON DELETE CASCADE,
    version           INTEGER NOT NULL DEFAULT 1,
    chunk_index       INTEGER NOT NULL,
    heading_path      TEXT,
    section_title     TEXT,
    text_content      TEXT NOT NULL,
    token_count       INTEGER NOT NULL DEFAULT 0,
    char_count        INTEGER NOT NULL DEFAULT 0,
    start_line        INTEGER NOT NULL DEFAULT 1,
    end_line          INTEGER NOT NULL DEFAULT 1,
    metadata_json     JSONB NOT NULL DEFAULT '{}'::jsonb,
    qdrant_point_id   TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, version, chunk_index)
);

CREATE TABLE IF NOT EXISTS vq_knowledge_objects (
    id                BIGSERIAL PRIMARY KEY,
    document_id       BIGINT NOT NULL REFERENCES vq_documents(id) ON DELETE CASCADE,
    version           INTEGER NOT NULL DEFAULT 1,
    object_type       TEXT NOT NULL,
    title             TEXT NOT NULL,
    body_text         TEXT NOT NULL,
    json_payload      JSONB NOT NULL DEFAULT '{}'::jsonb,
    confidence        DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    source_chunk_ids  JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_line_start INTEGER,
    source_line_end   INTEGER,
    content_hash      TEXT NOT NULL,
    qdrant_point_id   TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS vq_links (
    id                BIGSERIAL PRIMARY KEY,
    collection_name   TEXT NOT NULL,
    source_rel_path   TEXT NOT NULL,
    target_identifier TEXT NOT NULL,
    edge_type         TEXT NOT NULL DEFAULT 'links_to',
    anchor_text       TEXT NOT NULL DEFAULT '',
    source_kind       TEXT NOT NULL DEFAULT 'markdown',
    confidence        DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    metadata_json     JSONB NOT NULL DEFAULT '{}'::jsonb,
    stale             BOOLEAN NOT NULL DEFAULT FALSE,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (collection_name, source_rel_path, target_identifier, edge_type, anchor_text, source_kind)
);

CREATE TABLE IF NOT EXISTS vq_runtime_state (
    name              TEXT PRIMARY KEY,
    kind              TEXT NOT NULL,
    collection_name   TEXT,
    data_json         JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_vq_documents_rel_path ON vq_documents(collection_id, rel_path);
CREATE INDEX IF NOT EXISTS idx_vq_chunks_document ON vq_chunks(document_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_vq_chunks_pending ON vq_chunks(version, qdrant_point_id);
CREATE INDEX IF NOT EXISTS idx_vq_chunks_qdrant ON vq_chunks(qdrant_point_id);
CREATE INDEX IF NOT EXISTS idx_vq_chunks_text_fts ON vq_chunks USING GIN ((
    setweight(to_tsvector('english', COALESCE(text_content, '')), 'A') ||
    setweight(to_tsvector('simple', COALESCE(text_content, '')), 'B')
)) WHERE version = 1;
CREATE INDEX IF NOT EXISTS idx_vq_knowledge_document ON vq_knowledge_objects(document_id, object_type);
CREATE INDEX IF NOT EXISTS idx_vq_knowledge_pending ON vq_knowledge_objects(version, qdrant_point_id);
CREATE INDEX IF NOT EXISTS idx_vq_knowledge_qdrant ON vq_knowledge_objects(qdrant_point_id);
CREATE INDEX IF NOT EXISTS idx_vq_knowledge_body_fts ON vq_knowledge_objects USING GIN ((
    setweight(to_tsvector('english', COALESCE(body_text, '')), 'A') ||
    setweight(to_tsvector('simple', COALESCE(body_text, '')), 'B')
)) WHERE version = 1;
CREATE INDEX IF NOT EXISTS idx_vq_documents_title_fts ON vq_documents USING GIN ((
    setweight(to_tsvector('english', COALESCE(title, '')), 'A') ||
    setweight(to_tsvector('simple', COALESCE(rel_path, '')), 'B')
)) WHERE status = 'indexed';
CREATE INDEX IF NOT EXISTS idx_vq_links_source ON vq_links(collection_name, source_rel_path);
CREATE INDEX IF NOT EXISTS idx_vq_links_target ON vq_links(collection_name, target_identifier);
CREATE INDEX IF NOT EXISTS idx_vq_runtime_state_kind ON vq_runtime_state(kind, collection_name);
"""


@dataclass(frozen=True)
class Settings:
    db_dsn: str
    qdrant_url: str
    qdrant_collection: str
    embedding_dim: int


def config_dir(base_dir: Optional[Path] = None) -> Path:
    explicit = (os.getenv("VQ_CONFIG_DIR") or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    root = Path(base_dir or Path.cwd())
    return root / CONFIG_DIR_NAME


def config_path(base_dir: Optional[Path] = None) -> Path:
    return config_dir(base_dir) / CONFIG_FILE_NAME


def default_config() -> Dict[str, Any]:
    return {"collections": [], "contexts": []}


def ensure_local_config(base_dir: Optional[Path] = None) -> Path:
    directory = config_dir(base_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CONFIG_FILE_NAME
    if not path.exists():
        path.write_text(json.dumps(default_config(), indent=2) + "\n", encoding="utf-8")
    return path


def read_config(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    path = ensure_local_config(base_dir)
    return json.loads(path.read_text(encoding="utf-8"))


def write_config(config: Dict[str, Any], base_dir: Optional[Path] = None) -> Path:
    path = ensure_local_config(base_dir)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def target_to_parts(target: str) -> Tuple[str, str]:
    normalized = (target or "").strip()
    if not normalized.startswith("vaultq://"):
        raise ValueError("Context target must start with vaultq://")
    remainder = normalized[len("vaultq://") :].strip("/")
    if not remainder:
        raise ValueError("Context target must include a collection name")
    pieces = remainder.split("/", 1)
    collection_name = pieces[0].strip()
    path_prefix = pieces[1].strip("/") if len(pieces) > 1 else ""
    if not collection_name:
        raise ValueError("Context target must include a collection name")
    return collection_name, path_prefix


def get_settings() -> Settings:
    validate_embedding_configuration()
    dsn = (os.getenv("DATABASE_URL") or os.getenv("DB_DSN") or "").strip()
    if not dsn:
        dsn = (
            f"host={os.getenv('DB_HOST', '127.0.0.1')} "
            f"dbname={os.getenv('DB_NAME', 'vaultq')} "
            f"user={os.getenv('DB_USER', 'postgres')} "
            f"password={os.getenv('DB_PASS', 'password')} "
            f"port={os.getenv('DB_PORT', '5432')}"
        )
    return Settings(
        db_dsn=dsn,
        qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333").rstrip("/"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "vaultq"),
        embedding_dim=hybrid_dense_dim(),
    )


def _db_connect_timeout() -> int:
    try:
        return max(1, int(os.getenv("DB_CONNECT_TIMEOUT", "3")))
    except Exception:
        return 3


def connect(settings: Optional[Settings] = None):
    settings = settings or get_settings()
    return psycopg.connect(settings.db_dsn, row_factory=dict_row, connect_timeout=_db_connect_timeout())


def ensure_schema(settings: Optional[Settings] = None) -> None:
    settings = settings or get_settings()
    with connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(SQL_SCHEMA)
        conn.commit()


def ensure_qdrant_collection(settings: Optional[Settings] = None, reset: bool = False) -> None:
    settings = settings or get_settings()
    dense_name = dense_vector_name()
    sparse_name = sparse_vector_name()
    with httpx.Client(timeout=60.0) as client:
        if reset:
            delete_resp = client.delete(f"{settings.qdrant_url}/collections/{settings.qdrant_collection}")
            if delete_resp.status_code not in (200, 202, 404):
                raise RuntimeError(
                    f"Failed to delete Qdrant collection: {delete_resp.status_code} {delete_resp.text}"
                )
        vectors: Dict[str, Any] = {
            dense_name: {
                "size": settings.embedding_dim,
                "distance": "Cosine",
            }
        }
        payload: Dict[str, Any] = {"vectors": vectors}
        if use_qdrant_sparse_vectors():
            payload["sparse_vectors"] = {
                sparse_name: {"modifier": "idf"},
            }
        create_resp = client.put(
            f"{settings.qdrant_url}/collections/{settings.qdrant_collection}",
            json=payload,
        )
        if create_resp.status_code in (200, 201):
            return
        if create_resp.status_code == 409:
            existing_resp = client.get(f"{settings.qdrant_url}/collections/{settings.qdrant_collection}")
            existing_resp.raise_for_status()
            params = existing_resp.json().get("result", {}).get("config", {}).get("params", {})
            vectors = params.get("vectors") or {}
            sparse_vectors = params.get("sparse_vectors") or {}
            dense_config = vectors.get(dense_name) or {}
            dense_size = int(dense_config.get("size", 0) or 0)
            if dense_size != settings.embedding_dim:
                raise RuntimeError(
                    "Existing Qdrant collection has the wrong dense vector size; run `vq init --reset` "
                    f"to recreate it with size {settings.embedding_dim}."
                )
            if use_qdrant_sparse_vectors() and sparse_name not in sparse_vectors:
                raise RuntimeError(
                    "Existing Qdrant collection is missing the configured sparse vector; "
                    "run `vq init --reset` to recreate it."
                )
            return
        if create_resp.status_code not in (200, 201):
            raise RuntimeError(
                f"Failed to create Qdrant collection: {create_resp.status_code} {create_resp.text}"
            )


def clear_qdrant_point_ids(settings: Optional[Settings] = None) -> Dict[str, int]:
    settings = settings or get_settings()
    with connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE vq_chunks SET qdrant_point_id = NULL WHERE qdrant_point_id IS NOT NULL")
            chunks = cur.rowcount
            cur.execute(
                "UPDATE vq_knowledge_objects SET qdrant_point_id = NULL WHERE qdrant_point_id IS NOT NULL"
            )
            knowledge_objects = cur.rowcount
        conn.commit()
    return {"chunks": max(0, int(chunks or 0)), "knowledge_objects": max(0, int(knowledge_objects or 0))}


def delete_qdrant_points(point_ids: Sequence[str], settings: Optional[Settings] = None) -> int:
    ids = [str(point_id) for point_id in point_ids if str(point_id or "").strip()]
    if not ids:
        return 0
    settings = settings or get_settings()
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{settings.qdrant_url}/collections/{settings.qdrant_collection}/points/delete?wait=true",
            json={"points": ids},
        )
        if resp.status_code not in (200, 202):
            raise RuntimeError(f"Failed to delete Qdrant points: {resp.status_code} {resp.text}")
    return len(ids)


def upsert_collection_rows(base_dir: Optional[Path] = None, settings: Optional[Settings] = None) -> Dict[str, int]:
    config = read_config(base_dir)
    settings = settings or get_settings()
    configured = {
        row["name"]: {
            "path": str(Path(row["path"]).expanduser().resolve()),
            "pattern": row.get("pattern", "**/*.md"),
            "exclude_globs": row.get("exclude_globs", []),
        }
        for row in config.get("collections", [])
    }
    ids: Dict[str, int] = {}
    with connect(settings) as conn:
        with conn.cursor() as cur:
            for name, row in configured.items():
                cur.execute(
                    """
                    INSERT INTO vq_collections (name, root_path, pattern, exclude_globs, updated_at)
                    VALUES (%s, %s, %s, %s::jsonb, NOW())
                    ON CONFLICT (name)
                    DO UPDATE SET root_path = EXCLUDED.root_path,
                                  pattern = EXCLUDED.pattern,
                                  exclude_globs = EXCLUDED.exclude_globs,
                                  updated_at = NOW()
                    RETURNING id
                    """,
                    (name, row["path"], row["pattern"], json.dumps(row["exclude_globs"])),
                )
                ids[name] = int(cur.fetchone()["id"])

            cur.execute("DELETE FROM vq_contexts")
            for context in config.get("contexts", []):
                collection_name, path_prefix = target_to_parts(context["target"])
                collection_id = ids.get(collection_name)
                if collection_id is None:
                    continue
                cur.execute(
                    """
                    INSERT INTO vq_contexts (collection_id, path_prefix, context_text)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (collection_id, path_prefix, context_text) DO NOTHING
                    """,
                    (collection_id, path_prefix, context["text"]),
                )
        conn.commit()
    return ids


def collection_rows(settings: Optional[Settings] = None) -> List[Dict[str, Any]]:
    settings = settings or get_settings()
    with connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM vq_collections ORDER BY name")
            return list(cur.fetchall())


def status_snapshot(settings: Optional[Settings] = None) -> Dict[str, Any]:
    settings = settings or get_settings()
    embed_provider = resolve_embedding_provider()
    rerank_provider = resolve_rerank_provider()
    with connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS count FROM vq_collections")
            collections = int(cur.fetchone()["count"])
            cur.execute("SELECT COUNT(*) AS count FROM vq_documents")
            documents = int(cur.fetchone()["count"])
            cur.execute("SELECT COUNT(*) AS count FROM vq_chunks")
            chunks = int(cur.fetchone()["count"])
            cur.execute("SELECT COUNT(*) AS count FROM vq_knowledge_objects")
            knowledge_objects = int(cur.fetchone()["count"])
            cur.execute("SELECT COUNT(*) AS count FROM vq_chunks WHERE qdrant_point_id IS NULL")
            pending_chunk_embeddings = int(cur.fetchone()["count"])
            cur.execute("SELECT COUNT(*) AS count FROM vq_knowledge_objects WHERE qdrant_point_id IS NULL")
            pending_knowledge_embeddings = int(cur.fetchone()["count"])
    return {
        "collections": collections,
        "documents": documents,
        "chunks": chunks,
        "knowledge_objects": knowledge_objects,
        "pending_chunk_embeddings": pending_chunk_embeddings,
        "pending_knowledge_embeddings": pending_knowledge_embeddings,
        "qdrant_collection": settings.qdrant_collection,
        "qdrant_url": settings.qdrant_url,
        "embedding_provider": embedding_provider_name(),
        "embedding_model": hybrid_encoder_model_name(),
        "embedding_base_url": embed_provider.base_url,
        "embedding_dim": settings.embedding_dim,
        "reranker_model": reranker_model_name(),
        "rerank_base_url": rerank_provider.base_url,
        "contextual_embeddings": use_contextualized_chunk_embeddings(),
        "sparse_backend": sparse_backend() if use_sparse() else None,
        "sparse_vector_name": sparse_vector_name() if use_qdrant_sparse_vectors() else None,
    }
