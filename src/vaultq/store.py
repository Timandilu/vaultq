from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
import psycopg
from psycopg.rows import dict_row

from vaultq.embedding_provider import EmbeddingProviderConfig, load_env_file, resolve_embedding_provider
from vaultq.retrieval_models import late_interaction_dim, late_vector_name, use_multivector

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
    source_chunk_ids  JSONB,
    source_line_start INTEGER,
    source_line_end   INTEGER,
    content_hash      TEXT NOT NULL,
    qdrant_point_id   TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_vq_documents_rel_path ON vq_documents(collection_id, rel_path);
CREATE INDEX IF NOT EXISTS idx_vq_chunks_document ON vq_chunks(document_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_vq_chunks_qdrant ON vq_chunks(qdrant_point_id);
CREATE INDEX IF NOT EXISTS idx_vq_knowledge_document ON vq_knowledge_objects(document_id, object_type);
CREATE INDEX IF NOT EXISTS idx_vq_knowledge_qdrant ON vq_knowledge_objects(qdrant_point_id);
"""


@dataclass(frozen=True)
class Settings:
    db_dsn: str
    qdrant_url: str
    qdrant_collection: str
    embedding_dim: int
    provider: EmbeddingProviderConfig


def load_runtime_env(project_root: Path) -> None:
    env_path = project_root / ".env"
    if env_path.exists():
        load_env_file(env_path, override=False)


def config_dir(base_dir: Optional[Path] = None) -> Path:
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
    provider = resolve_embedding_provider()
    dsn = (os.getenv("DATABASE_URL") or os.getenv("DB_DSN") or "").strip()
    if not dsn:
        dsn = (
            f"host={os.getenv('DB_HOST', '127.0.0.1')} "
            f"dbname={os.getenv('DB_NAME', 'postgres')} "
            f"user={os.getenv('DB_USER', 'postgres')} "
            f"password={os.getenv('DB_PASS', 'password')} "
            f"port={os.getenv('DB_PORT', '5432')}"
        )
    return Settings(
        db_dsn=dsn,
        qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333").rstrip("/"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "vaultq"),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "3072")),
        provider=provider,
    )


def connect(settings: Optional[Settings] = None):
    settings = settings or get_settings()
    return psycopg.connect(settings.db_dsn, row_factory=dict_row)


def ensure_schema(settings: Optional[Settings] = None) -> None:
    settings = settings or get_settings()
    with connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(SQL_SCHEMA)
        conn.commit()


def ensure_qdrant_collection(settings: Optional[Settings] = None, reset: bool = False) -> None:
    settings = settings or get_settings()
    with httpx.Client(timeout=60.0) as client:
        for _ in range(30):
            try:
                resp = client.get(f"{settings.qdrant_url}/collections")
                if resp.status_code == 200:
                    break
            except Exception:
                pass
        if reset:
            delete_resp = client.delete(f"{settings.qdrant_url}/collections/{settings.qdrant_collection}")
            if delete_resp.status_code not in (200, 202, 404):
                raise RuntimeError(f"Failed to delete Qdrant collection: {delete_resp.status_code} {delete_resp.text}")
        vectors: Dict[str, Any] = {
            "dense": {
                "size": settings.embedding_dim,
                "distance": "Cosine",
            }
        }
        if use_multivector():
            late_dim = late_interaction_dim()
            if late_dim <= 0:
                raise RuntimeError("Could not resolve late interaction dimension")
            vectors[late_vector_name()] = {
                "size": late_dim,
                "distance": "Cosine",
                "multivector_config": {"comparator": "max_sim"},
                "hnsw_config": {"m": 0},
            }
        create_resp = client.put(
            f"{settings.qdrant_url}/collections/{settings.qdrant_collection}",
            json={
                "vectors": vectors,
                "sparse_vectors": {
                    "bm25": {"modifier": "idf"},
                },
            },
        )
        if create_resp.status_code not in (200, 201):
            raise RuntimeError(f"Failed to create Qdrant collection: {create_resp.status_code} {create_resp.text}")


def upsert_collection_rows(base_dir: Optional[Path] = None, settings: Optional[Settings] = None) -> Dict[str, int]:
    config = read_config(base_dir)
    settings = settings or get_settings()
    ids: Dict[str, int] = {}
    with connect(settings) as conn:
        with conn.cursor() as cur:
            for row in config.get("collections", []):
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
                    (
                        row["name"],
                        str(Path(row["path"]).expanduser().resolve()),
                        row.get("pattern", "**/*.md"),
                        json.dumps(row.get("exclude_globs", [])),
                    ),
                )
                ids[row["name"]] = int(cur.fetchone()["id"])
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
    }
