from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict

from fastmcp import FastMCP

from vaultq.embedding_provider import load_env_layers

load_env_layers()

MCP_INSTRUCTIONS = (
    "You have access to a markdown-vault retrieval system.\n"
    "Use vaultq_query() for the highest-quality hybrid retrieval path.\n"
    "Use vaultq_search() for keyword-only lookup.\n"
    "Use vaultq_get_doc() when you need a returned source document or cited chunk.\n"
    "Always cite the collection and relative path when answering from retrieved notes."
)

CONFIG_DIR_NAME = ".vaultq"
CONFIG_FILE_NAME = "config.json"
MAX_TOOL_LIMIT = 50


def _config_path() -> Path:
    explicit = (os.getenv("VQ_CONFIG_DIR") or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve() / CONFIG_FILE_NAME
    return Path.cwd().resolve() / CONFIG_DIR_NAME / CONFIG_FILE_NAME


def _read_local_config() -> Dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {"collections": [], "contexts": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_limit(limit: int) -> int:
    try:
        parsed = int(limit)
    except Exception:
        parsed = 5
    return max(1, min(MAX_TOOL_LIMIT, parsed))


def _redact_message(message: str) -> str:
    redacted = re.sub(r"(password=)[^\s]+", r"\1<redacted>", message, flags=re.IGNORECASE)
    redacted = re.sub(r"(Bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1<redacted>", redacted)
    redacted = re.sub(r"(api[_-]?key['\"\s:=]+)[A-Za-z0-9._~+/=-]+", r"\1<redacted>", redacted, flags=re.IGNORECASE)
    return redacted


def _success(data: Any) -> Dict[str, Any]:
    return {"ok": True, "data": data}


def _failure(exc: Exception, *, hint: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "type": exc.__class__.__name__,
            "message": _redact_message(str(exc)),
            "hint": hint,
        },
    }


def _run_readonly(func, *, hint: str) -> Dict[str, Any]:
    try:
        return _success(func())
    except Exception as exc:
        return _failure(exc, hint=hint)


def _get_internal_chunk(identifier: str) -> Dict[str, Any]:
    from vaultq.store import connect

    chunk_id = int(identifier.split(":", 1)[1])
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.id,
                    c.chunk_index,
                    c.heading_path,
                    c.section_title,
                    c.text_content,
                    c.start_line,
                    c.end_line,
                    c.qdrant_point_id,
                    d.rel_path,
                    d.title,
                    col.name AS collection_name
                FROM vq_chunks c
                JOIN vq_documents d ON d.id = c.document_id
                JOIN vq_collections col ON col.id = d.collection_id
                WHERE c.id = %s AND c.version = 1
                """,
                (chunk_id,),
            )
            row = cur.fetchone()
    if row is None:
        raise KeyError(f"Chunk not found: {identifier}")
    return {
        "id": f"chunk:{row['id']}",
        "point_id": row.get("qdrant_point_id"),
        "record_type": "chunk",
        "doc_type": "markdown_chunk",
        "collection_name": row.get("collection_name") or "",
        "rel_path": row.get("rel_path") or "",
        "title": row.get("title") or "",
        "heading_path": row.get("heading_path") or "",
        "section_title": row.get("section_title") or "",
        "chunk_index": row.get("chunk_index"),
        "start_line": row.get("start_line"),
        "end_line": row.get("end_line"),
        "text": row.get("text_content") or "",
    }


def _get_internal_knowledge_object(identifier: str) -> Dict[str, Any]:
    from vaultq.store import connect

    object_id = int(identifier.split(":", 1)[1])
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    ko.id,
                    ko.object_type,
                    ko.title AS knowledge_title,
                    ko.body_text,
                    ko.source_line_start,
                    ko.source_line_end,
                    ko.qdrant_point_id,
                    d.rel_path,
                    d.title,
                    col.name AS collection_name
                FROM vq_knowledge_objects ko
                JOIN vq_documents d ON d.id = ko.document_id
                JOIN vq_collections col ON col.id = d.collection_id
                WHERE ko.id = %s AND ko.version = 1
                """,
                (object_id,),
            )
            row = cur.fetchone()
    if row is None:
        raise KeyError(f"Knowledge object not found: {identifier}")
    return {
        "id": f"knowledge:{row['id']}",
        "point_id": row.get("qdrant_point_id"),
        "record_type": "knowledge_object",
        "doc_type": row.get("object_type") or "knowledge_object",
        "collection_name": row.get("collection_name") or "",
        "rel_path": row.get("rel_path") or "",
        "title": row.get("title") or "",
        "knowledge_title": row.get("knowledge_title") or "",
        "start_line": row.get("source_line_start"),
        "end_line": row.get("source_line_end"),
        "text": row.get("body_text") or "",
    }


def _get_doc_or_citation(identifier: str, *, full: bool) -> Dict[str, Any]:
    from vaultq.search import fetch as fetch_result
    from vaultq.search import get_document as get_document_record

    normalized = (identifier or "").strip()
    if not normalized:
        raise ValueError("identifier is required")
    if normalized.startswith("chunk:"):
        return _get_internal_chunk(normalized)
    if normalized.startswith("knowledge:"):
        return _get_internal_knowledge_object(normalized)
    if normalized.startswith("#") or "/" in normalized or normalized.endswith(".md"):
        return get_document_record(identifier=normalized, full=full)
    try:
        return fetch_result(normalized)
    except KeyError:
        return get_document_record(identifier=normalized, full=full)


def build_mcp() -> FastMCP:
    mcp = FastMCP(name="VaultQ", instructions=MCP_INSTRUCTIONS)

    @mcp.tool(name="vaultq_status")
    def vaultq_status() -> Dict[str, Any]:
        """Return current VaultQ store and retrieval status."""

        def run() -> Dict[str, Any]:
            from vaultq.search import health

            return health()

        return _run_readonly(
            run,
            hint="Check that Postgres is running and the VaultQ environment points at the intended database.",
        )

    @mcp.tool(name="vaultq_collection_list")
    def vaultq_collection_list() -> Dict[str, Any]:
        """List configured local VaultQ collections and contexts."""

        def run() -> Dict[str, Any]:
            config = _read_local_config()
            return {
                "config_path": str(_config_path()),
                "collections": config.get("collections", []),
                "contexts": config.get("contexts", []),
            }

        return _run_readonly(
            run,
            hint="Run `vq collection add /path/to/vault --name notes` or set VQ_CONFIG_DIR to the right config directory.",
        )

    @mcp.tool(name="vaultq_search")
    def vaultq_search(query: str, limit: int = 5) -> Dict[str, Any]:
        """Run keyword-only retrieval over the indexed markdown vault."""

        def run() -> Dict[str, Any]:
            from vaultq.search import search as search_records

            return search_records(query=query, limit=_normalize_limit(limit), retrieval_mode="keyword")

        return _run_readonly(
            run,
            hint="Check the query text, initialized schema, and configured Postgres connection.",
        )

    @mcp.tool(name="vaultq_query")
    def vaultq_query(query: str, limit: int = 5) -> Dict[str, Any]:
        """Run VaultQ hybrid retrieval with reranking and neighbor-window expansion."""

        def run() -> Dict[str, Any]:
            from vaultq.search import search as search_records

            return search_records(query=query, limit=_normalize_limit(limit), retrieval_mode="hybrid")

        return _run_readonly(
            run,
            hint="Check that Postgres, Qdrant, and the embedding/rerank provider settings are available.",
        )

    @mcp.tool(name="vaultq_get_doc")
    def vaultq_get_doc(identifier: str, full: bool = False) -> Dict[str, Any]:
        """Fetch a source document or cited chunk by rel_path, #document_id, point id, chunk:<id>, or knowledge:<id>."""

        return _run_readonly(
            lambda: _get_doc_or_citation(identifier, full=full),
            hint="Pass a result `id`, `rel_path`, `#document_id`, `chunk:<id>`, or `knowledge:<id>` returned by VaultQ.",
        )

    return mcp


def run_mcp(
    *,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 7070,
    show_banner: bool = True,
) -> None:
    mcp = build_mcp()
    transport = (transport or "stdio").strip().lower()
    if transport == "stdio":
        mcp.run(show_banner=show_banner)
        return
    mcp.run(transport=transport, host=host, port=port, show_banner=show_banner)


if __name__ == "__main__":
    run_mcp()
