from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

from fastmcp import FastMCP

from vaultq.embedding_provider import load_env_layers

load_env_layers()

MCP_INSTRUCTIONS = (
    "You have access to a markdown-vault retrieval system.\n"
    "Use vaultq_query() for the highest-quality hybrid retrieval path.\n"
    "Use vaultq_query(mode='focused') when you need concise, source-diverse related-work results without neighbor-window expansion.\n"
    "Use vaultq_related_work() before creating a new idea note when you need existing connected work and a suggested existing home.\n"
    "Use vaultq_semantic_clusters() when you need seed notes clustered by shared semantic anchors across the indexed corpus.\n"
    "Use vaultq_search() for keyword-only lookup.\n"
    "Use vaultq_get_doc() when you need a returned source document or cited chunk.\n"
    "Use vaultq_chunk_stats() when you need chunk length distribution, embedding coverage, or long-chunk outliers.\n"
    "Use vaultq_background_status() to inspect the DB-backed state of the MCP-tied background indexing worker.\n"
    "Hybrid/deep query results may include graph adjacency and session-diversity metadata; treat it as ranking evidence, not source text.\n"
    "Use vaultq_graph_backlinks() and vaultq_graph_neighbors() for explicit note-link context.\n"
    "Use vaultq_graph_traverse() when link type, direction, or multi-hop context matters.\n"
    "Use vaultq_write_status() before writing if you are unsure about allowed type/domain/tag/property values.\n"
    "Use vaultq_capture() only for agent-authored notes under the configured AI workspace.\n"
    "Use vaultq_agent_prepare() to prepare the disabled reversible review/promote runtime; it does not start a background worker.\n"
    "Use vaultq_agent_maintain() for one autonomous, reversible self-maintenance pass inside the agent workspace.\n"
"When VQ_SELF_MAINTAIN_INTERVAL_SECONDS is enabled, the MCP process also runs a state-gated self-maintenance loop that only acts after agent workspace changes.\n"
"When VQ_BACKGROUND_INDEX_COLLECTION is set, the MCP process runs low-impact background upkeep: no startup full index, path-only new-file discovery on a slow cadence, changed chunks daily, and 5-minute pending-embedding drains.\n"
"Do not manually wait on or drain pending embeddings unless the user explicitly needs same-turn semantic retrieval of newly written material.\n"
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


def _collection_root(collection_name: str = "") -> Path:
    config = _read_local_config()
    collections = config.get("collections", [])
    requested = (collection_name or "").strip()
    if requested:
        for row in collections:
            if row.get("name") == requested:
                return Path(row["path"]).expanduser().resolve()
        raise ValueError(f"Unknown collection: {requested}")
    if collections:
        return Path(collections[0]["path"]).expanduser().resolve()
    return Path.cwd().resolve()


def _collection_name_or_default(collection_name: str = "") -> str:
    requested = (collection_name or "").strip()
    if requested:
        return requested
    config = _read_local_config()
    collections = config.get("collections", [])
    if collections:
        return str(collections[0].get("name") or "")
    return ""


def _normalize_limit(limit: int) -> int:
    try:
        parsed = int(limit)
    except Exception:
        parsed = 5
    return max(1, min(MAX_TOOL_LIMIT, parsed))


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _background_index_settings() -> Dict[str, int | str] | None:
    collection_name = (os.getenv("VQ_BACKGROUND_INDEX_COLLECTION") or "").strip()
    if not collection_name and _env_enabled("VQ_BACKGROUND_INDEX_ENABLED"):
        collection_name = (os.getenv("VQ_SELF_MAINTAIN_COLLECTION") or "").strip()
    if not collection_name:
        return None
    pending_embed_interval = _env_int("VQ_BACKGROUND_PENDING_EMBED_SECONDS", 300)
    return {
        "collection_name": collection_name,
        "poll_interval": max(60, _env_int("VQ_BACKGROUND_INDEX_POLL_SECONDS", 300)),
        "new_file_index_delay_seconds": max(0, _env_int("VQ_BACKGROUND_NEW_FILE_DELAY_SECONDS", 600)),
        "changed_index_interval_seconds": max(300, _env_int("VQ_BACKGROUND_CHANGED_INDEX_SECONDS", 86400)),
        "embed_limit": max(1, _env_int("VQ_BACKGROUND_EMBED_LIMIT", 100)),
        "max_embed_batches": max(0, _env_int("VQ_BACKGROUND_MAX_EMBED_BATCHES", 1)),
        "pending_embed_interval_seconds": 0 if pending_embed_interval <= 0 else max(60, pending_embed_interval),
    }


def _redact_message(message: str) -> str:
    redacted = re.sub(r"(password=)[^\s]+", r"\1<redacted>", message, flags=re.IGNORECASE)
    redacted = re.sub(r"(Bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1<redacted>", redacted)
    redacted = re.sub(r"(api[_-]?key['\"\s:=]+)[A-Za-z0-9._~+/=-]+", r"\1<redacted>", redacted, flags=re.IGNORECASE)
    return redacted


def _start_self_maintenance_worker() -> threading.Thread | None:
    interval_seconds = _env_int("VQ_SELF_MAINTAIN_INTERVAL_SECONDS", 0)
    if interval_seconds <= 0:
        return None
    interval_seconds = max(60, interval_seconds)
    initial_delay_seconds = max(0, _env_int("VQ_SELF_MAINTAIN_INITIAL_DELAY_SECONDS", 60))
    max_actions = max(1, min(200, _env_int("VQ_SELF_MAINTAIN_MAX_ACTIONS", 20)))
    collection_name = (os.getenv("VQ_SELF_MAINTAIN_COLLECTION") or "").strip()
    state_path_raw = (os.getenv("VQ_SELF_MAINTAIN_STATE_FILE") or "").strip()
    state_path = Path(state_path_raw).expanduser().resolve() if state_path_raw else None

    def loop() -> None:
        if initial_delay_seconds:
            time.sleep(initial_delay_seconds)
        while True:
            try:
                from vaultq.agent_runtime import run_autonomous_maintenance_if_changed

                result = run_autonomous_maintenance_if_changed(
                    _collection_root(collection_name),
                    state_path=state_path,
                    max_actions=max_actions,
                )
                if result.get("ran"):
                    print(
                        f"VaultQ self-maintenance: {result.get('reason')} ({result.get('state_path')})",
                        file=sys.stderr,
                    )
            except Exception as exc:
                print(f"VaultQ self-maintenance failed: {_redact_message(str(exc))}", file=sys.stderr)
            time.sleep(interval_seconds)

    thread = threading.Thread(target=loop, name="vaultq-self-maintenance", daemon=True)
    thread.start()
    return thread


def _start_background_index_worker() -> threading.Thread | None:
    settings = _background_index_settings()
    if settings is None:
        return None
    initial_delay_seconds = max(0, _env_int("VQ_BACKGROUND_INDEX_INITIAL_DELAY_SECONDS", 30))
    state_name = ""
    try:
        from vaultq.background_index import background_state_name, record_background_index_state

        state_name = background_state_name(str(settings["collection_name"]))
        record_background_index_state(
            collection_name=str(settings["collection_name"]),
            worker="mcp",
            enabled=True,
            summary={
                "phase": "starting",
                "initial_delay_seconds": initial_delay_seconds,
                "poll_interval": int(settings["poll_interval"]),
                "new_file_index_delay_seconds": int(settings["new_file_index_delay_seconds"]),
                "changed_index_interval_seconds": int(settings["changed_index_interval_seconds"]),
                "embed_limit": int(settings["embed_limit"]),
                "max_embed_batches": int(settings["max_embed_batches"]),
                "pending_embed_interval_seconds": int(settings["pending_embed_interval_seconds"]),
            },
            name=state_name,
        )
    except Exception as exc:
        print(f"VaultQ background index state startup failed: {_redact_message(str(exc))}", file=sys.stderr)

    def loop() -> None:
        if initial_delay_seconds:
            time.sleep(initial_delay_seconds)
        while True:
            try:
                from vaultq.watch import run_watch

                run_watch(
                    collection_name=str(settings["collection_name"]),
                    poll_interval=float(settings["poll_interval"]),
                    skip_knowledge=True,
                    embed_limit=int(settings["embed_limit"]),
                    max_embed_batches=int(settings["max_embed_batches"]),
                    new_file_index_delay_seconds=float(settings["new_file_index_delay_seconds"]),
                    changed_index_interval_seconds=float(settings["changed_index_interval_seconds"]),
                    pending_embed_interval_seconds=float(settings["pending_embed_interval_seconds"]),
                    no_initial_sync=True,
                    state_name=state_name or None,
                    state_worker="mcp",
                    drain_existing_pending=True,
                    logger=lambda message: print(f"VaultQ background index: {message}", file=sys.stderr),
                )
            except Exception as exc:
                try:
                    from vaultq.background_index import background_state_name, record_background_index_state

                    record_background_index_state(
                        collection_name=str(settings["collection_name"]),
                        worker="mcp",
                        enabled=True,
                        summary={"loops": 0, "error": True},
                        last_error={"type": exc.__class__.__name__, "message": _redact_message(str(exc))},
                        name=background_state_name(str(settings["collection_name"])),
                    )
                except Exception:
                    pass
                print(f"VaultQ background index failed: {_redact_message(str(exc))}", file=sys.stderr)
                time.sleep(max(300, int(settings["poll_interval"])))

    thread = threading.Thread(target=loop, name="vaultq-background-index", daemon=True)
    thread.start()
    return thread


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


def _run_operation(name: str, params: Dict[str, Any], func, *, hint: str) -> Dict[str, Any]:
    from vaultq.operations import operation_by_name, summarize_params, validate_params

    try:
        operation = operation_by_name(name)
        clean_params = {key: value for key, value in params.items() if value is not None}
        validation_error = validate_params(operation, clean_params)
        if validation_error:
            raise ValueError(validation_error)
        result = func()
        if isinstance(result, dict):
            result.setdefault("_meta", {})["param_summary"] = summarize_params(name, clean_params)
            result["_meta"]["operation_scope"] = operation.scope
            result["_meta"]["mutating"] = operation.mutating
        return _success(result)
    except Exception as exc:
        return _failure(exc, hint=hint)


def _run_readonly(func, *, hint: str) -> Dict[str, Any]:
    try:
        return _success(func())
    except Exception as exc:
        return _failure(exc, hint=hint)


def _run_write(func, *, hint: str) -> Dict[str, Any]:
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


def _index_write_result(result: Dict[str, Any], *, collection_name: str = "") -> Dict[str, Any]:
    rel_path = str(result.get("path") or "").strip()
    if not rel_path:
        return result
    try:
        from vaultq.indexer import run_index_path

        stats = run_index_path(
            rel_path,
            collection_name=_collection_name_or_default(collection_name) or None,
            force=True,
            skip_knowledge=True,
        )
        result["indexed"] = stats.documents_indexed > 0
        result["index_stats"] = stats.__dict__
        if result["indexed"]:
            result["next_step"] = "local title/keyword retrieval is available; background embedding will handle semantic freshness unless same-turn semantic retrieval is required"
    except Exception as exc:
        result["indexed"] = False
        result["index_error"] = {
            "type": exc.__class__.__name__,
            "message": _redact_message(str(exc)),
        }
    return result


def build_mcp() -> FastMCP:
    mcp = FastMCP(name="VaultQ", instructions=MCP_INSTRUCTIONS)

    @mcp.tool(name="vaultq_status")
    def vaultq_status() -> Dict[str, Any]:
        """Return current VaultQ store and retrieval status."""

        def run() -> Dict[str, Any]:
            from vaultq.search import health

            return health()

        return _run_operation(
            "vaultq_status",
            {},
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

        return _run_operation(
            "vaultq_collection_list",
            {},
            run,
            hint="Run `vq collection add /path/to/vault --name notes` or set VQ_CONFIG_DIR to the right config directory.",
        )

    @mcp.tool(name="vaultq_operation_list")
    def vaultq_operation_list() -> Dict[str, Any]:
        """List registered VaultQ read and write operations."""

        def run() -> Dict[str, Any]:
            from vaultq.operations import list_operations

            return {"operations": list_operations()}

        return _run_operation("vaultq_operation_list", {}, run, hint="Check the VaultQ package installation.")

    @mcp.tool(name="vaultq_chunk_stats")
    def vaultq_chunk_stats(collection_name: str = "", sample_limit: int = 12) -> Dict[str, Any]:
        """Return chunk token/character length distribution and health assessment."""

        def run() -> Dict[str, Any]:
            from vaultq.chunk_stats import chunk_length_stats

            return chunk_length_stats(
                collection_name=collection_name or None,
                sample_limit=max(1, min(100, int(sample_limit or 12))),
            )

        return _run_operation(
            "vaultq_chunk_stats",
            {"collection_name": collection_name, "sample_limit": sample_limit},
            run,
            hint="Check that Postgres is running and the collection name exists.",
        )

    @mcp.tool(name="vaultq_background_status")
    def vaultq_background_status(collection_name: str = "") -> Dict[str, Any]:
        """Return DB-backed status for the MCP-tied background indexing worker."""

        def run() -> Dict[str, Any]:
            from vaultq.background_index import background_index_status

            return background_index_status(collection_name=collection_name or None)

        return _run_operation(
            "vaultq_background_status",
            {"collection_name": collection_name},
            run,
            hint="Start the HTTP MCP server with VQ_BACKGROUND_INDEX_COLLECTION set, then wait for the first worker loop.",
        )

    @mcp.tool(name="vaultq_search")
    def vaultq_search(query: str, limit: int = 5) -> Dict[str, Any]:
        """Run keyword-only retrieval over the indexed markdown vault."""

        def run() -> Dict[str, Any]:
            from vaultq.search import search as search_records

            return search_records(query=query, limit=_normalize_limit(limit), retrieval_mode="keyword")

        return _run_operation(
            "vaultq_search",
            {"query": query, "limit": limit},
            run,
            hint="Check the query text, initialized schema, and configured Postgres connection.",
        )

    @mcp.tool(name="vaultq_query")
    def vaultq_query(query: str, limit: int = 5, mode: str = "hybrid") -> Dict[str, Any]:
        """Run VaultQ retrieval with title, keyword, semantic, and optional rerank lanes. Use mode='focused' for concise source-diverse results."""

        def run() -> Dict[str, Any]:
            from vaultq.search import search as search_records

            return search_records(query=query, limit=_normalize_limit(limit), retrieval_mode=mode)

        return _run_operation(
            "vaultq_query",
            {"query": query, "limit": limit, "mode": mode},
            run,
            hint="Check that Postgres, Qdrant, and the embedding/rerank provider settings are available.",
        )

    @mcp.tool(name="vaultq_related_work")
    def vaultq_related_work(idea: str, limit: int = 6, mode: str = "focused") -> Dict[str, Any]:
        """Surface source-diverse existing notes that may connect to an idea before creating a new note."""

        def run() -> Dict[str, Any]:
            from vaultq.related import related_work

            return related_work(idea, limit=_normalize_limit(limit), mode=mode or "focused")

        return _run_operation(
            "vaultq_related_work",
            {"idea": idea, "limit": limit, "mode": mode},
            run,
            hint="Check retrieval services; use a concrete idea or working note title as input.",
        )

    @mcp.tool(name="vaultq_semantic_clusters")
    def vaultq_semantic_clusters(
        collection_name: str = "",
        seed_prefixes: list[str] | None = None,
        candidate_prefixes: list[str] | None = None,
        exclude_prefixes: list[str] | None = None,
        seed_limit: int = 25,
        related_limit: int = 5,
        min_score: float = 0.35,
        session_policy: str = "exclude",
        session_penalty: float = 0.65,
        session_max_per_seed: int = 1,
    ) -> Dict[str, Any]:
        """Cluster indexed seed nodes by shared semantic anchors in the corpus."""

        def run() -> Dict[str, Any]:
            from vaultq.semantic_clusters import semantic_node_clusters

            return semantic_node_clusters(
                collection_name=collection_name or None,
                seed_path_prefixes=seed_prefixes or None,
                candidate_path_prefixes=candidate_prefixes or None,
                exclude_path_prefixes=exclude_prefixes or None,
                seed_limit=max(1, min(200, int(seed_limit or 25))),
                related_limit=max(1, min(25, int(related_limit or 5))),
                min_score=max(0.0, float(min_score or 0.0)),
                session_policy=session_policy or "exclude",
                session_penalty=max(0.0, min(1.0, float(session_penalty or 0.65))),
                session_max_per_seed=max(0, min(25, int(session_max_per_seed or 0))),
            )

        return _run_operation(
            "vaultq_semantic_clusters",
            {
                "collection_name": collection_name,
                "seed_prefixes": seed_prefixes,
                "candidate_prefixes": candidate_prefixes,
                "exclude_prefixes": exclude_prefixes,
                "seed_limit": seed_limit,
                "related_limit": related_limit,
                "min_score": min_score,
                "session_policy": session_policy,
                "session_penalty": session_penalty,
                "session_max_per_seed": session_max_per_seed,
            },
            run,
            hint="Use seed_prefixes such as ['14_Agent_Workspace/Ideas'] and keep session_policy='exclude' unless you explicitly want session logs.",
        )

    @mcp.tool(name="vaultq_get_doc")
    def vaultq_get_doc(identifier: str, full: bool = False) -> Dict[str, Any]:
        """Fetch a source document or cited chunk by rel_path, #document_id, point id, chunk:<id>, or knowledge:<id>."""

        return _run_operation(
            "vaultq_get_doc",
            {"identifier": identifier, "full": full},
            lambda: _get_doc_or_citation(identifier, full=full),
            hint="Pass a result `id`, `rel_path`, `#document_id`, `chunk:<id>`, or `knowledge:<id>` returned by VaultQ.",
        )

    @mcp.tool(name="vaultq_write_status")
    def vaultq_write_status(collection_name: str = "") -> Dict[str, Any]:
        """Show write policy and property-schema status for a configured collection."""

        def run() -> Dict[str, Any]:
            from vaultq.ai_workspace import AI_WORKSPACE_DIRS, workspace_root
            from vaultq.policy import load_policy
            from vaultq.property_schema import load_property_schema

            root = _collection_root(collection_name)
            policy = load_policy(root)
            schema = load_property_schema(root)
            return {
                "root": str(root),
                "write_enabled": policy.write_enabled,
                "policy": policy.to_dict(),
                "property_schema": schema.to_dict(),
                "ai_workspace": str(workspace_root(root)),
                "ai_workspace_dirs": list(AI_WORKSPACE_DIRS),
            }

        return _run_operation(
            "vaultq_write_status",
            {"collection_name": collection_name},
            run,
            hint="Run `vq policy init --mode ai_workspace` for the intended collection.",
        )

    @mcp.tool(name="vaultq_graph_extract")
    def vaultq_graph_extract(collection_name: str = "") -> Dict[str, Any]:
        """Extract wikilinks, markdown links, and frontmatter graph edges into the VaultQ link table."""

        def run() -> Dict[str, Any]:
            from vaultq.graph import extract_graph

            return extract_graph(collection_name=collection_name or None)

        return _run_operation(
            "vaultq_graph_extract",
            {"collection_name": collection_name},
            run,
            hint="Check that Postgres is running and the configured collection path exists.",
        )

    @mcp.tool(name="vaultq_graph_neighbors")
    def vaultq_graph_neighbors(identifier: str, limit: int = 50, collection_name: str = "") -> Dict[str, Any]:
        """List outgoing graph links for a vault-relative note path."""

        def run() -> Dict[str, Any]:
            from vaultq.graph import neighbors

            return neighbors(identifier, collection_name=collection_name or None, limit=_normalize_limit(limit))

        return _run_operation(
            "vaultq_graph_neighbors",
            {"identifier": identifier, "limit": limit, "collection_name": collection_name},
            run,
            hint="Run `vaultq_graph_extract` first if the link table is empty.",
        )

    @mcp.tool(name="vaultq_graph_backlinks")
    def vaultq_graph_backlinks(identifier: str, limit: int = 50, collection_name: str = "") -> Dict[str, Any]:
        """List backlinks for a note, stem, or link target identifier."""

        def run() -> Dict[str, Any]:
            from vaultq.graph import backlinks

            return backlinks(identifier, collection_name=collection_name or None, limit=_normalize_limit(limit))

        return _run_operation(
            "vaultq_graph_backlinks",
            {"identifier": identifier, "limit": limit, "collection_name": collection_name},
            run,
            hint="Run `vaultq_graph_extract` first if the link table is empty.",
        )

    @mcp.tool(name="vaultq_graph_traverse")
    def vaultq_graph_traverse(
        identifier: str,
        depth: int = 2,
        direction: str = "out",
        edge_type: str = "",
        limit: int = 200,
        collection_name: str = "",
    ) -> Dict[str, Any]:
        """Traverse graph edges with depth, direction, and optional edge-type filtering."""

        def run() -> Dict[str, Any]:
            from vaultq.graph import traverse

            return traverse(
                identifier,
                collection_name=collection_name or None,
                depth=max(1, min(5, int(depth))),
                direction=direction,
                edge_type=edge_type or None,
                limit=_normalize_limit(limit),
            )

        return _run_operation(
            "vaultq_graph_traverse",
            {
                "identifier": identifier,
                "depth": depth,
                "direction": direction,
                "edge_type": edge_type,
                "limit": limit,
                "collection_name": collection_name,
            },
            run,
            hint="Run `vaultq_graph_extract` first if the link table is empty.",
        )

    @mcp.tool(name="vaultq_capture")
    def vaultq_capture(
        text: str,
        kind: str = "note",
        domain: str = "general",
        title: str = "",
        collection_name: str = "",
    ) -> Dict[str, Any]:
        """Create an agent-authored note under the configured AI workspace."""

        def run() -> Dict[str, Any]:
            from vaultq.write_ops import capture_note

            result = capture_note(_collection_root(collection_name), text=text, kind=kind, domain=domain, title=title or None)
            return _index_write_result(result, collection_name=collection_name)

        return _run_operation(
            "vaultq_capture",
            {"text": text, "kind": kind, "domain": domain, "title": title, "collection_name": collection_name},
            run,
            hint="Check write policy, domain/type rules, and the configured collection root.",
        )

    @mcp.tool(name="vaultq_note_put")
    def vaultq_note_put(rel_path: str, markdown: str, collection_name: str = "") -> Dict[str, Any]:
        """Create or replace a markdown note under an allowed vault-relative path."""

        def run() -> Dict[str, Any]:
            from vaultq.write_ops import note_put

            result = note_put(_collection_root(collection_name), rel_path=rel_path, markdown=markdown, agent_generated=True)
            return _index_write_result(result, collection_name=collection_name)

        return _run_operation(
            "vaultq_note_put",
            {"rel_path": rel_path, "markdown": markdown, "collection_name": collection_name},
            run,
            hint="Use an allowed AI workspace path and valid vault frontmatter.",
        )

    @mcp.tool(name="vaultq_note_append")
    def vaultq_note_append(rel_path: str, text: str, heading: str = "", collection_name: str = "") -> Dict[str, Any]:
        """Append text to an existing allowed note."""

        def run() -> Dict[str, Any]:
            from vaultq.write_ops import note_append

            result = note_append(_collection_root(collection_name), rel_path=rel_path, text=text, heading=heading or None)
            return _index_write_result(result, collection_name=collection_name)

        return _run_operation(
            "vaultq_note_append",
            {"rel_path": rel_path, "text": text, "heading": heading, "collection_name": collection_name},
            run,
            hint="Use an existing allowed AI workspace note path.",
        )

    @mcp.tool(name="vaultq_note_propose")
    def vaultq_note_propose(title: str, body: str, collection_name: str = "") -> Dict[str, Any]:
        """Write a proposal note under the AI workspace proposals folder."""

        def run() -> Dict[str, Any]:
            from vaultq.write_ops import propose_note

            result = propose_note(_collection_root(collection_name), title=title, body=body)
            return _index_write_result(result, collection_name=collection_name)

        return _run_operation(
            "vaultq_note_propose",
            {"title": title, "body": body, "collection_name": collection_name},
            run,
            hint="Use proposal notes for non-canonical suggestions that need human review.",
        )

    @mcp.tool(name="vaultq_property_propose")
    def vaultq_property_propose(
        name: str,
        reason: str,
        kind: str = "property",
        collection_name: str = "",
    ) -> Dict[str, Any]:
        """Propose a property or tag rule change without approving it."""

        def run() -> Dict[str, Any]:
            from vaultq.write_ops import propose_property

            result = propose_property(_collection_root(collection_name), property_name=name, reason=reason, tag=kind == "tag")
            return _index_write_result(result, collection_name=collection_name)

        return _run_operation(
            "vaultq_property_propose",
            {"name": name, "reason": reason, "kind": kind, "collection_name": collection_name},
            run,
            hint="Check write policy and use kind='property' or kind='tag'.",
        )

    @mcp.tool(name="vaultq_think")
    def vaultq_think(
        question: str,
        limit: int = 5,
        save_report: bool = False,
        collection_name: str = "",
    ) -> Dict[str, Any]:
        """Synthesize a lightweight cited answer from retrieval results, optionally saving a report."""

        def run() -> Dict[str, Any]:
            from vaultq.think import think

            root = _collection_root(collection_name) if save_report else None
            return think(question, limit=_normalize_limit(limit), save_report=save_report, root=root)

        return _run_operation(
            "vaultq_think",
            {"question": question, "limit": limit, "save_report": save_report, "collection_name": collection_name},
            run,
            hint="Check retrieval services; if save_report=true, ensure write policy is initialized.",
        )

    @mcp.tool(name="vaultq_maintain")
    def vaultq_maintain(collection_name: str = "", write_report: bool = False) -> Dict[str, Any]:
        """Inspect the AI workspace and optionally write a maintenance report."""

        def run() -> Dict[str, Any]:
            from vaultq.maintenance import maintenance_report

            return maintenance_report(_collection_root(collection_name), write_report=write_report)

        return _run_operation(
            "vaultq_maintain",
            {"collection_name": collection_name, "write_report": write_report},
            run,
            hint="Check configured collection root and write policy if write_report=true.",
        )

    @mcp.tool(name="vaultq_agent_prepare")
    def vaultq_agent_prepare(collection_name: str = "", write_sheet: bool = True) -> Dict[str, Any]:
        """Prepare the disabled reversible review/promote background agent runtime without starting it."""

        def run() -> Dict[str, Any]:
            from vaultq.agent_runtime import prepare_background_agent

            return prepare_background_agent(_collection_root(collection_name), write_sheet=write_sheet)

        return _run_operation(
            "vaultq_agent_prepare",
            {"collection_name": collection_name, "write_sheet": write_sheet},
            run,
            hint="Check configured collection root and write policy if write_sheet=true.",
        )

    @mcp.tool(name="vaultq_agent_maintain")
    def vaultq_agent_maintain(collection_name: str = "", dry_run: bool = False, max_actions: int = 20) -> Dict[str, Any]:
        """Run one autonomous, reversible self-maintenance pass inside the agent workspace."""

        def run() -> Dict[str, Any]:
            from vaultq.agent_runtime import run_autonomous_maintenance

            return run_autonomous_maintenance(
                _collection_root(collection_name),
                dry_run=dry_run,
                max_actions=max(1, min(200, int(max_actions or 20))),
            )

        return _run_operation(
            "vaultq_agent_maintain",
            {"collection_name": collection_name, "dry_run": dry_run, "max_actions": max_actions},
            run,
            hint="Use dry_run=true first if you want to inspect proposed internal workspace promotions.",
        )

    return mcp


def run_mcp(
    *,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 7073,
    show_banner: bool = True,
) -> None:
    mcp = build_mcp()
    _start_self_maintenance_worker()
    _start_background_index_worker()
    transport = (transport or "stdio").strip().lower()
    if transport == "stdio":
        mcp.run(show_banner=show_banner)
        return
    mcp.run(transport=transport, host=host, port=port, show_banner=show_banner)


if __name__ == "__main__":
    run_mcp()
