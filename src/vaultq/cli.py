from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from vaultq.embedding_provider import load_env_layers

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_env_layers(_PROJECT_ROOT)


def _print(obj: Any, as_json: bool = False) -> None:
    if as_json or isinstance(obj, (dict, list)):
        print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    else:
        print(obj)


def _cmd_init(args) -> int:
    from vaultq.store import clear_qdrant_point_ids, config_path, ensure_local_config, ensure_qdrant_collection, ensure_schema

    ensure_local_config()
    ensure_schema()
    ensure_qdrant_collection(reset=args.reset)
    if args.reset:
        clear_qdrant_point_ids()
    print(f"VaultQ initialized at {config_path()}")
    return 0


def _cmd_collection_add(args) -> int:
    from vaultq.store import read_config, write_config

    config = read_config()
    path = str(Path(args.path).expanduser().resolve())
    config.setdefault("collections", [])
    config["collections"] = [row for row in config["collections"] if row["name"] != args.name]
    config["collections"].append(
        {
            "name": args.name,
            "path": path,
            "pattern": args.pattern,
            "exclude_globs": args.exclude or [],
        }
    )
    write_config(config)
    print(f"Added collection {args.name} -> {path}")
    return 0


def _cmd_collection_list(args) -> int:
    from vaultq.store import read_config

    config = read_config()
    _print(config.get("collections", []), as_json=args.json)
    return 0


def _cmd_context_add(args) -> int:
    from vaultq.store import read_config, target_to_parts, write_config

    collection_name, path_prefix = target_to_parts(args.target)
    config = read_config()
    names = {row["name"] for row in config.get("collections", [])}
    if collection_name not in names:
        raise ValueError(f"Unknown collection in target: {collection_name}")
    config.setdefault("contexts", [])
    config["contexts"] = [
        row for row in config["contexts"] if not (row["target"] == args.target and row["text"] == args.text)
    ]
    config["contexts"].append(
        {
            "target": f"vaultq://{collection_name}" + (f"/{path_prefix}" if path_prefix else ""),
            "text": args.text,
        }
    )
    write_config(config)
    print(f"Added context for {args.target}")
    return 0


def _cmd_context_list(args) -> int:
    from vaultq.store import read_config

    config = read_config()
    _print(config.get("contexts", []), as_json=args.json)
    return 0


def _cmd_status(args) -> int:
    from vaultq.store import status_snapshot

    _print(status_snapshot(), as_json=args.json)
    return 0


def _cmd_chunk_stats(args) -> int:
    from vaultq.chunk_stats import chunk_length_stats

    result = chunk_length_stats(collection_name=args.collection, sample_limit=args.sample_limit)
    _print(result, as_json=args.json)
    return 0


def _cmd_background_status(args) -> int:
    from vaultq.background_index import background_index_status

    result = background_index_status(collection_name=args.collection)
    _print(result, as_json=args.json)
    return 0


def _resolve_collection_root(*, root: Optional[str] = None, collection: Optional[str] = None) -> Path:
    if root:
        return Path(root).expanduser().resolve()
    from vaultq.store import read_config

    config = read_config()
    collections = config.get("collections", [])
    if collection:
        for row in collections:
            if row.get("name") == collection:
                return Path(row["path"]).expanduser().resolve()
        raise ValueError(f"Unknown collection: {collection}")
    if collections:
        return Path(collections[0]["path"]).expanduser().resolve()
    return Path.cwd().resolve()


def _cmd_policy(args) -> int:
    from vaultq.policy import load_policy

    root = _resolve_collection_root(root=args.root, collection=args.collection)
    _print(load_policy(root).to_dict(), as_json=args.json)
    return 0


def _cmd_policy_init(args) -> int:
    from vaultq.policy import init_policy

    root = _resolve_collection_root(root=args.root, collection=args.collection)
    _print(init_policy(root, args.mode, overwrite=args.force).to_dict(), as_json=args.json)
    return 0


def _cmd_property_list(args) -> int:
    from vaultq.property_schema import load_property_schema, write_schema_files

    root = _resolve_collection_root(root=args.root, collection=args.collection)
    schema = load_property_schema(root)
    result = schema.to_dict()
    if args.sync:
        result["written"] = write_schema_files(root)
    _print(result, as_json=args.json)
    return 0


def _cmd_property_validate(args) -> int:
    from vaultq.property_schema import parse_frontmatter, validate_frontmatter

    root = _resolve_collection_root(root=args.root, collection=args.collection)
    text = sys.stdin.read() if args.path == "-" else Path(args.path).read_text(encoding="utf-8")
    result = validate_frontmatter(parse_frontmatter(text), root=root, agent_generated=args.agent, proposal=args.proposal)
    _print(result, as_json=True)
    return 0


def _cmd_property_propose(args) -> int:
    from vaultq.write_ops import propose_property

    root = _resolve_collection_root(root=args.root, collection=args.collection)
    result = propose_property(root, property_name=args.property or args.tag, reason=args.reason, tag=bool(args.tag))
    _print(result, as_json=True)
    return 0


def _cmd_capture(args) -> int:
    from vaultq.write_ops import capture_note

    root = _resolve_collection_root(root=args.root, collection=args.collection)
    result = capture_note(
        root,
        text=args.text,
        kind=args.kind,
        domain=args.domain,
        title=args.title,
        source=args.source or [],
        topic=args.topic or [],
    )
    _print(result, as_json=args.json)
    return 0


def _cmd_note_put(args) -> int:
    from vaultq.write_ops import note_put

    root = _resolve_collection_root(root=args.root, collection=args.collection)
    markdown = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
    _print(note_put(root, rel_path=args.path, markdown=markdown, agent_generated=args.agent), as_json=True)
    return 0


def _cmd_note_append(args) -> int:
    from vaultq.write_ops import note_append

    root = _resolve_collection_root(root=args.root, collection=args.collection)
    _print(note_append(root, rel_path=args.path, text=args.text, heading=args.heading), as_json=True)
    return 0


def _cmd_note_propose(args) -> int:
    from vaultq.write_ops import propose_note

    root = _resolve_collection_root(root=args.root, collection=args.collection)
    body = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
    _print(propose_note(root, title=args.title, body=body), as_json=True)
    return 0


def _cmd_operations(args) -> int:
    from vaultq.operations import list_operations

    _print(list_operations(), as_json=args.json)
    return 0


def _cmd_write_status(args) -> int:
    from vaultq.ai_workspace import AI_WORKSPACE_DIRS, workspace_root
    from vaultq.policy import load_policy
    from vaultq.property_schema import load_property_schema

    root = _resolve_collection_root(root=args.root, collection=args.collection)
    policy = load_policy(root)
    schema = load_property_schema(root)
    result = {
        "root": str(root),
        "write_enabled": policy.write_enabled,
        "default_mode": policy.default_mode.value,
        "policy": policy.to_dict(),
        "property_schema": schema.to_dict(),
        "ai_workspace": str(workspace_root(root)),
        "ai_workspace_dirs": list(AI_WORKSPACE_DIRS),
    }
    _print(result, as_json=args.json)
    return 0


def _cmd_graph_extract(args) -> int:
    from vaultq.graph import extract_graph

    _print(extract_graph(collection_name=args.collection), as_json=True)
    return 0


def _cmd_graph_neighbors(args) -> int:
    from vaultq.graph import neighbors

    result = neighbors(args.identifier, collection_name=args.collection, limit=args.limit)
    _print(result, as_json=args.json)
    return 0


def _cmd_graph_backlinks(args) -> int:
    from vaultq.graph import backlinks

    result = backlinks(args.identifier, collection_name=args.collection, limit=args.limit)
    _print(result, as_json=args.json)
    return 0


def _cmd_graph_traverse(args) -> int:
    from vaultq.graph import traverse

    result = traverse(
        args.identifier,
        collection_name=args.collection,
        depth=args.depth,
        direction=args.direction,
        edge_type=args.type,
        limit=args.limit,
    )
    _print(result, as_json=args.json)
    return 0


def _cmd_think(args) -> int:
    from vaultq.think import think

    root = _resolve_collection_root(root=args.root, collection=args.collection) if args.save_report else None
    result = think(args.question, limit=args.limit, save_report=args.save_report, root=root)
    _print(result, as_json=args.json)
    return 0


def _cmd_maintain(args) -> int:
    from vaultq.maintenance import maintenance_report

    root = _resolve_collection_root(root=args.root, collection=args.collection)
    result = maintenance_report(root, write_report=args.write_report)
    _print(result, as_json=args.json)
    return 0


def _cmd_agent_prepare(args) -> int:
    from vaultq.agent_runtime import prepare_background_agent

    root = _resolve_collection_root(root=args.root, collection=args.collection)
    result = prepare_background_agent(root, write_sheet=args.write_sheet)
    _print(result, as_json=args.json)
    return 0


def _cmd_agent_maintain(args) -> int:
    from vaultq.agent_runtime import run_autonomous_maintenance

    root = _resolve_collection_root(root=args.root, collection=args.collection)
    result = run_autonomous_maintenance(root, dry_run=args.dry_run, max_actions=args.max_actions)
    _print(result, as_json=args.json)
    return 0


def _cmd_doctor(args) -> int:
    from vaultq.doctor import run_doctor

    result = run_doctor(live=args.live, query=args.query)
    _print(result, as_json=getattr(args, "json", True))
    return 0


def _cmd_watch(args) -> int:
    from vaultq.watch import run_watch

    result = run_watch(
        collection_name=args.collection,
        poll_interval=args.interval,
        skip_knowledge=not args.with_knowledge,
        embed_limit=args.embed_limit,
        max_embed_batches=args.max_embed_batches,
        new_file_index_delay_seconds=args.new_file_index_delay_seconds,
        changed_index_interval_seconds=args.changed_index_interval_seconds,
        pending_embed_interval_seconds=args.pending_embed_interval_seconds,
        once=args.once,
        no_initial_sync=args.no_initial_sync,
        max_loops=args.max_loops,
        logger=None if args.json else print,
    )
    if args.json:
        _print(result, as_json=True)
    return 0


def _cmd_index(args) -> int:
    from vaultq.indexer import run_index

    stats = run_index(
        collection_name=args.collection,
        force=args.force,
        skip_knowledge=args.skip_knowledge,
        limit=args.limit,
    )
    _print(stats.__dict__, as_json=args.json)
    return 0


def _cmd_embed(args) -> int:
    from vaultq.embed import embed_pending

    result = embed_pending(kinds=args.kinds, limit=args.limit)
    _print(result, as_json=args.json)
    return 0


def _cmd_query(args, mode: str = "hybrid") -> int:
    from vaultq.search import search

    retrieval_mode = getattr(args, "mode", None) or mode
    result = search(args.query, limit=args.limit, retrieval_mode=retrieval_mode)
    _print(result, as_json=args.json)
    return 0


def _cmd_related(args) -> int:
    from vaultq.related import related_work

    result = related_work(args.idea, limit=args.limit, mode=args.mode)
    _print(result, as_json=args.json)
    return 0


def _cmd_semantic_clusters(args) -> int:
    from vaultq.semantic_clusters import semantic_node_clusters

    result = semantic_node_clusters(
        collection_name=args.collection,
        seed_path_prefixes=args.seed_prefix or None,
        candidate_path_prefixes=args.candidate_prefix or None,
        exclude_path_prefixes=args.exclude_prefix or None,
        seed_limit=args.seed_limit,
        related_limit=args.related_limit,
        min_score=args.min_score,
        session_policy=args.session_policy,
        session_penalty=args.session_penalty,
        session_max_per_seed=args.session_max_per_seed,
    )
    _print(result, as_json=args.json)
    return 0


def _cmd_fetch(args) -> int:
    from vaultq.search import fetch

    result = fetch(args.point_id)
    _print(result, as_json=args.json)
    return 0


def _cmd_get(args) -> int:
    from vaultq.search import get_document

    result = get_document(args.identifier, full=args.full)
    _print(result, as_json=args.json)
    return 0


def _cmd_tui(args) -> int:
    from vaultq.tui import run_tui

    run_tui()
    return 0


def _cmd_mcp(args) -> int:
    from vaultq.mcp_server import run_mcp

    run_mcp(transport=args.transport, host=args.host, port=args.port, show_banner=not args.no_banner)
    return 0


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VaultQ: TUI, CLI, and MCP toolkit for markdown-vault ingestion and hybrid retrieval"
    )
    sub = parser.add_subparsers(dest="command")

    init_parser = sub.add_parser("init", help="Initialize local config, Postgres schema, and Qdrant collection")
    init_parser.add_argument("--reset", action="store_true", help="Reset the Qdrant collection before creating it")
    init_parser.set_defaults(func=_cmd_init)

    collection_parser = sub.add_parser("collection", help="Manage collections")
    collection_sub = collection_parser.add_subparsers(dest="collection_command", required=True)
    collection_add = collection_sub.add_parser("add", help="Add a markdown collection")
    collection_add.add_argument("path")
    collection_add.add_argument("--name", required=True)
    collection_add.add_argument("--pattern", default="**/*.md")
    collection_add.add_argument("--exclude", action="append")
    collection_add.set_defaults(func=_cmd_collection_add)
    collection_list = collection_sub.add_parser("list", help="List configured collections")
    collection_list.add_argument("--json", action="store_true")
    collection_list.set_defaults(func=_cmd_collection_list)

    context_parser = sub.add_parser("context", help="Manage path-prefix context")
    context_sub = context_parser.add_subparsers(dest="context_command", required=True)
    context_add = context_sub.add_parser("add", help="Add a collection or path-prefix context")
    context_add.add_argument("target")
    context_add.add_argument("text")
    context_add.set_defaults(func=_cmd_context_add)
    context_list = context_sub.add_parser("list", help="List configured contexts")
    context_list.add_argument("--json", action="store_true")
    context_list.set_defaults(func=_cmd_context_list)

    status_parser = sub.add_parser("status", help="Show current store status")
    status_parser.add_argument("--json", action="store_true")
    status_parser.set_defaults(func=_cmd_status)

    chunks_parser = sub.add_parser("chunks", help="Inspect chunk length and embedding health")
    chunks_sub = chunks_parser.add_subparsers(dest="chunks_command", required=True)
    chunks_stats = chunks_sub.add_parser("stats", help="Show chunk token/character length distribution")
    chunks_stats.add_argument("--collection", help="Configured collection name")
    chunks_stats.add_argument("--sample-limit", type=int, default=12, help="Number of longest chunks to include")
    chunks_stats.add_argument("--json", action="store_true")
    chunks_stats.set_defaults(func=_cmd_chunk_stats)

    background_parser = sub.add_parser("background", help="Inspect MCP-tied background indexing")
    background_sub = background_parser.add_subparsers(dest="background_command", required=True)
    background_status = background_sub.add_parser("status", help="Show DB-backed background index worker status")
    background_status.add_argument("--collection", help="Configured collection name")
    background_status.add_argument("--json", action="store_true")
    background_status.set_defaults(func=_cmd_background_status)

    operations_parser = sub.add_parser("operations", help="List registered VaultQ operations")
    operations_parser.add_argument("--json", action="store_true")
    operations_parser.set_defaults(func=_cmd_operations)

    write_status_parser = sub.add_parser("write-status", help="Show write policy, property schema, and AI workspace status")
    write_status_parser.add_argument("--root", help="Vault root; defaults to first configured collection")
    write_status_parser.add_argument("--collection", help="Configured collection name")
    write_status_parser.add_argument("--json", action="store_true")
    write_status_parser.set_defaults(func=_cmd_write_status)

    graph_parser = sub.add_parser("graph", help="Extract and inspect vault links")
    graph_sub = graph_parser.add_subparsers(dest="graph_command", required=True)
    graph_extract = graph_sub.add_parser("extract", help="Extract wikilinks, markdown links, and frontmatter graph edges")
    graph_extract.add_argument("--collection", help="Configured collection name")
    graph_extract.set_defaults(func=_cmd_graph_extract)
    graph_neighbors = graph_sub.add_parser("neighbors", help="List outgoing graph links for a note")
    graph_neighbors.add_argument("identifier")
    graph_neighbors.add_argument("--collection", help="Configured collection name")
    graph_neighbors.add_argument("--limit", type=int, default=50)
    graph_neighbors.add_argument("--json", action="store_true")
    graph_neighbors.set_defaults(func=_cmd_graph_neighbors)
    graph_backlinks = graph_sub.add_parser("backlinks", help="List backlinks for a note or identifier")
    graph_backlinks.add_argument("identifier")
    graph_backlinks.add_argument("--collection", help="Configured collection name")
    graph_backlinks.add_argument("--limit", type=int, default=50)
    graph_backlinks.add_argument("--json", action="store_true")
    graph_backlinks.set_defaults(func=_cmd_graph_backlinks)
    graph_traverse = graph_sub.add_parser("traverse", help="Traverse graph edges with depth, type, and direction filters")
    graph_traverse.add_argument("identifier")
    graph_traverse.add_argument("--collection", help="Configured collection name")
    graph_traverse.add_argument("--depth", type=int, default=2)
    graph_traverse.add_argument("--direction", choices=("out", "in", "both"), default="out")
    graph_traverse.add_argument("--type", help="Only follow one edge type")
    graph_traverse.add_argument("--limit", type=int, default=200)
    graph_traverse.add_argument("--json", action="store_true")
    graph_traverse.set_defaults(func=_cmd_graph_traverse)

    policy_parser = sub.add_parser("policy", help="Inspect or initialize write policy")
    policy_sub = policy_parser.add_subparsers(dest="policy_command")
    policy_parser.add_argument("--root", help="Vault root; defaults to first configured collection")
    policy_parser.add_argument("--collection", help="Configured collection name")
    policy_parser.add_argument("--json", action="store_true")
    policy_parser.set_defaults(func=_cmd_policy)
    policy_init = policy_sub.add_parser("init", help="Create .vaultq/policy.json")
    policy_init.add_argument("--root", help="Vault root; defaults to first configured collection")
    policy_init.add_argument("--collection", help="Configured collection name")
    policy_init.add_argument("--mode", choices=("read_only", "ai_workspace", "proposal", "trusted_local", "admin"), default="ai_workspace")
    policy_init.add_argument("--force", action="store_true")
    policy_init.add_argument("--json", action="store_true")
    policy_init.set_defaults(func=_cmd_policy_init)

    property_parser = sub.add_parser("property", help="Inspect and validate vault property rules")
    property_sub = property_parser.add_subparsers(dest="property_command", required=True)
    property_list = property_sub.add_parser("list", help="List approved properties, types, and domains")
    property_list.add_argument("--root", help="Vault root; defaults to first configured collection")
    property_list.add_argument("--collection", help="Configured collection name")
    property_list.add_argument("--sync", action="store_true", help="Write .vaultq/property_schema.json and tag_rules.json")
    property_list.add_argument("--json", action="store_true")
    property_list.set_defaults(func=_cmd_property_list)
    property_validate = property_sub.add_parser("validate", help="Validate markdown frontmatter")
    property_validate.add_argument("path", help="Markdown path, or - for stdin")
    property_validate.add_argument("--root", help="Vault root; defaults to first configured collection")
    property_validate.add_argument("--collection", help="Configured collection name")
    property_validate.add_argument("--agent", action="store_true", help="Require agent-generated note fields")
    property_validate.add_argument("--proposal", action="store_true", help="Allow proposal-time unknown properties")
    property_validate.set_defaults(func=_cmd_property_validate)
    property_propose = property_sub.add_parser("propose", help="Propose a new property")
    property_propose.add_argument("--root", help="Vault root; defaults to first configured collection")
    property_propose.add_argument("--collection", help="Configured collection name")
    property_propose.add_argument("--property", required=True)
    property_propose.add_argument("--reason", required=True)
    property_propose.set_defaults(func=_cmd_property_propose)
    tag_propose = property_sub.add_parser("propose-tag", help="Propose a new tag")
    tag_propose.add_argument("--root", help="Vault root; defaults to first configured collection")
    tag_propose.add_argument("--collection", help="Configured collection name")
    tag_propose.add_argument("--tag", required=True)
    tag_propose.add_argument("--reason", required=True)
    tag_propose.set_defaults(func=_cmd_property_propose)

    doctor_parser = sub.add_parser("doctor", help="Validate backend services and optionally live provider calls")
    doctor_parser.add_argument("--live", action="store_true", help="Also call the configured embedding/rerank APIs")
    doctor_parser.add_argument("--query", default="deployment checklist", help="Sample query for live provider checks")
    doctor_parser.add_argument("--json", action="store_true", default=True)
    doctor_parser.set_defaults(func=_cmd_doctor)

    watch_parser = sub.add_parser("watch", help="Poll collections, auto-index changed markdown, and batch embeds")
    watch_parser.add_argument("--collection", help="Watch only one configured collection")
    watch_parser.add_argument(
        "--interval",
        type=float,
        default=_env_float("VQ_WATCH_POLL_SECONDS", _env_float("VQ_BACKGROUND_INDEX_POLL_SECONDS", 300.0)),
        help="Idle scheduler wakeup interval in seconds; filesystem scans run only when their cadence is due",
    )
    watch_parser.add_argument(
        "--with-knowledge",
        action="store_true",
        help="Also run grounded knowledge extraction during watch indexing",
    )
    watch_parser.add_argument(
        "--embed-limit",
        type=int,
        default=_env_int("VQ_BACKGROUND_EMBED_LIMIT", 100),
        help="Max vectors per embed batch",
    )
    watch_parser.add_argument(
        "--max-embed-batches",
        type=int,
        default=_env_int("VQ_BACKGROUND_MAX_EMBED_BATCHES", 1),
        help="Max embed batches per watch cycle; use 0 to drain the queue fully",
    )
    watch_parser.add_argument(
        "--new-file-index-delay-seconds",
        type=float,
        default=_env_float("VQ_BACKGROUND_NEW_FILE_DELAY_SECONDS", 600.0),
        help="Cadence for path-only new-file discovery, batched indexing, and embedding",
    )
    watch_parser.add_argument(
        "--changed-index-interval-seconds",
        type=float,
        default=_env_float("VQ_BACKGROUND_CHANGED_INDEX_SECONDS", 86400.0),
        help="Minimum interval before refreshing changed/deleted-file chunks with a full incremental index",
    )
    watch_parser.add_argument(
        "--pending-embed-interval-seconds",
        type=float,
        default=_env_float("VQ_BACKGROUND_PENDING_EMBED_SECONDS", 300.0),
        help="Cadence for checking pending embeddings left by other processes; use 0 to disable",
    )
    watch_parser.add_argument("--once", action="store_true", help="Run a single watch cycle and exit")
    watch_parser.add_argument(
        "--no-initial-sync",
        action="store_true",
        help="Start from the current path-only filesystem baseline instead of indexing immediately",
    )
    watch_parser.add_argument(
        "--max-loops",
        type=int,
        help="Stop after this many watch cycles; useful for smoke tests",
    )
    watch_parser.add_argument("--json", action="store_true", help="Print the final watch summary as JSON")
    watch_parser.set_defaults(func=_cmd_watch)

    index_parser = sub.add_parser("index", help="Index markdown files and optionally extract knowledge objects")
    index_parser.add_argument("--collection")
    index_parser.add_argument("--force", action="store_true")
    index_parser.add_argument("--skip-knowledge", action="store_true")
    index_parser.add_argument("--limit", type=int)
    index_parser.add_argument("--json", action="store_true")
    index_parser.set_defaults(func=_cmd_index)

    embed_parser = sub.add_parser("embed", help="Embed pending chunks and knowledge objects into Qdrant")
    embed_parser.add_argument("--kinds", choices=("chunks", "knowledge", "both"), default="both")
    embed_parser.add_argument("--limit", type=int, default=100)
    embed_parser.add_argument("--json", action="store_true")
    embed_parser.set_defaults(func=_cmd_embed)

    capture_parser = sub.add_parser("capture", help="Create an agent-authored note under the AI workspace")
    capture_parser.add_argument("text")
    capture_parser.add_argument("--root", help="Vault root; defaults to first configured collection")
    capture_parser.add_argument("--collection", help="Configured collection name")
    capture_parser.add_argument("--kind", choices=("idea", "note", "question", "finding", "report", "draft"), default="note")
    capture_parser.add_argument("--domain", default="general")
    capture_parser.add_argument("--title")
    capture_parser.add_argument("--source", action="append")
    capture_parser.add_argument("--topic", action="append")
    capture_parser.add_argument("--json", action="store_true")
    capture_parser.set_defaults(func=_cmd_capture)

    note_parser = sub.add_parser("note", help="Write allowed vault notes")
    note_sub = note_parser.add_subparsers(dest="note_command", required=True)
    note_put_parser = note_sub.add_parser("put", help="Create or replace an allowed note")
    note_put_parser.add_argument("path")
    note_put_parser.add_argument("input", help="Markdown file path, or - for stdin")
    note_put_parser.add_argument("--root", help="Vault root; defaults to first configured collection")
    note_put_parser.add_argument("--collection", help="Configured collection name")
    note_put_parser.add_argument("--agent", action="store_true", default=True)
    note_put_parser.set_defaults(func=_cmd_note_put)
    note_append_parser = note_sub.add_parser("append", help="Append to an allowed note")
    note_append_parser.add_argument("path")
    note_append_parser.add_argument("text")
    note_append_parser.add_argument("--root", help="Vault root; defaults to first configured collection")
    note_append_parser.add_argument("--collection", help="Configured collection name")
    note_append_parser.add_argument("--heading")
    note_append_parser.set_defaults(func=_cmd_note_append)
    note_propose_parser = note_sub.add_parser("propose", help="Write a proposal note under the AI workspace")
    note_propose_parser.add_argument("title")
    note_propose_parser.add_argument("input", help="Markdown body file path, or - for stdin")
    note_propose_parser.add_argument("--root", help="Vault root; defaults to first configured collection")
    note_propose_parser.add_argument("--collection", help="Configured collection name")
    note_propose_parser.set_defaults(func=_cmd_note_propose)

    think_parser = sub.add_parser("think", help="Synthesize a lightweight cited answer from retrieval results")
    think_parser.add_argument("question")
    think_parser.add_argument("-n", "--limit", type=int, default=5)
    think_parser.add_argument("--save-report", action="store_true", help="Save the synthesis under the AI workspace reports folder")
    think_parser.add_argument("--root", help="Vault root; required only when saving a report without a configured collection")
    think_parser.add_argument("--collection", help="Configured collection name")
    think_parser.add_argument("--json", action="store_true")
    think_parser.set_defaults(func=_cmd_think)

    maintain_parser = sub.add_parser("maintain", help="Inspect the AI workspace and optionally write a maintenance report")
    maintain_parser.add_argument("--root", help="Vault root; defaults to first configured collection")
    maintain_parser.add_argument("--collection", help="Configured collection name")
    maintain_parser.add_argument("--write-report", action="store_true")
    maintain_parser.add_argument("--json", action="store_true")
    maintain_parser.set_defaults(func=_cmd_maintain)

    agent_parser = sub.add_parser("agent", help="Prepare disabled background agent runtimes")
    agent_sub = agent_parser.add_subparsers(dest="agent_command", required=True)
    agent_prepare = agent_sub.add_parser("prepare", help="Prepare reversible review/promote folders and config without starting a worker")
    agent_prepare.add_argument("--root", help="Vault root; defaults to first configured collection")
    agent_prepare.add_argument("--collection", help="Configured collection name")
    agent_prepare.add_argument("--write-sheet", dest="write_sheet", action="store_true", default=True, help="Write an initial reversible review/promote sheet")
    agent_prepare.add_argument("--no-write-sheet", dest="write_sheet", action="store_false", help="Prepare config and folders without writing a review sheet")
    agent_prepare.add_argument("--json", action="store_true")
    agent_prepare.set_defaults(func=_cmd_agent_prepare)
    agent_maintain = agent_sub.add_parser("maintain", help="Run one autonomous reversible self-maintenance pass")
    agent_maintain.add_argument("--root", help="Vault root; defaults to first configured collection")
    agent_maintain.add_argument("--collection", help="Configured collection name")
    agent_maintain.add_argument("--dry-run", action="store_true", help="Report proposed workspace promotions without moving files")
    agent_maintain.add_argument("--max-actions", type=int, default=20)
    agent_maintain.add_argument("--json", action="store_true")
    agent_maintain.set_defaults(func=_cmd_agent_maintain)

    query_parser = sub.add_parser("query", help="Hybrid retrieval with reranking")
    query_parser.add_argument("query")
    query_parser.add_argument("-n", "--limit", type=int, default=5)
    query_parser.add_argument("--mode", choices=("fast", "focused", "balanced", "deep", "hybrid"), default="hybrid")
    query_parser.add_argument("--json", action="store_true")
    query_parser.set_defaults(func=_cmd_query)

    related_parser = sub.add_parser("related", help="Surface related existing work before creating a new note")
    related_parser.add_argument("idea")
    related_parser.add_argument("-n", "--limit", type=int, default=6)
    related_parser.add_argument("--mode", choices=("fast", "focused", "balanced", "deep", "hybrid"), default="focused")
    related_parser.add_argument("--json", action="store_true")
    related_parser.set_defaults(func=_cmd_related)

    clusters_parser = sub.add_parser("clusters", help="Build semantic node clusters")
    clusters_sub = clusters_parser.add_subparsers(dest="clusters_command", required=True)
    semantic_clusters = clusters_sub.add_parser("semantic", help="Cluster indexed seed nodes by semantic anchors")
    semantic_clusters.add_argument("--collection", help="Configured collection name")
    semantic_clusters.add_argument("--seed-prefix", action="append", help="Vault-relative seed path prefix; repeatable")
    semantic_clusters.add_argument("--candidate-prefix", action="append", help="Restrict related candidates to path prefix; repeatable")
    semantic_clusters.add_argument("--exclude-prefix", action="append", help="Exclude related candidate path prefix; repeatable")
    semantic_clusters.add_argument("--seed-limit", type=int, default=25)
    semantic_clusters.add_argument("--related-limit", type=int, default=5)
    semantic_clusters.add_argument("--min-score", type=float, default=0.35)
    semantic_clusters.add_argument("--session-policy", choices=("exclude", "penalize", "include"), default="exclude")
    semantic_clusters.add_argument("--session-penalty", type=float, default=0.65)
    semantic_clusters.add_argument("--session-max-per-seed", type=int, default=1, help="Max penalized session-log candidates per seed")
    semantic_clusters.add_argument("--json", action="store_true")
    semantic_clusters.set_defaults(func=_cmd_semantic_clusters)

    search_parser = sub.add_parser("search", help="Keyword search only")
    search_parser.add_argument("query")
    search_parser.add_argument("-n", "--limit", type=int, default=5)
    search_parser.add_argument("--json", action="store_true")
    search_parser.set_defaults(func=lambda args: _cmd_query(args, "keyword"))

    vsearch_parser = sub.add_parser("vsearch", help="Dense semantic search only")
    vsearch_parser.add_argument("query")
    vsearch_parser.add_argument("-n", "--limit", type=int, default=5)
    vsearch_parser.add_argument("--json", action="store_true")
    vsearch_parser.set_defaults(func=lambda args: _cmd_query(args, "semantic"))

    fetch_parser = sub.add_parser("fetch", help="Fetch a result by Qdrant point id")
    fetch_parser.add_argument("point_id")
    fetch_parser.add_argument("--json", action="store_true")
    fetch_parser.set_defaults(func=_cmd_fetch)

    get_parser = sub.add_parser("get", help="Fetch a document by relative path or #id")
    get_parser.add_argument("identifier")
    get_parser.add_argument("--full", action="store_true")
    get_parser.add_argument("--json", action="store_true")
    get_parser.set_defaults(func=_cmd_get)

    tui_parser = sub.add_parser("tui", help="Launch the Textual operator UI")
    tui_parser.set_defaults(func=_cmd_tui)

    mcp_parser = sub.add_parser("mcp", help="Run the VaultQ MCP server")
    mcp_parser.add_argument("--transport", choices=("stdio", "http"), default=os.getenv("MCP_TRANSPORT", "stdio"))
    mcp_parser.add_argument("--host", default=os.getenv("MCP_HOST", "127.0.0.1"))
    mcp_parser.add_argument("--port", type=int, default=_env_int("MCP_PORT", 7073))
    mcp_parser.add_argument("--no-banner", action="store_true")
    mcp_parser.set_defaults(func=_cmd_mcp)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        from vaultq.tui import run_tui

        run_tui()
        return 0
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
