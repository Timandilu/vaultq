from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from vaultq.embedding_provider import load_env_layers

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_env_layers(_PROJECT_ROOT)


def _print(obj: Any, as_json: bool = False) -> None:
    if as_json or isinstance(obj, (dict, list)):
        print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    else:
        print(obj)


def _cmd_init(args) -> int:
    from vaultq.store import config_path, ensure_local_config, ensure_qdrant_collection, ensure_schema

    ensure_local_config()
    ensure_schema()
    ensure_qdrant_collection(reset=args.reset)
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


def _cmd_doctor(args) -> int:
    from vaultq.doctor import run_doctor

    result = run_doctor(live=args.live, query=args.query)
    _print(result, as_json=True)
    return 0


def _cmd_watch(args) -> int:
    from vaultq.watch import run_watch

    result = run_watch(
        collection_name=args.collection,
        poll_interval=args.interval,
        skip_knowledge=not args.with_knowledge,
        embed_limit=args.embed_limit,
        max_embed_batches=args.max_embed_batches,
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


def _cmd_query(args, mode: str) -> int:
    from vaultq.search import search

    result = search(args.query, limit=args.limit, retrieval_mode=mode)
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

    doctor_parser = sub.add_parser("doctor", help="Validate backend services and optionally live provider calls")
    doctor_parser.add_argument("--live", action="store_true", help="Also call the configured embedding/rerank APIs")
    doctor_parser.add_argument("--query", default="deployment checklist", help="Sample query for live provider checks")
    doctor_parser.set_defaults(func=_cmd_doctor)

    watch_parser = sub.add_parser("watch", help="Poll collections, auto-index changed markdown, and batch embeds")
    watch_parser.add_argument("--collection", help="Watch only one configured collection")
    watch_parser.add_argument("--interval", type=float, default=2.0, help="Polling interval in seconds")
    watch_parser.add_argument(
        "--with-knowledge",
        action="store_true",
        help="Also run grounded knowledge extraction during watch indexing",
    )
    watch_parser.add_argument("--embed-limit", type=int, default=200, help="Max vectors per embed batch")
    watch_parser.add_argument(
        "--max-embed-batches",
        type=int,
        default=4,
        help="Max embed batches per watch cycle; use 0 to drain the queue fully",
    )
    watch_parser.add_argument("--once", action="store_true", help="Run a single watch cycle and exit")
    watch_parser.add_argument(
        "--no-initial-sync",
        action="store_true",
        help="Start from the current filesystem baseline instead of indexing immediately",
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

    query_parser = sub.add_parser("query", help="Hybrid retrieval with reranking")
    query_parser.add_argument("query")
    query_parser.add_argument("-n", "--limit", type=int, default=5)
    query_parser.add_argument("--json", action="store_true")
    query_parser.set_defaults(func=lambda args: _cmd_query(args, "hybrid"))

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
    mcp_parser.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    mcp_parser.add_argument("--host", default="127.0.0.1")
    mcp_parser.add_argument("--port", type=int, default=7070)
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
