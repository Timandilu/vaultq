from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from vaultq.embedding_provider import load_env_file

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = _PROJECT_ROOT / ".env"
_SHARED_ENV_PATH = _PROJECT_ROOT.parent / ".ai-embedding.env"
if _ENV_PATH.exists():
    load_env_file(_ENV_PATH, override=False)
if _SHARED_ENV_PATH.exists():
    load_env_file(_SHARED_ENV_PATH, override=False)

from vaultq.embed import embed_pending
from vaultq.indexer import run_index
from vaultq.search import get_document, search
from vaultq.store import (
    collection_rows,
    config_path,
    ensure_local_config,
    ensure_qdrant_collection,
    ensure_schema,
    read_config,
    status_snapshot,
    target_to_parts,
    write_config,
)


def _print(obj: Any, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    else:
        if isinstance(obj, (dict, list)):
            print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
        else:
            print(obj)


def _cmd_init(args) -> int:
    ensure_local_config()
    ensure_schema()
    ensure_qdrant_collection(reset=args.reset)
    print(f"VaultQ initialized at {config_path()}")
    return 0


def _cmd_collection_add(args) -> int:
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
    rows = collection_rows()
    _print(rows, as_json=args.json)
    return 0


def _cmd_context_add(args) -> int:
    collection_name, path_prefix = target_to_parts(args.target)
    config = read_config()
    config.setdefault("contexts", [])
    config["contexts"].append({"target": f"vaultq://{collection_name}" + (f"/{path_prefix}" if path_prefix else ""), "text": args.text})
    write_config(config)
    print(f"Added context for {args.target}")
    return 0


def _cmd_status(args) -> int:
    _print(status_snapshot(), as_json=args.json)
    return 0


def _cmd_index(args) -> int:
    stats = run_index(
        collection_name=args.collection,
        force=args.force,
        skip_knowledge=args.skip_knowledge,
        limit=args.limit,
    )
    _print(stats.__dict__, as_json=args.json)
    return 0


def _cmd_embed(args) -> int:
    result = embed_pending(kinds=args.kinds, limit=args.limit)
    _print(result, as_json=args.json)
    return 0


def _cmd_query(args, mode: str) -> int:
    result = search(args.query, limit=args.limit, retrieval_mode=mode)
    _print(result, as_json=args.json)
    return 0


def _cmd_get(args) -> int:
    result = get_document(args.identifier, full=args.full)
    _print(result, as_json=args.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VaultQ CLI")
    sub = parser.add_subparsers(dest="command", required=True)

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
    collection_list = collection_sub.add_parser("list", help="List collections")
    collection_list.add_argument("--json", action="store_true")
    collection_list.set_defaults(func=_cmd_collection_list)

    context_add = sub.add_parser("context", help="Manage contexts")
    context_sub = context_add.add_subparsers(dest="context_command", required=True)
    context_add_cmd = context_sub.add_parser("add", help="Add context for a collection or path prefix")
    context_add_cmd.add_argument("target")
    context_add_cmd.add_argument("text")
    context_add_cmd.set_defaults(func=_cmd_context_add)

    status_parser = sub.add_parser("status", help="Show store status")
    status_parser.add_argument("--json", action="store_true")
    status_parser.set_defaults(func=_cmd_status)

    index_parser = sub.add_parser("index", help="Index markdown files and optionally enrich them")
    index_parser.add_argument("--collection")
    index_parser.add_argument("--force", action="store_true")
    index_parser.add_argument("--skip-knowledge", action="store_true")
    index_parser.add_argument("--limit", type=int)
    index_parser.add_argument("--json", action="store_true")
    index_parser.set_defaults(func=_cmd_index)

    embed_parser = sub.add_parser("embed", help="Embed pending chunks and knowledge objects")
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

    get_parser = sub.add_parser("get", help="Fetch a document by relative path or #id")
    get_parser.add_argument("identifier")
    get_parser.add_argument("--full", action="store_true")
    get_parser.add_argument("--json", action="store_true")
    get_parser.set_defaults(func=_cmd_get)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
