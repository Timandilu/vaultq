from __future__ import annotations

import fnmatch
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event
from typing import Callable, Dict, Optional, Tuple

FileStamp = Tuple[int, int]
Logger = Optional[Callable[[str], None]]
CONFIG_DIR_NAME = ".vaultq"
CONFIG_FILE_NAME = "config.json"


@dataclass
class WatchSummary:
    loops: int = 0
    collections_watched: int = 0
    files_seen: int = 0
    cycles_with_changes: int = 0
    collections_indexed: int = 0
    documents_indexed: int = 0
    documents_deleted: int = 0
    chunks_written: int = 0
    knowledge_objects_written: int = 0
    embed_batches: int = 0
    embeddings_written: int = 0
    initial_sync: bool = False
    stopped: bool = False


def _log(logger: Logger, message: str) -> None:
    if logger is not None:
        logger(message)


def _config_path(base_dir: Optional[Path] = None) -> Path:
    explicit = (os.getenv("VQ_CONFIG_DIR") or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve() / CONFIG_FILE_NAME
    return Path(base_dir or Path.cwd()).resolve() / CONFIG_DIR_NAME / CONFIG_FILE_NAME


def _read_config(base_dir: Optional[Path] = None) -> Dict[str, object]:
    path = _config_path(base_dir)
    if not path.exists():
        return {"collections": [], "contexts": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _matches_excludes(rel_path: str, patterns: list[str]) -> bool:
    normalized = rel_path.replace("\\", "/")
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in patterns)


def _iter_markdown_files(root: Path, pattern: str, exclude_globs: list[str]):
    for path in root.glob(pattern):
        if not path.is_file():
            continue
        rel_path = path.relative_to(root).as_posix()
        if _matches_excludes(rel_path, exclude_globs):
            continue
        yield path


def _collection_snapshot(collection: Dict[str, object]) -> Dict[str, FileStamp]:
    root = Path(str(collection.get("path") or "")).expanduser().resolve()
    pattern = str(collection.get("pattern") or "**/*.md")
    exclude_globs = list(collection.get("exclude_globs") or [])
    snapshot: Dict[str, FileStamp] = {}
    if not root.exists():
        return snapshot
    for path in _iter_markdown_files(root, pattern, exclude_globs):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        rel_path = path.relative_to(root).as_posix()
        snapshot[rel_path] = (int(stat.st_mtime_ns), int(stat.st_size))
    return snapshot


def _build_watch_snapshot(
    *,
    base_dir: Optional[Path] = None,
    collection_name: Optional[str] = None,
) -> Dict[str, Dict[str, FileStamp]]:
    config = _read_config(base_dir)
    snapshots: Dict[str, Dict[str, FileStamp]] = {}
    for collection in config.get("collections", []):
        name = str(collection.get("name") or "").strip()
        if not name:
            continue
        if collection_name and name != collection_name:
            continue
        snapshots[name] = _collection_snapshot(collection)
    return snapshots


def _changed_collections(
    previous: Dict[str, Dict[str, FileStamp]],
    current: Dict[str, Dict[str, FileStamp]],
) -> list[str]:
    changed: list[str] = []
    for name, snapshot in current.items():
        if previous.get(name) != snapshot:
            changed.append(name)
    return changed


def _drain_embeddings(
    *,
    skip_knowledge: bool,
    embed_limit: int,
    max_embed_batches: int,
    logger: Logger,
) -> Dict[str, int]:
    from vaultq.embed import embed_pending
    from vaultq.store import status_snapshot

    kinds = "chunks" if skip_knowledge else "both"
    batch_count = 0
    embeddings_written = 0
    while True:
        if max_embed_batches > 0 and batch_count >= max_embed_batches:
            break
        pending = status_snapshot()
        pending_total = int(pending.get("pending_chunk_embeddings") or 0)
        if not skip_knowledge:
            pending_total += int(pending.get("pending_knowledge_embeddings") or 0)
        if pending_total <= 0:
            break
        result = embed_pending(kinds=kinds, limit=embed_limit)
        embedded = int(result.get("embedded") or 0)
        if embedded <= 0:
            break
        batch_count += 1
        embeddings_written += embedded
        _log(
            logger,
            f"watch: embed batch {batch_count} completed ({embedded} vectors; "
            f"chunks={result.get('chunks', 0)}, knowledge={result.get('knowledge_objects', 0)})",
        )
        if embedded < embed_limit:
            break
    return {"embed_batches": batch_count, "embeddings_written": embeddings_written}


def run_watch(
    *,
    base_dir: Optional[Path] = None,
    collection_name: Optional[str] = None,
    poll_interval: float = 2.0,
    skip_knowledge: bool = True,
    embed_limit: int = 200,
    max_embed_batches: int = 4,
    once: bool = False,
    no_initial_sync: bool = False,
    max_loops: Optional[int] = None,
    stop_event: Optional[Event] = None,
    logger: Logger = None,
) -> Dict[str, int | bool]:
    summary = WatchSummary(initial_sync=not no_initial_sync)
    previous_snapshot: Optional[Dict[str, Dict[str, FileStamp]]] = None
    if no_initial_sync:
        previous_snapshot = _build_watch_snapshot(base_dir=base_dir, collection_name=collection_name)
        files_seen = sum(len(rows) for rows in previous_snapshot.values())
        _log(
            logger,
            f"watch: baseline captured for {len(previous_snapshot)} collections and {files_seen} markdown files",
        )

    while True:
        if stop_event is not None and stop_event.is_set():
            summary.stopped = True
            break

        current_snapshot = _build_watch_snapshot(base_dir=base_dir, collection_name=collection_name)
        summary.loops += 1
        summary.collections_watched = len(current_snapshot)
        summary.files_seen = sum(len(rows) for rows in current_snapshot.values())

        if previous_snapshot is None:
            changed = sorted(current_snapshot.keys())
            if changed:
                _log(logger, f"watch: initial sync for {', '.join(changed)}")
        else:
            changed = _changed_collections(previous_snapshot, current_snapshot)
            if changed:
                _log(logger, f"watch: changes detected in {', '.join(changed)}")

        if changed:
            summary.cycles_with_changes += 1
            from vaultq.indexer import run_index

            for name in changed:
                stats = run_index(
                    base_dir=base_dir,
                    collection_name=name,
                    skip_knowledge=skip_knowledge,
                )
                summary.collections_indexed += 1
                summary.documents_indexed += stats.documents_indexed
                summary.documents_deleted += stats.documents_deleted
                summary.chunks_written += stats.chunks_written
                summary.knowledge_objects_written += stats.knowledge_objects_written
                _log(
                    logger,
                    f"watch: indexed {name} "
                    f"(indexed={stats.documents_indexed}, deleted={stats.documents_deleted}, "
                    f"skipped={stats.documents_skipped}, chunks={stats.chunks_written})",
                )
            embed_stats = _drain_embeddings(
                skip_knowledge=skip_knowledge,
                embed_limit=embed_limit,
                max_embed_batches=max_embed_batches,
                logger=logger,
            )
            summary.embed_batches += int(embed_stats["embed_batches"])
            summary.embeddings_written += int(embed_stats["embeddings_written"])

        previous_snapshot = current_snapshot

        if once:
            break
        if max_loops is not None and summary.loops >= max_loops:
            break
        if stop_event is not None:
            if stop_event.wait(max(0.1, poll_interval)):
                summary.stopped = True
                break
        else:
            time.sleep(max(0.1, poll_interval))

    return asdict(summary)
