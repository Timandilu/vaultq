from __future__ import annotations

import fnmatch
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
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
    new_file_scans: int = 0
    path_index_runs: int = 0
    changed_index_runs: int = 0
    new_files_queued: int = 0
    changed_collections_queued: int = 0
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


def _collection_path_snapshot(collection: Dict[str, object]) -> set[str]:
    root = Path(str(collection.get("path") or "")).expanduser().resolve()
    pattern = str(collection.get("pattern") or "**/*.md")
    exclude_globs = list(collection.get("exclude_globs") or [])
    snapshot: set[str] = set()
    if not root.exists():
        return snapshot
    for path in _iter_markdown_files(root, pattern, exclude_globs):
        snapshot.add(path.relative_to(root).as_posix())
    return snapshot


def _build_path_snapshot(
    *,
    base_dir: Optional[Path] = None,
    collection_name: Optional[str] = None,
) -> Dict[str, set[str]]:
    config = _read_config(base_dir)
    snapshots: Dict[str, set[str]] = {}
    for collection in config.get("collections", []):
        name = str(collection.get("name") or "").strip()
        if not name:
            continue
        if collection_name and name != collection_name:
            continue
        snapshots[name] = _collection_path_snapshot(collection)
    return snapshots


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


def _snapshot_diffs(
    previous: Dict[str, Dict[str, FileStamp]],
    current: Dict[str, Dict[str, FileStamp]],
) -> Dict[str, Dict[str, list[str]]]:
    diffs: Dict[str, Dict[str, list[str]]] = {}
    for name in sorted(set(previous) | set(current)):
        before = previous.get(name, {})
        after = current.get(name, {})
        new_paths = sorted(path for path in after if path not in before)
        changed_paths = sorted(path for path in after if path in before and after[path] != before[path])
        deleted_paths = sorted(path for path in before if path not in after)
        if new_paths or changed_paths or deleted_paths:
            diffs[name] = {
                "new": new_paths,
                "changed": changed_paths,
                "deleted": deleted_paths,
            }
    return diffs


def _path_snapshot_diffs(
    previous: Dict[str, set[str]],
    current: Dict[str, set[str]],
) -> Dict[str, Dict[str, list[str]]]:
    diffs: Dict[str, Dict[str, list[str]]] = {}
    for name in sorted(set(previous) | set(current)):
        before = previous.get(name, set())
        after = current.get(name, set())
        new_paths = sorted(after - before)
        deleted_paths = sorted(before - after)
        if new_paths or deleted_paths:
            diffs[name] = {
                "new": new_paths,
                "deleted": deleted_paths,
            }
    return diffs


def _record_path_summary(summary: WatchSummary, snapshot: Dict[str, set[str]]) -> None:
    summary.collections_watched = len(snapshot)
    summary.files_seen = sum(len(rows) for rows in snapshot.values())


def _add_index_stats(summary: WatchSummary, stats) -> None:
    summary.documents_indexed += stats.documents_indexed
    summary.documents_deleted += stats.documents_deleted
    summary.chunks_written += stats.chunks_written
    summary.knowledge_objects_written += stats.knowledge_objects_written


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
    pending_after = status_snapshot()
    pending_remaining = int(pending_after.get("pending_chunk_embeddings") or 0)
    if not skip_knowledge:
        pending_remaining += int(pending_after.get("pending_knowledge_embeddings") or 0)
    return {
        "embed_batches": batch_count,
        "embeddings_written": embeddings_written,
        "pending_remaining": pending_remaining,
    }


def run_watch(
    *,
    base_dir: Optional[Path] = None,
    collection_name: Optional[str] = None,
    poll_interval: float = 2.0,
    skip_knowledge: bool = True,
    embed_limit: int = 200,
    max_embed_batches: int = 4,
    new_file_index_delay_seconds: float = 600.0,
    changed_index_interval_seconds: float = 86400.0,
    once: bool = False,
    no_initial_sync: bool = False,
    max_loops: Optional[int] = None,
    stop_event: Optional[Event] = None,
    clock: Callable[[], float] = time.monotonic,
    state_name: Optional[str] = None,
    state_worker: str = "cli",
    drain_existing_pending: bool = False,
    logger: Logger = None,
) -> Dict[str, int | bool]:
    summary = WatchSummary(initial_sync=not no_initial_sync)
    previous_path_snapshot: Optional[Dict[str, set[str]]] = None
    next_new_file_scan_at: Optional[float] = None
    next_changed_index_at: Optional[float] = None
    next_pending_embed_at: Optional[float] = None
    checked_existing_pending = False
    new_file_index_delay_seconds = max(0.0, float(new_file_index_delay_seconds))
    changed_index_interval_seconds = max(0.0, float(changed_index_interval_seconds))
    if no_initial_sync:
        previous_path_snapshot = _build_path_snapshot(base_dir=base_dir, collection_name=collection_name)
        _record_path_summary(summary, previous_path_snapshot)
        _log(
            logger,
            f"watch: baseline captured for {summary.collections_watched} collections and {summary.files_seen} markdown files",
        )

    while True:
        if stop_event is not None and stop_event.is_set():
            summary.stopped = True
            break

        now = clock()
        summary.loops += 1
        if drain_existing_pending and not checked_existing_pending:
            checked_existing_pending = True
            try:
                pending_snapshot = status_snapshot()
                pending_existing = int(pending_snapshot.get("pending_chunk_embeddings") or 0)
                if not skip_knowledge:
                    pending_existing += int(pending_snapshot.get("pending_knowledge_embeddings") or 0)
                if pending_existing > 0:
                    next_pending_embed_at = now + new_file_index_delay_seconds
                    _log(logger, f"watch: scheduled pending embedding drain ({pending_existing} vectors)")
            except Exception as exc:
                _log(logger, f"watch: pending embedding check failed ({exc.__class__.__name__}: {exc})")
        initial_index_collections: list[str] = []
        due_new_paths: Dict[str, list[str]] = {}
        due_changed_collections: list[str] = []
        due_pending_embed = next_pending_embed_at is not None and now >= next_pending_embed_at
        state_next_new_file_scan_at: Optional[datetime] = None
        state_next_changed_refresh_at: Optional[datetime] = None

        if next_new_file_scan_at is None:
            next_new_file_scan_at = now + new_file_index_delay_seconds
        if next_changed_index_at is None:
            next_changed_index_at = now + changed_index_interval_seconds

        if previous_path_snapshot is None:
            previous_path_snapshot = _build_path_snapshot(base_dir=base_dir, collection_name=collection_name)
            _record_path_summary(summary, previous_path_snapshot)
            initial_index_collections = sorted(previous_path_snapshot.keys())
            if initial_index_collections:
                summary.cycles_with_changes += 1
                _log(logger, f"watch: initial sync for {', '.join(initial_index_collections)}")
        else:
            _record_path_summary(summary, previous_path_snapshot)
            if now >= next_new_file_scan_at:
                current_path_snapshot = _build_path_snapshot(base_dir=base_dir, collection_name=collection_name)
                summary.new_file_scans += 1
                _record_path_summary(summary, current_path_snapshot)
                diffs = _path_snapshot_diffs(previous_path_snapshot, current_path_snapshot)
                changed = sorted(diffs.keys())
                if changed:
                    detail = []
                    for name, diff in diffs.items():
                        if collection_name and name != collection_name:
                            continue
                        due_new_paths[name] = diff["new"]
                        detail.append(f"{name} new={len(diff['new'])} deleted={len(diff['deleted'])}")
                    if detail:
                        summary.cycles_with_changes += 1
                        _log(logger, "watch: path changes discovered (" + "; ".join(detail) + ")")
                previous_path_snapshot = current_path_snapshot
                next_new_file_scan_at = now + new_file_index_delay_seconds

            if now >= next_changed_index_at:
                due_changed_collections = sorted(previous_path_snapshot.keys())
                next_changed_index_at = now + changed_index_interval_seconds

        wall_now = datetime.now().astimezone()
        if next_new_file_scan_at is not None:
            state_next_new_file_scan_at = wall_now + timedelta(seconds=max(0.0, next_new_file_scan_at - now))
        if next_changed_index_at is not None:
            state_next_changed_refresh_at = wall_now + timedelta(seconds=max(0.0, next_changed_index_at - now))

        summary.new_files_queued = 0
        summary.changed_collections_queued = 0

        if initial_index_collections:
            from vaultq.indexer import run_index

            for name in initial_index_collections:
                stats = run_index(
                    base_dir=base_dir,
                    collection_name=name,
                    skip_knowledge=skip_knowledge,
                )
                summary.collections_indexed += 1
                _add_index_stats(summary, stats)
                _log(
                    logger,
                    f"watch: indexed {name} "
                    f"(indexed={stats.documents_indexed}, deleted={stats.documents_deleted}, "
                    f"skipped={stats.documents_skipped}, chunks={stats.chunks_written})",
                )

        indexed_due_work = False
        if due_new_paths:
            from vaultq.indexer import run_index_paths

            for name, paths in due_new_paths.items():
                if not paths:
                    continue
                stats = run_index_paths(
                    paths,
                    base_dir=base_dir,
                    collection_name=name,
                    force=True,
                    skip_knowledge=skip_knowledge,
                )
                summary.path_index_runs += len(paths)
                _add_index_stats(summary, stats)
                _log(
                    logger,
                    f"watch: indexed new paths {name}:{len(paths)} "
                    f"(indexed={stats.documents_indexed}, chunks={stats.chunks_written})",
                )
            indexed_due_work = True

        if due_changed_collections:
            from vaultq.indexer import run_refresh_changed_paths

            for name in due_changed_collections:
                stats = run_refresh_changed_paths(
                    base_dir=base_dir,
                    collection_name=name,
                    skip_knowledge=skip_knowledge,
                )
                summary.collections_indexed += 1
                summary.changed_index_runs += 1
                _add_index_stats(summary, stats)
                _log(
                    logger,
                    f"watch: daily changed-chunk refresh for {name} "
                    f"(indexed={stats.documents_indexed}, deleted={stats.documents_deleted}, "
                    f"skipped={stats.documents_skipped}, chunks={stats.chunks_written})",
                )
            indexed_due_work = True

        if initial_index_collections or indexed_due_work or due_pending_embed:
            embed_stats = _drain_embeddings(
                skip_knowledge=skip_knowledge,
                embed_limit=embed_limit,
                max_embed_batches=max_embed_batches,
                logger=logger,
            )
            summary.embed_batches += int(embed_stats["embed_batches"])
            summary.embeddings_written += int(embed_stats["embeddings_written"])
            if int(embed_stats.get("pending_remaining") or 0) > 0:
                next_pending_embed_at = now + new_file_index_delay_seconds
            else:
                next_pending_embed_at = None

        if state_name:
            try:
                from vaultq.background_index import record_background_index_state

                record_background_index_state(
                    collection_name=collection_name or "all",
                    worker=state_worker,
                    enabled=True,
                    summary=asdict(summary),
                    next_new_file_scan_at=state_next_new_file_scan_at,
                    next_changed_refresh_at=state_next_changed_refresh_at,
                    last_error=None,
                    name=state_name,
                )
            except Exception as exc:
                _log(logger, f"watch: background state write failed ({exc.__class__.__name__}: {exc})")

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
