from __future__ import annotations

from vaultq import watch
from vaultq.indexer import IndexStats


def _forbid_full_snapshots(monkeypatch) -> None:
    def fail_full_snapshot(**_) -> dict:
        raise AssertionError("lightweight watch must not build full mtime snapshots in idle polling")

    monkeypatch.setattr(watch, "_build_watch_snapshot", fail_full_snapshot)


def test_watch_does_not_rescan_files_on_every_idle_poll(monkeypatch) -> None:
    path_snapshots = [{"second_brain": {"existing.md"}}]
    path_scan_calls = 0
    now_values = [0.0, 60.0, 120.0]

    _forbid_full_snapshots(monkeypatch)
    monkeypatch.setattr(watch.time, "sleep", lambda _: None)

    def fake_path_snapshot(**_) -> dict[str, set[str]]:
        nonlocal path_scan_calls
        path_scan_calls += 1
        return path_snapshots[-1]

    monkeypatch.setattr(watch, "_build_path_snapshot", fake_path_snapshot, raising=False)

    summary = watch.run_watch(
        collection_name="second_brain",
        no_initial_sync=True,
        max_loops=3,
        poll_interval=60,
        new_file_index_delay_seconds=600,
        changed_index_interval_seconds=86400,
        clock=lambda: now_values.pop(0),
    )

    assert path_scan_calls == 1
    assert summary["new_file_scans"] == 0
    assert summary["path_index_runs"] == 0
    assert summary["changed_index_runs"] == 0


def test_watch_batches_new_file_indexing_on_lagged_path_scan(monkeypatch) -> None:
    path_snapshots = [
        {"second_brain": {"existing.md"}},
        {"second_brain": {"existing.md", "idea-a.md", "idea-b.md"}},
    ]
    now_values = [0.0, 600.0]
    indexed_batches: list[list[str]] = []
    embed_calls: list[dict[str, int]] = []

    _forbid_full_snapshots(monkeypatch)
    monkeypatch.setattr(watch.time, "sleep", lambda _: None)
    monkeypatch.setattr(watch, "_build_path_snapshot", lambda **_: path_snapshots.pop(0), raising=False)

    def fake_run_index_paths(rel_paths, **_) -> IndexStats:
        indexed_batches.append(list(rel_paths))
        return IndexStats(documents_scanned=len(rel_paths), documents_indexed=len(rel_paths), chunks_written=4)

    def fake_drain_embeddings(**kwargs) -> dict[str, int]:
        embed_calls.append({"embed_limit": kwargs["embed_limit"], "max_embed_batches": kwargs["max_embed_batches"]})
        return {"embed_batches": 1, "embeddings_written": 4, "pending_remaining": 0}

    monkeypatch.setattr("vaultq.indexer.run_index_paths", fake_run_index_paths, raising=False)
    monkeypatch.setattr(watch, "_drain_embeddings", fake_drain_embeddings)

    summary = watch.run_watch(
        collection_name="second_brain",
        no_initial_sync=True,
        max_loops=2,
        poll_interval=60,
        new_file_index_delay_seconds=600,
        changed_index_interval_seconds=86400,
        clock=lambda: now_values.pop(0),
    )

    assert indexed_batches == [["idea-a.md", "idea-b.md"]]
    assert embed_calls == [{"embed_limit": 200, "max_embed_batches": 4}]
    assert summary["new_file_scans"] == 1
    assert summary["path_index_runs"] == 2
    assert summary["changed_index_runs"] == 0
    assert summary["embeddings_written"] == 4


def test_watch_runs_daily_changed_refresh_without_frequent_full_snapshots(monkeypatch) -> None:
    path_snapshots = [{"second_brain": {"changed.md"}}]
    now_values = [0.0, 86399.0, 86400.0]
    collection_indexes: list[str] = []
    embed_calls = 0

    _forbid_full_snapshots(monkeypatch)
    monkeypatch.setattr(watch.time, "sleep", lambda _: None)
    monkeypatch.setattr(watch, "_build_path_snapshot", lambda **_: path_snapshots[-1], raising=False)

    def fake_run_refresh_changed_paths(*, collection_name: str, **_) -> IndexStats:
        collection_indexes.append(collection_name)
        return IndexStats(documents_scanned=1, documents_indexed=1, chunks_written=3)

    def fake_drain_embeddings(**_) -> dict[str, int]:
        nonlocal embed_calls
        embed_calls += 1
        return {"embed_batches": 1, "embeddings_written": 3, "pending_remaining": 0}

    monkeypatch.setattr("vaultq.indexer.run_refresh_changed_paths", fake_run_refresh_changed_paths)
    monkeypatch.setattr(watch, "_drain_embeddings", fake_drain_embeddings)

    summary = watch.run_watch(
        collection_name="second_brain",
        no_initial_sync=True,
        max_loops=3,
        poll_interval=60,
        new_file_index_delay_seconds=600,
        changed_index_interval_seconds=86400,
        clock=lambda: now_values.pop(0),
    )

    assert collection_indexes == ["second_brain"]
    assert embed_calls == 1
    assert summary["new_file_scans"] == 1
    assert summary["path_index_runs"] == 0
    assert summary["changed_index_runs"] == 1
    assert summary["embeddings_written"] == 3


def test_watch_continues_capped_embedding_batches_without_full_snapshots(monkeypatch) -> None:
    path_snapshots = [
        {"second_brain": {"existing.md"}},
        {"second_brain": {"existing.md", "new.md"}},
    ]
    now_values = [0.0, 600.0, 1200.0]
    indexed_batches: list[list[str]] = []
    pending_remaining = [5, 0]
    path_scan_calls = 0

    _forbid_full_snapshots(monkeypatch)
    monkeypatch.setattr(watch.time, "sleep", lambda _: None)

    def fake_path_snapshot(**_) -> dict[str, set[str]]:
        nonlocal path_scan_calls
        path_scan_calls += 1
        return path_snapshots[min(path_scan_calls - 1, len(path_snapshots) - 1)]

    def fake_run_index_paths(rel_paths, **_) -> IndexStats:
        indexed_batches.append(list(rel_paths))
        return IndexStats(documents_scanned=len(rel_paths), documents_indexed=len(rel_paths), chunks_written=6)

    def fake_drain_embeddings(**_) -> dict[str, int]:
        return {"embed_batches": 1, "embeddings_written": 3, "pending_remaining": pending_remaining.pop(0)}

    monkeypatch.setattr(watch, "_build_path_snapshot", fake_path_snapshot, raising=False)
    monkeypatch.setattr("vaultq.indexer.run_index_paths", fake_run_index_paths, raising=False)
    monkeypatch.setattr(watch, "_drain_embeddings", fake_drain_embeddings)

    summary = watch.run_watch(
        collection_name="second_brain",
        no_initial_sync=True,
        max_loops=3,
        poll_interval=60,
        new_file_index_delay_seconds=600,
        changed_index_interval_seconds=86400,
        clock=lambda: now_values.pop(0),
    )

    assert path_scan_calls <= 3
    assert indexed_batches == [["new.md"]]
    assert summary["path_index_runs"] == 1
    assert summary["new_file_scans"] == 2
    assert summary["embed_batches"] == 2
    assert summary["embeddings_written"] == 6
