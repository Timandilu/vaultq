from __future__ import annotations

from pathlib import Path

import pytest

from vaultq import indexer
from vaultq.indexer import IndexStats
from vaultq.write_lock import IndexWriteLockTimeout, acquire_index_write_lock


def test_index_write_lock_rejects_concurrent_acquire(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VQ_WRITE_LOCK_PATH", str(tmp_path / "index-write.lock"))

    with acquire_index_write_lock(base_dir=tmp_path, owner="outer", timeout_seconds=0):
        with pytest.raises(IndexWriteLockTimeout):
            with acquire_index_write_lock(base_dir=tmp_path, owner="inner", timeout_seconds=0):
                pass


def test_run_index_paths_holds_writer_lock_around_db_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeLock:
        def __enter__(self):
            events.append("lock_enter")
            return self

        def __exit__(self, *_):
            events.append("lock_exit")

    def fake_lock(**kwargs):
        events.append(f"lock_call:{kwargs.get('owner')}")
        return FakeLock()

    def fake_upsert_collection_rows(base_dir=None):
        events.append("upsert")
        return {}

    def fake_read_config(base_dir=None):
        events.append("read_config")
        return {"collections": []}

    monkeypatch.setattr(indexer, "acquire_index_write_lock", fake_lock, raising=False)
    monkeypatch.setattr(indexer, "upsert_collection_rows", fake_upsert_collection_rows)
    monkeypatch.setattr(indexer, "read_config", fake_read_config)

    stats = indexer.run_index_paths(["note.md"], base_dir=tmp_path)

    assert isinstance(stats, IndexStats)
    assert events == ["lock_call:run_index_paths", "lock_enter", "upsert", "read_config", "lock_exit"]
