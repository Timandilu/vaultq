from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from vaultq import indexer
from vaultq.indexer import (
    IndexStats,
    _classify_refresh_paths,
    _clean_text,
    _json_dumps,
    _parse_frontmatter,
)


def test_frontmatter_dates_are_json_serializable_for_indexing() -> None:
    frontmatter, body = _parse_frontmatter(
        """---
type: note
domain: ai
created: 2026-06-03
---

# Test
"""
    )

    assert frontmatter["created"] == date(2026, 6, 3)
    assert "# Test" in body
    assert _json_dumps(frontmatter) == '{"type": "note", "domain": "ai", "created": "2026-06-03"}'


def test_nul_bytes_are_sanitized_before_postgres_text_jsonb() -> None:
    assert _clean_text("a\x00b") == "a\ufffdb"
    assert "\\u0000" not in _json_dumps({"bad": "a\x00b"})
    assert "a\ufffdb" in _json_dumps({"bad": "a\x00b"})


def test_run_index_path_delegates_to_batched_indexer(monkeypatch) -> None:
    calls: list[tuple[list[str], str | None]] = []

    def fake_run_index_paths(rel_paths, **kwargs) -> IndexStats:
        calls.append((list(rel_paths), kwargs.get("collection_name")))
        return IndexStats(documents_scanned=len(rel_paths))

    monkeypatch.setattr(indexer, "run_index_paths", fake_run_index_paths)

    stats = indexer.run_index_path("new.md", collection_name="second_brain")

    assert calls == [(["new.md"], "second_brain")]
    assert stats.documents_scanned == 1


def test_refresh_classifier_uses_stat_metadata_without_reading_unchanged_files(tmp_path: Path) -> None:
    unchanged = tmp_path / "unchanged.md"
    changed = tmp_path / "changed.md"
    missing_metadata = tmp_path / "legacy.md"
    legacy_no_size = tmp_path / "legacy-no-size.md"
    legacy_stale = tmp_path / "legacy-stale.md"
    new_path = tmp_path / "new.md"
    for path in (unchanged, changed, missing_metadata, legacy_no_size, legacy_stale, new_path):
        path.write_text("# Note\n", encoding="utf-8")

    unchanged_stat = unchanged.stat()
    changed_stat = changed.stat()
    legacy_stat = missing_metadata.stat()
    legacy_no_size_stat = legacy_no_size.stat()
    legacy_stale_stat = legacy_stale.stat()
    rows = [
        {
            "id": 1,
            "rel_path": "unchanged.md",
            "metadata_json": {
                "file_mtime_ns": unchanged_stat.st_mtime_ns,
                "file_size": unchanged_stat.st_size,
            },
        },
        {
            "id": 2,
            "rel_path": "changed.md",
            "metadata_json": {
                "file_mtime_ns": changed_stat.st_mtime_ns - 1,
                "file_size": changed_stat.st_size,
            },
        },
        {
            "id": 3,
            "rel_path": "legacy.md",
            "metadata_json": {
                "file_size": legacy_stat.st_size,
            },
        },
        {
            "id": 4,
            "rel_path": "legacy-no-size.md",
            "metadata_json": {},
            "updated_at": datetime.fromtimestamp((legacy_no_size_stat.st_mtime_ns + 1_000_000_000) / 1_000_000_000, tz=timezone.utc),
        },
        {
            "id": 5,
            "rel_path": "legacy-stale.md",
            "metadata_json": {},
            "updated_at": datetime.fromtimestamp((legacy_stale_stat.st_mtime_ns - 5_000_000_000) / 1_000_000_000, tz=timezone.utc),
        },
        {
            "id": 6,
            "rel_path": "deleted.md",
            "metadata_json": {
                "file_mtime_ns": 1,
                "file_size": 1,
            },
        },
    ]

    plan = _classify_refresh_paths(
        root=tmp_path,
        current_paths={
            "unchanged.md": unchanged,
            "changed.md": changed,
            "legacy.md": missing_metadata,
            "legacy-no-size.md": legacy_no_size,
            "legacy-stale.md": legacy_stale,
            "new.md": new_path,
        },
        indexed_rows=rows,
    )

    assert plan.unchanged_paths == ["unchanged.md"]
    assert plan.changed_paths == ["changed.md", "legacy-stale.md"]
    assert plan.new_paths == ["new.md"]
    assert plan.deleted_documents == [{"id": 6, "rel_path": "deleted.md"}]
    assert plan.stamp_backfills == [
        {"id": 4, "rel_path": "legacy-no-size.md"},
        {"id": 3, "rel_path": "legacy.md"},
    ]
