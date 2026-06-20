from __future__ import annotations

from datetime import datetime, timezone

from vaultq.background_index import build_background_state_payload


def test_background_state_payload_is_db_status_shaped() -> None:
    payload = build_background_state_payload(
        collection_name="second_brain",
        worker="mcp",
        enabled=True,
        summary={
            "loops": 4,
            "files_seen": 2914,
            "new_file_scans": 1,
            "path_index_runs": 2,
            "changed_index_runs": 0,
            "embed_batches": 1,
            "embeddings_written": 8,
        },
        next_new_file_scan_at=datetime(2026, 6, 4, 12, 10, tzinfo=timezone.utc),
        next_changed_refresh_at=datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc),
        last_error=None,
    )

    assert payload["kind"] == "background_index"
    assert payload["collection_name"] == "second_brain"
    assert payload["worker"] == "mcp"
    assert payload["enabled"] is True
    assert payload["summary"]["path_index_runs"] == 2
    assert payload["next_new_file_scan_at"] == "2026-06-04T12:10:00+00:00"
    assert payload["next_changed_refresh_at"] == "2026-06-05T12:00:00+00:00"
    assert payload["last_error"] is None
