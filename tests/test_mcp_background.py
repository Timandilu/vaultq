from __future__ import annotations

from vaultq import mcp_server


def test_background_index_settings_are_disabled_without_collection(monkeypatch) -> None:
    for key in (
        "VQ_BACKGROUND_INDEX_COLLECTION",
        "VQ_BACKGROUND_INDEX_ENABLED",
        "VQ_SELF_MAINTAIN_COLLECTION",
    ):
        monkeypatch.delenv(key, raising=False)

    assert mcp_server._background_index_settings() is None


def test_background_index_settings_use_low_impact_defaults(monkeypatch) -> None:
    monkeypatch.setenv("VQ_BACKGROUND_INDEX_COLLECTION", "second_brain")
    for key in (
        "VQ_BACKGROUND_INDEX_POLL_SECONDS",
        "VQ_BACKGROUND_NEW_FILE_DELAY_SECONDS",
        "VQ_BACKGROUND_CHANGED_INDEX_SECONDS",
        "VQ_BACKGROUND_PENDING_EMBED_SECONDS",
        "VQ_BACKGROUND_EMBED_LIMIT",
        "VQ_BACKGROUND_MAX_EMBED_BATCHES",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = mcp_server._background_index_settings()

    assert settings == {
        "collection_name": "second_brain",
        "poll_interval": 300,
        "new_file_index_delay_seconds": 600,
        "changed_index_interval_seconds": 86400,
        "embed_limit": 100,
        "max_embed_batches": 1,
        "pending_embed_interval_seconds": 300,
    }
