from __future__ import annotations

from vaultq.store import Settings, clear_qdrant_point_ids, connect


def test_connect_uses_short_configurable_timeout(monkeypatch) -> None:
    calls = {}

    def fake_connect(dsn, **kwargs):
        calls["dsn"] = dsn
        calls.update(kwargs)
        return "connection"

    monkeypatch.setenv("DB_CONNECT_TIMEOUT", "7")
    monkeypatch.setattr("vaultq.store.psycopg.connect", fake_connect)

    result = connect(Settings("dbname=vaultq", "http://127.0.0.1:6333", "vaultq", 384))

    assert result == "connection"
    assert calls["dsn"] == "dbname=vaultq"
    assert calls["connect_timeout"] == 7
    assert "row_factory" in calls


def test_clear_qdrant_point_ids_resets_stale_db_markers(monkeypatch) -> None:
    statements = []

    class FakeCursor:
        rowcount = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, sql):
            statements.append(sql)
            self.rowcount = 3 if "vq_chunks" in sql else 2

    class FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def cursor(self):
            return FakeCursor()

        def commit(self):
            statements.append("COMMIT")

    monkeypatch.setattr("vaultq.store.connect", lambda _settings: FakeConn())

    result = clear_qdrant_point_ids(Settings("dbname=vaultq", "http://127.0.0.1:6333", "vaultq", 1024))

    assert result == {"chunks": 3, "knowledge_objects": 2}
    assert any("UPDATE vq_chunks SET qdrant_point_id = NULL" in sql for sql in statements)
    assert any("UPDATE vq_knowledge_objects SET qdrant_point_id = NULL" in sql for sql in statements)
