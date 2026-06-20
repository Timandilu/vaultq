from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

from vaultq.store import connect, ensure_schema, status_snapshot

BACKGROUND_INDEX_KIND = "background_index"


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def background_state_name(collection_name: str | None = None) -> str:
    clean = (collection_name or "default").strip() or "default"
    return f"{BACKGROUND_INDEX_KIND}:{clean}"


def build_background_state_payload(
    *,
    collection_name: str,
    worker: str,
    enabled: bool,
    summary: Dict[str, Any],
    next_new_file_scan_at: datetime | None,
    next_changed_refresh_at: datetime | None,
    last_error: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "kind": BACKGROUND_INDEX_KIND,
        "collection_name": collection_name,
        "worker": worker,
        "enabled": bool(enabled),
        "updated_at": datetime.now().astimezone().isoformat(),
        "summary": dict(summary),
        "next_new_file_scan_at": _iso(next_new_file_scan_at),
        "next_changed_refresh_at": _iso(next_changed_refresh_at),
        "last_error": last_error,
    }


def record_background_index_state(
    *,
    collection_name: str,
    worker: str = "mcp",
    enabled: bool = True,
    summary: Dict[str, Any],
    next_new_file_scan_at: datetime | None = None,
    next_changed_refresh_at: datetime | None = None,
    last_error: Optional[Dict[str, Any]] = None,
    name: str | None = None,
) -> Dict[str, Any]:
    ensure_schema()
    payload = build_background_state_payload(
        collection_name=collection_name,
        worker=worker,
        enabled=enabled,
        summary=summary,
        next_new_file_scan_at=next_new_file_scan_at,
        next_changed_refresh_at=next_changed_refresh_at,
        last_error=last_error,
    )
    state_name = name or background_state_name(collection_name)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO vq_runtime_state (name, kind, collection_name, data_json, updated_at)
                VALUES (%s, %s, %s, %s::jsonb, NOW())
                ON CONFLICT (name) DO UPDATE SET
                    kind = EXCLUDED.kind,
                    collection_name = EXCLUDED.collection_name,
                    data_json = EXCLUDED.data_json,
                    updated_at = NOW()
                """,
                (state_name, BACKGROUND_INDEX_KIND, collection_name, json.dumps(payload, ensure_ascii=False)),
            )
        conn.commit()
    return payload


def background_index_status(collection_name: str | None = None) -> Dict[str, Any]:
    ensure_schema()
    params: tuple[Any, ...]
    where = "WHERE kind = %s"
    params = (BACKGROUND_INDEX_KIND,)
    if collection_name:
        where += " AND collection_name = %s"
        params = (BACKGROUND_INDEX_KIND, collection_name)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT name, collection_name, data_json, updated_at
                FROM vq_runtime_state
                {where}
                ORDER BY updated_at DESC
                """,
                params,
            )
            rows = list(cur.fetchall())
    return {
        "ok": True,
        "kind": BACKGROUND_INDEX_KIND,
        "collection_name": collection_name,
        "states": rows,
        "store": status_snapshot(),
    }
