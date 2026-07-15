from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional


class IndexWriteLockTimeout(TimeoutError):
    """Raised when another VaultQ writer keeps the index write lock too long."""


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def write_lock_path(base_dir: Optional[Path] = None) -> Path:
    explicit = (os.getenv("VQ_WRITE_LOCK_PATH") or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()

    config_dir = (os.getenv("VQ_CONFIG_DIR") or "").strip()
    if config_dir:
        return Path(config_dir).expanduser().resolve() / "index-write.lock"

    root = Path(base_dir or Path.cwd()).expanduser().resolve()
    return root / ".vaultq" / "index-write.lock"


def _lock_payload(*, token: str, owner: str) -> str:
    return json.dumps(
        {
            "token": token,
            "owner": owner,
            "pid": os.getpid(),
            "acquired_at": datetime.now(timezone.utc).isoformat(),
        },
        ensure_ascii=True,
        sort_keys=True,
    )


def _try_create_lock(path: Path, *, token: str, owner: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False

    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(_lock_payload(token=token, owner=owner))
        handle.write("\n")
    return True


def _read_lock(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _is_stale(path: Path, *, stale_seconds: float) -> bool:
    if stale_seconds <= 0:
        return False
    try:
        return (time.time() - path.stat().st_mtime) > stale_seconds
    except FileNotFoundError:
        return False


@contextmanager
def acquire_index_write_lock(
    *,
    base_dir: Optional[Path] = None,
    owner: str = "vaultq",
    timeout_seconds: Optional[float] = None,
    poll_seconds: Optional[float] = None,
    stale_seconds: Optional[float] = None,
) -> Iterator[Path]:
    timeout = _env_float("VQ_INDEX_WRITE_LOCK_TIMEOUT_SECONDS", 300.0) if timeout_seconds is None else float(timeout_seconds)
    poll = _env_float("VQ_INDEX_WRITE_LOCK_POLL_SECONDS", 0.25) if poll_seconds is None else float(poll_seconds)
    stale_after = _env_float("VQ_INDEX_WRITE_LOCK_STALE_SECONDS", 21600.0) if stale_seconds is None else float(stale_seconds)
    path = write_lock_path(base_dir)
    token = uuid.uuid4().hex
    deadline = time.monotonic() + max(0.0, timeout)

    while True:
        if _try_create_lock(path, token=token, owner=owner):
            break

        if _is_stale(path, stale_seconds=stale_after):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
            continue

        if timeout <= 0 or time.monotonic() >= deadline:
            holder = _read_lock(path)
            holder_text = f" owner={holder.get('owner')} pid={holder.get('pid')}" if holder else ""
            raise IndexWriteLockTimeout(f"Timed out waiting for VaultQ index write lock: {path}{holder_text}")

        time.sleep(max(0.01, min(poll, deadline - time.monotonic())))

    try:
        yield path
    finally:
        holder = _read_lock(path)
        if holder.get("token") == token:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
