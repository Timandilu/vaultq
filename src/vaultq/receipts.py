from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


SECRET_PATTERNS = (
    re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/=-]+", re.IGNORECASE),
    re.compile(r"(api[_-]?key['\"\s:=]+)[A-Za-z0-9._~+/=-]+", re.IGNORECASE),
    re.compile(r"(password['\"\s:=]+)[^\s'\"}]+", re.IGNORECASE),
)


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if not isinstance(value, str):
        return value
    text = value
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(r"\1<redacted>", text)
    return text


def operation_id(operation: str, rel_path: str, now: datetime | None = None) -> str:
    now = now or datetime.now().astimezone()
    digest = hashlib.sha256(f"{operation}:{rel_path}:{now.isoformat()}".encode("utf-8")).hexdigest()[:10]
    return f"vqop_{now.strftime('%Y%m%dT%H%M%S')}_{digest}"


def write_receipt(root: Path, payload: Dict[str, Any]) -> str:
    root = Path(root).expanduser().resolve()
    now = datetime.now().astimezone()
    op_id = str(payload.get("operation_id") or operation_id(str(payload.get("operation") or "write"), str(payload.get("path") or ""), now))
    receipt_dir = root / ".vaultq" / "receipts" / now.strftime("%Y-%m-%d")
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"{op_id}.json"
    body = dict(payload)
    body["operation_id"] = op_id
    body.setdefault("created_at", now.isoformat())
    receipt_path.write_text(json.dumps(redact(body), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return receipt_path.relative_to(root).as_posix()
