from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


class PropertyValidationError(ValueError):
    pass


DEFAULT_TYPES = {
    "article",
    "concept",
    "note",
    "read",
    "report",
    "plan",
    "project",
    "resource",
    "draft",
    "system",
    "moc",
    "runbook",
    "sop",
}
DEFAULT_DOMAINS = {"ai", "ecom", "content", "strategy", "personal", "trading", "sport", "system", "general"}
DEFAULT_CREATED_BY = {"human", "agent", "mixed"}
BASE_PROPERTIES = {"type", "domain", "created", "created_by", "source", "project", "topic", "tags", "status"}
RESTRICTED_PROPERTIES = {"status"}
TAG_PREFIX_DENY = ("type/", "status/", "domain/")


@dataclass(frozen=True)
class PropertySchema:
    root: Path
    protocol_path: Path
    obsidian_types_path: Path
    allowed_types: tuple[str, ...]
    allowed_domains: tuple[str, ...]
    allowed_created_by: tuple[str, ...]
    allowed_properties: tuple[str, ...]
    obsidian_types: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": str(self.root),
            "protocol_path": str(self.protocol_path),
            "obsidian_types_path": str(self.obsidian_types_path),
            "allowed_types": list(self.allowed_types),
            "allowed_domains": list(self.allowed_domains),
            "allowed_created_by": list(self.allowed_created_by),
            "allowed_properties": list(self.allowed_properties),
            "restricted_properties": sorted(RESTRICTED_PROPERTIES),
            "obsidian_types": self.obsidian_types,
        }


def _parse_pipe_values(text: str, field: str, fallback: Iterable[str]) -> tuple[str, ...]:
    candidates = re.findall(rf"(?m)^\s*{re.escape(field)}\s*:\s*([^\n]+)$", text)
    values: list[str] = []
    for values_text in candidates:
        if "|" not in values_text:
            continue
        parsed = []
        for item in values_text.split("|"):
            value = item.strip().strip("`")
            if not value or " " in value or value.lower().startswith("optional"):
                continue
            parsed.append(value)
        if parsed:
            values = parsed
            break
    return tuple(dict.fromkeys(values)) or tuple(sorted(fallback))


def _load_obsidian_types(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    types = data.get("types") or {}
    return {str(key): str(value) for key, value in types.items()}


def load_property_schema(root: Path) -> PropertySchema:
    root = Path(root).expanduser().resolve()
    protocol_path = root / "00_System" / "Protocol - Layout and Tagging.md"
    protocol_text = protocol_path.read_text(encoding="utf-8") if protocol_path.exists() else ""
    obsidian_types_path = root / ".obsidian" / "types.json"
    obsidian_types = _load_obsidian_types(obsidian_types_path)
    allowed_properties = set(BASE_PROPERTIES)
    allowed_properties.update(obsidian_types.keys())
    return PropertySchema(
        root=root,
        protocol_path=protocol_path,
        obsidian_types_path=obsidian_types_path,
        allowed_types=_parse_pipe_values(protocol_text, "type", DEFAULT_TYPES),
        allowed_domains=_parse_pipe_values(protocol_text, "domain", DEFAULT_DOMAINS),
        allowed_created_by=tuple(sorted(DEFAULT_CREATED_BY)),
        allowed_properties=tuple(sorted(allowed_properties)),
        obsidian_types=obsidian_types,
    )


def write_schema_files(root: Path) -> Dict[str, str]:
    schema = load_property_schema(root)
    vaultq_dir = Path(root) / ".vaultq"
    vaultq_dir.mkdir(parents=True, exist_ok=True)
    property_path = vaultq_dir / "property_schema.json"
    tag_path = vaultq_dir / "tag_rules.json"
    property_path.write_text(json.dumps(schema.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not tag_path.exists():
        tag_path.write_text(
            json.dumps(
                {
                    "deny_prefixes": list(TAG_PREFIX_DENY),
                    "allow_existing_inventory": True,
                    "tags_optional": True,
                    "notes": [
                        "Do not mirror frontmatter as tags.",
                        "Use topic for retrieval themes that type/domain/folder do not cover.",
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return {"property_schema": str(property_path), "tag_rules": str(tag_path)}


def parse_frontmatter(markdown_text: str) -> Dict[str, Any]:
    match = re.match(r"^---\n(.*?)\n---\n?", markdown_text or "", flags=re.DOTALL)
    if not match:
        return {}
    try:
        data = yaml.safe_load(match.group(1)) or {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def frontmatter_to_markdown(frontmatter: Dict[str, Any]) -> str:
    return yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()


def _validate_date(value: Any, field: str, errors: list[str]) -> None:
    if isinstance(value, date):
        return
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        errors.append(f"`{field}` must be a YYYY-MM-DD date.")


def _normalize_tags(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item).strip().lstrip("#") for item in value if str(item).strip()]
    return [str(value).strip().lstrip("#")]


def validate_frontmatter(
    frontmatter: Dict[str, Any],
    *,
    schema: Optional[PropertySchema] = None,
    root: Optional[Path] = None,
    agent_generated: bool = False,
    proposal: bool = False,
) -> Dict[str, Any]:
    schema = schema or load_property_schema(root or Path.cwd())
    errors: list[str] = []
    warnings: list[str] = []
    for key in frontmatter:
        if key not in schema.allowed_properties and not proposal:
            errors.append(f"Unknown frontmatter property `{key}`. Use property proposal before writing it.")
    for key in ("type", "domain", "created"):
        if not frontmatter.get(key):
            errors.append(f"Missing required property `{key}`.")
    if agent_generated:
        if not frontmatter.get("created_by"):
            errors.append("Agent-generated notes must set `created_by: agent`.")
        elif frontmatter.get("created_by") != "agent":
            errors.append("Agent-generated notes must use `created_by: agent`.")
    created_by = frontmatter.get("created_by")
    if created_by and created_by not in schema.allowed_created_by:
        errors.append(f"`created_by` must be one of: {', '.join(schema.allowed_created_by)}.")
    note_type = frontmatter.get("type")
    if note_type and note_type not in schema.allowed_types:
        errors.append(f"`type` must be one of: {', '.join(schema.allowed_types)}.")
    if note_type == "local-chat-session":
        errors.append("`local-chat-session` is pipeline-only and must not be set manually.")
    domain = frontmatter.get("domain")
    if domain and domain not in schema.allowed_domains:
        errors.append(f"`domain` must be one of: {', '.join(schema.allowed_domains)}.")
    if "created" in frontmatter:
        _validate_date(frontmatter.get("created"), "created", errors)
    if "status" in frontmatter and not proposal and note_type != "article":
        errors.append("`status` is restricted and only allowed for protocol-approved pipeline contexts.")
    for tag in _normalize_tags(frontmatter.get("tags")):
        clean = tag.lower()
        if clean.startswith(TAG_PREFIX_DENY):
            errors.append(f"Tag `{tag}` mirrors frontmatter and is not allowed.")
    if errors:
        raise PropertyValidationError("; ".join(errors))
    return {"ok": True, "warnings": warnings, "schema": schema.to_dict()}
