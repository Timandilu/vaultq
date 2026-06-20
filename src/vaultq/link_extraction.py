from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, List

from vaultq.property_schema import parse_frontmatter


@dataclass(frozen=True)
class ExtractedLink:
    target_identifier: str
    edge_type: str
    anchor_text: str
    source_kind: str
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_identifier": self.target_identifier,
            "edge_type": self.edge_type,
            "anchor_text": self.anchor_text,
            "source_kind": self.source_kind,
            "confidence": self.confidence,
        }


def _strip_code_blocks(text: str) -> str:
    content = text or ""
    out: list[str] = []
    i = 0
    while i < len(content):
        if content.startswith("```", i):
            end = content.find("```", i + 3)
            if end == -1:
                out.append(" " * (len(content) - i))
                break
            out.append(" " * (end + 3 - i))
            i = end + 3
            continue
        if content[i] == "`":
            end = content.find("`", i + 1)
            if end != -1 and "\n" not in content[i + 1 : end]:
                out.append(" " * (end + 1 - i))
                i = end + 1
                continue
        out.append(content[i])
        i += 1
    return "".join(out)


def _normalize_target(target: str) -> str:
    value = (target or "").strip()
    value = value.split("#", 1)[0].strip()
    value = value.replace("\\", "/")
    if value.endswith(".md"):
        value = value[:-3]
    try:
        return PurePosixPath(value).as_posix().strip("/")
    except Exception:
        return value.strip("/")


def normalize_basename(value: str) -> str:
    normalized = _normalize_target(value).strip().lower()
    if not normalized:
        return ""
    return normalized.rsplit("/", 1)[-1]


def build_basename_index(paths: Iterable[str]) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for raw_path in paths:
        path = _normalize_target(str(raw_path or ""))
        basename = normalize_basename(path)
        if not basename:
            continue
        bucket = index.setdefault(basename, [])
        if path not in bucket:
            bucket.append(path)
    for basename, bucket in index.items():
        index[basename] = sorted(bucket, key=lambda item: (len(item), item.lower(), item))
    return index


def query_basename_index(index: Dict[str, List[str]], query: str, *, limit: int = 5) -> List[str]:
    basename = normalize_basename(query)
    if not basename:
        return []
    return list(index.get(basename, [])[: max(1, int(limit or 1))])


def _field_values(value: Any) -> Iterable[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        output = []
        for item in value:
            if isinstance(item, dict):
                for key in ("slug", "path", "name", "title"):
                    if str(item.get(key) or "").strip():
                        output.append(str(item[key]).strip())
                        break
            elif str(item).strip():
                output.append(str(item).strip())
        return output
    if isinstance(value, dict):
        for key in ("slug", "path", "name", "title"):
            if str(value.get(key) or "").strip():
                return [str(value[key]).strip()]
        return []
    return [str(value).strip()]


def extract_links(markdown_text: str, *, graph_fields: Iterable[str] = ("source", "project", "topic")) -> List[ExtractedLink]:
    body = _strip_code_blocks(markdown_text)
    links: list[ExtractedLink] = []
    seen: set[tuple[str, str, str, str]] = set()

    def add(target: str, edge_type: str, anchor: str, source_kind: str, confidence: float = 1.0) -> None:
        normalized = _normalize_target(target)
        if not normalized:
            return
        key = (normalized, edge_type, anchor, source_kind)
        if key in seen:
            return
        seen.add(key)
        links.append(ExtractedLink(normalized, edge_type, anchor, source_kind, confidence))

    qualified_ranges: list[tuple[int, int]] = []
    for match in re.finditer(r"\[\[([a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?):([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]", body, flags=re.IGNORECASE):
        source_id, target = match.group(1), match.group(2)
        anchor = match.group(3) or target
        add(target, "links_to", anchor, f"wikilink:{source_id}")
        qualified_ranges.append((match.start(), match.end()))

    masked = list(body)
    for start, end in qualified_ranges:
        for index in range(start, min(end, len(masked))):
            masked[index] = " "
    unqualified_body = "".join(masked)

    for match in re.finditer(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]", unqualified_body):
        target = match.group(1)
        anchor = match.group(2) or target
        add(target, "links_to", anchor, "wikilink")

    for match in re.finditer(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", body):
        anchor, target = match.groups()
        if re.match(r"^[a-z][a-z0-9+.-]*://", target, flags=re.IGNORECASE):
            continue
        if target.startswith("#"):
            continue
        add(target, "links_to", anchor, "markdown_link")

    frontmatter = parse_frontmatter(markdown_text)
    field_edge_types = {
        "source": "derived_from",
        "sources": "derived_from",
        "project": "belongs_to_project",
        "projects": "belongs_to_project",
        "related": "related_to",
        "see_also": "related_to",
        "supports": "supports",
        "contradicts": "contradicts",
        "questions": "questions",
        "topic": "mentions",
    }
    fields = set(graph_fields) | set(field_edge_types)
    for field in fields:
        for value in _field_values(frontmatter.get(field)):
            edge_type = field_edge_types.get(field, "mentions")
            add(value, edge_type, field, f"frontmatter:{field}", 0.9)
    return links
