from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class ParamDef:
    type: str
    required: bool = False
    description: str = ""
    default: Any = None
    enum: Optional[tuple[str, ...]] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"type": self.type}
        if self.required:
            data["required"] = True
        if self.description:
            data["description"] = self.description
        if self.default is not None:
            data["default"] = self.default
        if self.enum:
            data["enum"] = list(self.enum)
        return data


@dataclass(frozen=True)
class Operation:
    name: str
    scope: str
    description: str
    params: Dict[str, ParamDef]
    mutating: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "scope": self.scope,
            "description": self.description,
            "params": {key: value.to_dict() for key, value in self.params.items()},
            "mutating": self.mutating,
        }


def _p(type_name: str, *, required: bool = False, description: str = "", default: Any = None, enum: Optional[Iterable[str]] = None) -> ParamDef:
    return ParamDef(type_name, required, description, default, tuple(enum) if enum else None)


OPERATIONS = (
    Operation("vaultq_status", "read", "Return current VaultQ store and write-policy status.", {}),
    Operation("vaultq_collection_list", "read", "List configured collections and contexts.", {}),
    Operation("vaultq_operation_list", "read", "List registered VaultQ read and write operations.", {}),
    Operation(
        "vaultq_chunk_stats",
        "read",
        "Return chunk token/character length distribution and health assessment.",
        {"collection_name": _p("string"), "sample_limit": _p("number", default=12)},
    ),
    Operation(
        "vaultq_background_status",
        "read",
        "Return DB-backed status for the MCP-tied background indexing worker.",
        {"collection_name": _p("string")},
    ),
    Operation(
        "vaultq_search",
        "read",
        "Run keyword-only retrieval.",
        {"query": _p("string", required=True), "limit": _p("number", default=5)},
    ),
    Operation(
        "vaultq_query",
        "read",
        "Run query-first retrieval; focused mode returns concise, source-diverse related-work evidence.",
        {"query": _p("string", required=True), "limit": _p("number", default=5), "mode": _p("string", enum=("fast", "focused", "balanced", "deep", "hybrid"))},
    ),
    Operation(
        "vaultq_related_work",
        "read",
        "Surface source-diverse existing notes that may connect to an idea before creating a new note.",
        {"idea": _p("string", required=True), "limit": _p("number", default=6), "mode": _p("string", default="focused", enum=("fast", "focused", "balanced", "deep", "hybrid"))},
    ),
    Operation(
        "vaultq_semantic_clusters",
        "read",
        "Cluster any seed node set by semantic anchors in the indexed corpus.",
        {
            "collection_name": _p("string"),
            "seed_prefixes": _p("array"),
            "candidate_prefixes": _p("array"),
            "exclude_prefixes": _p("array"),
            "seed_limit": _p("number", default=25),
            "related_limit": _p("number", default=5),
            "min_score": _p("number", default=0.35),
            "session_policy": _p("string", default="exclude", enum=("exclude", "penalize", "include")),
            "session_penalty": _p("number", default=0.65),
            "session_max_per_seed": _p("number", default=1),
        },
    ),
    Operation(
        "vaultq_get_doc",
        "read",
        "Fetch a source document or citation target.",
        {"identifier": _p("string", required=True), "full": _p("boolean", default=False)},
    ),
    Operation("vaultq_write_status", "write", "Show write policy and AI workspace status.", {"collection_name": _p("string")}),
    Operation(
        "vaultq_capture",
        "write",
        "Create an agent-authored note under the AI workspace.",
        {
            "text": _p("string", required=True),
            "kind": _p("string", default="note", enum=("idea", "note", "question", "finding", "report", "draft")),
            "domain": _p("string", default="general"),
            "title": _p("string"),
            "collection_name": _p("string"),
        },
        mutating=True,
    ),
    Operation(
        "vaultq_note_put",
        "write",
        "Create or replace a markdown note under an allowed path.",
        {"rel_path": _p("string", required=True), "markdown": _p("string", required=True), "collection_name": _p("string")},
        mutating=True,
    ),
    Operation(
        "vaultq_note_append",
        "write",
        "Append text to an existing allowed note.",
        {"rel_path": _p("string", required=True), "text": _p("string", required=True), "heading": _p("string"), "collection_name": _p("string")},
        mutating=True,
    ),
    Operation(
        "vaultq_note_propose",
        "write",
        "Write a proposal note under the AI workspace proposals folder.",
        {"title": _p("string", required=True), "body": _p("string", required=True), "collection_name": _p("string")},
        mutating=True,
    ),
    Operation(
        "vaultq_property_propose",
        "write",
        "Propose a property or tag rule change.",
        {"name": _p("string", required=True), "reason": _p("string", required=True), "kind": _p("string", default="property", enum=("property", "tag")), "collection_name": _p("string")},
        mutating=True,
    ),
    Operation(
        "vaultq_graph_neighbors",
        "read",
        "List outgoing graph links for a note.",
        {"identifier": _p("string", required=True), "limit": _p("number", default=50), "collection_name": _p("string")},
    ),
    Operation(
        "vaultq_graph_extract",
        "index",
        "Extract graph links from configured markdown collections into the local VaultQ index.",
        {"collection_name": _p("string")},
        mutating=True,
    ),
    Operation(
        "vaultq_graph_backlinks",
        "read",
        "List backlinks for a note or identifier.",
        {"identifier": _p("string", required=True), "limit": _p("number", default=50), "collection_name": _p("string")},
    ),
    Operation(
        "vaultq_graph_traverse",
        "read",
        "Traverse graph edges with depth, type, and direction filters.",
        {
            "identifier": _p("string", required=True),
            "depth": _p("number", default=2),
            "direction": _p("string", default="out", enum=("out", "in", "both")),
            "edge_type": _p("string"),
            "limit": _p("number", default=200),
            "collection_name": _p("string"),
        },
    ),
    Operation(
        "vaultq_think",
        "read",
        "Synthesize a lightweight cited answer from retrieval results.",
        {
            "question": _p("string", required=True),
            "limit": _p("number", default=5),
            "save_report": _p("boolean", default=False),
            "collection_name": _p("string"),
        },
    ),
    Operation(
        "vaultq_maintain",
        "write",
        "Inspect the AI workspace and optionally write a maintenance report.",
        {"collection_name": _p("string"), "write_report": _p("boolean", default=False)},
        mutating=True,
    ),
    Operation(
        "vaultq_agent_prepare",
        "write",
        "Prepare a disabled, reversible review/promote background agent runtime.",
        {"collection_name": _p("string"), "write_sheet": _p("boolean", default=True)},
        mutating=True,
    ),
    Operation(
        "vaultq_agent_maintain",
        "write",
        "Run one autonomous, reversible self-maintenance pass inside the agent workspace.",
        {
            "collection_name": _p("string"),
            "dry_run": _p("boolean", default=False),
            "max_actions": _p("number", default=20),
        },
        mutating=True,
    ),
)


def list_operations() -> list[Dict[str, Any]]:
    return [operation.to_dict() for operation in OPERATIONS]


def operation_by_name(name: str) -> Operation:
    for operation in OPERATIONS:
        if operation.name == name:
            return operation
    raise KeyError(f"Unknown operation: {name}")


def _matches_type(value: Any, type_name: str) -> bool:
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    return True


def validate_params(operation: Operation, params: Dict[str, Any]) -> Optional[str]:
    declared = operation.params
    for key in params:
        if key not in declared:
            return f"Unknown parameter: {key}"
    for key, definition in declared.items():
        value = params.get(key)
        if definition.required and value is None:
            return f"Missing required parameter: {key}"
        if value is None:
            continue
        if not _matches_type(value, definition.type):
            return f'Parameter "{key}" must be a {definition.type}'
        if definition.enum and value not in definition.enum:
            return f'Parameter "{key}" must be one of: {", ".join(definition.enum)}'
    return None


def summarize_params(operation_name: str, params: Any) -> Optional[Dict[str, Any]]:
    if params is None:
        return None
    approx_bytes: Optional[int]
    try:
        raw_len = len(str(params).encode("utf-8"))
        approx_bytes = ((raw_len + 1023) // 1024) * 1024
    except Exception:
        approx_bytes = None
    if isinstance(params, list):
        return {"redacted": True, "kind": "array", "length": len(params), "approx_bytes": approx_bytes}
    if isinstance(params, dict):
        try:
            operation = operation_by_name(operation_name)
            allowed = set(operation.params)
        except KeyError:
            allowed = set()
        declared = sorted(key for key in params if key in allowed)
        unknown = len([key for key in params if key not in allowed])
        return {
            "redacted": True,
            "kind": "object",
            "declared_keys": declared,
            "unknown_key_count": unknown,
            "approx_bytes": approx_bytes,
        }
    return {"redacted": True, "kind": type(params).__name__, "approx_bytes": approx_bytes}
