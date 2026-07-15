from __future__ import annotations

from vaultq.operations import operation_by_name, summarize_params, validate_params


def test_operation_schema_validates_required_params() -> None:
    op = operation_by_name("vaultq_capture")

    assert validate_params(op, {}) == "Missing required parameter: text"
    assert validate_params(op, {"text": "x", "limit": 5}) is not None
    assert validate_params(op, {"text": "x", "kind": "idea", "domain": "ai"}) is None


def test_param_summary_reports_shape_without_values() -> None:
    summary = summarize_params("vaultq_capture", {"text": "private note", "kind": "idea", "weird_secret": "abc"})

    assert summary["redacted"] is True
    assert summary["declared_keys"] == ["kind", "text"]
    assert summary["unknown_key_count"] == 1
    assert "private note" not in str(summary)


def test_agent_maintain_operation_contract_is_declared() -> None:
    op = operation_by_name("vaultq_agent_maintain")

    assert op.mutating is True
    assert validate_params(op, {"dry_run": True, "max_actions": 10, "collection_name": "second_brain"}) is None


def test_chunk_stats_operation_contract_is_declared() -> None:
    op = operation_by_name("vaultq_chunk_stats")

    assert op.scope == "read"
    assert op.mutating is False
    assert validate_params(op, {"collection_name": "second_brain", "sample_limit": 5}) is None
    assert validate_params(op, {"sample_limit": "five"}) == 'Parameter "sample_limit" must be a number'


def test_background_status_operation_contract_is_declared() -> None:
    op = operation_by_name("vaultq_background_status")

    assert op.scope == "read"
    assert op.mutating is False
    assert validate_params(op, {"collection_name": "second_brain"}) is None


def test_related_work_operation_contract_is_declared() -> None:
    op = operation_by_name("vaultq_related_work")

    assert op.scope == "read"
    assert op.mutating is False
    assert validate_params(op, {"idea": "agent revenue loop", "limit": 5}) is None
    assert validate_params(op, {"idea": "x", "limit": "five"}) == 'Parameter "limit" must be a number'


def test_semantic_clusters_operation_contract_is_declared() -> None:
    op = operation_by_name("vaultq_semantic_clusters")

    assert op.scope == "read"
    assert op.mutating is False
    assert validate_params(
        op,
        {
            "collection_name": "second_brain",
            "seed_prefixes": ["14_Agent_Workspace/Ideas"],
            "seed_limit": 10,
            "related_limit": 4,
            "min_score": 0.35,
            "session_policy": "penalize",
            "session_max_per_seed": 1,
        },
    ) is None
    assert validate_params(op, {"session_policy": "flood"}) == 'Parameter "session_policy" must be one of: exclude, penalize, include'
