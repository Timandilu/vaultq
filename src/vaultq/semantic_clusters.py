from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

from vaultq.search import search
from vaultq.session_rerank import SESSION_PREFIX, adaptive_session_rerank, clean_path as _clean_path
from vaultq.store import connect


AGENT_WORKSPACE_PREFIX = "14_Agent_Workspace/"
IMPORTED_MARKERS = (
    "import",
    "imported",
    "blog",
    "blogs",
    "article",
    "articles",
    "library",
    "libraries",
    "readwise",
    "clipping",
    "clippings",
    "source",
    "sources",
    "resource",
    "resources",
    "text",
    "texts",
)
def source_class(rel_path: str) -> str:
    normalized = _clean_path(rel_path)
    if normalized.startswith(SESSION_PREFIX):
        return "agent_session"
    if normalized.startswith(AGENT_WORKSPACE_PREFIX):
        return "agent_workspace"
    parts = {part.lower() for part in normalized.replace("-", "_").replace(" ", "_").split("/")}
    flat = normalized.lower()
    if any(marker in parts or marker in flat for marker in IMPORTED_MARKERS):
        return "imported_text"
    return "vault_note"


def _matches_prefix(rel_path: str, prefixes: Optional[Sequence[str]]) -> bool:
    if not prefixes:
        return True
    normalized = _clean_path(rel_path)
    return any(normalized.startswith(_clean_path(prefix)) for prefix in prefixes if str(prefix or "").strip())


def _seed_profile(seed: Dict[str, Any], *, max_chars: int = 1600) -> str:
    text = str(seed.get("text") or seed.get("markdown_text") or "").strip()
    pieces = [
        str(seed.get("title") or "").strip(),
        f"type: {seed.get('node_type') or seed.get('type') or ''}".strip(),
        f"domain: {seed.get('domain') or ''}".strip(),
        text[:max_chars],
    ]
    return "\n\n".join(piece for piece in pieces if piece and piece != "type:" and piece != "domain:")


def _seed_contract(seed: Dict[str, Any]) -> Dict[str, Any]:
    rel_path = _clean_path(seed.get("rel_path") or seed.get("path"))
    return {
        "rel_path": rel_path,
        "title": str(seed.get("title") or rel_path),
        "node_type": str(seed.get("node_type") or seed.get("type") or ""),
        "domain": str(seed.get("domain") or ""),
        "source_class": source_class(rel_path),
    }


def _related_contract(
    row: Dict[str, Any],
    *,
    effective_score: float,
    session_policy: str,
    seed_rel_path: str,
    session_rerank: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    evidence = dict(row.get("evidence") or {})
    rel_path = _clean_path(row.get("rel_path") or evidence.get("rel_path"))
    contract = {
        "id": str(row.get("id") or ""),
        "rel_path": rel_path,
        "title": str(row.get("title") or rel_path),
        "score": round(float(row.get("score") or 0.0), 6),
        "effective_score": round(effective_score, 6),
        "source_class": source_class(rel_path),
        "session_policy": session_policy if source_class(rel_path) == "agent_session" else None,
        "connection_count": 1,
        "seed_paths": [seed_rel_path],
        "evidence": evidence,
        "text": str(row.get("text") or "")[:1200],
    }
    if session_rerank and contract["source_class"] == "agent_session":
        contract["session_rerank"] = session_rerank
    return contract


def _merge_related(existing: Dict[str, Any], incoming: Dict[str, Any]) -> None:
    existing["score"] = max(float(existing.get("score") or 0.0), float(incoming.get("score") or 0.0))
    existing["effective_score"] = max(
        float(existing.get("effective_score") or 0.0),
        float(incoming.get("effective_score") or 0.0),
    )
    existing["connection_count"] = int(existing.get("connection_count") or 0) + 1
    seed_paths = list(existing.get("seed_paths") or [])
    for seed_path in incoming.get("seed_paths") or []:
        if seed_path not in seed_paths:
            seed_paths.append(seed_path)
    existing["seed_paths"] = sorted(seed_paths)
    if incoming.get("session_rerank"):
        current = existing.get("session_rerank") or {}
        if float(incoming.get("effective_score") or 0.0) >= float(existing.get("effective_score") or 0.0):
            existing["session_rerank"] = incoming["session_rerank"]


def _adaptive_session_rerank(row: Dict[str, Any], *, query: str, session_penalty: float) -> Dict[str, Any]:
    evidence = dict(row.get("evidence") or {})
    rel_path = _clean_path(row.get("rel_path") or evidence.get("rel_path"))
    title = str(row.get("title") or rel_path)
    text = str(row.get("text") or "")
    return adaptive_session_rerank(
        query=query,
        rel_path=rel_path,
        title=title,
        text=text,
        evidence=evidence,
        session_penalty=session_penalty,
    )


def _session_quota(related_limit: int, session_max_per_seed: Optional[int]) -> int:
    if session_max_per_seed is not None:
        return max(0, min(related_limit, int(session_max_per_seed)))
    return max(1, min(2, related_limit // 3 or 1))


def _apply_penalized_session_quota(
    rows: Sequence[Dict[str, Any]],
    *,
    related_limit: int,
    session_max_per_seed: Optional[int],
) -> list[Dict[str, Any]]:
    quota = _session_quota(related_limit, session_max_per_seed)
    selected: list[Dict[str, Any]] = []
    session_count = 0
    for row in rows:
        if row.get("source_class") == "agent_session":
            if session_count >= quota:
                continue
            session_count += 1
        selected.append(row)
        if len(selected) >= related_limit:
            break
    return selected


def _candidate_rows_for_seed(
    seed: Dict[str, Any],
    *,
    related_limit: int,
    min_score: float,
    session_policy: str,
    session_penalty: float,
    session_max_per_seed: Optional[int],
    candidate_path_prefixes: Optional[Sequence[str]],
    exclude_path_prefixes: Optional[Sequence[str]],
) -> list[Dict[str, Any]]:
    seed_rel_path = _clean_path(seed.get("rel_path") or seed.get("path"))
    query = _seed_profile(seed)
    if not query:
        return []
    retrieval = search(query, limit=max(related_limit * 4, related_limit, 8), retrieval_mode="semantic")
    rows: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in retrieval.get("results", []):
        evidence = dict(row.get("evidence") or {})
        rel_path = _clean_path(row.get("rel_path") or evidence.get("rel_path"))
        if not rel_path or rel_path == seed_rel_path or rel_path in seen:
            continue
        if not _matches_prefix(rel_path, candidate_path_prefixes):
            continue
        if exclude_path_prefixes and _matches_prefix(rel_path, exclude_path_prefixes):
            continue
        row_source_class = source_class(rel_path)
        if row_source_class == "agent_session" and session_policy == "exclude":
            continue
        score = float(row.get("score") or 0.0)
        effective_score = score
        session_rerank = None
        if row_source_class == "agent_session" and session_policy == "penalize":
            session_rerank = _adaptive_session_rerank(row, query=query, session_penalty=session_penalty)
            effective_score *= float(session_rerank["factor"])
        if effective_score < min_score:
            continue
        seen.add(rel_path)
        rows.append(
            _related_contract(
                row,
                effective_score=effective_score,
                session_policy=session_policy,
                seed_rel_path=seed_rel_path,
                session_rerank=session_rerank,
            )
        )
    ordered = sorted(rows, key=lambda item: (-float(item.get("effective_score") or 0.0), item["rel_path"]))
    if session_policy == "penalize":
        return _apply_penalized_session_quota(
            ordered,
            related_limit=related_limit,
            session_max_per_seed=session_max_per_seed,
        )
    return ordered[:related_limit]


def semantic_cluster_seed_nodes(
    seeds: Sequence[Dict[str, Any]],
    *,
    related_limit: int = 5,
    min_score: float = 0.35,
    session_policy: str = "exclude",
    session_penalty: float = 0.65,
    session_max_per_seed: Optional[int] = 1,
    candidate_path_prefixes: Optional[Sequence[str]] = None,
    exclude_path_prefixes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    related_limit = max(1, min(25, int(related_limit or 5)))
    min_score = max(0.0, float(min_score or 0.0))
    session_policy = (session_policy or "exclude").strip().lower()
    if session_policy not in {"exclude", "penalize", "include"}:
        raise ValueError("session_policy must be one of: exclude, penalize, include")
    session_penalty = max(0.0, min(1.0, float(session_penalty or 0.65)))

    clusters: dict[str, Dict[str, Any]] = {}
    ordered_seeds = sorted((_seed_contract(seed) | {"text": seed.get("text") or seed.get("markdown_text") or ""} for seed in seeds), key=lambda item: item["rel_path"])
    for seed in ordered_seeds:
        related = _candidate_rows_for_seed(
            seed,
            related_limit=related_limit,
            min_score=min_score,
            session_policy=session_policy,
            session_penalty=session_penalty,
            session_max_per_seed=session_max_per_seed,
            candidate_path_prefixes=candidate_path_prefixes,
            exclude_path_prefixes=exclude_path_prefixes,
        )
        anchor = related[0] if related else {
            "rel_path": seed["rel_path"],
            "title": seed["title"],
            "source_class": seed["source_class"],
            "effective_score": 0.0,
        }
        key = str(anchor["rel_path"])
        cluster = clusters.setdefault(
            key,
            {
                "label": str(anchor.get("title") or key),
                "anchor": {
                    "rel_path": str(anchor["rel_path"]),
                    "title": str(anchor.get("title") or anchor["rel_path"]),
                    "source_class": str(anchor.get("source_class") or source_class(str(anchor["rel_path"]))),
                },
                "seed_nodes": {},
                "related_nodes": {},
            },
        )
        cluster["seed_nodes"][seed["rel_path"]] = {key: value for key, value in seed.items() if key != "text"}
        for row in related:
            existing = cluster["related_nodes"].get(row["rel_path"])
            if existing:
                _merge_related(existing, row)
            else:
                cluster["related_nodes"][row["rel_path"]] = row

    cluster_rows = []
    for cluster in clusters.values():
        seed_nodes = sorted(cluster["seed_nodes"].values(), key=lambda item: item["rel_path"])
        related_nodes = sorted(
            cluster["related_nodes"].values(),
            key=lambda item: (-float(item.get("effective_score") or 0.0), -int(item.get("connection_count") or 0), item["rel_path"]),
        )
        cluster_rows.append(
            {
                "label": cluster["label"],
                "anchor": cluster["anchor"],
                "count": len(seed_nodes),
                "seed_nodes": seed_nodes,
                "related_nodes": related_nodes[:related_limit],
            }
        )
    cluster_rows.sort(
        key=lambda item: (
            -int(item["count"]),
            -max((float(node.get("effective_score") or 0.0) for node in item["related_nodes"]), default=0.0),
            item["anchor"]["rel_path"],
        )
    )
    return {
        "ok": True,
        "cluster_algorithm": "semantic_seed_search",
        "seed_count": len(ordered_seeds),
        "cluster_count": len(cluster_rows),
        "session_policy": session_policy,
        "session_penalty": session_penalty if session_policy == "penalize" else None,
        "session_rerank_mode": "adaptive" if session_policy == "penalize" else None,
        "session_max_per_seed": _session_quota(related_limit, session_max_per_seed) if session_policy == "penalize" else None,
        "clusters": cluster_rows,
    }


def indexed_seed_nodes(
    *,
    collection_name: str | None = None,
    seed_path_prefixes: Optional[Sequence[str]] = None,
    seed_limit: int = 25,
) -> list[Dict[str, Any]]:
    seed_limit = max(1, min(200, int(seed_limit or 25)))
    clauses = ["d.status IN ('indexed', 'enriched')"]
    params: list[Any] = []
    if collection_name:
        clauses.append("col.name = %s")
        params.append(collection_name)
    if seed_path_prefixes:
        prefix_clauses = []
        for prefix in seed_path_prefixes:
            clean = _clean_path(prefix)
            if not clean:
                continue
            prefix_clauses.append("d.rel_path LIKE %s")
            params.append(clean.rstrip("/") + "/%")
        if prefix_clauses:
            clauses.append("(" + " OR ".join(prefix_clauses) + ")")
    params.append(seed_limit)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    col.name AS collection_name,
                    d.rel_path,
                    d.title,
                    d.markdown_text,
                    d.frontmatter_json
                FROM vq_documents d
                JOIN vq_collections col ON col.id = d.collection_id
                WHERE {" AND ".join(clauses)}
                ORDER BY d.updated_at DESC, d.rel_path ASC
                LIMIT %s
                """,
                params,
            )
            rows = list(cur.fetchall())
    seeds = []
    for row in rows:
        frontmatter = row.get("frontmatter_json") or {}
        seeds.append(
            {
                "collection_name": row.get("collection_name") or "",
                "rel_path": row.get("rel_path") or "",
                "title": row.get("title") or "",
                "node_type": frontmatter.get("type") or "",
                "domain": frontmatter.get("domain") or "",
                "text": row.get("markdown_text") or "",
            }
        )
    return seeds


def semantic_node_clusters(
    *,
    collection_name: str | None = None,
    seed_path_prefixes: Optional[Sequence[str]] = None,
    seed_limit: int = 25,
    related_limit: int = 5,
    min_score: float = 0.35,
    session_policy: str = "exclude",
    session_penalty: float = 0.65,
    session_max_per_seed: Optional[int] = 1,
    candidate_path_prefixes: Optional[Sequence[str]] = None,
    exclude_path_prefixes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    seeds = indexed_seed_nodes(
        collection_name=collection_name,
        seed_path_prefixes=seed_path_prefixes,
        seed_limit=seed_limit,
    )
    result = semantic_cluster_seed_nodes(
        seeds,
        related_limit=related_limit,
        min_score=min_score,
        session_policy=session_policy,
        session_penalty=session_penalty,
        session_max_per_seed=session_max_per_seed,
        candidate_path_prefixes=candidate_path_prefixes,
        exclude_path_prefixes=exclude_path_prefixes,
    )
    result["collection_name"] = collection_name
    result["seed_path_prefixes"] = list(seed_path_prefixes or [])
    return result
