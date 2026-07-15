from __future__ import annotations

import time
from typing import Any, Dict, List

import httpx

from vaultq.embedding_provider import resolve_embedding_provider, resolve_rerank_provider
from vaultq.store import connect, get_settings


def _ok(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": True, **payload}


def _fail(exc: Exception) -> Dict[str, Any]:
    return {"ok": False, "error": str(exc)}


def _provider_variant_probe(query: str, api_key: str, current_provider: str) -> Dict[str, Any]:
    if not api_key:
        return {"ok": None, "skipped": True, "reason": "no API key configured"}

    candidates = [
        {
            "provider": "atlas",
            "base_url": "https://ai.mongodb.com/v1",
            "contextual_payload": {
                "model": "voyage-context-3",
                "input_type": "query",
                "inputs": [[query]],
                "output_dimension": 1024,
            },
            "rerank_payload": {
                "model": "rerank-2.5",
                "query": query,
                "documents": ["run migrations and deploy", "buy groceries"],
                "top_k": 2,
                "return_documents": False,
                "truncation": True,
            },
        },
        {
            "provider": "voyage",
            "base_url": "https://api.voyageai.com/v1",
            "contextual_payload": {
                "model": "voyage-context-3",
                "input_type": "query",
                "inputs": [[query]],
                "output_dimension": 1024,
            },
            "rerank_payload": {
                "model": "rerank-2.5",
                "query": query,
                "documents": ["run migrations and deploy", "buy groceries"],
            },
        },
    ]

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    results: List[Dict[str, Any]] = []
    with httpx.Client(timeout=20.0) as client:
        for candidate in candidates:
            if candidate["provider"] == current_provider:
                continue
            contextual_resp = client.post(
                f"{candidate['base_url']}/contextualizedembeddings",
                headers=headers,
                json=candidate["contextual_payload"],
            )
            rerank_resp = client.post(
                f"{candidate['base_url']}/rerank",
                headers=headers,
                json=candidate["rerank_payload"],
            )
            ok = contextual_resp.status_code == 200 and rerank_resp.status_code == 200
            results.append(
                {
                    "provider": candidate["provider"],
                    "base_url": candidate["base_url"],
                    "contextual_status": contextual_resp.status_code,
                    "rerank_status": rerank_resp.status_code,
                    "ok": ok,
                }
            )
    recommended = next((row for row in results if row["ok"]), None)
    return {
        "ok": bool(recommended),
        "recommended_provider": (recommended or {}).get("provider"),
        "recommended_base_url": (recommended or {}).get("base_url"),
        "results": results,
    }


def run_doctor(*, live: bool = False, query: str = "deployment checklist") -> Dict[str, Any]:
    settings = get_settings()
    embed_provider = resolve_embedding_provider()
    rerank_provider = resolve_rerank_provider()

    report: Dict[str, Any] = {
        "config": {
            "embedding_provider": embed_provider.provider,
            "embedding_base_url": embed_provider.base_url,
            "embedding_model": embed_provider.model,
            "embedding_dim": settings.embedding_dim,
            "contextual_embeddings": embed_provider.model.lower().startswith("voyage-context"),
            "rerank_base_url": rerank_provider.base_url,
            "reranker_model": rerank_provider.model,
            "has_embedding_api_key": bool(embed_provider.api_key),
            "has_rerank_api_key": bool(rerank_provider.api_key),
            "qdrant_url": settings.qdrant_url,
            "qdrant_collection": settings.qdrant_collection,
        }
    }

    try:
        started = time.perf_counter()
        with connect(settings) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 AS ok")
                row = cur.fetchone()
        report["postgres"] = _ok(
            {
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                "result": row["ok"],
            }
        )
    except Exception as exc:
        report["postgres"] = _fail(exc)

    try:
        started = time.perf_counter()
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{settings.qdrant_url}/collections")
            resp.raise_for_status()
            collections: List[Dict[str, Any]] = resp.json().get("result", {}).get("collections", [])
        report["qdrant"] = _ok(
            {
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                "collections": [row.get("name") for row in collections],
            }
        )
    except Exception as exc:
        report["qdrant"] = _fail(exc)

    if not live:
        report["providers"] = {"ok": None, "skipped": True, "reason": "live checks disabled"}
        return report

    provider_checks: Dict[str, Any] = {}
    try:
        from vaultq.retrieval_models import (
            encode_hybrid_document_groups,
            encode_hybrid_query,
            rerank_documents,
        )

        started = time.perf_counter()
        query_vectors = encode_hybrid_query(query)
        provider_checks["query_embedding"] = _ok(
            {
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                "dense_dim": len(query_vectors.get("dense_vector") or []),
                "has_sparse_payload": bool(query_vectors.get("sparse_vector")),
            }
        )

        started = time.perf_counter()
        grouped = encode_hybrid_document_groups(
            [
                [
                    "Deployment requires migrations, smoke tests, and a staged rollout.",
                    "Rollback should be prepared before the final production cutover.",
                ]
            ]
        )
        first_vector = (((grouped or [[]])[0] or [{}])[0] or {}).get("dense_vector") or []
        provider_checks["document_group_embedding"] = _ok(
            {
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                "group_count": len(grouped),
                "dense_dim": len(first_vector),
            }
        )

        started = time.perf_counter()
        reranked = rerank_documents(
            query,
            [
                "Run migrations, perform smoke tests, then deploy the service.",
                "Buy milk, eggs, and bread on the way home.",
            ],
            top_n=2,
        )
        provider_checks["rerank"] = _ok(
            {
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                "results": reranked,
            }
        )
        provider_checks["ok"] = True
    except Exception as exc:
        provider_checks["ok"] = False
        provider_checks["error"] = str(exc)
        if "forbidden" in str(exc).lower() or "403" in str(exc):
            provider_checks["provider_probe"] = _provider_variant_probe(query, embed_provider.api_key, embed_provider.provider)

    report["providers"] = provider_checks
    return report
