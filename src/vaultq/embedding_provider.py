from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

try:
    from dotenv import load_dotenv as _load_dotenv_impl
except Exception:
    _load_dotenv_impl = None

_PROVIDER_DEFAULTS = {
    "voyage": {
        "base_url": "https://api.voyageai.com/v1",
        "embed_model": "voyage-context-3",
        "rerank_model": "rerank-2.5",
    },
    "atlas": {
        "base_url": "https://ai.mongodb.com/v1",
        "embed_model": "voyage-context-3",
        "rerank_model": "rerank-2.5",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "embed_model": "text-embedding-3-large",
        "rerank_model": "",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "embed_model": "openai/text-embedding-3-large",
        "rerank_model": "",
    },
    "custom": {
        "base_url": "https://api.openai.com/v1",
        "embed_model": "text-embedding-3-large",
        "rerank_model": "",
    },
}

_EMBED_ENV_KEYS = {
    "ATLAS_API_KEY",
    "EMBED_API_KEY",
    "EMBED_APP_NAME",
    "EMBED_BASE_URL",
    "EMBED_MODEL",
    "EMBED_PROVIDER",
    "EMBED_SITE_URL",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "ENABLE_CONTEXTUAL_EMBEDDINGS",
    "ENABLE_RERANKER",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_APP_NAME",
    "OPENROUTER_SITE_URL",
    "RERANK_API_KEY",
    "RERANK_BASE_URL",
    "RERANK_CANDIDATES",
    "RERANKER_MODEL",
    "VOYAGE_API_KEY",
}


@dataclass(frozen=True)
class ApiProviderConfig:
    provider: str
    base_url: str
    api_key: str
    model: str
    headers: Dict[str, str]


def _simple_load_dotenv(path: Path, override: bool = False) -> bool:
    if not path.exists():
        return False
    loaded = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = os.path.expandvars(value)
            loaded = True
    return loaded


def load_env_file(path: Path, override: bool = False) -> bool:
    if _load_dotenv_impl is not None:
        return bool(_load_dotenv_impl(dotenv_path=path, override=override))
    return _simple_load_dotenv(path, override=override)


def _selected_load_dotenv(path: Path, allowed_keys: set[str], override: bool = False) -> bool:
    if not path.exists():
        return False
    loaded = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in allowed_keys:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if override or not _clean(os.getenv(key)):
            os.environ[key] = os.path.expandvars(value)
            loaded = True
    return loaded


def _embedding_env_fallbacks(root: Path) -> list[Path]:
    candidates: list[Path] = []
    explicit = (os.getenv("VQ_EMBED_ENV_FILE") or "").strip()
    if explicit:
        candidates.append(Path(explicit).expanduser())
    parent = root.parent
    candidates.extend(
        [
            parent / "copyvector_mcp" / "copyvector_embed_pipeline" / ".env",
            parent / "copyvector_mcp" / "mcp_runtime" / ".env",
            parent / "markvis_ecom_mcp" / ".env",
            parent / ".ai-embedding.env",
        ]
    )
    return list(dict.fromkeys(path.resolve() for path in candidates if path))


def load_env_layers(project_root: Optional[Path] = None) -> None:
    root = Path(project_root or Path(__file__).resolve().parents[2])
    repo_env = root / ".env"
    repo_env_local = root / ".env.local"
    explicit = (os.getenv("VQ_ENV_FILE") or "").strip()

    if repo_env.exists():
        load_env_file(repo_env, override=False)
    if repo_env_local.exists():
        load_env_file(repo_env_local, override=True)
    if explicit:
        explicit_path = Path(explicit).expanduser().resolve()
        if explicit_path.exists():
            load_env_file(explicit_path, override=True)
    for fallback_path in _embedding_env_fallbacks(root):
        if fallback_path.exists():
            _selected_load_dotenv(fallback_path, _EMBED_ENV_KEYS, override=False)


def _clean(value: Optional[str]) -> str:
    return (value or "").strip().strip('"').strip("'")


def _first_env(*names: str) -> str:
    for name in names:
        if not name:
            continue
        value = _clean(os.getenv(name))
        if value:
            return value
    return ""


def _detect_provider_from_base(base_url: str) -> str:
    lowered = (base_url or "").lower()
    if "api.voyageai.com" in lowered:
        return "voyage"
    if "ai.mongodb.com" in lowered:
        return "atlas"
    if "openrouter.ai" in lowered:
        return "openrouter"
    if "api.openai.com" in lowered:
        return "openai"
    return "custom"


def _normalize_model_for_base(model: str, base_url: str) -> str:
    model = _clean(model)
    if not model:
        return ""
    if "/" in model and "openrouter.ai" not in (base_url or "").lower():
        _, normalized = model.split("/", 1)
        if normalized:
            return normalized
    return model


def _api_key_for_provider(base_url: str, *preferred_env_names: str) -> str:
    for env_name in preferred_env_names:
        value = _clean(os.getenv(env_name))
        if value:
            return value
    generic = _clean(os.getenv("EMBED_API_KEY"))
    if generic:
        return generic
    lowered = (base_url or "").lower()
    if "api.voyageai.com" in lowered:
        return _first_env("VOYAGE_API_KEY", "EMBED_API_KEY")
    if "ai.mongodb.com" in lowered:
        return (
            _first_env(
                "VOYAGE_API_KEY",
                "ATLAS_API_KEY",
                "MONGODB_ATLAS_API_KEY",
                "OPENAI_API_KEY",
                "OPENROUTER_API_KEY",
            )
        )
    if "openrouter.ai" in lowered:
        return _first_env("OPENROUTER_API_KEY", "OPENAI_API_KEY")
    return _first_env("OPENAI_API_KEY", "OPENROUTER_API_KEY")


def _provider_headers(
    base_url: str,
    *,
    api_key: str,
    default_app_name: str,
    site_env_name: str,
    app_env_name: str,
) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"
    if "openrouter.ai" in (base_url or "").lower():
        site_url = _first_env(site_env_name, "OPENROUTER_SITE_URL") or "http://localhost"
        app_name = _first_env(app_env_name, "OPENROUTER_APP_NAME") or default_app_name
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name
    return headers


def resolve_embedding_provider(default_app_name: str = "VaultQ") -> ApiProviderConfig:
    base_url = (
        _first_env("EMBED_BASE_URL", "OPENAI_BASE_URL")
        or _PROVIDER_DEFAULTS["atlas"]["base_url"]
    ).rstrip("/")
    provider = _clean(os.getenv("EMBED_PROVIDER")).lower() or _detect_provider_from_base(base_url)
    model = _first_env("EMBED_MODEL", "EMBEDDING_MODEL") or _PROVIDER_DEFAULTS.get(provider, _PROVIDER_DEFAULTS["custom"])["embed_model"]
    model = _normalize_model_for_base(model, base_url)
    api_key = _api_key_for_provider(base_url, "EMBED_API_KEY", "VOYAGE_API_KEY")
    headers = _provider_headers(
        base_url,
        api_key=api_key,
        default_app_name=default_app_name,
        site_env_name="EMBED_SITE_URL",
        app_env_name="EMBED_APP_NAME",
    )
    return ApiProviderConfig(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=model,
        headers=headers,
    )


def resolve_rerank_provider(default_app_name: str = "VaultQ Rerank") -> ApiProviderConfig:
    embed_provider = resolve_embedding_provider(default_app_name="VaultQ")
    base_url = (_first_env("RERANK_BASE_URL") or embed_provider.base_url).rstrip("/")
    provider = _detect_provider_from_base(base_url)
    default_model = _PROVIDER_DEFAULTS.get(provider, _PROVIDER_DEFAULTS["custom"])["rerank_model"]
    model = _first_env("RERANKER_MODEL") or default_model
    model = _normalize_model_for_base(model, base_url)
    api_key = _api_key_for_provider(base_url, "RERANK_API_KEY", "EMBED_API_KEY", "VOYAGE_API_KEY")
    headers = _provider_headers(
        base_url,
        api_key=api_key,
        default_app_name=default_app_name,
        site_env_name="RERANK_SITE_URL",
        app_env_name="RERANK_APP_NAME",
    )
    return ApiProviderConfig(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=model,
        headers=headers,
    )


def resolve_llm_provider(default_app_name: str = "VaultQ Enrichment") -> ApiProviderConfig:
    base_url = (_first_env("LLM_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
    provider = _clean(os.getenv("LLM_PROVIDER")).lower() or _detect_provider_from_base(base_url)
    model = _first_env("LLM_MODEL", "KNOWLEDGE_MODEL") or "google/gemma-4-31b-it:free"
    model = _normalize_model_for_base(model, base_url)
    api_key = (
        _first_env("LLM_API_KEY")
        or _api_key_for_provider(base_url, "LLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")
    )
    headers = _provider_headers(
        base_url,
        api_key=api_key,
        default_app_name=default_app_name,
        site_env_name="LLM_SITE_URL",
        app_env_name="LLM_APP_NAME",
    )
    return ApiProviderConfig(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=model,
        headers=headers,
    )
