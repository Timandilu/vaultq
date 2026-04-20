from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Optional

try:
    from dotenv import load_dotenv as _load_dotenv_impl
except Exception:
    _load_dotenv_impl = None

_PROVIDER_DEFAULTS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "text-embedding-3-large",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/text-embedding-3-large",
    },
    "custom": {
        "base_url": "https://api.openai.com/v1",
        "model": "text-embedding-3-large",
    },
}


@dataclass(frozen=True)
class EmbeddingProviderConfig:
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
            line = line[len("export "):].strip()
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


def _normalize_model_for_base(model: str, base_url: str) -> str:
    model = _clean(model)
    if not model:
        return "text-embedding-3-large"
    if "/" in model and "openrouter.ai" not in base_url.lower():
        _, normalized = model.split("/", 1)
        if normalized:
            return normalized
    return model


def resolve_embedding_provider(default_app_name: str = "VaultQ", default_site_url: str = "http://localhost") -> EmbeddingProviderConfig:
    provider = _clean(os.getenv("EMBED_PROVIDER", "openai")).lower() or "openai"
    base_url = _first_env("EMBED_BASE_URL", "OPENAI_BASE_URL") or _PROVIDER_DEFAULTS.get(provider, _PROVIDER_DEFAULTS["custom"])["base_url"]
    api_key = _first_env("EMBED_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
    model = _first_env("EMBED_MODEL", "EMBEDDING_MODEL") or _PROVIDER_DEFAULTS.get(provider, _PROVIDER_DEFAULTS["custom"])["model"]
    model = _normalize_model_for_base(model, base_url)

    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"
    if provider == "openrouter" or "openrouter.ai" in base_url.lower():
        site_url = _first_env("OPENROUTER_SITE_URL") or default_site_url
        app_name = _first_env("OPENROUTER_APP_NAME") or default_app_name
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name

    return EmbeddingProviderConfig(
        provider=provider,
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=model,
        headers=headers,
    )
