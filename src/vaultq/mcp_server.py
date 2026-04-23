from __future__ import annotations

from typing import Any, Dict

from fastmcp import FastMCP

from vaultq.embedding_provider import load_env_layers
from vaultq.search import fetch as fetch_result
from vaultq.search import get_document as get_document_record
from vaultq.search import health, search as search_records

load_env_layers()

MCP_INSTRUCTIONS = (
    "You have access to a markdown-vault retrieval system.\n"
    "Use search() to find relevant chunks or knowledge objects.\n"
    "Use fetch() to retrieve the full result payload for a point.\n"
    "Use get_document() when you need the stored source document.\n"
    "Always cite the collection and relative path when answering from retrieved notes."
)


def build_mcp() -> FastMCP:
    mcp = FastMCP(name="VaultQ", instructions=MCP_INSTRUCTIONS)

    @mcp.tool()
    def status() -> Dict[str, Any]:
        """Return the current VaultQ store and retrieval status."""
        return health()

    @mcp.tool()
    def search(query: str, limit: int = 5, retrieval_mode: str = "hybrid") -> Dict[str, Any]:
        """Search the indexed markdown vault using hybrid, semantic, or keyword retrieval."""
        return search_records(query=query, limit=limit, retrieval_mode=retrieval_mode)

    @mcp.tool()
    def fetch(point_id: str) -> Dict[str, Any]:
        """Fetch a single result by Qdrant point id and expand its neighboring context."""
        return fetch_result(point_id)

    @mcp.tool()
    def get_document(identifier: str, full: bool = False) -> Dict[str, Any]:
        """Fetch a stored document by relative path or #id."""
        return get_document_record(identifier=identifier, full=full)

    return mcp


def run_mcp(*, transport: str = "stdio", host: str = "127.0.0.1", port: int = 7070, show_banner: bool = True) -> None:
    mcp = build_mcp()
    transport = (transport or "stdio").strip().lower()
    if transport == "stdio":
        mcp.run(show_banner=show_banner)
        return
    mcp.run(transport=transport, host=host, port=port, show_banner=show_banner)


if __name__ == "__main__":
    run_mcp()
