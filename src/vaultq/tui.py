from __future__ import annotations

import json
from pathlib import Path
from threading import Event
from typing import Any, Dict, List, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Button, DataTable, Footer, Header, Input, Log, Markdown, Static, TabbedContent, TabPane

from vaultq.doctor import run_doctor
from vaultq.embed import embed_pending
from vaultq.indexer import run_index
from vaultq.search import search
from vaultq.store import (
    config_path,
    ensure_local_config,
    ensure_qdrant_collection,
    ensure_schema,
    read_config,
    status_snapshot,
    write_config,
)
from vaultq.watch import run_watch


class VaultQTui(App[None]):
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("i", "initialize", "Init"),
    ]

    CSS = """
    Screen {
        layout: vertical;
        background: #0f1115;
        color: #f5f3ec;
    }

    #hero {
        padding: 1 2;
        height: auto;
        border: round #8fb0ff;
        background: linear-gradient(90deg, #10131a, #162033, #1d2a44);
        margin: 1 2 0 2;
    }

    #hero-title {
        text-style: bold;
        color: #f8f4df;
    }

    #hero-copy {
        color: #d8def2;
    }

    TabbedContent {
        margin: 1 2 2 2;
        height: 1fr;
    }

    TabPane {
        padding: 1;
    }

    .panel {
        border: round #3f567f;
        padding: 1;
        height: 1fr;
    }

    .row {
        height: auto;
        margin-bottom: 1;
    }

    .button-row {
        height: auto;
        margin-bottom: 1;
    }

    .cta {
        margin-right: 1;
        width: 1fr;
    }

    Input {
        margin-right: 1;
    }

    DataTable {
        height: 1fr;
        border: round #2e4668;
    }

    Markdown {
        height: 1fr;
        border: round #394863;
        padding: 1;
        background: #10141c;
    }

    Log {
        height: 1fr;
        border: round #394863;
        background: #0e1218;
    }
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        super().__init__()
        self.base_dir = Path(base_dir or Path.cwd()).resolve()
        self.search_results: List[Dict[str, Any]] = []
        self.watch_stop_event: Optional[Event] = None
        self.watch_running = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(
            "[b]VaultQ[/b]\nPortable markdown-vault ingestion, contextual Voyage embeddings, hybrid retrieval, and MCP export.",
            id="hero",
        )
        with TabbedContent(initial="overview"):
            with TabPane("Overview", id="overview"):
                yield Markdown(id="overview_markdown")
            with TabPane("Collections", id="collections"):
                yield Input(placeholder="Collection name", id="collection_name", classes="row")
                yield Input(placeholder="/path/to/vault", id="collection_path", classes="row")
                yield Input(placeholder="Pattern, e.g. **/*.md", value="**/*.md", id="collection_pattern", classes="row")
                yield Input(
                    placeholder="Exclude globs, comma separated",
                    value=".obsidian/**, .git/**, node_modules/**",
                    id="collection_excludes",
                    classes="row",
                )
                with Horizontal(classes="button-row"):
                    yield Button("Add / Update Collection", id="add_collection", variant="primary", classes="cta")
                yield DataTable(id="collections_table", classes="panel")
            with TabPane("Pipeline", id="pipeline"):
                with Horizontal(classes="button-row"):
                    yield Button("Initialize Store", id="init_store", variant="primary", classes="cta")
                    yield Button("Index With Knowledge", id="index_full", classes="cta")
                    yield Button("Index Chunks Only", id="index_skip_knowledge", classes="cta")
                    yield Button("Embed Pending", id="embed_pending", classes="cta")
                    yield Button("Start Watch", id="start_watch", classes="cta")
                    yield Button("Stop Watch", id="stop_watch", classes="cta")
                    yield Button("Run Doctor", id="run_doctor", classes="cta")
                    yield Button("Refresh Status", id="refresh_status", classes="cta")
                yield Markdown(id="status_markdown", classes="panel")
                yield Log(id="pipeline_log", classes="panel", auto_scroll=True)
            with TabPane("Search", id="search"):
                yield Input(placeholder="Search the vault", id="search_query", classes="row")
                with Horizontal(classes="button-row"):
                    yield Button("Hybrid Search", id="search_hybrid", variant="primary", classes="cta")
                    yield Button("Semantic Search", id="search_semantic", classes="cta")
                    yield Button("Keyword Search", id="search_keyword", classes="cta")
                yield DataTable(id="results_table", classes="panel")
                yield Markdown(id="result_markdown", classes="panel")
            with TabPane("MCP", id="mcp"):
                yield Markdown(id="mcp_markdown")
        yield Footer()

    def on_mount(self) -> None:
        collections_table = self.query_one("#collections_table", DataTable)
        collections_table.cursor_type = "row"
        collections_table.add_columns("Name", "Path", "Pattern", "Excludes")

        results_table = self.query_one("#results_table", DataTable)
        results_table.cursor_type = "row"
        results_table.add_columns("Score", "Type", "Path", "Section")

        self._refresh_collections_view()
        self._refresh_overview()
        self._refresh_mcp_view()
        self._log("VaultQ TUI ready.")

    def action_refresh(self) -> None:
        self._refresh_collections_view()
        self._refresh_overview()
        self._refresh_mcp_view()
        self._log("Refreshed dashboard.")

    def action_initialize(self) -> None:
        self._run_job("initialize", self._job_initialize, refresh=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "add_collection":
            self._add_collection()
        elif button_id == "init_store":
            self._run_job("initialize", self._job_initialize, refresh=True)
        elif button_id == "index_full":
            self._run_job("index", lambda: run_index(base_dir=self.base_dir, skip_knowledge=False), refresh=True)
        elif button_id == "index_skip_knowledge":
            self._run_job(
                "index chunks-only",
                lambda: run_index(base_dir=self.base_dir, skip_knowledge=True),
                refresh=True,
            )
        elif button_id == "embed_pending":
            self._run_job("embed", lambda: embed_pending(limit=400), refresh=True)
        elif button_id == "start_watch":
            self._start_watch()
        elif button_id == "stop_watch":
            self._stop_watch()
        elif button_id == "run_doctor":
            self._run_job("doctor", lambda: run_doctor(live=True), refresh=False)
        elif button_id == "refresh_status":
            self._refresh_overview()
        elif button_id == "search_hybrid":
            self._run_search("hybrid")
        elif button_id == "search_semantic":
            self._run_search("semantic")
        elif button_id == "search_keyword":
            self._run_search("keyword")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "results_table":
            return
        selected_id = str(getattr(event.row_key, "value", event.row_key))
        for result in self.search_results:
            if result["id"] == selected_id:
                self.query_one("#result_markdown", Markdown).update(self._result_markdown(result))
                break

    def _job_initialize(self) -> Dict[str, str]:
        ensure_local_config(self.base_dir)
        ensure_schema()
        ensure_qdrant_collection()
        return {"config_path": str(config_path(self.base_dir)), "status": "initialized"}

    def _run_job(self, label: str, func, *, refresh: bool = False) -> None:
        self._log(f"{label}: started")

        def job() -> None:
            try:
                result = func()
            except Exception as exc:
                self.call_from_thread(self._job_failed, label, exc)
                return
            self.call_from_thread(self._job_finished, label, result, refresh)

        self.run_worker(job, thread=True, exclusive=False)

    def _job_failed(self, label: str, exc: Exception) -> None:
        self._log(f"{label}: failed - {exc}")
        self._refresh_overview()

    def _job_finished(self, label: str, result: Any, refresh: bool) -> None:
        rendered = json.dumps(result, indent=2, ensure_ascii=False, default=str) if isinstance(result, (dict, list)) else str(result)
        self._log(f"{label}: completed\n{rendered}")
        if refresh:
            self._refresh_overview()
            self._refresh_collections_view()

    def _add_collection(self) -> None:
        name = self.query_one("#collection_name", Input).value.strip()
        path = self.query_one("#collection_path", Input).value.strip()
        pattern = self.query_one("#collection_pattern", Input).value.strip() or "**/*.md"
        excludes_raw = self.query_one("#collection_excludes", Input).value.strip()
        excludes = [item.strip() for item in excludes_raw.split(",") if item.strip()]
        if not name or not path:
            self._log("add collection: name and path are required")
            return
        config = read_config(self.base_dir)
        config.setdefault("collections", [])
        config["collections"] = [row for row in config["collections"] if row["name"] != name]
        config["collections"].append(
            {
                "name": name,
                "path": str(Path(path).expanduser().resolve()),
                "pattern": pattern,
                "exclude_globs": excludes,
            }
        )
        write_config(config, self.base_dir)
        self._log(f"collection updated: {name}")
        self._refresh_collections_view()

    def _run_search(self, retrieval_mode: str) -> None:
        query = self.query_one("#search_query", Input).value.strip()
        if not query:
            self._log("search: query is required")
            return

        def job() -> None:
            try:
                result = search(query, limit=8, retrieval_mode=retrieval_mode)
            except Exception as exc:
                self.call_from_thread(self._job_failed, f"search {retrieval_mode}", exc)
                return
            self.call_from_thread(self._apply_search_result, result)

        self._log(f"search {retrieval_mode}: started")
        self.run_worker(job, thread=True, exclusive=False)

    def _apply_search_result(self, payload: Dict[str, Any]) -> None:
        self.search_results = list(payload.get("results") or [])
        table = self.query_one("#results_table", DataTable)
        table.clear(columns=False)
        for result in self.search_results:
            section = result.get("heading_path") or result.get("knowledge_title") or result.get("doc_type") or ""
            table.add_row(
                f"{float(result.get('score') or 0.0):.4f}",
                result.get("doc_type") or "",
                result.get("rel_path") or "",
                section,
                key=result["id"],
            )
        if self.search_results:
            self.query_one("#result_markdown", Markdown).update(self._result_markdown(self.search_results[0]))
            self._log(
                f"search completed: {len(self.search_results)} results in {payload.get('elapsed_ms', 0)} ms"
            )
        else:
            self.query_one("#result_markdown", Markdown).update("## No results\n\nTry a broader query or run indexing first.")
            self._log("search completed: no results")

    def _start_watch(self) -> None:
        if self.watch_running:
            self._log("watch: already running")
            return
        self.watch_stop_event = Event()
        self.watch_running = True
        self._log("watch: starting (chunks-only, interval=2s, embed_limit=200)")

        def worker() -> None:
            try:
                result = run_watch(
                    base_dir=self.base_dir,
                    poll_interval=2.0,
                    skip_knowledge=True,
                    embed_limit=200,
                    max_embed_batches=4,
                    stop_event=self.watch_stop_event,
                    logger=lambda message: self.call_from_thread(self._log, message),
                )
            except Exception as exc:
                self.call_from_thread(self._watch_failed, exc)
                return
            self.call_from_thread(self._watch_finished, result)

        self.run_worker(worker, thread=True, exclusive=False)

    def _stop_watch(self) -> None:
        if not self.watch_running or self.watch_stop_event is None:
            self._log("watch: not running")
            return
        self.watch_stop_event.set()
        self._log("watch: stop requested")

    def _watch_failed(self, exc: Exception) -> None:
        self.watch_running = False
        self.watch_stop_event = None
        self._log(f"watch: failed - {exc}")
        self._refresh_overview()

    def _watch_finished(self, result: Dict[str, Any]) -> None:
        self.watch_running = False
        self.watch_stop_event = None
        self._log("watch: stopped\n" + json.dumps(result, indent=2, ensure_ascii=False, default=str))
        self._refresh_overview()

    def _refresh_collections_view(self) -> None:
        table = self.query_one("#collections_table", DataTable)
        table.clear(columns=False)
        config = read_config(self.base_dir)
        for row in config.get("collections", []):
            table.add_row(
                row.get("name") or "",
                row.get("path") or "",
                row.get("pattern") or "",
                ", ".join(row.get("exclude_globs") or []),
                key=row.get("name") or None,
            )

    def _refresh_overview(self) -> None:
        try:
            status = status_snapshot()
            configured = read_config(self.base_dir).get("collections", [])
            collection_names = ", ".join(row["name"] for row in configured) if configured else "none"
            markdown = "\n".join(
                [
                    "# VaultQ",
                    "",
                    f"- Config path: `{config_path(self.base_dir)}`",
                    f"- Collections in DB: `{collection_names}`",
                    f"- Documents: `{status['documents']}`",
                    f"- Chunks: `{status['chunks']}`",
                    f"- Knowledge objects: `{status['knowledge_objects']}`",
                    f"- Pending chunk embeddings: `{status['pending_chunk_embeddings']}`",
                    f"- Pending knowledge embeddings: `{status['pending_knowledge_embeddings']}`",
                    f"- Watch mode: `{'running' if self.watch_running else 'stopped'}`",
                    f"- Dense model: `{status['embedding_model']}`",
                    f"- Provider: `{status['embedding_provider']}`",
                    f"- Reranker: `{status['reranker_model']}`",
                    f"- Contextual embeddings: `{status['contextual_embeddings']}`",
                    f"- Qdrant collection: `{status['qdrant_collection']}`",
                    f"- Qdrant URL: `{status['qdrant_url']}`",
                ]
            )
        except Exception as exc:
            markdown = "\n".join(
                [
                    "# VaultQ",
                    "",
                    "The store is not initialized yet or the backing services are unavailable.",
                    "",
                    f"- Config path: `{config_path(self.base_dir)}`",
                    f"- Error: `{exc}`",
                ]
            )
        self.query_one("#overview_markdown", Markdown).update(markdown)
        self.query_one("#status_markdown", Markdown).update(markdown)

    def _refresh_mcp_view(self) -> None:
        self.query_one("#mcp_markdown", Markdown).update(
            "\n".join(
                [
                    "# MCP",
                    "",
                    "Run VaultQ as an MCP server from the repo root or any configured runtime directory.",
                    "",
                    "```bash",
                    "vq mcp --transport stdio",
                    "```",
                    "",
                    "HTTP transport:",
                    "",
                    "```bash",
                    "vq mcp --transport http --host 127.0.0.1 --port 7070",
                    "```",
                    "",
                    "Tool surface:",
                    "",
                    "- `status`",
                    "- `search`",
                    "- `fetch`",
                    "- `get_document`",
                ]
            )
        )

    def _result_markdown(self, result: Dict[str, Any]) -> str:
        metadata = result.get("metadata") or {}
        metadata_lines = [f"- **{key}**: `{value}`" for key, value in metadata.items()]
        return "\n".join(
            [
                f"# {result.get('title') or result.get('knowledge_title') or result.get('rel_path') or 'Result'}",
                "",
                f"- Collection: `{result.get('collection_name')}`",
                f"- Path: `{result.get('rel_path')}`",
                f"- Type: `{result.get('doc_type')}`",
                f"- Score: `{result.get('score')}`",
                "",
                "## Text",
                "",
                result.get("text") or "_No text returned._",
                "",
                "## Metadata",
                "",
                *(metadata_lines or ["- _No extra metadata_"]),
            ]
        )

    def _log(self, message: str) -> None:
        self.query_one("#pipeline_log", Log).write_line(message)


def run_tui(base_dir: Optional[Path] = None) -> None:
    VaultQTui(base_dir=base_dir).run()
