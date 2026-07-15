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
from vaultq.chunk_stats import chunk_length_stats
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
            "[b]VaultQ[/b]\nAgent-native markdown vault, Atlas/Voyage retrieval, graph signals, MCP tools, and autonomous workspace maintenance.",
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
                    value=".obsidian/**, .trash/**, .vaultq/**, 88_Agents/sessions/**",
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
            with TabPane("Agent", id="agent"):
                with Horizontal(classes="button-row"):
                    yield Button("Prepare Runtime", id="agent_prepare", variant="primary", classes="cta")
                    yield Button("Dry Run Maintenance", id="agent_maintain_dry", classes="cta")
                    yield Button("Run Maintenance", id="agent_maintain", classes="cta")
                    yield Button("Refresh Agent Status", id="agent_status", classes="cta")
                yield Markdown(id="agent_markdown", classes="panel")
            with TabPane("Search", id="search"):
                yield Input(placeholder="Search the vault", id="search_query", classes="row")
                with Horizontal(classes="button-row"):
                    yield Button("Related Work", id="search_related", variant="primary", classes="cta")
                    yield Button("Focused Search", id="search_focused", variant="primary", classes="cta")
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
        self._refresh_agent_view()
        self._log("VaultQ TUI ready.")

    def action_refresh(self) -> None:
        self._refresh_collections_view()
        self._refresh_overview()
        self._refresh_mcp_view()
        self._refresh_agent_view()
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
        elif button_id == "agent_prepare":
            self._run_job("agent prepare", self._job_agent_prepare, refresh=True)
        elif button_id == "agent_maintain_dry":
            self._run_job("agent maintenance dry-run", lambda: self._job_agent_maintain(dry_run=True), refresh=True)
        elif button_id == "agent_maintain":
            self._run_job("agent maintenance", lambda: self._job_agent_maintain(dry_run=False), refresh=True)
        elif button_id == "agent_status":
            self._refresh_agent_view()
        elif button_id == "search_related":
            self._run_related()
        elif button_id == "search_focused":
            self._run_search("focused")
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

    def _default_collection_root(self) -> Optional[Path]:
        config = read_config(self.base_dir)
        collections = config.get("collections") or []
        if not collections:
            return None
        return Path(collections[0]["path"]).expanduser().resolve()

    def _job_agent_prepare(self) -> Dict[str, Any]:
        from vaultq.agent_runtime import prepare_background_agent

        root = self._default_collection_root()
        if root is None:
            raise ValueError("Configure a collection before preparing the agent runtime.")
        return prepare_background_agent(root, write_sheet=True)

    def _job_agent_maintain(self, *, dry_run: bool) -> Dict[str, Any]:
        from vaultq.agent_runtime import run_autonomous_maintenance

        root = self._default_collection_root()
        if root is None:
            raise ValueError("Configure a collection before running agent maintenance.")
        return run_autonomous_maintenance(root, dry_run=dry_run, max_actions=20)

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
            self._refresh_agent_view()

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

    def _run_related(self) -> None:
        query = self.query_one("#search_query", Input).value.strip()
        if not query:
            self._log("related work: idea/query is required")
            return

        def job() -> None:
            try:
                from vaultq.related import related_work

                result = related_work(query, limit=8)
            except Exception as exc:
                self.call_from_thread(self._job_failed, "related work", exc)
                return
            self.call_from_thread(self._apply_related_result, result)

        self._log("related work: started")
        self.run_worker(job, thread=True, exclusive=False)

    def _apply_related_result(self, payload: Dict[str, Any]) -> None:
        self.search_results = [
            {
                **row,
                "doc_type": "related",
                "heading_path": row.get("connection_reason") or "",
            }
            for row in payload.get("connections", [])
        ]
        table = self.query_one("#results_table", DataTable)
        table.clear(columns=False)
        for index, result in enumerate(self.search_results):
            table.add_row(
                f"{float(result.get('score') or 0.0):.4f}",
                "related",
                result.get("rel_path") or "",
                result.get("connection_reason") or "",
                key=str(result.get("id") or f"related-{index}"),
            )
        if self.search_results:
            self.query_one("#result_markdown", Markdown).update(
                "\n".join(
                    [
                        "## Related Work",
                        "",
                        f"- Suggested home: `{(payload.get('suggested_existing_home') or {}).get('rel_path') or 'none'}`",
                        f"- Write default: `{payload.get('write_default')}`",
                        "",
                        self._result_markdown(self.search_results[0]),
                    ]
                )
            )
            self._log(f"related work completed: {len(self.search_results)} connection(s)")
        else:
            self.query_one("#result_markdown", Markdown).update("## No related work\n\nCreate only under the agent workspace if a note is still needed.")
            self._log("related work completed: no connections")

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
        self._log("watch: starting (low-impact, idle_tick=300s, new_file_scan=600s, changed_refresh=86400s)")

        def worker() -> None:
            try:
                from vaultq.background_index import background_state_name

                result = run_watch(
                    base_dir=self.base_dir,
                    poll_interval=300.0,
                    skip_knowledge=True,
                    embed_limit=100,
                    max_embed_batches=1,
                    new_file_index_delay_seconds=600.0,
                    changed_index_interval_seconds=86400.0,
                    no_initial_sync=True,
                    stop_event=self.watch_stop_event,
                    state_name=background_state_name("all"),
                    state_worker="tui",
                    drain_existing_pending=True,
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
            from vaultq.background_index import background_index_status

            background_status = background_index_status()
            background_states = background_status.get("states") or []
            background_summary = "none"
            if background_states:
                latest = background_states[0]
                data = latest.get("data_json") or {}
                background_summary = f"{latest.get('collection_name') or 'unknown'} updated {latest.get('updated_at')} loops={data.get('summary', {}).get('loops', 0)}"
            chunk_stats = chunk_length_stats(sample_limit=5)
            chunk_summary = chunk_stats["summary"]
            chunk_assessment = chunk_stats["assessment"]
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
                    f"- Chunk p50/p95/max: `{chunk_summary['p50_tokens']:.0f}` / `{chunk_summary['p95_tokens']:.0f}` / `{chunk_summary['max_tokens']}` tokens",
                    f"- Chunk health: `{chunk_assessment['overall']}` (`{chunk_assessment['over_max_pct']}%` over max)",
                    f"- Knowledge objects: `{status['knowledge_objects']}`",
                    f"- Pending chunk embeddings: `{status['pending_chunk_embeddings']}`",
                    f"- Pending knowledge embeddings: `{status['pending_knowledge_embeddings']}`",
                    f"- Watch mode: `{'running' if self.watch_running else 'stopped'}`",
                    f"- Background index state: `{background_summary}`",
                    f"- Dense model: `{status['embedding_model']}`",
                    f"- Provider: `{status['embedding_provider']}`",
                    f"- Reranker: `{status['reranker_model']}`",
                    f"- Contextual embeddings: `{status['contextual_embeddings']}`",
                    f"- Qdrant collection: `{status['qdrant_collection']}`",
                    f"- Qdrant URL: `{status['qdrant_url']}`",
                    f"- Agent workspace: `14_Agent_Workspace`",
                    f"- Session transcript index: `excluded via 88_Agents/sessions/**`",
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

    def _refresh_agent_view(self) -> None:
        try:
            from vaultq.agent_runtime import AGENT_RUNTIME_FILE, runtime_config
            from vaultq.maintenance import inspect_ai_workspace
            from vaultq.policy import load_policy
            from vaultq.property_schema import load_property_schema

            root = self._default_collection_root()
            if root is None:
                raise ValueError("No collection configured.")
            policy = load_policy(root)
            schema = load_property_schema(root)
            workspace = inspect_ai_workspace(root)
            runtime_path = root / AGENT_RUNTIME_FILE
            runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else runtime_config(root)
            markdown = "\n".join(
                [
                    "# Agent Runtime",
                    "",
                    f"- Workspace: `{runtime['workspace_root']}`",
                    f"- Mode: `{runtime['mode']}`",
                    f"- Runtime config: `{runtime_path}`",
                    f"- Background enabled: `{runtime['enabled']}`",
                    f"- Prepared only: `{runtime['prepared_only']}`",
                    f"- Allowed write roots: `{', '.join(policy.allowed_roots)}`",
                    f"- Markdown files in workspace: `{workspace['markdown_files']}`",
                    f"- Proposal files: `{workspace['proposal_files']}`",
                    f"- Report/run files: `{workspace['report_files']}`",
                    f"- Required agent provenance: `created_by: agent`",
                    f"- Allowed created_by values: `{', '.join(schema.allowed_created_by)}`",
                    "",
                    "## Autonomous Maintenance",
                    "",
                    "A live pass promotes agent-authored notes inside `14_Agent_Workspace`, writes promotion ledger entries, writes idea clusters, and updates `Human Updates/Latest Agent Workspace Update.md`. It does not mutate canonical vault folders.",
                ]
            )
        except Exception as exc:
            markdown = "\n".join(
                [
                    "# Agent Runtime",
                    "",
                    "Agent status is unavailable.",
                    "",
                    f"- Error: `{exc}`",
                ]
            )
        self.query_one("#agent_markdown", Markdown).update(markdown)

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
                    "vq mcp --transport http --host 127.0.0.1 --port 7073",
                    "```",
                    "",
                    "Tool surface:",
                    "",
                    "- `vaultq_status`",
                    "- `vaultq_collection_list`",
                    "- `vaultq_operation_list`",
                    "- `vaultq_chunk_stats`",
                    "- `vaultq_background_status`",
                    "- `vaultq_search`",
                    "- `vaultq_query`",
                    "- `vaultq_related_work`",
                    "- `vaultq_semantic_clusters`",
                    "- `vaultq_get_doc`",
                    "- `vaultq_write_status`",
                    "- `vaultq_capture`",
                    "- `vaultq_note_put`",
                    "- `vaultq_note_append`",
                    "- `vaultq_note_propose`",
                    "- `vaultq_property_propose`",
                    "- `vaultq_graph_extract`",
                    "- `vaultq_graph_neighbors`",
                    "- `vaultq_graph_backlinks`",
                    "- `vaultq_graph_traverse`",
                    "- `vaultq_think`",
                    "- `vaultq_maintain`",
                    "- `vaultq_agent_prepare`",
                    "- `vaultq_agent_maintain`",
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
