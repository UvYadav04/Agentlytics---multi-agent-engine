# analyzerEngine

The agent engine behind DataAnalyzer: ingestion, the three agents (Orchestrator, Tabular, Document), the tools they call, the sandboxed Python executor, the vector store, and the multi-provider LLM abstraction. Everything a chat message needs to turn into an investigated, cited answer lives here.

It's substantial enough to be its own repository — its own `requirements.txt`, its own `.env`, its own internal import style, its own dependency surface (docling, LlamaParse, DuckDB, Chroma, autogen). This README is the entry point for working in this folder specifically; for how it fits into the rest of the product see [`../README.md`](../README.md) and [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md).

## Not a standalone service

`analyzerEngine` is a library, not a process — nothing here binds a port or runs on its own. It's imported by `worker_service` only, never by `api_service` (see `../README.md`'s "Key things to know" section for why). A slow or stuck agent run can block a queue worker; it can never block the HTTP layer, because the HTTP layer never imports this code.

One consequence worth knowing before you touch anything here: every internal import in this package is **bare** — `from agents.orchestrator import OrchestratorAgent`, `from tools.tabular.tabular_tools import TabularTools`, `from ingestion.manager import IngestionManager` — never `from analyzerEngine.agents...`. That only resolves because `worker_service/engine_bootstrap.py` inserts `analyzerEngine/` directly onto `sys.path` before anything here is imported. If you ever import the same module both ways (`analyzerEngine.vectordb.chroma_store` from outside and `vectordb.chroma_store` from inside), Python treats them as two unrelated modules with two separate singletons — see `ARCHITECTURE.md` §5.2 for the bug that came from exactly this. Any script you run directly from inside this folder (the seed script, a REPL, a one-off) needs that same `sys.path` insert first; importing `worker_service.engine_bootstrap` does it as a side effect.

## Layout

| Directory | What it is |
|---|---|
| `agents/` | The three agents — `OrchestratorAgent`, `TabularAgent`, `DocumentAgent` — each an autogen `AssistantAgent` wrapping a tool class. See [`agents/README.md`](agents/README.md). |
| `tools/` | The tool implementations the agents call: `tools/tabular` (DuckDB queries + sandboxed Python), `tools/document` (RAG search/verification over Chroma), `tools/orchestrator` (file catalog, memory, capability gating, deliverable generation), `tools/reporting` (CSV/dashboard/report writers), `tools/hypothesis` (LLM-generated investigation angles). |
| `ingestion/` | Turns an uploaded CSV/XLSX/PDF/TXT into Parquet + vector-store chunks. See [`ingestion/README.md`](ingestion/README.md). |
| `vectordb/` | `ChromaVectorStore` (Chroma Cloud client behind `BaseVectorStore`) and `CrossEncoderReranker` (Hugging Face Inference API). |
| `sandbox/` | The Docker-based warm pool of Python containers `run_python` executes user-requested code in — `sandbox_manager.py` (pool lifecycle), `sandbox_server.py`/`execution_engine.py` (what runs inside each container), `sandbox_client.py` (HTTP-over-Unix-socket to a container), `path_resolver.py` (the only place artifact ids get turned into filesystem paths — validates every segment, no raw path ever crosses this boundary). See `ARCHITECTURE.md` §5.10. |
| `llm_provider/` | The one place every LLM call in this codebase goes through. `LLMProvider.get_client()` builds a provider client (`providers/{openai,anthropic,azure,groq,gemini,deepinfra}_client.py`) and layers `RetryingChatCompletionClient` (exponential backoff + jitter), an optional `FallbackChatCompletionClient` (primary → a second provider on failure), and `LangfuseTracedChatCompletionClient` (Langfuse Cloud tracing, no-op if unconfigured) around it. No agent or tool ever imports an SDK directly. |
| `sample_data/` | `dummy.csv` / `dummy.xlsx` / `dummy.pdf` — the sample files every new empty workspace gets cloned from (see `Server/shared/dummy_files.py` and `Server/scripts/seed_dummy_template.py`), plus `generate_samples.py`, the generator that produced them. |
| `config.py` | `get_settings()` — a thin `dotenv`-backed reader every module in this package calls instead of `os.getenv` directly. Loads `.env` in this directory (via `python-dotenv`'s `load_dotenv()`, so it also picks up an already-exported process env). |

## How a request actually flows

1. A file is uploaded → `worker_service/tasks/ingestion.py`'s `run_ingestion` job calls `ingestion.manager.IngestionManager.ingest_file()`, which picks an ingestor from `ingestion/registry.py` by extension and produces Parquet (`storage`) and/or vector-store chunks (`vector_store`).
2. `worker_service/tasks/investigation.py` builds a `tools.orchestrator.file_catalog.FileCatalog` from every `ready` file in the workspace and constructs an `agents.orchestrator.OrchestratorAgent` around it.
3. `OrchestratorAgent.run(query, workspace_id, ...)` talks to the user directly and never runs SQL or RAG itself — it delegates to a fresh `TabularAgent`/`DocumentAgent`, scoped to exactly the files it assigns, via `invoke_tabular_agent`/`invoke_document_agent`.
4. Those sub-agents run their own tool-calling loop (`tools/tabular/tabular_tools.py` → DuckDB views over Parquet + `run_python` in the sandbox pool; `tools/document/document_tools.py` → Chroma similarity search, optionally reranked), then hand a transcript to a second, tool-free `AssistantAgent` whose only job is to emit findings as JSON — see `agents/README.md` for why that split exists.
5. The Orchestrator folds those findings into `InvestigationState`, optionally calls `generate_csv`/`generate_dashboard`/`generate_report` (capability-gated — see `agents/orchestrator/capabilities.py`) to produce a deliverable file, and returns an `OrchestratorResult` (`final_answer`, `confidence`, `artifact_refs`, `open_questions`).
6. `worker_service/tasks/investigation.py` persists that result, and `worker_service/tasks/chat_title.py`/`update_chat_memory` run afterward, asynchronously, to title the chat and extract any durable user preferences into `tools/orchestrator/memory.py`'s `LongTermMemory`.

None of the pieces in step 4/5 hold state between calls — the sandbox pool reloads every table fresh from the parquet volume per `run_python` call (`ARCHITECTURE.md` §5.10), and a new `TabularAgent`/`DocumentAgent` instance is constructed per orchestrator delegation, not reused across turns.

## Environment

`analyzerEngine/.env` (copy from `.env.example` in this directory) is loaded by `worker_service` only — `api_service` never imports this package, so it never needs these vars. Every key has an inline comment in `.env.example`; the groups worth knowing about:

- **LLM keys** — `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`/`GROQ_API_KEY`/`GEMINI_API_KEY`/`DEEPINFRA_API_KEY`. You only need the ones your configured providers actually use.
- **Per-agent provider/model overrides** — `ORCHESTRATOR_PROVIDER`/`_MODEL`, `TABULAR_AGENT_PROVIDER`/`_MODEL`, `DOCUMENT_AGENT_PROVIDER`/`_MODEL`, `DOCUMENT_PROCESSOR_PROVIDER`/`_MODEL`, `HYPOTHESIS_PROVIDER`/`_MODEL`. Each agent can run on a different provider/model; `DEFAULT_LLM_PROVIDER`/`DEFAULT_MODEL` are the fallback if an agent-specific override is blank. `FALLBACK_LLM_PROVIDER` (defaults to `groq`) is the provider `OrchestratorAgent`/`DocumentAgent` fall back to if their primary provider call fails outright.
- **Chroma Cloud** — `CHROMA_API_KEY`/`CHROMA_TENANT`/`CHROMA_DATABASE`, required for `ChromaVectorStore` (document RAG and PDF table pointer-chunks).
- **LlamaParse** — `LLAMAPARSE_API_KEY`, required for PDF ingestion (see `ingestion/README.md` for why parsing is hosted rather than local).
- **Jev decision model** — `TYPESAFE_API_KEY` (required to call Jev), plus optional `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL` (defaults to `jev-latest`), `JEV_ENABLED`, `JEV_TIMEOUT_SECONDS`, `JEV_MAX_RETRIES` and `JEV_BACKOFF_SECONDS`. See [Decision model (Jev)](#decision-model-jev).
- **Reranker** — `RERANKER_MODEL` (defaults to a MiniLM cross-encoder) + `HF_API_TOKEN`, required only if `DocumentAgent` is constructed with a `CrossEncoderReranker`.
- **Sandbox tuning** — `SANDBOX_IDLE_TIMEOUT_SECONDS`/`SANDBOX_HEALTH_TIMEOUT_SECONDS`/`SANDBOX_REAP_INTERVAL_SECONDS`/`SANDBOX_POOL_MIN_SIZE`/`SANDBOX_POOL_MAX_SIZE`/`SANDBOX_ACQUIRE_TIMEOUT_SECONDS`. All six are read by `sandbox/sandbox_manager.py` via plain `os.environ.get()`, not through `config.py`'s `Settings` — they only pick up values from this `.env` file because `config.py`'s `load_dotenv()` populates `os.environ` as a side effect, and something else in the process has to import `config` before `sandbox_manager` for that to have happened. In practice that ordering already holds (agents/tools import `config` early), but if you ever see one of these fall back to its hardcoded default unexpectedly, check import order first.
- **Langfuse** — `LANGFUSE_HOST`/`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`, optional and no-op if blank (every LLM call already goes through `LangfuseTracedChatCompletionClient` regardless — these just decide whether it actually exports anywhere).
- **`AGENT_LOG_LEVEL`** — controls `agents/logger.py`'s shared `logging.getLogger("agent")`, which every agent writes into (`logs/agents.log` + console).

## Decision model (Jev)

`jev/` is a standalone client for TypeSafe's Jev, a System One decision model: you send a `state` and a map of typed questions, and get typed answers with probabilities instead of generated text. It has no dependency on agents, tools or the LLM providers, so anything in the engine (or `api_service`/`shared`, via `analyzerEngine.jev`) can import it.

Question types:

- `Choice(instructions, options)` picks one option from 2-255 options (a list, or a dict of option to description). The answer has `choice`, `probabilities`, `confidence`, `pick(min_confidence, fallback)` and `ranked()`.
- `Score(instructions, levels)` rates the state on 2-10 ordered levels. The answer has `score`, `legend`, `probabilities`, `confidence` and `nearest_label()`.
- `Noul(instructions, yes=..., no=...)` asks a yes/no question. The answer has `noul` (probability of yes), `is_yes(threshold)`, `is_no(threshold)` and `is_uncertain(margin)`.

Ask every question that shares the same state in one call - Jev evaluates them in parallel:

```python
from jev import Choice, Noul, Score, decide

result = await decide(
    {"query": query, "has_files": True},
    {
        "route": Choice("Which agent should handle `query`?", {
            "tabular": "Needs computation over tables",
            "document": "Answer already exists in uploaded documents",
            "orchestrator": "Needs several capabilities or is unclear",
        }),
        "wants_file": Noul("Does `query` ask for a downloadable file such as a CSV or report?"),
        "complexity": Score("How much investigation does `query` need?", ["Single lookup", "A few steps", "Open-ended"]),
    },
)
route = result.choice("route").pick(0.7, fallback="orchestrator")
```

Single-question shortcuts are `choose`, `rate` and `yes_no` (and `choose_sync`, `rate_sync`, `yes_no_sync`, `decide_sync` for synchronous code). `try_decide`/`try_decide_sync` return `None` instead of raising when Jev is not configured or the call fails, so existing logic can stay as the fallback. Errors are `JevConfigError`, `JevValidationError`, `JevAuthenticationError`, `JevRequestError`, `JevUnavailableError` and `JevTimeoutError`, all subclasses of `JevError`. Rate limits (429), overloads (529), 5xx responses and timeouts are retried with exponential backoff. Logs record question ids, model, latency and token usage only - never the state.

## Tool surface

Every tool the Orchestrator can call is registered in `agents/orchestrator/capabilities.py`. Most are always available (`CORE_TOOLS`: file discovery, `invoke_tabular_agent`/`invoke_document_agent`, `generate_hypotheses`, `recall_user_info`, ...) so the model doesn't have to think about permissions for the common path. A small set of deliverable tools (`generate_csv`, `generate_report`, more added the same way) are capability-gated instead — kept out of every turn's prompt until the model explicitly asks for them (either by piggybacking `next_capabilities=["report"]` onto a call it's already making, or via the standalone `request_capabilities` tool when it has nothing else to call at that moment). This keeps the base prompt from growing every time a new deliverable type is added. `agents/README.md` has the full walkthrough, including how `TabularAgent`/`DocumentAgent` themselves are invoked and what each of their own tool loops looks like.

## Deeper reading

- [`agents/README.md`](agents/README.md) — each agent's tool loop, the formatter-split pattern, model fallback, long-term memory, and how to add a new agent.
- [`ingestion/README.md`](ingestion/README.md) — the CSV/XLSX/PDF/TXT pipelines, the LlamaParse-over-local-docling decision, table extraction, and how to add a new file type.
- [`../README.md`](../README.md) — the three-service split, local setup, and the two `.env` files.
- [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md) — full system design and the problems that shaped it (§5 in particular — the dual-singleton bug, the sandbox pool rewrite, the reranker swap).
