# StockWise assistant

The LangGraph assistant runs independently of the forecasting API. It reads local
market history, calls the LSTM prediction endpoint, searches news, and returns
Plotly figures to Streamlit. MLflow records the graph and model/tool calls in a
dedicated **GenAI** experiment named `stock-assistant`.

## Start locally

Run commands from the repository root in the project virtual environment:

```powershell
python -m pip install -e ".[dev,assistant,gemini,frontend]"
```

Add these settings to your existing `.env` (do not overwrite your existing keys):

```dotenv
ASSISTANT_PROVIDER=gemini
ASSISTANT_MODEL=gemini-2.5-flash
GOOGLE_API_KEY=your-key
TAVILY_API_KEY=your-tavily-key
ASSISTANT_FORECAST_URL=https://stock-forecast-api.orangemoss-9dc4289a.uksouth.azurecontainerapps.io
ASSISTANT_URL=http://127.0.0.1:8001
ASSISTANT_DATA_ROOT=data/raw
ASSISTANT_MLFLOW_EXPERIMENT=stock-assistant
ASSISTANT_TRACING_ENABLED=true
```

Choose a model available to your account. `ASSISTANT_SIMPLE_MODEL` is optional;
when omitted, graders use the same model. Gemini also accepts `GEMINI_API_KEY`.
Existing `GENERATIVE_MODEL` and `SIMPLE_TASK_MODEL` remain Gemini compatibility
settings. Azure and Ollama use the explicit `ASSISTANT_*` settings.

Start each service in a separate terminal:

```powershell
# Assistant service (one worker while conversation memory is in process).
python -m uvicorn stock_assistant.api:app --host 127.0.0.1 --port 8001 --env-file .env

# Standalone chat page, or use the relocated apps/front/home.py for the full UI.
python -m streamlit run apps/front/pages/chatbot.py

# Dedicated assistant database; training keeps its existing store.
python -m mlflow ui --backend-store-uri sqlite:///mlflow-assistant.db --host 127.0.0.1 --port 5000
```

Open the Streamlit URL shown in the terminal and MLflow at http://127.0.0.1:5000.
Select `stock-assistant`, whose workflow type is GenAI, then inspect **Traces**.
Training stays in `stock-return-forecasting`. The setup uses the MLflow
`mlflow.experimentKind=genai_development` tag and LangChain/LangGraph autologging.
It records node/tool timing, inputs, outputs, errors, and model token usage where
the provider reports it. It does not promise exact provider billing totals.

`ASSISTANT_MLFLOW_TRACKING_URI` configures the assistant store; if empty, the
assistant uses an absolute path to repository-root `mlflow-assistant.db`.
`MLFLOW_TRACKING_URI` remains the training setting. To share a compatible tracking
server, explicitly set both URIs to it. Point the assistant UI at its database.
The existing training database was backed up and verified unchanged; its revision
`dc11669786a5` is not recognized by the currently installed MLflow migration chain.
The assistant uses a separate store so it does not depend on repairing that store. Initialization errors are surfaced;
MLflow runtime export failures also need attention in service logs. Traces contain
conversation/tool content: keep this local store private. Set
`ASSISTANT_TRACING_ENABLED=false` to disable assistant tracing before startup.

The API exposes `/health`, `/chat`, and `/docs`:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/chat -Method Post -ContentType application/json -Body '{"question":"Plot BIAT and explain its LSTM forecast with recent news","thread_id":"demo-1"}'
```

The response contains `text`, `sources`, `charts`, and `thread_id`. Reuse the thread
ID for follow-ups. Memory is currently in process and disappears on restart; this
is a local development service, with no user authentication or durable multiuser
conversation store. Do not expose it publicly as-is.

## Provider choices and student credits

| Provider | Install extra | Required configuration |
|---|---|---|
| Gemini | `gemini` | `ASSISTANT_PROVIDER=gemini`, model, Google/Gemini API key |
| Ollama | `ollama` | `ASSISTANT_PROVIDER=ollama`, locally pulled tool-capable model, optional `OLLAMA_BASE_URL` |
| Azure OpenAI | `azure-llm` | `ASSISTANT_PROVIDER=azure`, model **deployment name**, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION` |

For Ollama, install and start Ollama separately, pull a model supporting tool
calling and structured output, and set `ASSISTANT_MODEL` to that local model name.
The default server URL is `http://localhost:11434`. Local inference uses your RAM,
VRAM and electricity; news search still requires Tavily. The optional Pinecone
archive uses Gemini embeddings and is not a fully local retrieval pipeline.

Microsoft currently advertises **$100 Azure credit usable within 12 months**, and
access to Azure OpenAI, for Azure for Students. This is shared Azure credit, not a
separate unlimited LLM allowance. It excludes Azure Marketplace offers. Actual
remaining credit, model availability, deployment quota and subscription eligibility
must be checked in your Azure account before deploying a model. No Azure resources
are provisioned by this change. [Azure for Students](https://azure.microsoft.com/en-us/free/students/)

Gemini offers a free tier for selected models, subject to account/model quotas.
The free-tier pricing table also specifies data-use conditions; confirm those for
your chosen model before sending private content. Your available quota is not
verified by the local tests. [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)

## Tools and forecasting semantics

- `get_stock_data`, `list_available_companies`: validated local CSVs under
  `ASSISTANT_DATA_ROOT`, defaulting to the forecast data root or `data/raw`.
  No DVC pull, training, promotion or cloud mutation occurs during a chat.
- `analyze_stock_performance`, `compare_stocks`, `top_movers`: deterministic
  calculations over those bars.
- `predict_stock`: HTTP POST to `<ASSISTANT_FORECAST_URL>/v1/predict`. It requests
  the LSTM return model, forwards `FORECAST_API_KEY` as `X-API-Key` if configured,
  validates the reply and preserves its evidence. A `model_version` may be supplied
  when `/models` lists multiple evaluated versions. The API's approval checks apply.
- `list_forecast_models`: available versions from the authenticated `/models` endpoint.
- `plot_stock`, `plot_comparison`: historical charts and indicator overlays.
- `plot_prediction`: a single predicted log-return bar. It includes the as-of
  date, version and deployment status; it does not invent future dates or a price
  path. `plot_stock(include_forecast=true)` adds this as a separate chart.
- `web_search`: Tavily news search with dated source URLs. Missing credentials
  return an explicit unavailable result.
- `retrieve_news`: optional Pinecone archive; install `.[retrieval]` and configure
  its existing index, embedding model and keys. Without it, use web search.

Forecast horizons are **1, 5 or 20 observations**, interpreted using the returned
`horizon_basis`. A five-observation forecast is one cumulative forecast, not five
daily prices. A missing `forecast_for` stays missing. The agent must preserve
unapproved/stale/unavailable outcomes and communicate uncertainty. History may
still be charted when no serving model is available.

The graph prepares turn state, routes `on_topic` to the agent and `off_topic` to the off-topic responder, alternates agent/tool
steps, grades the answer, and optionally retries or searches the web. The default
retry count is two; `AGENT_MAX_RETRIES=0` reduces extra generation attempts.
`ASSISTANT_RECURSION_LIMIT=40` bounds graph steps. All model clients initialize
on demand. Provider switching requires restarting the service.

## Repository layout and migration

| Package/directory | Responsibility |
|---|---|
| `stock_assistant/` | LangGraph workflow, model providers, HTTP forecast client, tools, MLflow tracing, assistant API |
| `stock_forecasting/` | Training, evaluation, release controls, model registry, prediction API; public neural classes in `neural/` |
| `market_data/` | Validated local history adapter; older dashboard data adapter retained separately |
| `apps/front/` | Streamlit UI |
| `apps/back/` | Existing dashboard API and cache; `/chat` and mounted `/forecast` |
| `legacy/` | Earlier pickle model/scaler handlers, factories and endpoint |
| `scripts/`, `configs/`, `infrastructure/` | Operational entry points, configuration, deployment assets |
| `tests/assistant/`, `tests/forecasting/` | Assistant and forecasting checks |

The former `src/` package is removed. Imports, Docker entry points, CI paths and
the benchmark notebook now use the names above. Saved neural state-dict bundles
continue to use the current loader. Historical Python pickles referring to `src.*`
may need regeneration; they are not the new assistant's serving format.
`pyproject.toml` is the dependency source of truth; requirements and Conda manifests
delegate to its extras. Existing model/data artifacts and the training experiment
are not moved or promoted.

```powershell
$env:MPLBACKEND = "Agg"
python -m pytest tests/assistant tests/forecasting tests/prediction_model/test_torch_models.py -q
python -m ruff check stock_assistant market_data/history.py stock_forecasting
```

Tests mock external model/news providers, verify the HTTP boundary, and persist a
real offline LangGraph trace to an isolated MLflow database. They do not establish
that a live key has quota or that a deployed model is reachable.

On the current Windows environment, the two CUDA tests pass their assertions but
their Python process exits with native shutdown code `0xC0000409`. This reproduces
with only those two tests; the assistant-only suite exits normally. For a clean
CPU run, set `$env:CUDA_VISIBLE_DEVICES = "-1"` before starting pytest; the two GPU
tests will be skipped. The existing Torch/CUDA installation was retained.

References: [MLflow LangChain/LangGraph tracing](https://www.mlflow.org/docs/latest/genai/flavors/langchain/),
[Ollama integration](https://docs.langchain.com/oss/python/integrations/providers/ollama/),
[Azure chat integration](https://reference.langchain.com/python/langchain-openai/chat_models/azure/AzureChatOpenAI).

The assistant requires an explicit remote forecast URL and never falls back to a
local model. The current client uses the forecasting API's `/v1/predict` request
and response contract with `FORECAST_API_KEY`. An Azure ML managed `/score`
endpoint is a different contract and needs an adapter for its deployed scoring
script; do not set its URL as this API base URL. Registering models in Azure ML
and downloading them for local inference is not remote deployment inference.


## Azure single-user deployment (14 October 2026)

The Azure rollout uses the existing forecasting API and registry, an Azure OpenAI
`stock-assistant` deployment (GPT-4.1 mini, Global Standard), and a separate
Workload Profiles Container Apps environment with only the Consumption profile.
Express environments do not support the required sidecars, Easy Auth or Azure Files.

`infrastructure/azure/assistant-llm.bicep` defines the LLM. The complete app is in
`infrastructure/azure/stockwise-app.bicep`. It separates the frontend, dashboard API,
assistant, MLflow and gateway into containers in one single-replica app. Only the
gateway is exposed. A browser password prompt protects the app and MLflow over
HTTPS. Microsoft sign-in is disabled because tenant administrator consent was
required. Share neither the password nor the private deployment parameters.

- Frontend: Streamlit, through the authenticated gateway.
- Dashboard: `apps.back.dashboard:app`, validated date/closing-price history plus remote forecasts.
  Display history does not require valid opening/high/low fields; model-input OHLC
  validation remains strict. Missing deployed models do not prevent history charts.
- Assistant: `stock_assistant.api:app`, Azure LLM plus the existing Azure forecast API.
- MLflow: `/mlflow/` behind the same password protection; the assistant uses the
  private loopback tracking endpoint. This is separate from forecasting MLflow in Azure ML.
- Persistence: a dedicated Azure Files share holds the market CSV snapshot and
  MLflow traces/artifacts. The small demo uses MLflow's explicitly enabled file
  store with a single worker, not SQLite over SMB. Move to a supported SQL tracking
  backend before scaling beyond this demo. Existing local traces are not migrated.
- Chat history is in memory and resets on restart/scale-to-zero. Streamlit sessions
  likewise do not survive restarts. Market data is a deployment snapshot and must
  be refreshed from the maintained datasets; no automatic refresh job is added.

The app scales from zero to one replica (2 vCPU / 4 GiB total across containers).
Close browser tabs when finished: active WebSocket sessions can keep it running.
The subscription has a monthly budget target of 20 billing-currency units, not a
spending cap or automatic shutdown; no email notifications were configured.
Azure retail pricing checked for UK South lists GPT-4.1 mini Global Standard at
USD 0.40 / million input tokens and USD 1.60 / million output tokens. Include all
agent/tool iterations and graders when estimating token use. Registry, build,
compute, storage, existing forecast resources and third-party search cost extra.

The assistant Dockerfile accepts `--build-arg ASSISTANT_EXTRAS=assistant,azure-llm,retrieval`.
It omits training-only PyTorch/Matplotlib/statsmodels; inference stays remote.
Builds use a curated source context without `.env`, local models or datasets, and
images are deployed by digest. Credentials stay in Container Apps secrets and
ignored local deployment parameters (`artifacts/stockwise.parameters.json`).
Never commit that generated parameters file. The gateway uses HTTP Basic authentication
over HTTPS. The username is `stockwise`; the generated password is saved locally as
`STOCKWISE_APP_PASSWORD` in `.env`. Azure stores only its salted SHA-512 crypt hash.
Protect and rotate the password; browsers may retain it until closed.

For future changes, build new image tags in ACR, update their digests in private
parameters, validate the Bicep deployment, then deploy. Keep external ingress off
until gateway password protection has been verified. Microsoft sign-in is disabled.
The deployed frontend must point at the new dashboard API, not the legacy backend.

Deployed URLs:

- App: https://stockwise-app.blueriver-6d32164c.uksouth.azurecontainerapps.io/
- Assistant MLflow: https://stockwise-app.blueriver-6d32164c.uksouth.azurecontainerapps.io/mlflow/

Use username `stockwise` and the local `.env` value `STOCKWISE_APP_PASSWORD`.
The first request after scale-to-zero can take longer while containers start.

### Market terminal

The dashboard lists only locally chartable stocks present in the remote `/models`
catalog with an LSTM return model at horizon 1. It pins each prediction request to
the catalog model version. A catalog outage does not fall back to all CSV symbols;
a temporarily unavailable forecast is shown explicitly alongside its history.

The terminal supports candles/lines, volume, selectable SMA and EMA windows
(5/10/20/50/100/200 bars), Bollinger Bands (20 bars, two population standard
deviations), Wilder RSI (14), and MACD (12/26/9). Indicators are computed before
cropping the selected time range, with warmup gaps preserved. Invalid OHLC bars
are excluded from candles; invalid volume values remain missing. Display logic
never relaxes model training or inference validation. Price units and observation
dates come from the source snapshot, not a live exchange feed.

Frontend chart calculations live in `apps/front/trading_charts.py`; the theme is
in `.streamlit/config.toml`. The chart includes separate aligned volume/momentum
panels, configurable overlays, zoom/pan, and CSV export. Use Refresh to invalidate
the local response cache when deployments or market data change.

Daily data publication, forecast refresh and the proposed multi-horizon model are
documented in [dynamic_forecasts.md](dynamic_forecasts.md).


### Live research activity

The chat uses `POST /chat/stream` with the same question/thread ID body as
`POST /chat`. It returns newline-delimited JSON (`application/x-ndjson`):
`status`, `progress`, periodic `heartbeat`, then exactly one `result` or `error`
for a connected request. Progress has a unique execution `id`, a public `label`,
`status` (`running`, `completed`, `failed`), and an optional allowlisted tool name.
Parallel calls and retries have separate IDs. The final result keeps the existing
`text`, `sources`, `charts`, and `thread_id` fields. `/chat` remains available.

The UI shows a live collapsible activity panel and retains completed activity
with the answer in browser-session history. Progress does not include prompts,
tool arguments/results, provider errors, or model reasoning. Answers appear after
final grading; intermediate drafts are deliberately not token-streamed.

Restart the assistant and frontend together to enable this feature locally.
The frontend reads the assistant stream server-to-server, so the public gateway
continues to carry Streamlit's WebSocket updates. Streaming responses disable
proxy buffering and send idle heartbeats. Disconnecting stops delivery and skips
queued turns that have not started; an already-running model/tool operation may
finish and update conversation state. It is not a provider cancellation request.
