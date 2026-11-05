# StockWise web

React + TypeScript, Vite, Tailwind CSS. Node 22.12+ is required.
The Python services remain independent; the browser never needs model/API keys.

TuniScope is StockWise's research assistant. The navbar includes a light/dark
theme switch; the initial theme follows the system and explicit choices persist
in this browser. Charts follow the selected theme as well.

## Local development

From the repository root, run the APIs in separate terminals with the existing
virtual environment and `.env`:

```powershell
python -m uvicorn apps.back.dashboard:app --host 127.0.0.1 --port 8002 --env-file .env
python -m uvicorn stock_assistant.api:app --host 127.0.0.1 --port 8001 --env-file .env
```

Then start the frontend in another terminal:

```powershell
cd apps/front
npm ci
npm run dev
```

Open http://127.0.0.1:5173. Vite proxies `/market-api` to port 8002 and
`/assistant-api` to port 8001. Change those targets in `vite.config.ts` if needed.
Keep keys in the repository-root `.env`, not in `VITE_*` variables (which are public).
The forecasting service remains remote; it does not need to run locally.
If the market catalog fails locally, check that the dashboard API is running on
port 8002 with `--env-file .env`. The catalog requires `ASSISTANT_FORECAST_URL`,
`FORECAST_API_KEY`, and local market history; Vite alone cannot provide it.

## Structure

- `src/pages/Market.tsx`: individual stock charts, forecasts and history export.
- `src/pages/MarketList.tsx`: searchable/sortable deployed-stock snapshots.
- `src/components/StockResearch.tsx`: suggested questions and dated news links.
- `src/pages/Chat.tsx`: conversation UI, tool activity and final answers.
- `src/api.ts`: incremental NDJSON transport with cancellation and EOF detection.
- `src/components/Plot.tsx`: lazy-loaded Plotly rendering and resize/cleanup.
- `src/styles.css`: Tailwind theme and shared styles.
- `../back/trading_charts.py`: server-side aggregation and technical indicators.

Chats live in memory until tab reload/close. Server conversation checkpoints also
remain in process. The stop button stops receiving; a running model operation can
still finish on the server. Only final graded answers are shown, not draft tokens.
Historical charts support daily/monthly/yearly bars; this does not change the
deployed forecast horizon. Missing data and forecast outages remain explicit.

`#market` is the stock list; `#market/BIAT` is a directly linkable stock page.
The list calls `GET /market` once and never runs per-row forecasts. Each detail
page uses the existing history/chart APIs and places the deployed forecast beside
the chart. Suggested questions start a fresh assistant conversation and submit the
stock-specific prompt. An active assistant turn finishes before a queued suggestion
starts. News uses `GET /stock/{symbol}/news`, the server-side Tavily configuration,
and a 15-minute in-memory cache. Missing coverage, provider failures, and undated
articles are explicitly labelled; headlines are links, not generated summaries.
Market snapshots are historical observations, not live prices or an order book.

## Verify and ship

```powershell
npm test
npm run build
npm run test:e2e
```

The production image builds static assets and serves them with unprivileged nginx
on port 8501. Build from the repository root:

```powershell
docker build -f apps/front/Dockerfile -t stockwise-frontend .
```

For Azure, deploy the updated frontend, gateway and assistant/dashboard images
together, plus the updated Bicep readiness probe. The gateway retains password
authentication and proxies the two API prefixes with chat buffering disabled.
MLflow routes are unchanged. Do not expose the static container alone as the app:
production API routing and authentication belong to the gateway. `npm run preview`
is only a static-build preview, not a replacement for the production gateway.

The two browser smoke tests use mocked services (no Azure calls or token spend).
Install the test browser once with `npx playwright install chromium`. They cover
market loading/grouping/export and the chat response/chart/history flow. The three
unit tests cover the streaming parser. Existing Python indicator tests are reused;
only two tests were added for the new chart HTTP endpoint.

Build configuration follows [Vite](https://vite.dev/guide/) and
[Tailwind's Vite integration](https://tailwindcss.com/docs/installation/using-vite).
