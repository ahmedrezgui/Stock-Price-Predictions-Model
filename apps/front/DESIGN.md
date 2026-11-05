# StockWise interface audit and redesign

## Audit (2026-11-05)

React 19 / TypeScript, Vite 6, Tailwind 4 via the Vite plugin. Lucide icons,
react-markdown/GFM and lazy-loaded Plotly already cover the design requirements.
Hash routes: home, market list, market/{symbol}, chat. Chat stays mounted during
navigation; sessions and progress are in memory. The authenticated production
gateway and development proxy keep API credentials on the server.

Working: deployed-stock filtering, sortable market data, server-side chart
aggregation/indicators, CSV export, forecast error isolation, sourced news,
suggestion-to-chat handoff, NDJSON progress, cancellation, theme persistence.

Problems: oversized marketing headline with no market context; repetitive surface
treatment; missing shared loading/retry patterns; small controls; sparse list
context; raw JSON forecast details; activity rendered as plain status strings;
mobile sidebar lacks keyboard containment; chart styles duplicate CSS colors.

## Priorities

1. Shared dark/light tokens, self-hosted interface typography, numerical formatting,
   compact shell, skip link, focus treatment, explicit loading/empty/error patterns.
2. Real-data home snapshot; list breadth and filters; stock workspace with one
   forecast panel, readable metadata, truthful availability and single-step view.
3. Reusable chat messages/activity, source cards, copy action, refined composer,
   keyboard-friendly mobile conversation drawer, reduced-motion interactions.
4. Chart theme alignment, responsive browser inspection, targeted flow tests.

## Integrity boundaries

No new backend contracts, demo prices, fabricated evaluations, uncertainty bands,
buy/sell controls, or order books. Deployed LSTM horizon remains one observation.
Historical data is not live and dates may differ across stocks. Forecast charts
only show an estimate beside history when its as-of date matches the last bar;
the estimate is a separate diamond, never a fabricated future price path.
Missing confidence intervals and evaluation metrics are disclosed as unavailable.
News and agent results remain API data. Preserve all existing public paths and
the user's navigation preferences: logo opens home, two text navigation items,
subtle emerald active tab, icon-only theme toggle.

## Validation

Use the existing build/type check, streaming unit tests, and browser flows. Extend
browser coverage for home data, forecasting success, loading/error recovery,
keyboard navigation and mobile/light mode. Test fixtures stay in tests only.

Completed: production build and TypeScript check; three stream-parser unit cases;
three focused Playwright flows covering home error recovery, dated single-point
forecasts, market filtering, interval requests, export, unavailable forecasts,
suggestion handoff, progress/sources/comparison charts, conversation preservation,
theme persistence, mobile overflow and drawer keyboard containment. Screenshots
in `test-results/` cover desktop and mobile, dark and light themes. Browser tests
use an isolated Vite server on 5174 rather than the user's development server.

Only added runtime dependency: `@fontsource-variable/inter`, self-hosted typography
without external font requests. Existing Plotly is retained and loaded dynamically;
its full distribution still produces Vite's large-chunk warning. A tailored chart
bundle is a future optimization, requiring coverage of all agent plot types.

Backend contracts and deployment/auth settings are unchanged by this redesign.
Browser checks use API fixtures and do not establish live Azure model availability.
No new Azure deployment was performed. Forecast freshness depends on publishing
new market observations; historical evaluations and other horizons require backend
support. Chat history remains session-only, and streaming exposes tool activity
and a final answer rather than token-by-token prose.

## Rendered product review (second pass)

Reviewed Chromium renders at 1440x1000, 820x1180 and 390x844. The first design
still buried forecasts in a long metadata sidebar, repeated too many bordered
prompt cards, used undersized response text, and let default chart heights crowd
the chat viewport. Tablet forecast cards were particularly sparse.

Refinements: forecast-first horizontal result/evidence/context layout, tablet
reflow and mobile result hierarchy; expandable methodology; compact error/retry
layout; editorial prompt rows; larger assistant text and aligned reading/composer
widths; auto-growing single-line composer; quieter activity treatment; horizontal
default legends and compact chart sizing/date ticks. Existing explicit chart
layouts and all data traces remain intact.

Live local-backend inspection covered the 25-stock directory, BIAT candles, SMA,
volume and RSI, plus the stale-forecast state. It caught table overflow and a
timezone-induced observation-date shift. Both are fixed, with regression checks.
The application has no separate predictions route: this review improves the
forecast experience within stock analysis. Successful forecasts and completed
assistant turns were also exercised with controlled API fixtures, not live model
calls. Fixtures are test-only. After refinement, no page overflow was observed
across the four live pages at all three sizes. Dark and light screenshots were
inspected, including the final mobile forecast/error layouts.

Validation: production build/typecheck, four unit cases, three browser journeys
(including tablet/mobile table scrolling, plot-size settling and keyboard flow).
No additional runtime dependencies or backend changes. Forecast availability
still requires fresh published inputs; the UI explicitly reports that condition.
