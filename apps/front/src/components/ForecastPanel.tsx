import { useMemo } from "react";
import { ArrowUpRight, Clock3, Info, Sparkles, RefreshCw } from "lucide-react";
import type { Bar, Stock, Figure } from "../types";
import { dateLabel, number, percent } from "../format";
import Plot from "./Plot";
import { useTheme } from "../theme";

export default function ForecastPanel({
  stock,
  symbol,
  rows,
  retry,
  askQuestion,
}: {
  stock: Stock;
  symbol: string;
  rows: Bar[];
  retry: () => void;
  askQuestion: (question: string) => void;
}) {
  const forecast = stock.forecast;
  const { theme } = useTheme();
  const chart = useMemo<Figure | undefined>(() => {
    if (
      !forecast ||
      typeof forecast.implied_price !== "number" ||
      !Number.isFinite(forecast.implied_price) ||
      rows.at(-1)?.date.slice(0, 10) !== forecast.as_of?.slice(0, 10)
    )
      return;
    const recent = rows.slice(-12),
      dates = recent.map((r) => r.date.slice(0, 10));
    const tokens = getComputedStyle(document.documentElement);
    return {
      data: [
        {
          type: "scatter",
          x: dates,
          y: recent.map((r) => r.cloture),
          mode: "lines",
          name: "Observed close",
          line: {
            color: tokens.getPropertyValue("--color-accent").trim(),
            width: 2,
          },
        },
        {
          type: "scatter",
          x: ["Next observation"],
          y: [forecast.implied_price],
          mode: "markers",
          name: "LSTM estimate",
          marker: {
            color: tokens.getPropertyValue("--color-link").trim(),
            size: 10,
            symbol: "diamond",
          },
        },
      ],
      layout: {
        height: 210,
        margin: { l: 35, r: 15, t: 35, b: 30 },
        legend: { orientation: "h", x: 0, y: 1.2 },
        xaxis: {
          type: "category",
          categoryorder: "array",
          categoryarray: [...dates, "Next observation"],
          tickvals: [dates[0], dates.at(-1)!, "Next observation"],
          ticktext: [dateLabel(dates[0]).slice(0, 6), "", "Next"],
          tickangle: 0,
        },
        yaxis: { tickformat: ".3~f", zeroline: false },
        shapes: [
          {
            type: "line",
            xref: "x",
            yref: "paper",
            x0: dates.at(-1)!,
            x1: dates.at(-1)!,
            y0: 0,
            y1: 1,
            line: { color: "#64748b", dash: "dot", width: 1 },
          },
        ],
      },
    };
  }, [forecast, rows, theme]);
  return (
    <section
      className="panel forecast-panel"
      aria-label={`${symbol} LSTM forecast`}
    >
      <div className="section-heading">
        <div>
          <p className="eyebrow">MODEL OUTLOOK / {symbol}</p>
          <h2>Next-observation forecast</h2>
        </div>
        <div className="forecast-config">
          <span className="badge">LSTM</span>
          <span>
            <Clock3 size={14} />
            {forecast?.forecast_horizon_observations ?? 1} observed bar
          </span>
        </div>
      </div>
      {forecast ? (
        <div className="forecast-body">
          <div className="forecast-result">
            <p className="text-muted">Implied next price</p>
            <p className="forecast-value">{number(forecast.implied_price)}</p>
            <p
              className={
                (forecast.predicted_log_return ?? 0) < 0
                  ? "negative"
                  : "positive"
              }
            >
              {typeof forecast.predicted_log_return === "number"
                ? percent(Math.expm1(forecast.predicted_log_return) * 100, 3)
                : "Return unavailable"}
              <span className="return-label">estimated return</span>
            </p>
            <p className="forecast-origin">
              Source price units · {forecast.price_basis || "unverified"} basis
            </p>
          </div>
          <div className="forecast-visual">
            {chart ? (
              <>
                <Plot
                  figure={chart}
                  height={210}
                  description={`${symbol}: recent observed closes and one next-observation LSTM estimate. Dotted boundary marks the last observation; no future path is implied.`}
                />
                <p className="chart-caption">
                  Diamond: model estimate. Dotted line: last observed bar.
                </p>
              </>
            ) : (
              <p className="forecast-note">
                The forecast origin does not match the latest history, or its
                implied price is unavailable. No forecast point is plotted.
              </p>
            )}
          </div>
          <div className="forecast-context">
            <dl className="forecast-facts">
              <div>
                <dt>Forecast as of</dt>
                <dd>{dateLabel(forecast.as_of)}</dd>
              </div>
              {forecast.forecast_for && (
                <div>
                  <dt>Target date</dt>
                  <dd>{dateLabel(forecast.forecast_for)}</dd>
                </div>
              )}
              <div>
                <dt>Horizon basis</dt>
                <dd>
                  {forecast.horizon_basis === "exchange_sessions"
                    ? "Exchange sessions"
                    : "Observed bars"}
                </dd>
              </div>
              <div>
                <dt>Price basis</dt>
                <dd>{forecast.price_basis || "Unverified"}</dd>
              </div>
            </dl>
            <details className="model-details">
              <summary>Model & methodology</summary>
              <p className="methodology-note">
                News is not an input to this LSTM estimate. Confidence intervals
                and historical evaluation metrics are not provided by this
                endpoint. Other horizons and model comparisons are not currently
                served.
              </p>
              <dl>
                <dt>Version</dt>
                <dd>{forecast.model_version || "Unavailable"}</dd>
                <dt>Deployment</dt>
                <dd>
                  {String(
                    forecast.deployment_status || "Unavailable",
                  ).replaceAll("_", " ")}
                </dd>
                <dt>Predicted log return</dt>
                <dd>{number(forecast.predicted_log_return, 6)}</dd>
              </dl>
            </details>
          </div>
        </div>
      ) : (
        <div className="forecast-unavailable">
          <div className="status-icon">
            <Clock3 size={22} />
          </div>
          <div>
            <h3>Forecast not ready</h3>
            <p>
              The deployed forecast is temporarily unavailable. History is still
              available.
            </p>
            {stock.forecast_error && (
              <p className="text-xs">{stock.forecast_error}</p>
            )}
          </div>
          <button className="secondary" onClick={retry}>
            <RefreshCw size={14} />
            Retry forecast
          </button>
        </div>
      )}
      <div className="forecast-bottom">
        <p>
          <Info size={14} />
          One-step model estimate, not a guaranteed outcome. No confidence
          interval supplied.
        </p>
        <button
          className="ai-action"
          onClick={() =>
            askQuestion(
              `Explain ${symbol}'s latest deployed LSTM forecast and its limitations.`,
            )
          }
        >
          <Sparkles size={16} />
          Explain with TuniScope
          <ArrowUpRight size={15} />
        </button>
      </div>
    </section>
  );
}
