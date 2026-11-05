import { useEffect, useMemo, useState } from "react";
import { RefreshCw, Download, SlidersHorizontal } from "lucide-react";
import { getJson } from "../api";
import Plot from "../components/Plot";
import ForecastPanel from "../components/ForecastPanel";
import { LoadingState, ErrorState } from "../components/States";
import { number, dateLabel } from "../format";
import StockResearch from "../components/StockResearch";
import type { Figure, Stock } from "../types";

export default function Market({
  navigate,
  initialSymbol,
  askQuestion,
}: {
  navigate: (page: string) => void;
  initialSymbol: string;
  askQuestion: (question: string) => void;
}) {
  const [companies, setCompanies] = useState<string[]>([]),
    [symbol] = useState(initialSymbol);
  const [stock, setStock] = useState<Stock>(),
    [figure, setFigure] = useState<Figure>();
  const [error, setError] = useState(""),
    [chartError, setChartError] = useState("");
  const [refresh, setRefresh] = useState(0),
    [catalogLoading, setCatalogLoading] = useState(true);
  const [period, setPeriod] = useState("3M"),
    [interval, setInterval] = useState("Day"),
    [style, setStyle] = useState("Candles"),
    [momentum, setMomentum] = useState("RSI");
  const [sma, setSma] = useState([20, 50]),
    [ema, setEma] = useState<number[]>([]),
    [bands, setBands] = useState(false),
    [volume, setVolume] = useState(true);
  useEffect(() => {
    const controller = new AbortController();
    setError("");
    setCatalogLoading(true);
    getJson<string[]>("/market-api/companies", controller.signal)
      .then((list) => {
        setCompanies(list);
      })
      .catch((e) => {
        if (!controller.signal.aborted) {
          setError(e.message);
          setCompanies([]);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setCatalogLoading(false);
      });
    return () => controller.abort();
  }, [refresh]);
  useEffect(() => {
    if (!symbol || catalogLoading) return;
    const controller = new AbortController();
    setStock(undefined);
    setError("");
    getJson<Stock>(
      `/market-api/stock/${encodeURIComponent(symbol)}`,
      controller.signal,
    )
      .then((data) => {
        if (!data.data.length)
          throw new Error("No history is available for this stock.");
        setStock(data);
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      });
    return () => controller.abort();
  }, [symbol, refresh, catalogLoading]);
  useEffect(() => {
    if (!symbol || catalogLoading) return;
    const controller = new AbortController();
    setFigure(undefined);
    setChartError("");
    const query = new URLSearchParams({
      period,
      interval,
      style,
      momentum,
      sma: sma.join(","),
      ema: ema.join(","),
      bands: String(bands),
      volume: String(volume),
    });
    getJson<Figure>(
      `/market-api/stock/${encodeURIComponent(symbol)}/chart?${query}`,
      controller.signal,
    )
      .then(setFigure)
      .catch((e) => {
        if (!controller.signal.aborted) setChartError(e.message);
      });
    return () => controller.abort();
  }, [
    symbol,
    refresh,
    catalogLoading,
    period,
    interval,
    style,
    momentum,
    sma,
    ema,
    bands,
    volume,
  ]);
  const rows = useMemo(
    () =>
      stock ? [...stock.data].sort((a, b) => a.date.localeCompare(b.date)) : [],
    [stock],
  );
  const last = rows.at(-1),
    previous = rows.at(-2);
  const change =
    last && previous ? (last.cloture / previous.cloture - 1) * 100 : undefined;
  const recent = last
    ? rows.filter(
        (row) =>
          new Date(row.date).getTime() >=
          new Date(last.date).getTime() - 365.25 * 86400000,
      )
    : [];
  function download() {
    if (!stock) return;
    const escape = (value: unknown) =>
      '"' + String(value ?? "").replaceAll('"', '""') + '"';
    const csv = [
      stock.columns.join(","),
      ...rows.map((row) =>
        stock.columns
          .map((key) => escape(row[key as keyof typeof row]))
          .join(","),
      ),
    ].join("\r\n");
    const url = URL.createObjectURL(
      new Blob([csv], { type: "text/csv;charset=utf-8" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `${symbol}-history.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return (
    <main className="page-shell stock-page">
      <div className="stock-heading flex justify-between items-end mb-4">
        <div>
          <button
            className="text-sm text-muted hover:text-accent mb-4"
            onClick={() => navigate("market")}
          >
            ← All stocks
          </button>
          <h1 className="text-3xl font-semibold flex items-center gap-3">
            <span className="stock-avatar">{symbol.slice(0, 2)}</span>
            {symbol}
          </h1>
          <p className="text-xs text-muted mt-2">
            Tunisian equity · Deployed LSTM
          </p>
        </div>
        <button className="secondary" onClick={() => setRefresh((n) => n + 1)}>
          <RefreshCw size={15} /> Refresh
        </button>
      </div>
      {error && (
        <ErrorState message={error} retry={() => setRefresh((n) => n + 1)} />
      )}
      {catalogLoading ? (
        <p className="text-muted py-10" role="status">
          Loading deployed stocks…
        </p>
      ) : !companies.length && !error ? (
        <p className="notice">
          No stocks with deployed models and available history.
        </p>
      ) : companies.length > 0 ? (
        <>
          <div className="stock-selector flex flex-wrap gap-4 items-center mb-3">
            <label className="flex items-center gap-3 text-sm text-muted">
              Stock
              <select
                aria-label="Stock"
                value={symbol}
                onChange={(e) =>
                  navigate(`market/${encodeURIComponent(e.target.value)}`)
                }
              >
                {companies.map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>
            </label>
            <span className="text-xs text-muted">
              {companies.length} stocks · Historical snapshot
            </span>
          </div>
          {!stock && !error && (
            <p role="status" className="text-muted py-10">
              Loading history and forecast…
            </p>
          )}
          {stock && last && (
            <>
              <div className="stock-metrics">
                {[
                  [
                    `${symbol} / Last close`,
                    number(last.cloture),
                    change === undefined
                      ? ""
                      : `${change >= 0 ? "+" : ""}${number(change, 2)}% vs previous bar`,
                  ],
                  [
                    "Last volume",
                    number(last.volume, 0),
                    "Observed trading volume",
                  ],
                  [
                    "1Y closing range",
                    `${number(Math.min(...recent.map((r) => r.cloture)), 2)} – ${number(Math.max(...recent.map((r) => r.cloture)), 2)}`,
                    "Source price units",
                  ],
                  [
                    "Data as of",
                    dateLabel(last.date),
                    "Historical observation",
                  ],
                ].map(([label, value, detail]) => (
                  <section className="stock-metric" key={label}>
                    <p className="text-xs text-muted">{label}</p>
                    <p className="text-xl md:text-2xl font-semibold my-3 tabular-nums">
                      {value}
                    </p>
                    <p className="text-xs text-muted">{detail}</p>
                  </section>
                ))}
              </div>
              <div className="stock-detail-grid">
                <ForecastPanel
                  stock={stock}
                  symbol={symbol}
                  rows={rows}
                  retry={() => setRefresh((n) => n + 1)}
                  askQuestion={askQuestion}
                />
                <section className="panel price-panel overflow-hidden min-w-0">
                  <div className="p-5 border-b border-line flex flex-wrap items-center justify-between gap-4">
                    <h2 className="font-medium">
                      {symbol}{" "}
                      <span className="text-muted ml-2 font-normal">
                        Price action
                      </span>
                    </h2>
                    <div
                      className="flex flex-wrap gap-1"
                      aria-label="Chart range"
                    >
                      {["1M", "3M", "6M", "1Y", "5Y", "All"].map((p) => (
                        <button
                          key={p}
                          className={period === p ? "range active" : "range"}
                          aria-pressed={period === p}
                          onClick={() => setPeriod(p)}
                        >
                          {p}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="chart-controls flex flex-wrap gap-4 p-5">
                    {[
                      [
                        "Bar interval",
                        interval,
                        setInterval,
                        ["Day", "Month", "Year"],
                      ],
                      ["Price style", style, setStyle, ["Candles", "Line"]],
                      [
                        "Momentum",
                        momentum,
                        setMomentum,
                        ["RSI", "MACD", "None"],
                      ],
                    ].map(([label, value, setter, choices]) => (
                      <label className="field" key={String(label)}>
                        {String(label)}
                        <select
                          value={String(value)}
                          onChange={(e) =>
                            (setter as (v: string) => void)(e.target.value)
                          }
                        >
                          {(choices as string[]).map((v) => (
                            <option key={v}>{v}</option>
                          ))}
                        </select>
                      </label>
                    ))}
                  </div>
                  <details className="px-5 pb-3">
                    <summary className="text-sm text-muted cursor-pointer">
                      <SlidersHorizontal size={14} className="inline mr-2" />
                      Indicators & overlays
                    </summary>
                    <div className="flex flex-wrap gap-7 py-4">
                      {[
                        ["SMA", sma, setSma],
                        ["EMA", ema, setEma],
                      ].map(([name, selected, setter]) => (
                        <fieldset key={String(name)}>
                          <legend className="text-xs text-muted mb-2">
                            {String(name)} windows
                          </legend>
                          <div className="flex gap-3 flex-wrap">
                            {[5, 10, 20, 50, 100, 200].map((w) => (
                              <label className="text-sm flex gap-1" key={w}>
                                <input
                                  type="checkbox"
                                  checked={(selected as number[]).includes(w)}
                                  onChange={(e) =>
                                    (setter as (v: number[]) => void)(
                                      e.target.checked
                                        ? [...(selected as number[]), w]
                                        : (selected as number[]).filter(
                                            (v) => v !== w,
                                          ),
                                    )
                                  }
                                />
                                {w}
                              </label>
                            ))}
                          </div>
                        </fieldset>
                      ))}
                      <label className="text-sm">
                        <input
                          type="checkbox"
                          checked={bands}
                          onChange={(e) => setBands(e.target.checked)}
                        />{" "}
                        Bollinger Bands
                      </label>
                      <label className="text-sm">
                        <input
                          type="checkbox"
                          checked={volume}
                          onChange={(e) => setVolume(e.target.checked)}
                        />{" "}
                        Volume
                      </label>
                    </div>
                    <p className="text-xs text-muted">
                      Windows count selected bars. Indicators use full history
                      before cropping. Monthly/yearly edge periods can be
                      partial; the forecast horizon is unchanged.
                    </p>
                  </details>
                  {chartError ? (
                    <ErrorState
                      message={chartError}
                      retry={() => setRefresh((n) => n + 1)}
                    />
                  ) : figure ? (
                    <Plot
                      figure={figure}
                      description={`${symbol}: ${interval.toLowerCase()} historical price bars with selected technical indicators. The forecast is shown separately.`}
                    />
                  ) : (
                    <LoadingState chart label="Preparing chart" />
                  )}
                  {!!(
                    stock.quality?.invalid_ohlc_rows ||
                    stock.quality?.missing_volume_rows
                  ) && (
                    <p className="text-xs text-muted p-5">
                      {stock.quality?.invalid_ohlc_rows || 0} bars have
                      incomplete OHLC; {stock.quality?.missing_volume_rows || 0}{" "}
                      volume values are unavailable. Use line mode to see
                      closing prices.
                    </p>
                  )}
                </section>
              </div>
              <section className="py-8 border-b border-line">
                <h2 className="text-xl font-semibold mb-5">
                  {symbol} market information
                </h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                  {[
                    ["Last observed open", number(last.ouverture)],
                    ["Last observed high", number(last.haut)],
                    ["Last observed low", number(last.bas)],
                    ["Previous close", number(previous?.cloture)],
                  ].map(([label, value]) => (
                    <div key={label}>
                      <p className="text-xs text-muted mb-2">{label}</p>
                      <strong className="text-sm">{value}</strong>
                    </div>
                  ))}
                </div>
                <p className="text-xs text-muted mt-5">
                  Snapshot: {last.date.slice(0, 10)}. Daily observations; no
                  intraday feed or order-book data.
                </p>
              </section>
              <StockResearch symbol={symbol} askQuestion={askQuestion} />
              <details className="panel p-5 mt-5">
                <summary className="cursor-pointer text-sm">
                  Historical data / export
                </summary>
                <button className="secondary my-4" onClick={download}>
                  <Download size={15} />
                  Download CSV
                </button>
                <div className="overflow-auto max-h-80">
                  <table className="w-full text-sm text-right">
                    <thead>
                      <tr>
                        {stock.columns.map((c) => (
                          <th className="p-3 text-muted" key={c}>
                            {c}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {[...rows].reverse().map((row) => (
                        <tr className="border-t border-line" key={row.date}>
                          {stock.columns.map((c) => (
                            <td className="p-3" key={c}>
                              {c === "date"
                                ? row.date.slice(0, 10)
                                : number(row[c as keyof typeof row])}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            </>
          )}
        </>
      ) : null}
    </main>
  );
}
