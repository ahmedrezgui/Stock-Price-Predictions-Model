import { useEffect, useState } from "react";
import {
  ArrowUpRight,
  ArrowRight,
  ChartNoAxesCombined,
  Sparkles,
  Layers,
  Search,
} from "lucide-react";
import { getJson } from "../api";
import type { MarketRow, Figure } from "../types";
import { number, percent, dateLabel } from "../format";
import { EmptyState, ErrorState, LoadingState } from "../components/States";
import Plot from "../components/Plot";

export default function Overview({
  navigate,
  askQuestion,
}: {
  navigate: (page: string) => void;
  askQuestion: (question: string) => void;
}) {
  const [rows, setRows] = useState<MarketRow[]>([]),
    [selected, setSelected] = useState("");
  const [loading, setLoading] = useState(true),
    [error, setError] = useState(""),
    [retry, setRetry] = useState(0);
  const [figure, setFigure] = useState<Figure>(),
    [chartError, setChartError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    getJson<MarketRow[]>("/market-api/market", controller.signal)
      .then((data) => {
        setRows(data);
        setSelected((current) =>
          data.some((r) => r.symbol === current)
            ? current
            : data[0]?.symbol || "",
        );
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [retry]);
  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();
    setFigure(undefined);
    setChartError("");
    getJson<Figure>(
      `/market-api/stock/${encodeURIComponent(selected)}/chart?style=Line&period=3M&momentum=None&volume=false&sma=&ema=`,
      controller.signal,
    )
      .then(setFigure)
      .catch((e) => {
        if (!controller.signal.aborted) setChartError(e.message);
      });
    return () => controller.abort();
  }, [selected, retry]);
  const stock = rows.find((r) => r.symbol === selected);
  return (
    <main className="page-shell overview-page">
      <section className="overview-intro">
        <div>
          <p className="eyebrow">STOCKWISE / FINANCIAL INTELLIGENCE</p>
          <h1>
            Your market.
            <br />
            <span>A clearer perspective.</span>
          </h1>
          <p>
            From price action to model outlook. Bring your Tunisian market
            research into one focused workspace.
          </p>
          <button className="primary" onClick={() => navigate("market")}>
            Explore stocks <ArrowRight size={16} />
          </button>
        </div>
        <div className="intro-note">
          <Layers size={23} />
          <p>History. Forecasts. Context.</p>
          <span>
            Real market observations and deployed LSTM models, connected to an
            assistant that can explain the evidence.
          </span>
          <div className="flex gap-2 mt-5">
            <span className="badge">DAILY OBSERVATIONS</span>
            <span className="badge ai-badge">AI RESEARCH</span>
          </div>
        </div>
      </section>
      <section className="overview-workspace">
        <div className="panel min-w-0">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Market lens</p>
              <h2>Start with a stock</h2>
            </div>
            <ChartNoAxesCombined size={20} className="text-muted" />
          </div>
          {error ? (
            <ErrorState message={error} retry={() => setRetry((n) => n + 1)} />
          ) : loading ? (
            <LoadingState chart label="Loading market snapshot" />
          ) : !rows.length ? (
            <EmptyState
              title="No deployed stocks yet"
              detail="Stocks appear here when deployed models and history are available."
            />
          ) : (
            <>
              <div className="overview-quote">
                <label className="ticker-picker">
                  <Search size={15} />
                  <select
                    aria-label="Overview stock"
                    value={selected}
                    onChange={(e) => setSelected(e.target.value)}
                  >
                    {rows.map((row) => (
                      <option key={row.symbol}>{row.symbol}</option>
                    ))}
                  </select>
                </label>
                <div className="ml-auto text-right">
                  <strong className="quote-value">
                    {number(stock?.close)}
                  </strong>
                  <span
                    className={
                      (stock?.change_pct ?? 0) < 0 ? "negative" : "positive"
                    }
                  >
                    {percent(stock?.change_pct)} <small>last bar</small>
                  </span>
                </div>
              </div>
              {chartError ? (
                <ErrorState
                  message={chartError}
                  retry={() => setRetry((n) => n + 1)}
                />
              ) : figure ? (
                <Plot
                  figure={figure}
                  height={300}
                  description={`${selected}: closing prices over the latest three months of available history.`}
                />
              ) : (
                <LoadingState chart label="Loading price history" />
              )}
              <div className="panel-footer">
                <span>
                  As of {dateLabel(stock?.as_of)} · Source price units
                </span>
                <button
                  className="text-link"
                  onClick={() => navigate(`market/${selected}`)}
                >
                  Analyze {selected}
                  <ArrowUpRight size={14} />
                </button>
              </div>
            </>
          )}
        </div>
        <aside className="overview-research">
          <div className="ai-emblem">
            <Sparkles size={23} />
          </div>
          <p className="eyebrow ai-text mt-5">TuniScope / Research assistant</p>
          <h2>
            A question is a<br />
            good place to start.
          </h2>
          <p>
            Ask for an explanation, not just a number. Follow the tools, inspect
            the sources, and explore the charts behind each answer.
          </p>
          <button
            className="research-question"
            onClick={() =>
              askQuestion(
                selected
                  ? `Explain ${selected}'s latest deployed LSTM forecast and the date of its inputs.`
                  : "Which Tunisian stocks have deployed LSTM forecasts available?",
              )
            }
          >
            <span>
              {selected
                ? `What does the model suggest for ${selected}?`
                : "Which stocks can I explore?"}
            </span>
            <ArrowUpRight size={16} />
          </button>
          <button
            className="research-question"
            onClick={() =>
              askQuestion(
                selected
                  ? `Compare ${selected} with other available Tunisian stocks, using shared observation dates.`
                  : "Explain how to interpret the deployed LSTM forecasts and their limitations.",
              )
            }
          >
            <span>
              {selected
                ? `How does ${selected} compare with its peers?`
                : "How should I read a forecast?"}
            </span>
            <ArrowUpRight size={16} />
          </button>
          <button
            className="text-link ai-text"
            onClick={() => navigate("chat")}
          >
            Open TuniScope
            <ArrowRight size={15} />
          </button>
        </aside>
      </section>
      <section className="workflow-strip">
        {[
          [
            "01",
            "Explore the evidence",
            "Price, volume and technical indicators across observed market history.",
          ],
          [
            "02",
            "Read the model",
            "A deployed one-step LSTM estimate, with dates and limitations in view.",
          ],
          [
            "03",
            "Put it in context",
            "Ask TuniScope to research news, compare stocks and explain its sources.",
          ],
        ].map(([n, title, detail]) => (
          <div key={n}>
            <span>{n}</span>
            <div>
              <h3>{title}</h3>
              <p>{detail}</p>
            </div>
          </div>
        ))}
      </section>
      <p className="page-footnote">
        Research workspace · Historical observations, not live quotes. Forecasts
        are estimates, not guarantees.
      </p>
    </main>
  );
}
