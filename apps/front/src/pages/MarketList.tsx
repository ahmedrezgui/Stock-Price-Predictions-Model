import { useEffect, useMemo, useState } from "react";
import { ArrowUpRight, Search, RefreshCw, ArrowUpDown } from "lucide-react";
import { getJson } from "../api";
import { LoadingState, EmptyState, ErrorState } from "../components/States";
import type { MarketRow as Row } from "../types";
import { number } from "../format";

export default function MarketList({
  navigate,
}: {
  navigate: (page: string) => void;
}) {
  const [rows, setRows] = useState<Row[]>([]),
    [query, setQuery] = useState("");
  const [error, setError] = useState(""),
    [loading, setLoading] = useState(true),
    [refresh, setRefresh] = useState(0);
  const [sort, setSort] = useState<keyof Row>("symbol"),
    [asc, setAsc] = useState(true);
  const [filter, setFilter] = useState("All stocks");
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    getJson<Row[]>("/market-api/market", controller.signal)
      .then(setRows)
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [refresh]);
  const visible = useMemo(
    () =>
      rows
        .filter((r) => r.symbol.toLowerCase().includes(query.toLowerCase()))
        .filter(
          (r) =>
            filter === "All stocks" ||
            (filter === "Rising"
              ? (r.change_pct ?? 0) > 0
              : (r.change_pct ?? 0) < 0),
        )
        .sort((a, b) => {
          const x = a[sort],
            y = b[sort];
          if (x === null) return y === null ? 0 : 1;
          if (y === null) return -1;
          return (
            (typeof x === "number" && typeof y === "number"
              ? x - y
              : String(x).localeCompare(String(y))) * (asc ? 1 : -1)
          );
        }),
    [rows, query, sort, asc, filter],
  );
  const reorder = (key: keyof Row) => {
    setSort(key);
    setAsc(key === sort ? !asc : key === "symbol");
  };
  return (
    <main className="page-shell market-list-page">
      <div className="flex justify-between items-end gap-5 mb-8">
        <div>
          <p className="eyebrow">MARKETS / TUNISIA</p>
          <h1 className="text-3xl font-semibold mt-3">Market terminal</h1>
          <p className="text-muted text-sm mt-3">
            Tunisian stocks with deployed LSTM models. Select a stock to
            explore.
          </p>
        </div>
        <button
          className="icon-button"
          aria-label="Refresh stocks"
          onClick={() => setRefresh((n) => n + 1)}
        >
          <RefreshCw size={19} />
        </button>
      </div>
      {!loading && !error && rows.length > 0 && (
        <div className="market-breadth">
          <div>
            <span>Coverage</span>
            <strong>
              {rows.length}
              <small>deployed stocks</small>
            </strong>
          </div>
          <div>
            <span>Advancing</span>
            <strong className="positive">
              ↗ {rows.filter((r) => (r.change_pct ?? 0) > 0).length}
            </strong>
          </div>
          <div>
            <span>Declining</span>
            <strong className="negative">
              ↘ {rows.filter((r) => (r.change_pct ?? 0) < 0).length}
            </strong>
          </div>
          <p>
            Last available bar per stock.
            <br />
            Observation dates can differ.
          </p>
        </div>
      )}
      <section
        className="panel market-table-panel"
        aria-label="Deployed stock directory"
      >
        <div className="table-toolbar">
          <div className="flex items-center gap-3">
            <div className="segmented" aria-label="Filter stocks">
              {["All stocks", "Rising", "Falling"].map((f) => (
                <button
                  key={f}
                  aria-pressed={filter === f}
                  className={filter === f ? "selected" : ""}
                  onClick={() => setFilter(f)}
                >
                  {f}
                </button>
              ))}
            </div>
          </div>
          <label className="relative">
            <Search size={17} className="absolute left-3 top-3 text-muted" />
            <input
              className="pl-10!"
              aria-label="Search stocks"
              placeholder="Search stocks"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </label>
        </div>
        {error ? (
          <ErrorState message={error} retry={() => setRefresh((n) => n + 1)} />
        ) : loading ? (
          <LoadingState label="Loading market snapshot" />
        ) : (
          <div
            className="table-scroll"
            tabIndex={0}
            role="region"
            aria-label="Stock prices. Scroll horizontally to see all columns."
          >
            <table className="stock-table">
              <thead>
                <tr>
                  {(
                    [
                      ["symbol", "Stock"],
                      ["close", "Last close"],
                      ["change_pct", "Change / last bar"],
                      ["volume", "Volume / last bar"],
                      ["as_of", "Data as of"],
                    ] as [keyof Row, string][]
                  ).map(([key, label]) => (
                    <th
                      key={key}
                      aria-sort={
                        sort === key
                          ? asc
                            ? "ascending"
                            : "descending"
                          : "none"
                      }
                    >
                      <button
                        onClick={() => reorder(key)}
                        className="inline-flex items-center gap-2"
                      >
                        {label}
                        <ArrowUpDown size={12} />
                      </button>
                    </th>
                  ))}
                  <th>
                    <span className="sr-only">Action</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {visible.map((row) => (
                  <tr
                    key={row.symbol}
                    className="cursor-pointer"
                    onClick={() =>
                      navigate(`market/${encodeURIComponent(row.symbol)}`)
                    }
                  >
                    <td>
                      <a
                        href={`#market/${encodeURIComponent(row.symbol)}`}
                        className="flex items-center gap-3"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <span className="stock-avatar">
                          {row.symbol.slice(0, 2)}
                        </span>
                        <span>
                          <strong>{row.symbol}</strong>
                          <span className="block text-xs text-muted mt-1">
                            Tunisian equity
                          </span>
                        </span>
                      </a>
                    </td>
                    <td className="font-semibold tabular-nums">
                      {number(row.close)}
                    </td>
                    <td
                      className={
                        (row.change_pct ?? 0) >= 0
                          ? "text-accent"
                          : "text-rose-400"
                      }
                    >
                      {row.change_pct === null
                        ? "—"
                        : `${row.change_pct >= 0 ? "+" : ""}${number(row.change_pct, 2)}%`}
                    </td>
                    <td>{number(row.volume, 0)}</td>
                    <td className="text-muted text-xs">
                      {row.as_of || "Unavailable"}
                    </td>
                    <td>
                      <a
                        href={`#market/${encodeURIComponent(row.symbol)}`}
                        className="inline-flex items-center gap-2 text-sm font-medium"
                        onClick={(e) => e.stopPropagation()}
                        aria-label={`View ${row.symbol}`}
                      >
                        Details
                        <ArrowUpRight size={14} />
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!visible.length && (
              <EmptyState
                title={
                  rows.length
                    ? "No stocks match your filters"
                    : "No deployed stocks available"
                }
                detail={
                  rows.length
                    ? "Try another ticker or reset your filters."
                    : "Stocks appear when both model deployments and history are available."
                }
                action={
                  rows.length
                    ? {
                        label: "Reset filters",
                        run: () => {
                          setQuery("");
                          setFilter("All stocks");
                        },
                      }
                    : undefined
                }
              />
            )}
          </div>
        )}
        <div className="panel-footer">
          <span>
            {!loading && !error
              ? `${visible.length} stocks shown`
              : "Deployed stock directory"}
          </span>
          <span>Daily observations · Source price units</span>
        </div>
      </section>
      <p className="text-xs text-muted mt-6">
        Historical snapshots, not live quotes. Change compares the last two
        available observations; source price units are retained. Dates may
        differ by stock.
      </p>
    </main>
  );
}
