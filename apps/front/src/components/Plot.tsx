import { useEffect, useRef, useState } from "react";
import type { Figure } from "../types";
import { useTheme } from "../theme";
import { LoadingState } from "./States";

export default function Plot({
  figure,
  height,
  description = "Interactive market chart. Hover to inspect observations; use the chart toolbar to zoom or download.",
}: {
  figure: Figure;
  height?: number;
  description?: string;
}) {
  const { theme } = useTheme();
  const element = useRef<HTMLDivElement>(null);
  const [error, setError] = useState(false);
  const [loading, setLoading] = useState(true);
  const [compact, setCompact] = useState(false);
  useEffect(() => {
    const node = element.current;
    if (!node) return;
    const observer = new ResizeObserver(() => {
      if (node.clientWidth > 0) setCompact(node.clientWidth < 520);
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    let disposed = false;
    let cleanup = () => {};
    setError(false);
    setLoading(true);
    import("plotly.js-dist-min")
      .then(async ({ default: Plotly }) => {
        if (disposed || !element.current) return;
        const node = document.createElement("div");
        node.style.width = "100%";
        element.current.appendChild(node);
        cleanup = () => {
          Plotly.purge(node);
          node.remove();
        };
        // Plotly mutates figures; never let it mutate saved conversation state.
        const copy = structuredClone(figure);
        const tokens = getComputedStyle(document.documentElement);
        const ink = tokens.getPropertyValue("--color-muted").trim();
        const grid = tokens.getPropertyValue("--color-line").trim();
        const background = tokens.getPropertyValue("--color-panel").trim();
        copy.layout.xaxis ??= {};
        copy.layout.yaxis ??= {};
        for (const [key, value] of Object.entries(copy.layout)) {
          if (/^[xy]axis\d*$/.test(key) && value && typeof value === "object") {
            Object.assign(value, {
              gridcolor: grid,
              zerolinecolor: grid,
              tickfont: { color: ink },
            });
            if (compact && key.startsWith("xaxis")) {
              const axis = value as Partial<import("plotly.js").LayoutAxis>;
              if (Array.isArray(axis.tickvals) && axis.tickvals.length > 4) {
                const step = Math.ceil(axis.tickvals.length / 4);
                const labels = axis.ticktext;
                axis.tickvals = axis.tickvals.filter((_, i) => i % step === 0);
                if (Array.isArray(labels))
                  axis.ticktext = labels.filter((_, i) => i % step === 0);
              }
              if (
                !axis.ticktext &&
                Array.isArray(axis.tickvals) &&
                axis.tickvals.every((value) =>
                  /^\d{4}-\d{2}-\d{2}$/.test(String(value)),
                )
              ) {
                axis.ticktext = axis.tickvals.map((value) =>
                  String(value).slice(5),
                );
              }
              if (Array.isArray(axis.ticktext)) {
                axis.ticktext = axis.ticktext.map((label) =>
                  /^\d{4}-\d{2}-\d{2}$/.test(String(label))
                    ? String(label).slice(5)
                    : label,
                );
              }
              axis.tickangle = 0;
            }
          }
        }
        await Plotly.newPlot(
          node,
          copy.data,
          {
            margin: { l: 45, r: 25, t: 45, b: 45 },
            ...copy.layout,
            paper_bgcolor: background,
            plot_bgcolor: background,
            font: {
              ...copy.layout.font,
              color: ink,
              family: "Inter Variable, Inter, sans-serif",
              size: 11,
            },
            hoverlabel: {
              bgcolor: background,
              bordercolor: grid,
              font: { color: ink, family: "Inter Variable, Inter, sans-serif" },
            },
            modebar: {
              color: ink,
              activecolor: tokens.getPropertyValue("--color-accent").trim(),
              bgcolor: background,
            },
            legend: {
              orientation: "h",
              x: 0,
              y: 1.12,
              ...copy.layout.legend,
              font: { color: ink },
            },
            autosize: true,
            width: undefined,
            height:
              height ??
              (compact && copy.layout.height
                ? Math.min(copy.layout.height, 620)
                : copy.layout.height) ??
              320,
          },
          { responsive: true, displaylogo: false, scrollZoom: true },
        );
        if (disposed) {
          Plotly.purge(node);
          node.remove();
          return;
        }
        setLoading(false);
        const observer = new ResizeObserver(() => {
          if (node.isConnected && node.offsetWidth > 0) {
            Promise.resolve(Plotly.Plots.resize(node)).catch(() => {});
          }
        });
        observer.observe(element.current);
        cleanup = () => {
          observer.disconnect();
          Plotly.purge(node);
          node.remove();
        };
      })
      .catch(() => {
        if (!disposed) setError(true);
      });
    return () => {
      disposed = true;
      cleanup();
    };
  }, [figure, theme, height, compact]);
  return error ? (
    <p role="alert" className="notice">
      This chart could not be displayed.
    </p>
  ) : (
    <div
      className="plot-frame"
      role="group"
      aria-label={description}
      aria-busy={loading}
    >
      {loading && (
        <div className="plot-loading">
          <LoadingState label="Rendering chart" />
        </div>
      )}
      <div
        ref={element}
        className="w-full min-w-0"
        style={{ minHeight: height ?? 300 }}
        aria-label="Interactive market chart"
      />
    </div>
  );
}
