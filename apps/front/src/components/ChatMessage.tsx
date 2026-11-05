import { useState } from "react";
import { Sparkles, Copy, Check, ExternalLink } from "lucide-react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Message } from "../types";
import { safeLink, dateLabel } from "../format";
import Plot from "./Plot";
import ExecutionStatus from "./ExecutionStatus";
export default function ChatMessage({ message }: { message: Message }) {
  const [copied, setCopied] = useState(false),
    [copyError, setCopyError] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setCopyError(false);
    } catch {
      setCopyError(true);
    }
  }
  return (
    <article
      className={message.role === "user" ? "user-message" : "assistant-message"}
      aria-label={
        message.role === "user" ? "Your message" : "Assistant response"
      }
    >
      {message.role === "assistant" && (
        <>
          <div className="message-heading">
            <span className="assistant-avatar">
              <Sparkles size={15} />
            </span>
            <strong>TuniScope</strong>
            <span className="message-duration">
              {message.elapsed !== undefined &&
                `Response time: ${message.elapsed.toFixed(1)}s`}
            </span>
          </div>
          <ExecutionStatus activity={message.activity || []} />
        </>
      )}
      <div className={`markdown ${message.error ? "negative" : ""}`}>
        <Markdown
          remarkPlugins={[remarkGfm]}
          components={{
            a: ({ children, href }) => (
              <a
                href={safeLink(href)}
                target="_blank"
                rel="noopener noreferrer"
              >
                {children}
              </a>
            ),
          }}
        >
          {message.content}
        </Markdown>
      </div>
      {message.charts?.map((chart, i) => (
        <figure className="panel message-chart" key={i}>
          {chart.title && <figcaption>{chart.title}</figcaption>}
          <Plot
            figure={chart.figure}
            description={
              chart.description ||
              chart.title ||
              "Assistant-generated market chart"
            }
          />
          {chart.description && <p>{chart.description}</p>}
        </figure>
      ))}
      {!!message.sources?.length && (
        <details className="source-details">
          <summary>Sources ({message.sources.length})</summary>
          <div className="source-grid">
            {message.sources.map((source, i) => (
              <a
                className="source-card"
                href={safeLink(source.link)}
                key={i}
                target="_blank"
                rel="noopener noreferrer"
              >
                <span className="source-index">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span>
                  <strong>{source.title || source.link || "Source"}</strong>
                  {source.date && <small>{dateLabel(source.date)}</small>}
                </span>
                <ExternalLink size={13} />
              </a>
            ))}
          </div>
        </details>
      )}
      {message.role === "assistant" && (
        <div className="message-actions">
          <button
            className="icon-button"
            aria-label={copied ? "Answer copied" : "Copy answer"}
            title={copied ? "Copied" : "Copy answer"}
            onClick={copy}
          >
            {copied ? <Check size={15} /> : <Copy size={15} />}
          </button>
          <span role="status">
            {copyError
              ? "Copy unavailable. Select the answer text to copy it."
              : copied
                ? "Copied to clipboard"
                : ""}
          </span>
        </div>
      )}
    </article>
  );
}
