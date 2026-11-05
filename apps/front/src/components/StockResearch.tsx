import { useEffect, useState } from "react";
import { ArrowUpRight, Sparkles } from "lucide-react";
import { getJson } from "../api";
type News = {
  articles: { title: string; link: string; publisher: string; date?: string }[];
  unavailable: boolean;
  checked_at: string;
};
export default function StockResearch({
  symbol,
  askQuestion,
}: {
  symbol: string;
  askQuestion: (question: string) => void;
}) {
  const [news, setNews] = useState<News>(),
    [error, setError] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    setNews(undefined);
    setError(false);
    getJson<News>(
      `/market-api/stock/${encodeURIComponent(symbol)}/news`,
      controller.signal,
    )
      .then(setNews)
      .catch(() => {
        if (!controller.signal.aborted) setError(true);
      });
    return () => controller.abort();
  }, [symbol]);
  const questions = [
    `What recent news could affect ${symbol}? Include dated sources.`,
    `Explain ${symbol}'s latest deployed LSTM forecast and its limitations.`,
    `What do the volume and moving averages suggest for ${symbol}?`,
    `Compare ${symbol} with other available Tunisian stocks, rebased to 100.`,
    `How has ${symbol} performed over its last 20 trading observations?`,
    `What risks should I consider when interpreting ${symbol}'s forecast?`,
  ];
  return (
    <>
      <section className="research-launch">
        <div>
          <p className="eyebrow ai-text">CONTINUE YOUR RESEARCH</p>
          <h2 className="text-xl font-semibold flex items-center gap-3">
            <Sparkles className="ai-text" size={20} />
            Ask TuniScope about {symbol}
          </h2>
          <p className="text-sm text-muted mt-2 mb-5">
            Start a stock-focused conversation with a question below.
          </p>
        </div>
        <div className="research-prompts">
          {questions.map((q) => (
            <button
              key={q}
              className="research-question"
              onClick={() => askQuestion(q)}
            >
              <span>{q}</span>
              <ArrowUpRight size={17} className="shrink-0 text-accent" />
            </button>
          ))}
        </div>
      </section>
      <section className="py-10">
        <h2 className="text-2xl font-semibold">Recent {symbol} news</h2>
        <p className="text-xs text-muted mt-2">
          Web search results, with source links. Coverage and publication dates
          may be incomplete.
        </p>
        {error || news?.unavailable ? (
          <p className="notice">
            News is temporarily unavailable. Please try again later.
          </p>
        ) : !news ? (
          <p role="status" className="py-8 text-muted">
            Finding recent articles…
          </p>
        ) : !news.articles.length ? (
          <p className="py-8 text-muted">
            No recent articles were found for {symbol}.
          </p>
        ) : (
          <div className="grid md:grid-cols-2 gap-x-12">
            {news.articles.map((article) => (
              <article key={article.link} className="py-6 border-b border-line">
                <a
                  className="font-medium leading-relaxed hover:text-accent"
                  href={
                    /^https?:\/\//.test(article.link) ? article.link : undefined
                  }
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {article.title}
                  <ArrowUpRight size={14} className="inline ml-2" />
                </a>
                <p className="text-xs text-muted mt-3">
                  {article.publisher} ·{" "}
                  {article.date
                    ? article.date.slice(0, 16).replace("T", " ")
                    : "Date unavailable"}
                </p>
              </article>
            ))}
          </div>
        )}
      </section>
    </>
  );
}
