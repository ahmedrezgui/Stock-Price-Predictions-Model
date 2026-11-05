import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { Sun, Moon } from "lucide-react";
import Overview from "./pages/Overview";
import { ThemeProvider, useTheme } from "./theme";
import Market from "./pages/Market";
import MarketList from "./pages/MarketList";
import ChatPage from "./pages/Chat";
import "./styles.css";
import "@fontsource-variable/inter";

function App() {
  const { theme, toggle } = useTheme();
  const [page, setPage] = useState(location.hash.slice(1) || "overview");
  const [launch, setLaunch] = useState<{ id: string; question: string }>();
  const marketPage = page === "market" || page.startsWith("market/");
  const askQuestion = (question: string) => {
    setLaunch({ id: crypto.randomUUID(), question });
    navigate("chat");
  };
  const navigate = (next: string) => {
    location.hash = next;
    setPage(next);
  };
  useEffect(() => {
    const syncPage = () => setPage(location.hash.slice(1) || "overview");
    window.addEventListener("hashchange", syncPage);
    return () => window.removeEventListener("hashchange", syncPage);
  }, []);
  return (
    <div className="min-h-dvh">
      <a
        className="skip-link"
        href="#workspace"
        onClick={(e) => {
          e.preventDefault();
          document.getElementById("workspace")?.focus();
        }}
      >
        Skip to content
      </a>
      <header className="app-header">
        <button
          className="brand"
          aria-label="StockWise home"
          onClick={() => navigate("overview")}
        >
          <img src="/favicon.svg" alt="" className="brand-mark" />
          Stock<span>Wise</span>
          <span className="brand-dot" />
          <span className="brand-caption">RESEARCH WORKSPACE</span>
        </button>
        <nav aria-label="Main navigation">
          {[
            ["market", "Market terminal"],
            ["chat", "TuniScope"],
          ].map(([id, label]) => (
            <button
              key={id}
              className={
                page === id || (id === "market" && marketPage)
                  ? "nav-link active"
                  : "nav-link"
              }
              onClick={() => navigate(id)}
              aria-current={
                page === id || (id === "market" && marketPage)
                  ? "page"
                  : undefined
              }
            >
              {label}
            </button>
          ))}
        </nav>
        <button
          className="theme-toggle"
          onClick={toggle}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
        >
          {theme === "dark" ? <Sun size={20} /> : <Moon size={20} />}
        </button>
      </header>
      <div id="workspace" tabIndex={-1}>
        {/* Keep chat mounted when visiting the terminal so an active turn can finish. */}
        <div hidden={page !== "chat"}>
          <ChatPage launch={launch} visible={page === "chat"} />
        </div>
        {page === "market" && <MarketList navigate={navigate} />}
        {page.startsWith("market/") && (
          <Market
            key={page}
            initialSymbol={page.slice(7).toUpperCase()}
            navigate={navigate}
            askQuestion={askQuestion}
          />
        )}
        {page !== "chat" && !marketPage && (
          <Overview navigate={navigate} askQuestion={askQuestion} />
        )}
      </div>
    </div>
  );
}
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <App />
    </ThemeProvider>
  </StrictMode>,
);
