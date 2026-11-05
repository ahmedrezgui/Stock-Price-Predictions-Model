import { useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  Square,
  Plus,
  Trash2,
  PanelLeftClose,
  PanelLeftOpen,
  Sparkles,
  Search,
  ArrowUpRight,
} from "lucide-react";
import { ask } from "../api";
import ChatMessage from "../components/ChatMessage";
import ExecutionStatus from "../components/ExecutionStatus";
import type { Activity, Chat, Message } from "../types";

const createChat = (): Chat => ({
  id: crypto.randomUUID(),
  title: "New conversation",
  messages: [],
});
const suggestions = [
  [
    "Explore a forecast",
    "Plot BIAT and explain its latest deployed LSTM forecast.",
  ],
  [
    "Compare stocks",
    "Compare BIAT, AL and ATB over their shared recent history, rebased to 100.",
  ],
  [
    "Read the market",
    "What recent news is available for BIAT? Include dates and sources.",
  ],
];
export default function ChatPage({
  launch,
  visible,
}: {
  launch?: { id: string; question: string };
  visible: boolean;
}) {
  // Session-only history mirrors the prior UI. Server checkpoints are not durable.
  const [chats, setChats] = useState<Chat[]>(() => [createChat()]);
  const [active, setActive] = useState(""),
    [sidebar, setSidebar] = useState(
      () => window.matchMedia("(min-width: 768px)").matches,
    ),
    [search, setSearch] = useState("");
  const [input, setInput] = useState(""),
    [busy, setBusy] = useState(false),
    [activity, setActivity] = useState<Activity[]>([]);
  const [label, setLabel] = useState("Starting research…"),
    [elapsed, setElapsed] = useState(0);
  const controller = useRef<AbortController | null>(null),
    end = useRef<HTMLDivElement>(null),
    textarea = useRef<HTMLTextAreaElement>(null);
  const chat = chats.find((c) => c.id === active) || chats[0];
  const sidebarElement = useRef<HTMLElement>(null);
  const openSidebarButton = useRef<HTMLButtonElement>(null);
  const followLatest = useRef(true);
  useEffect(() => {
    if (!visible || !sidebar || !matchMedia("(max-width: 767px)").matches)
      return;
    const node = sidebarElement.current;
    const first = node?.querySelector<HTMLButtonElement>("button");
    first?.focus();
    const keydown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setSidebar(false);
        requestAnimationFrame(() => openSidebarButton.current?.focus());
      }
      if (event.key !== "Tab" || !node) return;
      const controls = [
        ...node.querySelectorAll<HTMLElement>(
          "button:not(:disabled), input, a[href]",
        ),
      ];
      const firstControl = controls[0],
        lastControl = controls.at(-1);
      if (event.shiftKey && document.activeElement === firstControl) {
        event.preventDefault();
        lastControl?.focus();
      } else if (!event.shiftKey && document.activeElement === lastControl) {
        event.preventDefault();
        firstControl?.focus();
      }
    };
    document.addEventListener("keydown", keydown);
    return () => document.removeEventListener("keydown", keydown);
  }, [sidebar, visible]);
  const handledLaunch = useRef<string | undefined>(undefined);
  useEffect(() => {
    if (!launch || busy || handledLaunch.current === launch.id) return;
    handledLaunch.current = launch.id;
    setChats((all) => [
      ...all.filter((c) => c.messages.length),
      { id: launch.id, title: launch.question, messages: [] },
    ]);
    setActive(launch.id);
    void send(launch.question, launch.id);
  }, [launch, busy]);
  useEffect(() => () => controller.current?.abort(), []);
  useEffect(() => {
    if (followLatest.current)
      end.current?.scrollIntoView({
        behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "instant"
          : "smooth",
        block: "end",
      });
  }, [chat.messages.length, activity.length]);
  function append(id: string, message: Message) {
    setChats((all) =>
      all.map((c) =>
        c.id === id ? { ...c, messages: [...c.messages, message] } : c,
      ),
    );
  }
  function newChat() {
    const next = createChat();
    setChats((all) => [...all.filter((c) => c.messages.length), next]);
    setActive(next.id);
    setInput("");
    if (textarea.current) textarea.current.style.height = "auto";
    textarea.current?.focus();
  }
  async function send(question = input, targetId = chat.id) {
    question = question.trim();
    if (!question || question.length > 16000 || controller.current) return;
    const id = targetId,
      request = new AbortController();
    controller.current = request;
    followLatest.current = true;
    setChats((all) =>
      all.map((c) =>
        c.id === id
          ? { ...c, title: c.messages.length ? c.title : question }
          : c,
      ),
    );
    append(id, { role: "user", content: question });
    setInput("");
    if (textarea.current) textarea.current.style.height = "auto";
    setBusy(true);
    setActivity([]);
    setLabel("Starting research…");
    setElapsed(0);
    const start = performance.now(),
      steps = new Map<string, Activity>();
    const timer = window.setInterval(
      () => setElapsed((performance.now() - start) / 1000),
      100,
    );
    const duration = () => (performance.now() - start) / 1000;
    const fail = (content: string) =>
      append(id, {
        role: "assistant",
        content,
        error: true,
        elapsed: duration(),
        activity: [...steps.values()].map((s) =>
          s.status === "running" ? { ...s, status: "interrupted" } : s,
        ),
      });
    try {
      await ask(question, id, request.signal, (event) => {
        if (event.type === "status") setLabel(event.label || "Working…");
        if (event.type === "progress" && event.id) {
          steps.set(event.id, {
            id: event.id,
            label: event.label || "Research",
            status: event.status || "running",
          });
          setActivity([...steps.values()]);
          setLabel(
            [...steps.values()].filter((s) => s.status === "running").at(-1)
              ?.label || "Continuing research…",
          );
        }
        if (event.type === "result")
          append(id, {
            role: "assistant",
            content: event.text || "",
            charts: event.charts,
            sources: event.sources,
            elapsed: duration(),
            activity: [...steps.values()],
          });
        if (event.type === "error")
          fail(event.message || "Research could not finish. Please try again.");
      });
    } catch (e) {
      fail(
        request.signal.aborted
          ? "Stopped receiving this response. The server may still finish the current research turn."
          : e instanceof Error
            ? e.message
            : "The connection was interrupted. Please try again.",
      );
    } finally {
      clearInterval(timer);
      controller.current = null;
      setBusy(false);
      textarea.current?.focus();
    }
  }
  return (
    <div className="chat-layout">
      {sidebar && (
        <button
          className="sidebar-backdrop"
          aria-label="Dismiss conversation sidebar"
          onClick={() => {
            setSidebar(false);
            requestAnimationFrame(() => openSidebarButton.current?.focus());
          }}
        />
      )}
      {sidebar && (
        <aside
          className="chat-sidebar"
          ref={sidebarElement}
          aria-label="Conversations"
        >
          <div className="flex justify-between items-center mb-6">
            <span className="text-sm font-medium">Your workspace</span>
            <button
              className="icon-button"
              onClick={() => {
                setSidebar(false);
                requestAnimationFrame(() => openSidebarButton.current?.focus());
              }}
              aria-label="Close sidebar"
            >
              <PanelLeftClose size={18} />
            </button>
          </div>
          <button
            className="sidebar-link new-chat"
            disabled={busy}
            onClick={newChat}
          >
            <Plus size={18} />
            New chat
          </button>
          <div className="relative my-5">
            <Search size={14} className="absolute left-3 top-3 text-muted" />
            <input
              className="w-full pl-9! text-xs"
              aria-label="Search chats"
              placeholder="Search chats"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <p className="eyebrow mb-3">Your chats</p>
          {!chats.some((c) => c.messages.length) && (
            <p className="sidebar-empty">
              Your research starts here. Conversations will appear as you ask
              questions.
            </p>
          )}
          <div className="flex-1 overflow-auto">
            {[...chats]
              .reverse()
              .filter(
                (c) =>
                  c.messages.length &&
                  c.title.toLowerCase().includes(search.toLowerCase()),
              )
              .map((c) => (
                <div
                  className={
                    "chat-item " + (c.id === chat.id ? "selected" : "")
                  }
                  key={c.id}
                >
                  <button
                    className="truncate flex-1 text-left min-w-0 py-3 pl-3"
                    title={c.title}
                    disabled={busy}
                    onClick={() => {
                      setActive(c.id);
                      setInput("");
                      if (textarea.current)
                        textarea.current.style.height = "auto";
                    }}
                  >
                    {c.title}
                  </button>
                  <button
                    className="icon-button shrink-0"
                    aria-label={`Delete ${c.title}`}
                    disabled={busy}
                    onClick={() => {
                      const remaining = chats.filter((x) => x.id !== c.id);
                      setChats(remaining.length ? remaining : [createChat()]);
                      if (c.id === chat.id) setActive("");
                    }}
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
          </div>
          <p className="text-[11px] text-muted leading-relaxed pt-5">
            Conversations stay in this tab until you reload. Deleting a chat
            removes its visible history.
          </p>
        </aside>
      )}
      <section className="chat-main">
        <div className="chat-topbar">
          {!sidebar && (
            <button
              className="icon-button"
              ref={openSidebarButton}
              onClick={() => setSidebar(true)}
              aria-label="Open sidebar"
            >
              <PanelLeftOpen size={19} />
            </button>
          )}
          <Sparkles size={15} className="ai-text" />
          TuniScope · AI research assistant
          <span className="ml-auto text-xs hidden sm:block">
            Grounded in your market data
          </span>
        </div>
        <div
          className="chat-scroll"
          onScroll={(e) => {
            const node = e.currentTarget;
            followLatest.current =
              node.scrollHeight - node.scrollTop - node.clientHeight < 100;
          }}
        >
          <div className="chat-thread">
            {!chat.messages.length ? (
              <div className="welcome">
                <div className="welcome-icon">
                  <Sparkles size={27} />
                </div>
                <p className="eyebrow ai-text mt-7">YOUR RESEARCH PARTNER</p>
                <h1 className="chat-welcome-title">
                  What can I help you research?
                </h1>
                <p className="welcome-description">
                  Explore a forecast, compare stocks, or make sense of the news.
                </p>
                <div className="chat-suggestions">
                  {suggestions.map(([title, prompt]) => (
                    <button
                      className="suggestion"
                      key={title}
                      onClick={() => void send(prompt)}
                    >
                      <span className="flex justify-between gap-2 mb-3 font-medium text-sm">
                        {title}
                        <ArrowUpRight size={15} />
                      </span>
                      <span className="text-xs text-muted leading-relaxed">
                        {prompt}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              chat.messages.map((message, index) => (
                <ChatMessage key={index} message={message} />
              ))
            )}
            {busy && (
              <div className="live-research">
                <div role="status" className="research-running">
                  <span className="research-pulse" />
                  {label}
                  <span className="text-muted ml-auto tabular-nums">
                    {elapsed.toFixed(1)}s
                  </span>
                </div>
                <ExecutionStatus activity={activity} />
              </div>
            )}
            <div ref={end} />
          </div>
        </div>
        <div className="composer-area">
          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              void send();
            }}
          >
            <textarea
              ref={textarea}
              aria-label="Message TuniScope"
              placeholder="Ask TuniScope about a stock…"
              value={input}
              maxLength={16000}
              aria-describedby="composer-help"
              rows={1}
              onChange={(e) => {
                setInput(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = `${Math.min(e.target.scrollHeight, 180)}px`;
              }}
              onKeyDown={(e) => {
                if (
                  e.key === "Enter" &&
                  !e.shiftKey &&
                  !e.nativeEvent.isComposing
                ) {
                  e.preventDefault();
                  void send();
                }
              }}
            />
            <div className="flex justify-between items-center px-3 pb-2">
              <span id="composer-help" className="composer-hint">
                Enter to send · Shift + Enter for a new line
              </span>
              {busy ? (
                <button
                  type="button"
                  className="send-button"
                  aria-label="Stop receiving response"
                  onClick={() => controller.current?.abort()}
                >
                  <Square size={15} />
                </button>
              ) : (
                <button
                  className="send-button"
                  type="submit"
                  disabled={!input.trim()}
                  aria-label="Send message"
                >
                  <ArrowUp size={19} />
                </button>
              )}
            </div>
          </form>
          <p className="text-center text-[10px] text-muted mt-3">
            TuniScope can make mistakes. Verify sources and forecast dates.
          </p>
        </div>
      </section>
    </div>
  );
}
