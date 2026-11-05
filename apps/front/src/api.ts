import type { StreamEvent } from "./types";

export async function getJson<T>(
  path: string,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(path, { signal });
  if (!response.ok) {
    if (import.meta.env.DEV && [500, 502, 504].includes(response.status)) {
      throw new Error(
        "Cannot reach the local dashboard API. Start it on port 8002, then refresh this page.",
      );
    }
    throw new Error(
      response.status === 503
        ? "The deployed model catalog or market history is temporarily unavailable. Please retry shortly."
        : "Market data is unavailable. Please try again shortly.",
    );
  }
  return response.json();
}

// Decode incrementally: HTTP chunks need not align with lines or UTF-8 characters.
export async function consumeStream(
  body: ReadableStream<Uint8Array>,
  onEvent: (event: StreamEvent) => void,
) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const dispatch = (line: string) => {
    if (!line.trim()) return false;
    const event: StreamEvent = JSON.parse(line);
    onEvent(event);
    return event.type === "result" || event.type === "error";
  };
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let end: number;
      while ((end = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, end);
        buffer = buffer.slice(end + 1);
        if (dispatch(line)) return;
      }
      if (done) {
        if (dispatch(buffer)) return;
        throw new Error(
          "The connection ended before the answer arrived. Please try again.",
        );
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

export async function ask(
  question: string,
  threadId: string,
  signal: AbortSignal,
  onEvent: (event: StreamEvent) => void,
) {
  const response = await fetch("/assistant-api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, thread_id: threadId }),
    signal,
  });
  if (!response.ok || !response.body)
    throw new Error("The assistant is unavailable. Please try again shortly.");
  await consumeStream(response.body, onEvent);
}
