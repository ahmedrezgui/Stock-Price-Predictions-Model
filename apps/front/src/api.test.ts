import { describe, expect, it } from "vitest";
import { consumeStream } from "./api";
import type { StreamEvent } from "./types";

function stream(text: string, size = 1) {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (let i = 0; i < bytes.length; i += size)
        controller.enqueue(bytes.slice(i, i + size));
      controller.close();
    },
  });
}
describe("assistant transport", () => {
  it("handles split UTF-8 and NDJSON boundaries, including a final line without newline", async () => {
    const events: StreamEvent[] = [];
    await consumeStream(
      stream(
        '\n{"type":"progress","label":"Étude","id":"1"}\n{"type":"heartbeat"}\n{"type":"result","text":"BIAT ↑"}',
      ),
      (e) => events.push(e),
    );
    expect(events.map((e) => e.type)).toEqual([
      "progress",
      "heartbeat",
      "result",
    ]);
    expect(events[0].label).toBe("Étude");
    expect(events[2].text).toBe("BIAT ↑");
  });
  it("rejects a disconnected stream instead of reporting success", async () => {
    await expect(
      consumeStream(stream('{"type":"heartbeat"}\n'), () => {}),
    ).rejects.toThrow("before the answer");
  });
  it("delivers terminal errors and stops consuming", async () => {
    const events: StreamEvent[] = [];
    await consumeStream(
      stream('{"type":"error","message":"Unavailable"}\ninvalid'),
      (e) => events.push(e),
    );
    expect(events).toEqual([{ type: "error", message: "Unavailable" }]);
  });
});
