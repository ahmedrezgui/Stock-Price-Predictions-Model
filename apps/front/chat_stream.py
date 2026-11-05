"""Read incremental assistant events without buffering the complete answer."""
import json
import requests


def stream_response(url, question, thread_id, timeout=180):
    with requests.post(
        f"{url}/chat/stream", json={"question": question, "thread_id": thread_id},
        stream=True, timeout=(10, timeout),
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if not line:
                continue
            event = json.loads(line)
            yield event
            if event.get("type") in {"result", "error"}:
                return
        raise requests.ConnectionError("Assistant stream ended without a final result")
