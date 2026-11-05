"""Public execution status only: never forward prompts, arguments or results."""
import json
from threading import Lock
from langchain_core.callbacks import BaseCallbackHandler

STAGES = {
    "check_topic_relevency": "Checking your question",
    "agent": "Preparing the response",
    "grade_answer": "Checking the answer against evidence",
    "web_fallback": "Searching for additional news",
    "off_topic": "Preparing a reply",
}
TOOLS = {
    "predict_stock": "Fetching the deployed forecast",
    "list_forecast_models": "Checking available forecast models",
    "get_stock_data": "Loading stock history",
    "list_available_companies": "Checking available stocks",
    "plot_stock": "Building the price chart",
    "plot_prediction": "Charting the forecast",
    "plot_comparison": "Building the comparison chart",
    "compare_stocks": "Comparing stock performance",
    "analyze_stock_performance": "Analyzing stock performance",
    "top_movers": "Finding market movers",
    "web_search": "Searching recent news",
    "retrieve_news": "Searching the news archive",
}


def failed_output(output):
    if getattr(output, "status", None) == "error":
        return True
    update = getattr(output, "update", None)
    if isinstance(update, dict):
        return any(failed_output(m) for m in update.get("messages", []))
    content = getattr(output, "content", output)
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except ValueError:
            return False
    return isinstance(content, dict) and bool(content.get("error"))


class ProgressHandler(BaseCallbackHandler):
    def __init__(self, emit):
        self.emit = emit
        self.active = {}
        self.lock = Lock()

    def start(self, run_id, label, tool=None):
        with self.lock:
            event = {"type": "progress", "id": str(run_id), "label": label,
                     "status": "running"}
            if tool:
                event["tool"] = tool
            self.active[str(run_id)] = event
            self.emit(event.copy())

    def finish(self, run_id, failed=False):
        with self.lock:
            event = self.active.pop(str(run_id), None)
            if event:
                self.emit({**event, "status": "failed" if failed else "completed"})

    def on_chain_start(self, serialized, inputs, *, run_id, **kwargs):
        name = kwargs.get("name") or (serialized or {}).get("name")
        if name in STAGES:
            self.start(run_id, STAGES[name])

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self.finish(run_id)

    def on_chain_error(self, error, *, run_id, **kwargs):
        self.finish(run_id, True)

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        name = (serialized or {}).get("name")
        self.start(run_id, TOOLS.get(name, "Running a research tool"), name if name in TOOLS else None)

    def on_tool_end(self, output, *, run_id, **kwargs):
        self.finish(run_id, failed_output(output))

    def on_tool_error(self, error, *, run_id, **kwargs):
        self.finish(run_id, True)
