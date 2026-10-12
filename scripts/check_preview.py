"""Check authenticated predictions and durable Blob storage without printing credentials."""

import json
import os
from pathlib import Path

import httpx

from stock_forecasting.cloud_storage import container, read_json


def main():
    private = Path(os.environ["LOCALAPPDATA"]) / "StockForecast"
    endpoint = (private / "preview-url.txt").read_text().strip()
    key = (private / "preview-api-key.txt").read_text().strip()
    with httpx.Client(timeout=120, follow_redirects=False) as client:
        request = {"symbol": "BIAT", "model": "lstm", "task": "return", "horizon": 1}
        assert client.post(endpoint + "/v1/predict", json=request).status_code == 401
        ready = client.get(endpoint + "/ready")
        ready.raise_for_status()
        response = client.post(endpoint + "/v1/predict", headers={"X-API-Key": key}, json=request)
        response.raise_for_status()
        result = response.json()
        assert result["deployment_status"] == "preview_unapproved"
        expected = json.loads(Path("deployment/preview.json").read_text())
        assert result["model_version"] == expected["bundle_version"]
        repeated = client.post(endpoint + "/v1/predict", headers={"X-API-Key": key}, json=request)
        repeated.raise_for_status()
        assert repeated.json()["forecast_id"] == result["forecast_id"]
    saved, _ = read_json(container("forecast-preview-monitoring"), f"forecasts/{result['forecast_id']}.json")
    assert saved["model_version"] == result["model_version"]
    assert saved["deployment_status"] == "preview_unapproved"
    print(
        json.dumps({"endpoint": endpoint, "authentication": "passed", "durable_forecast": "passed", **result}, indent=2)
    )


if __name__ == "__main__":
    main()
