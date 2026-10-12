"""Exercise every approved stock against HTTPS and confirm its durable forecast."""

import json
import os
from pathlib import Path

import httpx

from stock_forecasting.cloud_storage import container, read_json
from stock_forecasting.initial_release import authorization, STATUS


def main():
    private = Path(os.environ["LOCALAPPDATA"]) / "StockForecast"
    endpoint = (private / "api-url.txt").read_text().strip()
    key = (private / "api-key.txt").read_text().strip()
    approval, digest = authorization("configs/initial_release.json")
    records = []
    with httpx.Client(timeout=180, follow_redirects=False) as client:
        client.get(endpoint + "/docs").raise_for_status()
        schema = client.get(endpoint + "/openapi.json")
        schema.raise_for_status()
        assert "/v1/predict" in schema.json()["paths"]
        assert client.get(endpoint + "/v1/models").status_code == 401
        client.headers["X-API-Key"] = key
        ready = client.get(endpoint + "/ready")
        ready.raise_for_status()
        catalog = client.get(endpoint + "/v1/models")
        catalog.raise_for_status()
        assert {r["symbol"] for r in catalog.json()} == {r["symbol"] for r in approval["models"]}
        assert client.post(endpoint + "/v1/predict", json={"symbol": "UNKNOWN"}).status_code == 404
        assert client.post(endpoint + "/v1/predict", json={"symbol": "BIAT", "horizon": 2}).status_code == 422
        for item in approval["models"]:
            request = {"symbol": item["symbol"], "model": "lstm", "task": "return", "horizon": 1}
            response = client.post(endpoint + "/v1/predict", json=request)
            response.raise_for_status()
            result = response.json()
            assert result["model_version"] == item["bundle_version"]
            assert result["deployment_status"] == STATUS and result["approval_id"] == digest
            repeated = client.post(endpoint + "/v1/predict", json=request)
            repeated.raise_for_status()
            assert repeated.json()["forecast_id"] == result["forecast_id"]
            saved, _ = read_json(container("forecast-monitoring"), f"forecasts/{result['forecast_id']}.json")
            assert saved["model_version"] == item["bundle_version"]
            assert saved["deployment_status"] == STATUS and saved["approval_id"] == digest
            records.append(result)
            print(
                f"{item['symbol']}: prediction, identity, approval and persistence passed; as_of={result['as_of']}",
                flush=True,
            )
    path = Path("artifacts/azure/api-verification.json")
    path.write_text(
        json.dumps({"endpoint": endpoint, "models_checked": len(records), "predictions": records}, indent=2)
    )
    print(f"All {len(records)} models passed. Evidence: {path}")


if __name__ == "__main__":
    main()
