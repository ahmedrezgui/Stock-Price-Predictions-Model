# Forecast API v1

Start from the repository root:

```powershell
python -m uvicorn stock_forecasting.api:app --host 127.0.0.1 --port 8000
```

`GET /health` is process liveness; `GET /ready` checks loadable, checksum-valid
model bundles and local symbol files. It returns 503 when unavailable. It does
not promise that every dataset is current; prediction performs freshness and
feature validation. `GET /models` (alias `/v1/models`) returns exact available
family/task/horizon/version combinations. `/docs` exposes OpenAPI. Artifacts cache
by immutable path; restart on replacement (overwriting versions is unsupported).

```powershell
$models = Invoke-RestMethod http://127.0.0.1:8000/models
$chosen = $models | Where-Object { $_.symbol -eq 'BIAT' -and $_.model -eq 'lstm' -and $_.task -eq 'return' -and $_.horizon -eq 1 } | Select-Object -First 1
$body = @{symbol='BIAT'; model='lstm'; horizon=1; task='return'; model_version=$chosen.model_version} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/v1/predict -Method Post -ContentType application/json -Body $body
```

Choose a seed/version using validation and operational policy, not the best test
score. When multiple versions match, omission of model_version returns 409.
Unknown symbols/models return 404; schema or incompatible data return 422;
missing files return 503. Errors have `error.code`, `error.message`, and
`request_id`. Every response has X-Request-ID and X-Latency-Ms; logs contain
request metadata and never request secrets. Unexpected exceptions return a
generic 500 with server-side traceback.

`as_of` means exchange-local date of the latest completed input observation,
not current wall time. `forecast_for` is the exchange-calendar target date if a
verified calendar is configured; otherwise null. `forecast_horizon_sessions`
is likewise null when only observed bars are known; the separate observation
horizon and horizon_basis prevent misleading calendar claims. The return is a
decimal cumulative log-return point forecast. implied_price = latest close *
exp(forecast) on the same explicitly named price basis; it is not guaranteed.
For direction-only models return and price are null. Probability is a sigmoid
head output, marked uncalibrated; regression-only probability is null. No interval
is claimed. `data_version` fingerprints current canonical input;
`training_data_version` identifies the immutable training snapshot.

Set `FORECAST_MODEL_ROOT` and `FORECAST_DATA_ROOT` to read-only model/data
directories. Serving uses the exact bundle's feature groups, feature order,
feature version, target scaling and scaler. It neither downloads nor trains.
Set `FORECAST_ENV=production` and provide `FORECAST_API_KEY` from a secret store;
send `X-API-Key` over HTTPS. Nonlocal operation fails closed without a configured
key. Local authentication is optional. Entra/OAuth at an API gateway and rate
limits are the recommended next step for broader public access; not implemented.

The existing web `/companies`, `/stock/{company}` and `/rag` contracts remain
separate. Their historical price endpoint is legacy and has no verified exchange
calendar. Do not use it for new return forecasts. No LangGraph code was changed.
A future agent tool should call `/v1/predict`, preserve null fields and surface
price_basis, as_of, version and stale-data errors. It must not relabel observed
bars as sessions or probabilities as calibrated confidence.

Azure ML's optional scoring adapter accepts `{ "observations": [...] }`, with
canonical OHLCV records and enough historical warmup. It uses the same bundle
and features but has an explicit input-data contract, separate from symbol-based
FastAPI. Managed endpoints provide their own Entra authentication boundary.
