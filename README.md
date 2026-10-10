# Stock forecasting with PyTorch

Compare LSTM, GRU, ARIMA and simple baselines on Tunisian stock returns, then
train versioned candidates and serve predictions through FastAPI.

## Start with the benchmark

Open [Models_benchmark.ipynb](notebook/Models%20benchmark/Models_benchmark.ipynb)
for the complete workflow and saved outputs. Read the
[benchmark report](reports/benchmark/benchmark_report.md) for the model-choice
rationale, metrics, per-stock results and figures.

The experiment covers **BIAT, ATB, ARTES, NAKL and CIL**, using available history
from 2010 through October 2026. All models for a stock use the same forecast
origins. All stocks share calendar boundaries: training before 2022, validation
in 2022?2023 and test from 2024. Available trading rows can differ across stocks.
Future labels crossing split boundaries are purged; scalers fit training inputs only.

LSTM and GRU use the existing public model classes and shared training engine.
They receive matching features, sequence length, hidden size, optimizer settings,
three seeds (7, 19, 43), a 50-epoch maximum and patience 10. Early stopping restores
the best validation checkpoint, so actual epoch counts differ. The notebook
requires CUDA for all 60 neural fits and neural predictions; ARIMA runs on CPU
with training-only AIC order selection. Matching neural budgets does not imply
equal parameter counts or equal compute costs across architectures.

### Model choice

Lower relative RMSE is better; 1 is the historical-mean baseline. Scores average
seeds within each stock, then weight the five stocks equally.

| Return model | Validation relative RMSE | Test relative RMSE |
|---|---:|---:|
| Historical mean | 1.000000 | 1.000000 |
| ARIMA | 1.002528 | 0.996132 |
| LSTM | 0.997116 | 1.000522 |
| GRU | 1.000840 | 0.997867 |

**LSTM is the provisional validation-selected candidate.** Its validation advantage
is only 0.288%, and it beats the mean baseline on two of five stocks. ARIMA has the
lowest test error; LSTM slightly underperforms the baseline on test. These results
justify further evaluation of LSTM, not declaring it the production winner.
Freeze the configuration and compare against ARIMA and simple baselines on new,
unseen observations before making a deployment decision.

The report includes forecasts, training curves, confusion matrices, probability
and strategy diagnostics: one Markdown file and 31 PNGs. Displayed tables persist
in notebook outputs; full predictions live in kernel memory and require rerunning
after restart. Benchmarking does not save or promote a serving model.

## Local setup

Use Python 3.11 or newer and a CUDA-enabled PyTorch environment for the notebook.
The notebook checks CUDA availability and stops rather than falling back to CPU.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Raw datasets are not committed. Obtain the five CSVs in `data/raw/` through the
ilboursa download script or restore your local datasets before rerunning the
notebook. The report records each source checksum and actual date coverage.
Choose the project environment as the notebook kernel and run its cells in order.
The viewing cell can be rerun without repeating training.

## Routine training and API

```powershell
python -m scripts.train_model --config configs/training.json --symbols BIAT
python -m uvicorn stock_forecasting.api:app --host 127.0.0.1 --port 8000
```

Routine training uses its own configuration; the benchmark explicitly overrides
history, epoch budget and device. Review the selected configuration before
training a deployment candidate. The API exposes `/health`, `/ready`, `/models`
and `/v1/predict`, with interactive documentation at `/docs`. Serving uses saved
model bundles and frozen preprocessing, not the benchmark's in-memory predictions.

## Validation and remaining work

```powershell
python -m pytest tests/forecasting -q
python -m ruff check stock_forecasting tests/forecasting
```

Provider prices do not have verified corporate-action adjustments or exchange
calendars. Horizons count observed bars. Strategy plots include simplified costs
and execution assumptions and do not establish trading profitability. Previously
inspected historical test periods are retrospective evidence.

Azure integration, candidate promotion and monitoring are separate from this
benchmark milestone. A successful benchmark is not proof of a verified live Azure
deployment or performance-monitoring service. The `docs/` directory contains local
project guides and is intentionally excluded from Git; it is absent from a fresh
clone. The existing research CLI and ETL dependencies remain available.
