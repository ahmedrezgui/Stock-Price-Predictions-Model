from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import random
import time
import warnings

import numpy as np
from scipy.special import expit
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import Config


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class RecurrentNetwork(nn.Module):
    def __init__(self, input_size: int, config: Config):
        super().__init__()
        recurrent = nn.LSTM if config.family == "lstm" else nn.GRU
        self.encoder = recurrent(
            input_size,
            config.hidden_size,
            config.layers,
            batch_first=True,
            dropout=config.dropout if config.layers > 1 else 0,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.regression = nn.Linear(config.hidden_size, 1)
        self.direction = nn.Linear(config.hidden_size, 1)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        hidden = self.dropout(encoded[:, -1])
        return self.regression(hidden).squeeze(-1), self.direction(hidden).squeeze(-1)


class NeuralForecaster:
    def __init__(self, input_size: int, config: Config):
        self.config = config
        seed_everything(config.seed)
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device
        )
        self.network = RecurrentNetwork(input_size, config).to(self.device)
        self.target_mean, self.target_scale = 0.0, 1.0
        self.history: list[dict] = []
        self.training_seconds = 0.0

    def fit(self, x, y, x_validation, y_validation, *, fixed_epochs=False):
        c = self.config
        seed_everything(c.seed)
        self.training_mode = "fixed_epoch_refit" if fixed_epochs else "validation_selection"
        x, x_validation = np.asarray(x, dtype=np.float32), np.asarray(x_validation, dtype=np.float32)
        y, y_validation = np.asarray(y, dtype=np.float32), np.asarray(y_validation, dtype=np.float32)
        for inputs, labels in ((x, y), (x_validation, y_validation)):
            if inputs.ndim != 3 or any(n == 0 for n in inputs.shape) or labels.shape != (len(inputs),):
                raise ValueError("Expected nonempty sequences and aligned one-dimensional targets")
            if inputs.shape[2] != self.network.encoder.input_size:
                raise ValueError("Input feature count mismatch")
        self.history = []
        if not all(np.isfinite(a).all() for a in (x, y, x_validation, y_validation)):
            raise ValueError("Nonfinite training data")
        self.target_mean = float(np.mean(y))
        self.target_scale = max(float(np.std(y)), 1e-6)
        loader = DataLoader(
            TensorDataset(torch.as_tensor(x), torch.as_tensor(y, dtype=torch.float32)),
            batch_size=c.batch_size,
            shuffle=False,
        )
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        regression = nn.MSELoss() if c.loss == "mse" else nn.HuberLoss()
        classification = nn.BCEWithLogitsLoss()

        def loss_fn(inputs, labels):
            forecast, logits = self.network(inputs)
            reg = regression(forecast, (labels - self.target_mean) / self.target_scale)
            cls = classification(logits, (labels > 0).float())
            return (
                cls
                if c.task == "direction"
                else (reg + c.classification_weight * cls if c.task == "multitask" else reg)
            )

        best_loss, best, stale = float("inf"), None, 0
        start = time.perf_counter()
        for epoch in range(c.epochs):
            self.network.train()
            losses = []
            for inputs, labels in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(inputs.to(self.device), labels.to(self.device))
                if not torch.isfinite(loss):
                    raise ValueError("Training diverged")
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()
                losses.append((loss.item(), len(labels)))
            self.network.eval()
            with torch.inference_mode():
                validation_loss = float(
                    loss_fn(
                        torch.as_tensor(x_validation, device=self.device),
                        torch.as_tensor(y_validation, dtype=torch.float32, device=self.device),
                    )
                )
            if not fixed_epochs:
                scheduler.step(validation_loss)
            self.history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": sum(a * n for a, n in losses) / len(y),
                    "validation_loss": validation_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            if validation_loss < best_loss - 1e-7:
                best_loss, best, stale = validation_loss, deepcopy(self.network.state_dict()), 0
            else:
                stale += 1
            if not fixed_epochs and stale >= c.patience:
                break
        if not fixed_epochs:
            self.network.load_state_dict(best)
        self.training_seconds = time.perf_counter() - start
        return self

    def predict(self, x) -> dict:
        self.network.eval()
        returns, logits = [], []
        with torch.inference_mode():
            for batch in torch.as_tensor(x, dtype=torch.float32).split(self.config.batch_size):
                r, p = self.network(batch.to(self.device))
                returns.extend(r.cpu().numpy())
                logits.extend(p.cpu().numpy())
        return {
            "return": (np.asarray(returns) * self.target_scale + self.target_mean)
            if self.config.task != "direction"
            else None,
            "probability": expit(np.asarray(logits)) if self.config.task != "return" else None,
        }

    def metadata(self):
        return {
            "config": asdict(self.config),
            "target_mean": self.target_mean,
            "target_scale": self.target_scale,
            "parameters": sum(p.numel() for p in self.network.parameters()),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in self.network.parameters()),
            "training_seconds": self.training_seconds,
            "device": str(self.device),
            "history": self.history,
            "training_mode": getattr(self, "training_mode", "unfitted"),
        }


class ARIMAForecaster:
    """Train-only AIC order selection. Append realized returns through each origin.

    Coefficients are frozen between fits; h-step daily-return forecasts are summed.
    No probability estimates are claimed from point forecasts.
    """

    def fit(self, returns, candidates=((0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1))):
        start = time.perf_counter()
        self.selection = []
        best = None
        for order in candidates:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    fit = ARIMA(np.asarray(returns), order=order).fit()
                    converged = bool(fit.mle_retvals.get("converged", True))
                    self.selection.append(
                        {
                            "order": list(order),
                            "aic": float(fit.aic) if np.isfinite(fit.aic) else None,
                            "converged": converged,
                            "warnings": [str(w.message) for w in caught],
                        }
                    )
                    if converged and np.isfinite(fit.aic) and (best is None or fit.aic < best.aic):
                        best = fit
                except (ValueError, np.linalg.LinAlgError) as exc:
                    self.selection.append({"order": list(order), "error": str(exc)})
        if best is None:
            raise ValueError("All ARIMA candidates failed to converge")
        self.result = best
        self.order = best.model.order
        self.training_seconds = time.perf_counter() - start
        return self

    def rolling_predict(self, daily_returns: np.ndarray, origins: np.ndarray, train_last: int, horizon: int):
        result = self.result
        last = train_last
        predictions = []
        for origin in origins:
            if origin < last:
                raise ValueError("Forecast origins must be chronological and after fitting data")
            if origin > last:
                result = result.append(np.asarray(daily_returns[last + 1 : origin + 1]), refit=False)
            predictions.append(float(np.sum(result.forecast(horizon))))
            last = origin
        return np.asarray(predictions)

    def save(self, path: str | Path, metadata: dict | None = None) -> Path:
        """Statsmodels pickle is only for trusted local/registry artifacts."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        self.result.save(path / "arima.pkl")
        meta = {
            **(metadata or {}),
            "order": list(self.order),
            "selection": self.selection,
            "training_seconds": self.training_seconds,
            "trusted_artifact_only": True,
            "sha256": sha256((path / "arima.pkl").read_bytes()).hexdigest(),
        }
        (path / "metadata.json").write_text(json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if sha256((path / "arima.pkl").read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError("ARIMA artifact checksum mismatch")
        model = cls()
        model.result = ARIMAResults.load(path / "arima.pkl")
        model.order = tuple(meta["order"])
        model.selection = meta["selection"]
        model.training_seconds = meta["training_seconds"]
        return model


def naive_predictions(train_targets: np.ndarray, n: int) -> dict:
    probability = float(np.mean(train_targets > 0))
    return {
        "zero_return": {"return": np.zeros(n), "probability": None},
        "historical_mean": {"return": np.full(n, np.mean(train_targets)), "probability": None},
        "historical_probability": {"return": None, "probability": np.full(n, probability)},
        "majority_direction": {"direction": np.full(n, probability >= 0.5), "return": None, "probability": None},
    }


def create_forecaster(input_size, config):
    """All configured training paths use the existing public model classes."""
    from stock_forecasting.neural import LSTMModel, GRUModel

    cls = LSTMModel if config.family == "lstm" else GRUModel
    return cls.from_config(input_size, config)
