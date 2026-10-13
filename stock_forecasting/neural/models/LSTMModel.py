from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .IModel import IModel


class _LSTMNetwork(nn.Module):
    """Private PyTorch network stored in the model's existing `model` property."""

    def __init__(self, hidden_size=64):
        super().__init__()
        self.lstm = None
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, inputs):
        lstm_out, _ = self.lstm(inputs)
        last_hidden_state = lstm_out[:, -1, :]
        x = self.fc1(last_hidden_state)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)
        return output

    def __getstate__(self):
        # Pickle/joblib save CPU copies without moving the live training network.
        state = super().__getstate__().copy()
        state["_modules"] = deepcopy(state["_modules"])
        for module in state["_modules"].values():
            if module is not None:
                module.cpu()
        return state


class LSTMModel(IModel):
    def __init__(self, stock_name: str = None):
        super().__init__(stock_name)
        self.model = _LSTMNetwork()

    @classmethod
    def from_config(cls, input_size, config):
        """Construct the shared return/direction model; default constructor stays legacy."""
        if config.family != "lstm":
            raise ValueError("Configuration family does not match LSTMModel")
        from stock_forecasting.models import NeuralForecaster

        instance = cls(config.symbol)
        instance.model = NeuralForecaster(input_size, config)
        return instance

    def fit(self, x, y, x_validation, y_validation):
        self.model.fit(x, y, x_validation, y_validation)
        return self

    def fit_fixed(self, x, y):
        """Refit for a previously selected epoch budget; no validation selection."""
        self.model.fit(x, y, x, y, fixed_epochs=True)
        return self

    @property
    def network(self):
        return self.model.network

    @property
    def config(self):
        return self.model.config

    @property
    def history(self):
        return self.model.history

    @property
    def target_mean(self):
        return self.model.target_mean

    @target_mean.setter
    def target_mean(self, value):
        self.model.target_mean = value

    @property
    def target_scale(self):
        return self.model.target_scale

    @target_scale.setter
    def target_scale(self, value):
        self.model.target_scale = value

    def metadata(self):
        return self.model.metadata()

    def train(self, features, targets, last_trained_date):
        from stock_forecasting.models import NeuralForecaster

        if isinstance(self.model, NeuralForecaster):
            raise ValueError(
                "Configured forecasting models require fit with explicit validation, or fit_fixed for refit"
            )
        features = np.asarray(features, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        if features.ndim != 3 or any(size == 0 for size in features.shape):
            raise ValueError("Features must have nonempty shape (samples, timesteps, features).")
        if targets.shape not in ((len(features),), (len(features), 1)):
            raise ValueError("Targets must have shape (samples,) or (samples, 1).")
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise ValueError("Training data must contain only finite numeric values.")
        if self.model.lstm is None:
            self.model.lstm = nn.LSTM(
                input_size=features.shape[2],
                hidden_size=64,
                num_layers=2,
                batch_first=True,
                dropout=0.2,
            )
        elif features.shape[2] != self.model.lstm.input_size:
            raise ValueError(f"Expected {self.model.lstm.input_size} input features, got {features.shape[2]}.")

        print(f"features shape: {features.shape}")
        print(f"targets shape: {targets.shape}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.train()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(features.copy()), torch.from_numpy(targets.reshape(-1, 1).copy())),
            batch_size=32,
            shuffle=False,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, eps=1e-7)
        criterion = nn.MSELoss()
        for epoch in range(50):
            total_loss = 0.0
            for batch_features, batch_targets in loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(batch_features), batch_targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_features)
            print(f"Epoch {epoch + 1}/50 - loss: {total_loss / len(features):.6f} - device: {device}")
        self.last_trained_date = (
            max(last_trained_date, self.last_trained_date) if self.last_trained_date is not None else last_trained_date
        )

    def predict(self, input_data):
        from stock_forecasting.models import NeuralForecaster

        if isinstance(self.model, NeuralForecaster):
            return self.model.predict(input_data)
        if self.model.lstm is None:
            raise RuntimeError("Train the model before predicting.")
        inputs = np.asarray(input_data, dtype=np.float32)
        if inputs.ndim != 3 or inputs.shape[1] == 0:
            raise ValueError("Features must have shape (samples, timesteps, features).")
        if inputs.shape[2] != self.model.lstm.input_size:
            raise ValueError(f"Expected {self.model.lstm.input_size} input features, got {inputs.shape[2]}.")
        if not np.isfinite(inputs).all():
            raise ValueError("Features must contain only finite numeric values.")
        print(f"input shape: {inputs.shape}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        inputs = torch.from_numpy(inputs.copy())
        with torch.inference_mode():
            batches = [self.model(batch.to(device)).cpu().numpy() for batch in inputs.split(32) if len(batch)]
        res = np.concatenate(batches) if batches else np.empty((0, 1), dtype=np.float32)
        print(f"prediction result shape: {res.shape}")
        return res
