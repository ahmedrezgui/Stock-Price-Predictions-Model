from .IModel import IModel
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class GRUModel(IModel):
    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = deepcopy(self.model)
        if isinstance(state["model"], nn.Module):
            state["model"].cpu()
        return state

    def __init__(self, stock_name: str = None):
        super().__init__(stock_name)
        # PyTorch needs the feature count to construct GRU; infer it in train().
        self.model = nn.ModuleDict(
            {
                "dropout": nn.Dropout(0.2),
                "dense": nn.Linear(50, 1),
            }
        )

    @classmethod
    def from_config(cls, input_size, config):
        """Construct the shared return/direction model; default constructor stays legacy."""
        if config.family != "gru":
            raise ValueError("Configuration family does not match GRUModel")
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
        print(f"features shape: {features.shape}")
        print(f"targets shape: {targets.shape}")
        features = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        targets = torch.as_tensor(np.asarray(targets), dtype=torch.float32)
        if features.ndim != 3 or any(size == 0 for size in features.shape):
            raise ValueError("Features must have nonempty shape (samples, timesteps, features).")
        if targets.shape not in ((len(features),), (len(features), 1)):
            raise ValueError("Targets must have shape (samples,) or (samples, 1).")
        if not torch.isfinite(features).all() or not torch.isfinite(targets).all():
            raise ValueError("Training data must contain only finite numeric values.")
        if "gru" not in self.model:
            self.model["gru"] = nn.GRU(features.shape[2], 50, batch_first=True)
        elif features.shape[2] != self.model["gru"].input_size:
            raise ValueError("Input feature count differs from the trained GRU.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.train()
        loader = DataLoader(TensorDataset(features, targets.reshape(-1, 1)), batch_size=32, shuffle=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, eps=1e-7)
        criterion = nn.MSELoss()
        for epoch in range(50):
            total_loss = 0.0
            for batch_features, batch_targets in loader:
                optimizer.zero_grad()
                sequence, _ = self.model["gru"](batch_features.to(device))
                predictions = self.model["dense"](self.model["dropout"](sequence[:, -1, :]))
                loss = criterion(predictions, batch_targets.to(device))
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
        print(f"input shape: {input_data.shape}")
        if "gru" not in self.model:
            raise RuntimeError("Train the model before predicting.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        inputs = torch.as_tensor(np.asarray(input_data), dtype=torch.float32)
        if inputs.ndim != 3 or inputs.shape[1] == 0 or inputs.shape[2] != self.model["gru"].input_size:
            raise ValueError("Features must have shape (samples, timesteps, trained features).")
        if not torch.isfinite(inputs).all():
            raise ValueError("Features must contain only finite numeric values.")
        batches = []
        with torch.inference_mode():
            for batch in inputs.split(32):
                if len(batch):
                    sequence, _ = self.model["gru"](batch.to(device))
                    predictions = self.model["dense"](self.model["dropout"](sequence[:, -1, :]))
                    batches.append(predictions.cpu().numpy())
        res = np.concatenate(batches) if batches else np.empty((0, 1), dtype=np.float32)
        print(f"prediction result shape: {res.shape}")
        return res.squeeze()
