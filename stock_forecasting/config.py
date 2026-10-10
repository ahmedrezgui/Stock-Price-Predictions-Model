from dataclasses import asdict, dataclass, field
import json
import re
from pathlib import Path
from typing import Literal


@dataclass
class Config:
    symbol: str = "BIAT"
    horizon: int = 1
    sequence_length: int = 20
    feature_groups: list[str] = field(default_factory=lambda: ["price", "volatility"])
    family: Literal["lstm", "gru"] = "lstm"
    task: Literal["return", "direction", "multitask"] = "return"
    hidden_size: int = 32
    layers: int = 1
    dropout: float = 0.1
    epochs: int = 20
    patience: int = 5
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    loss: Literal["mse", "huber"] = "huber"
    classification_weight: float = 0.5
    seed: int = 7
    device: str = "auto"
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    adjustment: str = "unverified"
    calendar: str | None = None
    session_file: str | None = None
    as_of: str | None = None
    start: str | None = None
    end: str | None = None
    source: str = "ilboursa"
    cost_bps: float = 10.0
    threshold: float = 0.0

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,19}", self.symbol):
            raise ValueError("Invalid stock symbol")
        if self.horizon not in (1, 5, 20):
            raise ValueError("horizon must be 1, 5 or 20 observations/sessions")
        if min(self.sequence_length, self.hidden_size, self.layers, self.epochs, self.batch_size, self.patience) < 1:
            raise ValueError("Model sizes and training budgets must be positive")
        if self.family not in ("lstm", "gru") or self.task not in ("return", "direction", "multitask"):
            raise ValueError("Unsupported family/task")
        if self.loss not in ("mse", "huber"):
            raise ValueError("loss must be mse or huber")
        if not 0 < self.train_fraction < 1 or not 0 < self.validation_fraction < 1 - self.train_fraction:
            raise ValueError("Invalid chronological split fractions")
        if self.adjustment not in ("unverified", "raw", "adjusted"):
            raise ValueError("Unknown adjustment policy")
        if not 0 <= self.dropout < 1 or self.classification_weight <= 0:
            raise ValueError("Invalid dropout/loss weight")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.cost_bps < 0:
            raise ValueError("Invalid optimization/cost parameters")
        if self.threshold < 0:
            raise ValueError("Trading threshold must be nonnegative")
        if not set(self.feature_groups) <= {"price", "volatility", "volume", "technical"}:
            raise ValueError("Unknown feature group")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def read(cls, path: str | Path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
