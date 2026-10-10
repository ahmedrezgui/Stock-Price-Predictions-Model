from hashlib import sha256
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import uuid

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .config import Config
from .data import FEATURE_VERSION
from .models import NeuralForecaster, create_forecaster


def code_version():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip()
        )
        return {"git_commit": commit, "dirty_worktree": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "dirty_worktree": None}


def save_bundle(model: NeuralForecaster, scaler, columns, manifest, split, root: str | Path) -> Path:
    version = f"{model.config.symbol}-{model.config.family}-{model.config.task}-h{model.config.horizon}-{uuid.uuid4().hex[:12]}"
    path = Path(root) / version
    path.mkdir(parents=True, exist_ok=False)
    torch.save({k: v.cpu() for k, v in model.network.state_dict().items()}, path / "weights.pt")
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": version,
        "feature_version": FEATURE_VERSION,
        "feature_columns": columns,
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "target": "log(close[t+h]/close[t]); decimal units; positive iff >0",
        "dataset": manifest,
        "split": split,
        "model": model.metadata(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        **code_version(),
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    hashes = {name: sha256((path / name).read_bytes()).hexdigest() for name in ("weights.pt", "metadata.json")}
    (path / "checksums.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    return path


def load_bundle(path: str | Path):
    path = Path(path)
    checksums = json.loads((path / "checksums.json").read_text(encoding="utf-8"))
    for name in ("weights.pt", "metadata.json"):
        if sha256((path / name).read_bytes()).hexdigest() != checksums[name]:
            raise ValueError("Artifact checksum mismatch")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata["schema_version"] != 1 or metadata["feature_version"] != FEATURE_VERSION:
        raise ValueError("Incompatible feature or artifact version")
    config = Config(**{**metadata["model"]["config"], "device": "cpu"})
    model = create_forecaster(len(metadata["feature_columns"]), config)
    model.network.load_state_dict(torch.load(path / "weights.pt", map_location="cpu", weights_only=True))
    model.target_mean = metadata["model"]["target_mean"]
    model.target_scale = metadata["model"]["target_scale"]
    scaler = StandardScaler()
    scaler.mean_ = np.array(metadata["scaler"]["mean"])
    scaler.scale_ = np.array(metadata["scaler"]["scale"])
    scaler.n_features_in_ = len(scaler.mean_)
    if (
        len(scaler.mean_) != len(metadata["feature_columns"])
        or not np.isfinite(scaler.scale_).all()
        or (scaler.scale_ <= 0).any()
    ):
        raise ValueError("Incompatible scaler")
    return model, scaler, metadata
