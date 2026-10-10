"""Package only promoted bundles and their data for the root Dockerfile."""

import argparse
import json
import shutil
from pathlib import Path
from .artifacts import load_bundle
from .registry import read_registry, identity
from .api import PredictionRequest, predict_bundle


def package_release(models, data, output):
    models, data, output = Path(models), Path(data), Path(output)
    state = read_registry(models)
    if not state["active"]:
        raise ValueError("No promoted models to release")
    verified = []
    for key, version in state["active"].items():
        if Path(version).name != version:
            raise ValueError("Invalid version")
        model, _, meta = load_bundle(models / version)
        if key != identity(model.config):
            raise ValueError("Registry identity mismatch")
        request = PredictionRequest(
            symbol=model.config.symbol, model=model.config.family, task=model.config.task, horizon=model.config.horizon
        )
        predict_bundle(models / version, data, request)
        verified.append((version, model.config.symbol, meta["dataset"]["dataset_version"]))
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    (output / "data").mkdir()
    for version, symbol, _ in verified:
        shutil.copytree(models / version, output / "models" / version)
        shutil.copy2(data / f"{symbol}.csv", output / "data" / f"{symbol}.csv")
    (output / "models" / "active.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    (output / "release.json").write_text(json.dumps(verified, indent=2), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="deployment")
    args = parser.parse_args()
    print(package_release(args.models, args.data, args.output))


if __name__ == "__main__":
    main()
