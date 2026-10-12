"""Package an explicitly unapproved preview; never update production registry state."""

import argparse
import json
from pathlib import Path
import shutil

from .artifacts import load_bundle
from .registry import identity


def package_preview(
    symbol, output, results="artifacts/market/results.json", models="artifacts/models", data="data/processed"
):
    result = json.loads(Path(results).read_text())[symbol]
    if result["validation_objective"] > result["baseline_objective"]:
        raise ValueError("Preview requires a development-passing candidate")
    source = Path(models) / result["model_version"]
    model, _, metadata = load_bundle(source)
    if model.config.symbol != symbol:
        raise ValueError("Preview identity mismatch")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    (output / "data").mkdir()
    shutil.copytree(source, output / "models" / source.name)
    shutil.copy2(Path(data) / f"{symbol}.csv", output / "data" / f"{symbol}.csv")
    (output / "models/active.json").write_text(
        json.dumps(
            {"active": {identity(model.config): source.name}, "history": [], "deployment_status": "preview_unapproved"}
        ),
        encoding="utf-8",
    )
    (output / "preview.json").write_text(
        json.dumps(
            {
                "production_approved": False,
                "symbol": symbol,
                "bundle_version": metadata["model_version"],
                "source_run_id": result["mlflow_run_id"],
                "purpose": "Authenticated research preview",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BIAT")
    parser.add_argument("--output", default="deployment")
    args = parser.parse_args()
    print(package_preview(args.symbol, args.output))


if __name__ == "__main__":
    main()
