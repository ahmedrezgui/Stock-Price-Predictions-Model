"""Explicit SDK v2 actions; importing this module never contacts Azure."""

import argparse
import os
import shlex
from pathlib import Path


def ml_client():
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    return MLClient(
        DefaultAzureCredential(),
        os.environ["AZURE_SUBSCRIPTION_ID"],
        os.environ["AZURE_RESOURCE_GROUP"],
        os.environ["AZURE_ML_WORKSPACE"],
    )


def submit_training(config, dataset, compute, output="outputs", mlflow_uri=None, mlflow_experiment=None):
    from azure.ai.ml import Input, Output, command
    from azure.ai.ml.entities import CommandJobLimits, Environment

    client = ml_client()
    environment = Environment(
        name="stock-forecast-training",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
        conda_file="infrastructure/azure/environment.yml",
    )
    tracking_env = {}
    if mlflow_uri:
        tracking_env["MLFLOW_TRACKING_URI"] = mlflow_uri
    if mlflow_experiment:
        tracking_env["MLFLOW_EXPERIMENT_NAME"] = mlflow_experiment
    job = command(
        code=".",
        command=f"python -m scripts.train_model --data ${{{{inputs.data}}}} --config {shlex.quote(str(config))} --artifacts ${{{{outputs.models}}}} --output ${{{{outputs.models}}}}/reports",
        inputs={"data": Input(type="uri_file", path=dataset)},
        outputs={"models": Output(type="uri_folder", mode="rw_mount")},
        environment=environment,
        environment_variables=tracking_env,
        compute=compute,
        experiment_name=mlflow_experiment or "stock-return-forecasting",
        display_name="stock-return-training",
        limits=CommandJobLimits(timeout=3600),
    )
    return client.jobs.create_or_update(job)


def register_bundle(path, name):
    from azure.ai.ml.entities import Model
    from .artifacts import load_bundle

    _, _, meta = load_bundle(path)
    return ml_client().models.create_or_update(
        Model(
            name=name,
            path=path,
            type="custom_model",
            description="Verified recurrent weights and frozen preprocessing",
            tags={"dataset_version": meta["dataset"]["dataset_version"], "model_version": meta["model_version"]},
        )
    )


def deploy(endpoint_name, model_id, image, instance_type):
    # This older path could update a deployment without binding its weights to a gate.
    # Keep it closed until the Azure ML managed-endpoint adapter verifies that binding.
    raise ValueError(
        "Direct Azure ML deployment is disabled: use the gated forecast-release workflow "
        "for App Service. A gate-aware Azure ML managed-endpoint adapter is required."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Azure actions require explicit approval; this flag records the operator's decision"
    )
    parser.add_argument("action", choices=["train", "register", "download", "deploy"])
    parser.add_argument("--approve-cloud-action", action="store_true")
    parser.add_argument("--config", default="configs/forecast.json")
    parser.add_argument("--dataset")
    parser.add_argument("--compute")
    parser.add_argument("--bundle")
    parser.add_argument("--name")
    parser.add_argument("--version")
    parser.add_argument("--output", default="artifacts/downloaded")
    parser.add_argument("--model-id")
    parser.add_argument("--image")
    parser.add_argument("--instance-type")
    args = parser.parse_args()
    if args.action != "download" and not args.approve_cloud_action:
        parser.error("Explicit approval required before cloud writes, jobs or deployment")
    requirements = {
        "train": ("dataset", "compute"),
        "register": ("bundle", "name"),
        "download": ("name", "version"),
        "deploy": ("name", "model_id", "image", "instance_type"),
    }
    for key in requirements[args.action]:
        if not getattr(args, key):
            parser.error(f"--{key.replace('_', '-')} is required")
    if args.action == "train":
        if not Path(args.config).is_file():
            parser.error("Config file not found")
        result = submit_training(args.config, args.dataset, args.compute)
    elif args.action == "register":
        result = register_bundle(args.bundle, args.name)
    elif args.action == "download":
        result = ml_client().models.download(args.name, args.version, download_path=args.output)
    else:
        result = deploy(args.name, args.model_id, args.image, args.instance_type)
    print(getattr(result, "name", result))


if __name__ == "__main__":
    main()
