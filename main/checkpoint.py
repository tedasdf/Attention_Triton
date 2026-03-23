from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import torch
from omegaconf import OmegaConf


@dataclass
class CheckpointConfig:
    save_latest: bool = True
    save_best: bool = True
    save_step_checkpoints: bool = True
    step_checkpoint_interval: int = 1000
    keep_last_n_step_checkpoints: int = 3


def create_run_dir(
    output_root: str | Path, run_name: str | None = None
) -> dict[str, Path]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        import time

        run_name = time.strftime("run_%Y_%m_%d_%H%M%S")

    run_dir = output_root / run_name
    checkpoints_dir = run_dir / "checkpoints"

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "checkpoints_dir": checkpoints_dir,
    }


def write_config_snapshot(run_dir: Path, loaded_cfg, cfg, parser) -> None:
    snapshot_path = run_dir / "config_snapshot.yaml"

    snapshot_payload = OmegaConf.create(
        {
            "source_config_path": str(getattr(parser, "config_path", "")),
            "smoke_test": bool(getattr(parser, "smoke_test", False)),
            "resume": bool(getattr(parser, "resume", False)),
            "sweep": bool(getattr(parser, "sweep", False)),
            "loaded_config": OmegaConf.to_container(loaded_cfg, resolve=True),
            "resolved_hyperparameters": OmegaConf.to_container(cfg, resolve=True),
        }
    )

    OmegaConf.save(config=snapshot_payload, f=snapshot_path)


def write_run_info(
    run_dir: Path,
    output_root: Path,
    status: str,
    *,
    config_path: str | None = None,
    smoke_test: bool = False,
    resume: bool = False,
    sweep: bool = False,
    data_dir: str | Path | None = None,
    model_params: int | None = None,
    trainable_model_params: int | None = None,
    latest_epoch: int | None = None,
    latest_step: int | None = None,
    best_val_loss: float | None = None,
    error: str | None = None,
    wandb_info: dict[str, Any] | None = None,
) -> None:
    payload = {
        "status": status,
        "run_dir": str(run_dir),
        "output_root": str(output_root),
        "config_path": config_path,
        "smoke_test": smoke_test,
        "resume": resume,
        "sweep": sweep,
        "data_dir": str(data_dir) if data_dir is not None else None,
        "model_params": model_params,
        "trainable_model_params": trainable_model_params,
        "latest_epoch": latest_epoch,
        "latest_step": latest_step,
        "best_val_loss": best_val_loss,
        "error": error,
        "wandb": wandb_info
        or {
            "enabled": False,
            "project": None,
            "entity": None,
            "id": None,
            "name": None,
            "url": None,
        },
    }

    with open(run_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_run_registry(
    output_root: Path,
    run_dir: Path,
    *,
    config_path: str | None = None,
    status: str,
    wandb_info: dict[str, Any] | None = None,
) -> None:
    wandb_info = wandb_info or {}

    row = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "config_path": config_path,
        "status": status,
        "wandb_project": wandb_info.get("project"),
        "wandb_entity": wandb_info.get("entity"),
        "wandb_id": wandb_info.get("id"),
        "wandb_name": wandb_info.get("name"),
        "wandb_url": wandb_info.get("url"),
    }

    registry_path = output_root / "run_registry.jsonl"
    with open(registry_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def save_checkpoint(
    path: str | Path,
    model,
    opt,
    scheduler,
    epoch: int,
    step: int,
    ptr: int,
    best_val_loss: float,
    *,
    wandb_run=None,
    artifact_name: str | None = None,
    aliases: list[str] | None = None,
    upload_to_wandb: bool = False,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict() if opt is not None else None,
        "scheduler_state_dict": scheduler.state_dict()
        if scheduler is not None
        else None,
        "epoch": epoch,
        "step": step,
        "ptr": ptr,
        "best_val_loss": best_val_loss,
    }

    torch.save(checkpoint, path)

    if upload_to_wandb:
        if wandb_run is None:
            raise ValueError("upload_to_wandb=True but wandb_run is None")

        import wandb

        artifact = wandb.Artifact(
            name=artifact_name or f"{wandb_run.id}-checkpoint",
            type="model",
            metadata={
                "epoch": epoch,
                "step": step,
                "best_val_loss": best_val_loss,
                "local_path": str(path),
            },
        )
        artifact.add_file(str(path), name=path.name)
        wandb_run.log_artifact(artifact, aliases=aliases or ["latest"])

    return path


def load_checkpoint(
    path: str | Path,
    model,
    opt=None,
    scheduler=None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    path = Path(path)
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if opt is not None and checkpoint.get("optimizer_state_dict") is not None:
        opt.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "ptr": checkpoint.get("ptr", 0),
        "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
    }
