from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import shutil
import torch


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
    logs_dir = run_dir / "logs"
    artifacts_dir = run_dir / "artifacts"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "checkpoints_dir": checkpoints_dir,
        "logs_dir": logs_dir,
        "artifacts_dir": artifacts_dir,
    }


def write_run_metadata(
    run_dir: str | Path,
    run_name: str,
    data_dir: str | Path,
    tokenizer_path: str | Path,
    dataset_metadata_path: str | Path,
    hyperparameters: dict[str, Any],
    status: str = "running",
    best_val_loss: float | None = None,
    latest_step: int = 0,
    latest_epoch: int = 0,
) -> Path:
    run_dir = Path(run_dir)
    metadata_path = run_dir / "run_metadata.json"

    metadata = {
        "run_name": run_name,
        "data_dir": str(data_dir),
        "tokenizer_path": str(tokenizer_path),
        "dataset_metadata_path": str(dataset_metadata_path),
        "config_path": hyperparameters,
        "best_val_loss": best_val_loss,
        "latest_step": latest_step,
        "latest_epoch": latest_epoch,
        "status": status,
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


def update_run_metadata(
    run_dir: str | Path,
    **updates: Any,
) -> None:
    run_dir = Path(run_dir)
    metadata_path = run_dir / "run_metadata.json"

    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata.update(updates)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def save_config_snapshot(config_path: str | Path, run_dir: str | Path) -> Path:
    config_path = Path(config_path)
    run_dir = Path(run_dir)

    snapshot_path = run_dir / "config_snapshot.yaml"
    shutil.copy2(config_path, snapshot_path)
    return snapshot_path


def save_checkpoint(
    path: str | Path,
    model,
    opt,
    scheduler,
    epoch: int,
    step: int,
    ptr: int,
    best_val_loss: float,
    config: dict[str, Any],
    data_dir: str | Path,
    tokenizer_path: str | Path,
) -> None:
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
        "config": config,
        "data_dir": str(data_dir),
        "tokenizer_path": str(tokenizer_path),
    }

    torch.save(checkpoint, path)


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
        "config": checkpoint.get("config", {}),
        "data_dir": checkpoint.get("data_dir"),
        "tokenizer_path": checkpoint.get("tokenizer_path"),
    }


def maybe_save_step_checkpoint(
    ckpt_cfg: CheckpointConfig,
    checkpoints_dir: str | Path,
    model,
    opt,
    scheduler,
    epoch: int,
    step: int,
    ptr: int,
    best_val_loss: float,
    config: dict[str, Any],
    data_dir: str | Path,
    tokenizer_path: str | Path,
) -> None:
    if not ckpt_cfg.save_step_checkpoints:
        return
    if step % ckpt_cfg.step_checkpoint_interval != 0:
        return

    checkpoints_dir = Path(checkpoints_dir)
    step_path = checkpoints_dir / f"step_{step:07d}.pt"

    save_checkpoint(
        path=step_path,
        model=model,
        opt=opt,
        scheduler=scheduler,
        epoch=epoch,
        step=step,
        ptr=ptr,
        best_val_loss=best_val_loss,
        config=config,
        data_dir=data_dir,
        tokenizer_path=tokenizer_path,
    )

    step_ckpts = sorted(checkpoints_dir.glob("step_*.pt"))
    if len(step_ckpts) > ckpt_cfg.keep_last_n_step_checkpoints:
        for old_ckpt in step_ckpts[: -ckpt_cfg.keep_last_n_step_checkpoints]:
            old_ckpt.unlink(missing_ok=True)


def save_latest_checkpoint(
    ckpt_cfg: CheckpointConfig,
    checkpoints_dir: str | Path,
    model,
    opt,
    scheduler,
    epoch: int,
    step: int,
    ptr: int,
    best_val_loss: float,
    config: dict[str, Any],
    data_dir: str | Path,
    tokenizer_path: str | Path,
) -> None:
    if not ckpt_cfg.save_latest:
        return

    checkpoints_dir = Path(checkpoints_dir)
    latest_path = checkpoints_dir / "latest.pt"

    save_checkpoint(
        path=latest_path,
        model=model,
        opt=opt,
        scheduler=scheduler,
        epoch=epoch,
        step=step,
        ptr=ptr,
        best_val_loss=best_val_loss,
        config=config,
        data_dir=data_dir,
        tokenizer_path=tokenizer_path,
    )


def save_best_checkpoint(
    ckpt_cfg: CheckpointConfig,
    checkpoints_dir: str | Path,
    model,
    opt,
    scheduler,
    epoch: int,
    step: int,
    ptr: int,
    best_val_loss: float,
    config: dict[str, Any],
    data_dir: str | Path,
    tokenizer_path: str | Path,
) -> None:
    if not ckpt_cfg.save_best:
        return

    checkpoints_dir = Path(checkpoints_dir)
    best_path = checkpoints_dir / "best.pt"

    save_checkpoint(
        path=best_path,
        model=model,
        opt=opt,
        scheduler=scheduler,
        epoch=epoch,
        step=step,
        ptr=ptr,
        best_val_loss=best_val_loss,
        config=config,
        data_dir=data_dir,
        tokenizer_path=tokenizer_path,
    )
