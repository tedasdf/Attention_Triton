import random
import time
import json
from pathlib import Path
import math

import torch
from torch.nn import functional as F
from tqdm import tqdm
import structlog

# train.py
from main.checkpoint import (
    write_run_info,
    create_run_dir,
    save_checkpoint,
    write_config_snapshot,
    append_run_registry,
)
from model.config import GPTConfig, Hyperparameters
from model.transformer import GPT
from utils.data import BPETokenizer
from utils.logger import WandbLogger, configure_wandb_metrics
import wandb
import os
import numpy as np
from omegaconf import OmegaConf


def merge_sweep_config(cfg, sweep_cfg: dict):
    """
    Merge flat W&B sweep parameters into the OmegaConf hyperparameter config.
    Only keys already present in cfg should be overridden.
    """
    valid_keys = set(cfg.keys())
    filtered = {k: v for k, v in sweep_cfg.items() if k in valid_keys}
    return OmegaConf.merge(cfg, OmegaConf.create(filtered))


def configure_logging(log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    file_handler = open(log_file, "w")

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    class DualLogger:
        def __init__(self, file_handler):
            self.file_handler = file_handler
            self.logger = structlog.get_logger()

        def log(self, event, **kwargs):
            log_entry = json.dumps({"event": event, "timestamp": time.time(), **kwargs})
            self.file_handler.write(log_entry + "\n")
            self.file_handler.flush()

            if kwargs.get("prnt", True):
                if "step" in kwargs and "max_steps" in kwargs:
                    tqdm.write(
                        f"[{kwargs.get('step'):>5}/{kwargs.get('max_steps')}] {event}: loss={kwargs.get('loss', 'N/A'):.6f} time={kwargs.get('elapsed_time', 0):.2f}s"
                    )
                else:
                    parts = [
                        f"{k}={v}"
                        for k, v in kwargs.items()
                        if k not in ["prnt", "timestamp"]
                    ]
                    if parts:
                        tqdm.write(f"{event}: {', '.join(parts)}")
                    else:
                        tqdm.write(event)

    return DualLogger(file_handler)


logger = None


def get_batch(
    split_ids: torch.Tensor,
    ptr: int,
    block_size: int,
    batch_size: int,
    device: torch.device,
):
    span = block_size * batch_size + 1
    if ptr + span >= len(split_ids):
        ptr = 0
    batch = split_ids[ptr : ptr + span]
    x = batch[:-1].view(batch_size, block_size).to(device)
    y = batch[1:].view(batch_size, block_size).to(device)
    return x, y, ptr + block_size * batch_size


def iter_full_split(
    split_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device
):
    span = block_size * batch_size + 1
    for ptr in range(0, len(split_ids) - span + 1, span):
        batch = split_ids[ptr : ptr + span]
        x = batch[:-1].view(batch_size, block_size).to(device)
        y = batch[1:].view(batch_size, block_size).to(device)
        yield x, y


def cross_entropy_with_z_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    z_loss_weight: float = 1e-4,
    ignore_index: int = -100,
):
    """
    logits:  [B, T, V]
    targets: [B, T]
    """
    B, T, V = logits.shape

    flat_logits = logits.reshape(B * T, V)
    flat_targets = targets.reshape(B * T)

    ce_loss = F.cross_entropy(
        flat_logits,
        flat_targets,
        ignore_index=ignore_index,
    )

    if z_loss_weight == 0.0:
        return ce_loss, ce_loss.detach(), torch.tensor(0.0, device=logits.device)

    log_z = torch.logsumexp(flat_logits, dim=-1)  # [B*T]
    valid_mask = flat_targets != ignore_index

    if valid_mask.any():
        z_loss = (log_z[valid_mask] ** 2).mean()
    else:
        z_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    total_loss = ce_loss + z_loss_weight * z_loss
    return total_loss, ce_loss.detach(), z_loss.detach()


def main(parser):
    # -------------------------
    # config
    # -------------------------
    if parser.sweep:
        wandb.init(project=parser.wandb_project)
        config_path = wandb.config["config_path"]
    else:
        config_path = parser.config_path

    schema = OmegaConf.structured(Hyperparameters)
    loaded_cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(schema, loaded_cfg.hyperparameters)

    if parser.sweep:
        cfg = merge_sweep_config(cfg, dict(wandb.config))

    run_paths = create_run_dir(output_root=cfg.output_dir)
    run_dir = Path(run_paths["run_dir"])
    checkpoints_dir = Path(run_paths["checkpoints_dir"])
    output_dir = Path(cfg.output_dir)

    data_dir = Path(cfg.data_dir)
    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"
    tokenizer_path = data_dir / "vocab.json"
    metadata_path = data_dir / "metadata.json"

    def get_wandb_info():
        wb_run = getattr(wandb, "run", None)
        if wb_run is None:
            return None
        return {
            "enabled": True,
            "project": getattr(wb_run, "project", None),
            "entity": getattr(wb_run, "entity", None),
            "id": getattr(wb_run, "id", None),
            "name": getattr(wb_run, "name", None),
            "url": getattr(wb_run, "url", None),
        }

    print("-" * 30)
    print(f"📂 Data Dir: {data_dir.absolute()} | Exists: {data_dir.exists()}")
    print(f"📂 Output Dir: {output_dir.absolute()} | Exists: {output_dir.exists()}")
    print(f"📄 Train Path: {train_path} | Exists: {train_path.exists()}")
    print(f"📄 Val Path: {val_path} | Exists: {val_path.exists()}")
    print(f"📄 Tokenizer Path: {tokenizer_path} | Exists: {tokenizer_path.exists()}")
    print(f"📄 Metadata Path: {metadata_path} | Exists: {metadata_path.exists()}")
    print("-" * 30)

    if not parser.smoke_test:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key and not parser.sweep:
            wandb.login(key=wandb_api_key)
    else:
        print("🚀 SMOKE TEST MODE enabled.")

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device.type}")

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train.bin at {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing val.bin at {val_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing vocab.json at {tokenizer_path}")

    tok = BPETokenizer.load(tokenizer_path)

    train_ids = torch.from_numpy(
        np.fromfile(train_path, dtype=np.uint16).astype(np.int64)
    ).long()

    val_ids = torch.from_numpy(
        np.fromfile(val_path, dtype=np.uint16).astype(np.int64)
    ).long()

    # -------------------------
    # smoke test on token tensors directly
    # -------------------------
    if parser.smoke_test:
        min_tokens_needed = cfg.block_size * cfg.batch_size + 1
        smoke_train_tokens = min(
            len(train_ids),
            max(min_tokens_needed, cfg.block_size * cfg.batch_size * 2),
        )
        smoke_val_tokens = min(
            len(val_ids),
            max(min_tokens_needed, cfg.block_size * cfg.batch_size * 2),
        )

        if smoke_train_tokens < min_tokens_needed:
            raise ValueError(
                f"Smoke test train split too small. Need at least {min_tokens_needed} tokens, "
                f"got {smoke_train_tokens}."
            )
        if smoke_val_tokens < min_tokens_needed:
            raise ValueError(
                f"Smoke test val split too small. Need at least {min_tokens_needed} tokens, "
                f"got {smoke_val_tokens}."
            )

        train_ids = train_ids[:smoke_train_tokens]
        val_ids = val_ids[:smoke_val_tokens]

        print(f"SMOKE train tokens: {len(train_ids)}")
        print(f"SMOKE val tokens: {len(val_ids)}")

    print("vocab size:", tok.vocab_size)
    print("train min/max:", int(train_ids.min()), int(train_ids.max()))
    print("val min/max:", int(val_ids.min()), int(val_ids.max()))

    # -------------------------
    # filtered dataset metadata for wandb only
    # -------------------------
    dataset_metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            dataset_metadata = json.load(f)

        for key in [
            "dataset_path",
            "data_dir",
            "train_path",
            "val_path",
            "tokenizer_path",
            "metadata_path",
        ]:
            dataset_metadata.pop(key, None)

    token_budget_per_batch = cfg.block_size * cfg.batch_size
    batches = max(1, len(train_ids) // token_budget_per_batch)
    steps_per_epoch = max(1, math.ceil(batches / cfg.accumulation_steps))

    model_cfg = GPTConfig.from_flat(cfg)
    model = GPT(model_cfg).to(device)
    model_params = int(sum(p.numel() for p in model.parameters()))
    trainable_model_params = int(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )

    target_train_tokens = 20 * model_params
    tokens_per_epoch = max(1, len(train_ids))
    epochs = max(1, math.ceil(target_train_tokens / tokens_per_epoch))
    max_steps = epochs * steps_per_epoch
    eval_interval = max(1, batches // cfg.evals_per_epoch)

    if parser.smoke_test:
        epochs = 1
        batches = 1
        steps_per_epoch = 1
        max_steps = 1
        eval_interval = 1

    print(
        f"train_tokens={len(train_ids)}, val_tokens={len(val_ids)}, "
        f"epochs={epochs}, batches_per_epoch={batches}, vocab_size={tok.vocab_size}"
    )
    print(
        f"target_train_tokens={target_train_tokens}, "
        f"tokens_per_epoch={tokens_per_epoch}, computed_epochs={epochs}"
    )
    print(f"model parameters={model_params}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )

    warmup_steps = cfg.warmup_step
    cosine_steps = max(1, max_steps - warmup_steps)

    scheduler_decay = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cosine_steps
    )
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[scheduler_warmup, scheduler_decay],
        milestones=[warmup_steps],
    )

    use_bf16 = (
        device.type == "cuda"
        and getattr(cfg, "use_bfloat16", False)
        and torch.cuda.is_bf16_supported()
    )
    torch.set_float32_matmul_precision("high")

    def has_nonfinite_gradients(model) -> bool:
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return True
        return False

    def evaluate():
        model.eval()
        losses = 0.0
        total_tokens = 0

        with torch.no_grad():
            for xb, yb in iter_full_split(
                val_ids, cfg.block_size, cfg.batch_size, device
            ):
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_bf16,
                ):
                    logits, _ = model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(
                    logits.float().view(-1, V),
                    yb.view(-1),
                    reduction="sum",
                )
                losses += loss.item()
                total_tokens += yb.numel()

        model.train()
        return losses / total_tokens

    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb_cfg["model_params"] = model_params
    wandb_cfg["trainable_model_params"] = trainable_model_params
    wandb_cfg["run_dir"] = str(run_dir)
    wandb_cfg["checkpoints_dir"] = str(checkpoints_dir)

    if dataset_metadata:
        wandb_cfg["dataset_metadata"] = dataset_metadata

    if parser.sweep:
        configure_wandb_metrics()
        wandb.config.update(wandb_cfg, allow_val_change=True)

        class SweepLogger:
            def log_metrics(self, metrics, step=None):
                if step is not None:
                    metrics = dict(metrics)
                    metrics["global_step"] = int(step)
                wandb.log(metrics, step=step)

            def finish(self):
                wandb.finish()

        wandb_logger = SweepLogger()
    else:
        wandb_logger = WandbLogger(
            project=parser.wandb_project,
            config=wandb_cfg,
            enabled=not parser.smoke_test,
        )

    ptr = 0
    global_step = 0
    optimizer_step = 0
    best_val_loss = float("inf")
    nonfinite_events_total = 0
    z_loss_weight = cfg.z_loss_weight
    t0 = time.perf_counter()
    start_epoch = 1

    # write snapshot + minimal run info before training starts
    write_config_snapshot(
        run_dir=run_dir,
        loaded_cfg=loaded_cfg,
        cfg=cfg,
        parser=parser,
    )

    write_run_info(
        run_dir=run_dir,
        output_root=output_dir,
        status="started",
        config_path=str(config_path),
        smoke_test=bool(parser.smoke_test),
        resume=bool(parser.resume),
        sweep=bool(parser.sweep),
        data_dir=None,
        model_params=model_params,
        trainable_model_params=trainable_model_params,
        latest_epoch=start_epoch,
        latest_step=global_step,
        best_val_loss=None,
        wandb_info=get_wandb_info(),
    )

    append_run_registry(
        output_root=output_dir,
        run_dir=run_dir,
        config_path=str(config_path),
        status="started",
        wandb_info=get_wandb_info(),
    )

    try:
        if parser.resume:
            raise ValueError(
                "parser.resume=True, but resume source is not defined in this new flow yet. "
                "Use an explicit local checkpoint path or a W&B artifact before enabling resume."
            )

        for epoch in range(start_epoch, epochs + 1):
            for i in tqdm(range(1, batches + 1), desc=f"Epoch {epoch}/{epochs}"):
                global_step += 1

                step_start = time.perf_counter()

                data_start = time.perf_counter()
                xb, yb, ptr = get_batch(
                    train_ids, ptr, cfg.block_size, cfg.batch_size, device
                )
                data_time = time.perf_counter() - data_start

                compute_start = time.perf_counter()
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_bf16,
                ):
                    logits, loss = model(xb, yb)

                loss, ce_loss, z_loss = cross_entropy_with_z_loss(
                    logits,
                    yb,
                    z_loss_weight=z_loss_weight,
                    ignore_index=-100,
                )
                loss.backward()

                grad_norm = None
                optimizer_updated = False

                if (i % cfg.accumulation_steps == 0) or (i == batches):
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    scheduler.step()
                    opt.zero_grad(set_to_none=True)
                    optimizer_step += 1
                    optimizer_updated = True

                compute_time = time.perf_counter() - compute_start
                step_time = time.perf_counter() - step_start

                tokens_this_step = int(yb.numel())
                samples_this_step = int(xb.size(0))
                tokens_per_sec = tokens_this_step / max(step_time, 1e-8)
                samples_per_sec = samples_this_step / max(step_time, 1e-8)

                loss_is_finite = bool(torch.isfinite(loss).item())
                grads_are_finite = not has_nonfinite_gradients(model)
                nan_or_inf_flag = 0 if (loss_is_finite and grads_are_finite) else 1
                nonfinite_events_total += nan_or_inf_flag

                current_lr = float(scheduler.get_last_lr()[0])

                if torch.cuda.is_available():
                    gpu_mem_allocated_mb = float(
                        torch.cuda.memory_allocated() / (1024**2)
                    )
                    gpu_mem_reserved_mb = float(
                        torch.cuda.memory_reserved() / (1024**2)
                    )
                    gpu_peak_allocated_mb = float(
                        torch.cuda.max_memory_allocated() / (1024**2)
                    )
                    gpu_peak_reserved_mb = float(
                        torch.cuda.max_memory_reserved() / (1024**2)
                    )
                else:
                    gpu_mem_allocated_mb = 0.0
                    gpu_mem_reserved_mb = 0.0
                    gpu_peak_allocated_mb = 0.0
                    gpu_peak_reserved_mb = 0.0

                elapsed_time = time.perf_counter() - t0

                train_metrics = {
                    "epoch": int(epoch),
                    "train/loss": float(loss.item()),
                    "train/ce_loss": float(ce_loss.item())
                    if torch.is_tensor(ce_loss)
                    else float(ce_loss),
                    "train/z_loss": float(z_loss.item())
                    if torch.is_tensor(z_loss)
                    else float(z_loss),
                    "train/epoch": int(epoch),
                    "runtime/step_time_sec": float(step_time),
                    "runtime/data_time_sec": float(data_time),
                    "runtime/compute_time_sec": float(compute_time),
                    "runtime/tokens_per_sec": float(tokens_per_sec),
                    "runtime/samples_per_sec": float(samples_per_sec),
                    "runtime/lr": float(current_lr),
                    "runtime/gpu_mem_allocated_mb": float(gpu_mem_allocated_mb),
                    "runtime/gpu_mem_reserved_mb": float(gpu_mem_reserved_mb),
                    "runtime/gpu_peak_allocated_mb": float(gpu_peak_allocated_mb),
                    "runtime/gpu_peak_reserved_mb": float(gpu_peak_reserved_mb),
                    "runtime/elapsed_time_sec": float(elapsed_time),
                    "health/nan_or_inf_flag": int(nan_or_inf_flag),
                    "health/nonfinite_events_total": int(nonfinite_events_total),
                    "debug/optimizer_step": int(optimizer_step),
                    "debug/optimizer_updated": int(optimizer_updated),
                }

                if grad_norm is not None:
                    train_metrics["health/grad_norm"] = float(grad_norm)

                wandb_logger.log_metrics(train_metrics, step=global_step)

                if (
                    global_step == 1
                    or global_step % eval_interval == 0
                    or global_step == max_steps
                    or parser.smoke_test
                ):
                    eval_start = time.perf_counter()
                    val_loss = evaluate()
                    eval_time = time.perf_counter() - eval_start

                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss

                    checkpoint_start = time.perf_counter()

                    aliases = ["latest"]
                    if is_best:
                        aliases.append("best")

                    save_checkpoint(
                        path=checkpoints_dir / "latest.pt",
                        model=model,
                        opt=opt,
                        scheduler=scheduler,
                        epoch=epoch,
                        step=global_step,
                        ptr=ptr,
                        best_val_loss=best_val_loss,
                        wandb_run=getattr(wandb, "run", None),
                        artifact_name=f"{run_dir.name}-checkpoint",
                        aliases=aliases,
                        upload_to_wandb=(
                            not parser.smoke_test
                            and getattr(wandb, "run", None) is not None
                        ),
                    )

                    checkpoint_time = time.perf_counter() - checkpoint_start

                    write_run_info(
                        run_dir=run_dir,
                        output_root=output_dir,
                        status="running",
                        config_path=str(config_path),
                        smoke_test=bool(parser.smoke_test),
                        resume=bool(parser.resume),
                        sweep=bool(parser.sweep),
                        data_dir=None,
                        model_params=model_params,
                        trainable_model_params=trainable_model_params,
                        latest_epoch=epoch,
                        latest_step=global_step,
                        best_val_loss=float(best_val_loss),
                        wandb_info=get_wandb_info(),
                    )

                    wandb_logger.log_metrics(
                        {
                            "eval/loss": float(val_loss),
                            "eval/time_sec": float(eval_time),
                            "checkpoint/save_time_sec": float(checkpoint_time),
                        },
                        step=global_step,
                    )

        write_run_info(
            run_dir=run_dir,
            output_root=output_dir,
            status="completed",
            config_path=str(config_path),
            smoke_test=bool(parser.smoke_test),
            resume=bool(parser.resume),
            sweep=bool(parser.sweep),
            data_dir=None,
            model_params=model_params,
            trainable_model_params=trainable_model_params,
            latest_epoch=epoch if epochs > 0 else 0,
            latest_step=global_step,
            best_val_loss=None
            if best_val_loss == float("inf")
            else float(best_val_loss),
            wandb_info=get_wandb_info(),
        )

    except Exception as e:
        write_run_info(
            run_dir=run_dir,
            output_root=output_dir,
            status="failed",
            config_path=str(config_path),
            smoke_test=bool(parser.smoke_test),
            resume=bool(parser.resume),
            sweep=bool(parser.sweep),
            data_dir=None,
            model_params=model_params,
            trainable_model_params=trainable_model_params,
            latest_epoch=(epoch if "epoch" in locals() else None),
            latest_step=global_step,
            best_val_loss=None
            if best_val_loss == float("inf")
            else float(best_val_loss),
            error=str(e),
            wandb_info=get_wandb_info(),
        )
        raise

    finally:
        wandb_logger.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NTP Transformer Training Pipeline")

    parser.add_argument(
        "--config_path",
        type=str,
        default="main/config/base.yaml",
        help="Config path",
    )

    parser.add_argument(
        "--smoke-test", action="store_true", help="Run a quick 1-batch validation"
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Enable W&B sweep mode and override config from wandb.config",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="scaling_law",
        help="W&B project name",
    )

    parser.add_argument(
        "--resume", action="store_true", help="Resume previously stored checkpoint"
    )

    parser = parser.parse_args()

    try:
        main(parser)
    finally:
        if logger and hasattr(logger, "file_handler"):
            logger.file_handler.close()
