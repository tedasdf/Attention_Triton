import random
import time
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm import tqdm
import structlog

# train.py
from model.config import GPTConfig, Hyperparameters
from model.transformer import GPT
from utils.data import get_titles, BPETokenizer, train_tokenizer
from utils.logger import WandbLogger
import mlflow
import wandb
import os


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


def main(parser):
    h = Hyperparameters()

    data_dir = Path(parser.data_dir)
    output_dir = Path(parser.output_dir)
    vocab_path = data_dir / "vocab.json"

    print("-" * 30)
    print(f"📂 Data Dir: {data_dir.absolute()} | Exists: {data_dir.exists()}")
    print(f"📂 Output Dir: {output_dir.absolute()} | Exists: {output_dir.exists()}")
    print(f"📄 Vocab Path: {vocab_path} | Exists: {vocab_path.exists()}")
    print("-" * 30)
    # output_dir.mkdir(parents=True, exist_ok=True)

    # Inside your main()
    if parser.smoke_test:
        print("🚀 SMOKE TEST MODE: Running 1 epoch, 1 batch only.")
        h.epochs = 1
        h.num_titles = 100
        batches = 1  # Force it to just one iteration
        print("⚠️ WANDB_API_KEY not found in environment. WandB might fail.")
    else:
        # Use the environment variable if available
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)

    torch.manual_seed(h.seed)
    random.seed(h.seed)

    global logger
    logger = configure_logging(h.log_file)

    hyperparams_dict = vars(h)
    logger.log("hyperparameters_configured", **hyperparams_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log("device_info", device=device)

    train_titles, val_titles = get_titles(
        h.num_titles,
        h.seed,
        h.val_frac,
        smoke_test=parser.smoke_test,
        data_dir=data_dir,
    )

    # 2. Tokenizer: Don't retrain if not necessary
    if vocab_path.exists():
        logger.log("tokenizer_info", status="loading_existing", path=str(vocab_path))
        tok = BPETokenizer.load(vocab_path)  # Assumes you have a .load() method
    else:
        logger.log("tokenizer_info", status="training_new")
        vocab = train_tokenizer(
            train_titles + val_titles, h.vocab_size, eos_token="<eos>"
        )
        tok = BPETokenizer(vocab)

    eos_token = "<eos>"
    tok = BPETokenizer(
        train_tokenizer(train_titles + val_titles, h.vocab_size, eos_token=eos_token)
    )
    train_text = eos_token.join(train_titles) + eos_token
    val_text = eos_token.join(val_titles) + eos_token
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)

    batches = len(train_ids) // (h.block_size * h.batch_size)
    max_steps = h.epochs * batches
    eval_interval = batches // h.evals_per_epoch
    logger.log(
        "dataset_info",
        titles_count=len(train_titles),
        epochs=h.epochs,
        batches_per_epoch=batches,
        tokens_per_epoch=len(train_ids),
        vocab_size=tok.vocab_size,
    )

    model_cfg = GPTConfig.from_flat(h)
    model = GPT(model_cfg).to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("model_info", parameters_count=model_params)

    opt = torch.optim.SGD(model.parameters(), lr=h.lr, weight_decay=h.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)

    def evaluate():
        model.eval()
        losses = 0.0
        with torch.no_grad():
            for xb, yb in iter_full_split(val_ids, h.block_size, h.batch_size, device):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction="sum")
                losses += loss.item()
        model.train()
        return losses / len(val_text)

    wandb_logger = WandbLogger(
        project="ntp-transformer", config=h, enabled=not parser.smoke_test
    )

    ptr = 0
    step = 0
    t0 = time.time()
    for epoch in range(1, h.epochs + 1):
        for _ in tqdm(range(1, batches + 1), desc=f"Epoch {epoch}/{h.epochs}"):
            step += 1
            xb, yb, ptr = get_batch(train_ids, ptr, h.block_size, h.batch_size, device)
            _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            elapsed = time.time() - t0
            logger.log(
                "training_step",
                step=step,
                max_steps=max_steps,
                loss=loss.item(),
                elapsed_time=elapsed,
                prnt=False,
            )

            wandb_logger.log_metrics(
                {"train/loss": loss, "train/epoch": epoch, "system/lr": 1e-4},
                step=epoch,
            )

            if (
                step == 1 or step % eval_interval == 0 or step == max_steps
            ) or parser.smoke_test:
                val_loss = evaluate()
                logger.log(
                    "validation_step",
                    step=step,
                    max_steps=max_steps,
                    loss=val_loss,
                    elapsed_time=elapsed,
                )

    if not parser.smoke_test:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        with mlflow.start_run(run_name="production_candidate"):
            mlflow.log_param("total_epochs", 10)
            mlflow.pytorch.log_model(model, "ntp_model")

    if not parser.smoke_test:
        current_loss = val_loss
        best_loss_path = output_dir / "best_loss.txt"
        model_save_path = output_dir / "model.pt"  # Standard name for DVC to track
        metadata_path = Path(
            "run_metadata.env"
        )  # Keep in root for the shell script to find

        is_better = True
        if best_loss_path.exists():
            best_loss = float(best_loss_path.read_text().strip())
            if current_loss >= best_loss:
                is_better = False
                print(
                    f"📉 No improvement. Current: {current_loss:.4f} | Best: {best_loss:.4f}"
                )

        if is_better:
            print(f"🏆 New Best Model! Loss: {current_loss:.4f}")
            model_save_path.parent.mkdir(parents=True, exist_ok=True)

            # 1. Save weights
            torch.save(model.state_dict(), model_save_path)
            # 2. Update best_loss record
            best_loss_path.write_text(f"{current_loss:.4f}")

            # 3. Create metadata for the push_model.sh script
            # Note: We write this to the ROOT so the shell script sees it easily
            with open(metadata_path, "w") as f:
                f.write(
                    f"WANDB_RUN_NAME={wandb.run.name if wandb.run else 'offline'}\n"
                )
                f.write(
                    f"MLFLOW_RUN_ID={mlflow.active_run().info.run_id if mlflow.active_run() else 'none'}\n"
                )
                f.write(f"VAL_LOSS={current_loss:.4f}\n")

    wandb_logger.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NTP Transformer Training Pipeline")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to datasets (symlinked or local)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Where to save model.pt and best_loss.txt",
    )

    parser.add_argument(
        "--smoke-test", action="store_true", help="Run a quick 1-batch validation"
    )

    parser = parser.parse_args()

    try:
        main(parser)
    finally:
        if logger and hasattr(logger, "file_handler"):
            logger.file_handler.close()
