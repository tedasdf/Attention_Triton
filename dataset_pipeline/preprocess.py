from dataclasses import dataclass
from pathlib import Path

import json

from datasets import load_dataset
from omegaconf import OmegaConf
from tokenizers import Tokenizer
import torch
from tokenizers import models, trainers, pre_tokenizers, decoders


@dataclass
class DatasetConfig:
    source_type: str = "huggingface"
    dataset_name: str = "julien040/hacker-news-posts"
    dataset_split: str = "train"
    text_field: str = "title"
    output_dir: str = "artifacts/datasets/hn_v1"
    cache_dir: str = "data/hf_cache"
    num_samples: int = 100000
    seed: int = 1337
    val_frac: float = 0.1


@dataclass
class TokenizerConfig:
    vocab_size: int = 16000
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    reuse_existing: bool = True
    vocab_filename: str = "tokenizer.json"


@dataclass
class AppConfig:
    dataset: DatasetConfig
    tokenizer: TokenizerConfig


class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tk = tokenizer
        self.stoi = {tok: i for tok, i in tokenizer.get_vocab().items()}
        self.itos = {i: tok for tok, i in tokenizer.get_vocab().items()}

    def encode(self, s: str) -> list[int]:
        return self.tk.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self):
        return self.tk.get_vocab_size()

    @classmethod
    def load(cls, path: Path):
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: Path):
        self.tk.save(str(path))


def train_tokenizer(
    titles: list[str],
    vocab_size: int,
    unk_token: str = "<unk>",
    pad_token: str = "<pad>",
    eos_token: str = "<eos>",
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=[pad_token, eos_token, unk_token]
    )
    tokenizer.train_from_iterator(titles, trainer)
    return tokenizer


def load_config(config_path: str) -> AppConfig:
    schema = OmegaConf.structured(AppConfig)
    loaded_cfg = OmegaConf.load(config_path)
    merged = OmegaConf.merge(schema, loaded_cfg)
    cfg_dict = OmegaConf.to_container(merged, resolve=True)
    return AppConfig(
        dataset=DatasetConfig(**cfg_dict["dataset"]),
        tokenizer=TokenizerConfig(**cfg_dict["tokenizer"]),
    )


def download_and_preprocess_huggingface_dataset(
    cfg: DatasetConfig,
    smoke_test: bool = False,
) -> tuple[list[str], list[str]]:
    if cfg.source_type != "huggingface":
        raise ValueError(f"Expected source_type='huggingface', got {cfg.source_type}")

    streaming = not smoke_test

    ds = load_dataset(
        cfg.dataset_name,
        split=cfg.dataset_split,
        streaming=streaming,
    )

    ds = ds.shuffle(seed=cfg.seed, buffer_size=10_000)

    num_samples = 100 if smoke_test else cfg.num_samples
    sampled_rows = list(ds.take(num_samples))

    titles = [
        row["title"].strip()
        for row in sampled_rows
        if row.get("title") and row["title"].strip()
    ]

    n_train = int(len(titles) * (1 - cfg.val_frac))
    return titles[:n_train], titles[n_train:]


def write_dataset_metadata(
    cfg: DatasetConfig,
    tokenizer_cfg: TokenizerConfig,
    train_samples: list[str],
    val_samples: list[str],
    output_dir: Path,
) -> None:
    metadata = {
        "dataset_name": cfg.dataset_name,
        "source_type": cfg.source_type,
        "dataset_split": cfg.dataset_split,
        "text_field": cfg.text_field,
        "seed": cfg.seed,
        "val_frac": cfg.val_frac,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "tokenizer": {
            "vocab_size": tokenizer_cfg.vocab_size,
            "eos_token": tokenizer_cfg.eos_token,
            "reuse_existing": tokenizer_cfg.reuse_existing,
            "vocab_filename": tokenizer_cfg.vocab_filename,
        },
    }

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main(parser) -> None:
    app_cfg = load_config(parser.config_path)
    dataset_cfg = app_cfg.dataset
    tokenizer_cfg = app_cfg.tokenizer

    output_dir = Path(dataset_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_cfg.source_type == "huggingface":
        train_titles, val_titles = download_and_preprocess_huggingface_dataset(
            dataset_cfg,
            smoke_test=parser.smoke_test,
        )
    else:
        raise NotImplementedError(
            f"source_type '{dataset_cfg.source_type}' is not implemented yet."
        )

    # train_path = output_dir / "train.jsonl"
    # val_path = output_dir / "val.jsonl"

    # write_jsonl(train_samples, train_path)
    # write_jsonl(val_samples, val_path)
    write_dataset_metadata(
        dataset_cfg,
        tokenizer_cfg,
        train_titles,
        val_titles,
        output_dir,
    )

    print(f"Saved metadata to:      {output_dir / 'metadata.json'}")
    print(f"Train samples: {len(train_titles)}")
    print(f"Val samples:   {len(val_titles)}")

    eos_token = tokenizer_cfg.eos_token
    vocab_path = output_dir / tokenizer_cfg.vocab_filename

    if tokenizer_cfg.reuse_existing and vocab_path.exists():
        tok = BPETokenizer.load(vocab_path)
    else:
        vocab = train_tokenizer(
            train_titles, tokenizer_cfg.vocab_size, eos_token=eos_token
        )
        tok = BPETokenizer(vocab)
        tok.save(vocab_path)

    train_text = eos_token.join(train_titles) + eos_token
    val_text = eos_token.join(val_titles) + eos_token

    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)

    train_bin_path = output_dir / "train.bin"
    val_bin_path = output_dir / "val.bin"
    train_ids.numpy().tofile(train_bin_path)
    val_ids.numpy().tofile(val_bin_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset preprocessing pipeline")

    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/tokenizer.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run a small quick preprocessing sample",
    )

    args = parser.parse_args()
    main(args)
