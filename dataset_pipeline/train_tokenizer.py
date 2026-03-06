import os
import json
import time
import yaml
import ray

from tokenizers import Tokenizer
from tokenizers import models, trainers, pre_tokenizers, decoders

from utils.run_context import start_run

import hashlib


def md5_int_list(xs):
    b = ",".join(map(str, xs)).encode("utf-8")
    return hashlib.md5(b).hexdigest()


def write_tokenizer_sanity_suite(
    tokenizer,
    text_iter,
    out_dir: str,
    n_samples: int = 50,
    head_n: int = 64,
):
    os.makedirs(out_dir, exist_ok=True)

    tests = []
    for text in text_iter:
        enc = tokenizer.encode(text)
        ids = enc.ids
        tests.append(
            {
                "text": text,
                "ids_len": len(ids),
                "ids_head": ids[:head_n],
                "ids_md5": md5_int_list(ids),
            }
        )
        if len(tests) >= n_samples:
            break

    path = os.path.join(out_dir, "smoke_test.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_samples": n_samples,
                "head_n": head_n,
                "tests": tests,
            },
            f,
            indent=2,
        )

    print("wrote tokenizer sanity suite:", path)
    return path


def iter_training_text(
    snapshot_dir: str, fields: list[str], batch_size: int = 4096, debug: bool = False
):
    ds = ray.data.read_parquet(snapshot_dir)

    for batch in ds.iter_batches(batch_size=batch_size, batch_format="pyarrow"):
        cols = {
            f: batch[f].to_pylist() if f in batch.column_names else None for f in fields
        }

        n = batch.num_rows
        if debug:
            n = min(10, n)

        for i in range(n):
            parts = []
            for f in fields:
                arr = cols.get(f)
                if arr is None:
                    continue
                v = arr[i]
                if isinstance(v, str) and v.strip():
                    parts.append(v)
            text = "\n".join(parts)
            if text:
                yield text


def train_tokenizer(
    text_iter,
    vocab_size: int,
    special_tokens: list[str],
    unk_token: str = "<unk>",
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    tokenizer.train_from_iterator(text_iter, trainer)
    return tokenizer


def write_bundle(tokenizer: Tokenizer, out_dir: str, manifest: dict):
    os.makedirs(out_dir, exist_ok=True)

    tok_path = os.path.join(out_dir, "tokenizer.json")
    tokenizer.save(tok_path)

    # special tokens map + ids
    special = manifest.get("special_tokens", [])
    special_map = {t: tokenizer.token_to_id(t) for t in special}

    with open(os.path.join(out_dir, "special_tokens.json"), "w", encoding="utf-8") as f:
        json.dump({"tokens": special, "token_to_id": special_map}, f, indent=2)

    with open(
        os.path.join(out_dir, "tokenizer_manifest.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2)

    return tok_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train and save a ByteLevel BPE tokenizer bundle."
    )
    parser.add_argument("--config", type=str, default=r"configs\tokenizer.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    tok_cfg = raw["tokenizer"]
    snapshot_dir = tok_cfg["source_dir"]
    fields = tok_cfg.get("text_fields", ["instruction", "code_ref"])
    vocab_size = int(tok_cfg.get("vocab_size", 32000))
    version = tok_cfg.get("version", "tok_v0001")
    batch_size = int(tok_cfg.get("batch_size", 4096))
    debug = bool(tok_cfg.get("debug", False))
    out_root = tok_cfg.get("output_dir", os.path.join("artifacts", "tokenizer"))
    out_dir = os.path.join(out_root, version)

    special_tokens = tok_cfg.get("special_tokens", ["<pad>", "<eos>", "<unk>"])
    unk_token = "<unk>"  # keep consistent with special_tokens

    # stamp run
    ctx = start_run(
        phase="tokenizer",
        config_path=args.config,
        dataset_version_id=tok_cfg.get("dataset_version_id", None),
        extras={
            "version": version,
            "source_dir": snapshot_dir,
            "vocab_size": vocab_size,
        },
    )
    print("run_dir:", ctx.run_dir)

    # init ray once
    ray.init(**(tok_cfg.get("ray_init_kwargs", {}) or {}))

    text_iter = iter_training_text(
        snapshot_dir, fields, batch_size=batch_size, debug=debug
    )
    tokenizer = train_tokenizer(
        text_iter,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        unk_token=unk_token,
    )

    manifest = {
        "tokenizer_version": version,
        "created_at": time.time(),
        "source_dir": snapshot_dir,
        "text_fields": fields,
        "vocab_size": vocab_size,
        "special_tokens": special_tokens,
        "run_id": ctx.run_id,
    }

    tok_path = write_bundle(tokenizer, out_dir, manifest)
    print("wrote tokenizer bundle:", out_dir)
    print("tokenizer file:", tok_path)

    write_tokenizer_sanity_suite(
        tokenizer=tokenizer,
        text_iter=iter_training_text(
            snapshot_dir, fields, batch_size=4096, debug=False
        ),
        out_dir=out_dir,
        n_samples=50,
        head_n=64,
    )
