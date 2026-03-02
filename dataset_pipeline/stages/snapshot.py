import hashlib
from dataclasses import dataclass
from typing import Literal
import pandas as pd

import ray

# ----------------- hyperparams / config -----------------


@dataclass
class DedupConfig:
    mode: Literal["exact", "cluster"] = (
        "cluster"  # "exact" needs canon_hash; "cluster" needs cluster_id
    )
    key_col: str | None = None  # override grouping key if you want


# ----------------- helpers (same as before) -----------------


def _stable_md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()


def dedup_snapshot(ds, mode: str, key_col: str | None = None):
    if mode not in {"exact", "cluster"}:
        raise ValueError("mode must be 'exact' or 'cluster'")

    if key_col is None:
        key_col = "canon_hash" if mode == "exact" else "cluster_id"

    # Keep only rows with a non-empty key
    ds_k = ds.filter(lambda r: bool(r.get(key_col)))

    def pick_best_group(pdf: "pd.DataFrame") -> "pd.DataFrame":
        # Deterministic representative choice (no quality scoring):
        # 1) longest code_ref
        # 2) then longest instruction
        # 3) then smallest md5(id)
        code_len = pdf["code_ref"].fillna("").str.len()
        instr_len = pdf["instruction"].fillna("").str.len()
        id_hash = pdf["id"].fillna("").map(_stable_md5)

        pdf = pdf.assign(_code_len=code_len, _instr_len=instr_len, _id_hash=id_hash)
        pdf = pdf.sort_values(
            by=["_code_len", "_instr_len", "_id_hash"],
            ascending=[False, False, True],
            kind="mergesort",  # stable
        )
        # return single-row dataframe
        return pdf.head(1).drop(columns=["_code_len", "_instr_len", "_id_hash"])

    reps = ds_k.groupby(key_col).map_groups(pick_best_group, batch_format="pandas")
    return reps, key_col


# ----------------- main -----------------

if __name__ == "__main__":
    ray.init()

    input_dir = "intermediate/clustered_v0001"  # <-- uses your previous parquet output
    output_dir = "intermediate/snapshots_v0001"

    cfg = DedupConfig(
        mode="cluster",
        key_col=None,
    )

    ds_in = ray.data.read_parquet(input_dir)

    reps, used_key = dedup_snapshot(ds_in, mode=cfg.mode, key_col=cfg.key_col)

    keep_cols = ["dataset", "id", "instruction", "code_ref", "language", "has_code"]

    # keep cluster_id if you're doing cluster mode
    if cfg.mode == "cluster" and "cluster_id" in reps.schema().names:
        keep_cols.append("cluster_id")

    # keep canon_hash if you're doing exact mode
    if cfg.mode == "exact" and "canon_hash" in reps.schema().names:
        keep_cols.append("canon_hash")

    reps_out = reps.select_columns(keep_cols)

    reps_out.repartition(1).write_parquet(output_dir)

    print("input:", input_dir)
    print("mode:", cfg.mode, "| group_by:", used_key)
    print("wrote:", output_dir)
    print("sample:", reps.take(3))
