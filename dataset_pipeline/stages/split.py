import os
import ray
from .utils import _hash01


def run_stage_split(cfg, stage_paths: dict[str, str], snapshot_ds=None):
    """
    Input:  stage_paths["snapshot"] (05_snapshot/)
    Output: stage_paths["split"]/train, /val, /test
    Split is cluster-safe: uses cfg.split.key (default cluster_id)
    """
    # load snapshot if not provided
    if snapshot_ds is None:
        snapshot_ds = ray.data.read_parquet(stage_paths["snapshot"])

    split_cfg = getattr(cfg, "split", None)
    if split_cfg is None:
        raise ValueError("Missing split config. Add `split:` section to YAML.")

    key = getattr(split_cfg, "key", "cluster_id")
    train_p = float(getattr(split_cfg, "train", 0.90))
    val_p = float(getattr(split_cfg, "val", 0.05))
    test_p = float(getattr(split_cfg, "test", 0.05))
    algo = getattr(split_cfg, "hash", "md5")

    if abs((train_p + val_p + test_p) - 1.0) > 1e-6:
        raise ValueError("split.train + split.val + split.test must sum to 1.0")

    train_cut = train_p
    val_cut = train_p + val_p

    def assign_split(row):
        k = row.get(key)
        if not k:
            # fallback: use id to avoid crash; still deterministic
            k = row.get("id", "")
        u = _hash01(k, algo=algo)
        if u < train_cut:
            row["split"] = "train"
        elif u < val_cut:
            row["split"] = "val"
        else:
            row["split"] = "test"
        return row

    ds = snapshot_ds.map(assign_split)

    out_root = stage_paths["split"]
    train_dir = os.path.join(out_root, "train")
    val_dir = os.path.join(out_root, "val")
    test_dir = os.path.join(out_root, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    ds.filter(lambda r: r["split"] == "train").drop_columns(["split"]).write_parquet(
        train_dir
    )
    ds.filter(lambda r: r["split"] == "val").drop_columns(["split"]).write_parquet(
        val_dir
    )
    ds.filter(lambda r: r["split"] == "test").drop_columns(["split"]).write_parquet(
        test_dir
    )

    print("wrote split:", out_root)
    return {
        "train": train_dir,
        "val": val_dir,
        "test": test_dir,
    }
