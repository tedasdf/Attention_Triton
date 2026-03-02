import ray

from stages.cluster_ids import build_id_to_cluster, make_add_cluster_id
from stages.snapshot import dedup_snapshot
from stages.lsh import lsh_pairs_map_batches
from stages.minihash import make_add_minhash
from stages.canonicalize import canonicalize

from config import load_pipeline_config


def data_preprocess(
    input_dir,
    out_dir,
    cfg_path="./config/pipeline.yaml",
):
    cfg = load_pipeline_config(cfg_path)

    # init ray ONCE here
    ray.init(**(cfg.run.ray.get("init_kwargs", {}) if cfg.run.ray else {}))

    ds = ray.data.read_parquet(input_dir)

    def add_node_types(row):
        if row.get("language") != "python" or not row.get("has_code"):
            row["parse_ok"] = False
            row["node_types"] = []
            row["parse_err"] = "not_python_or_no_code"
            return row

        res = canonicalize(row["code_ref"], cfg.canonicalize)
        row["parse_ok"] = res.ok
        row["node_types"] = res.rep if res.ok else []
        row["parse_err"] = res.err
        return row

    ds2 = ds.map(add_node_types)
    ds_canon = ds2.filter(lambda r: r.get("parse_ok", False))

    # MinHash
    ds_sig = ds_canon.map(make_add_minhash(cfg.minhash))
    ds_minihash = ds_sig.filter(lambda r: r.get("sig_ok", False))

    # LSH (prototype: per-batch)
    def lsh_pairs_map_batches_safe(batch, k, b, max_bucket_size):
        try:
            return lsh_pairs_map_batches(
                batch, k=k, b=b, max_bucket_size=max_bucket_size
            )
        except Exception as e:
            print("LSH UDF error:", type(e).__name__, e)
            return {"id1": [], "id2": []}

    pairs_ds = (
        ds_minihash.select_columns(["id", "sig", "sig_ok"])
        .map_batches(
            lsh_pairs_map_batches_safe,
            batch_format="default",
            batch_size=cfg.lsh.batch_size,
            fn_kwargs={
                "k": cfg.lsh.k,
                "b": cfg.lsh.b,
                "max_bucket_size": cfg.lsh.max_bucket_size,
            },
        )
        .groupby(["id1", "id2"])
        .count()
        .drop_columns(["count()"])
    )

    pairs = pairs_ds.take_all()

    # Cluster IDs (local union-find)
    id_to_cluster = build_id_to_cluster(pairs)
    ds_with_cluster = ds_minihash.map(make_add_cluster_id(id_to_cluster))

    # Snapshot dedup
    reps, used_key = dedup_snapshot(
        ds_with_cluster,
        mode=cfg.snapshot.mode,
        key_col=cfg.snapshot.key_col,
    )

    # Keep only safe columns for writing
    keep_cols = ["dataset", "id", "instruction", "code_ref", "language", "has_code"]
    reps_out = reps.select_columns([c for c in keep_cols if c in reps.schema().names])

    # Write snapshot
    reps_out.write_parquet(out_dir)

    print("group_by:", used_key)
    print("wrote:", out_dir)
    print("sample:", reps_out.take(3))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Ray data preprocessing pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config/pipeline.yaml",
        help="Path to pipeline YAML config.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Path to pipeline YAML config.",
    )
    parser.add_argument(
        "---output_dir",
        type=str,
        default=None,
        help="Path to pipeline YAML config.",
    )
    args = parser.parse_args()

    data_preprocess(
        cfg_path=args.config, input_dir=args.input_dir, out_dir=args.output_dir
    )
