import ray
import os

from .stages.report_quality import run_data_quality_report
from .stages.snapshot import run_stage_snapshot
from .stages.cluster import run_stage_cluster_map
from .data_utils import require_stage_output, preflight, write_done
from .stages.canonicalize import run_stage_canonicalize
from .stages.minihash import run_stage_minhash
from .stages.pairs import run_stage_lsh
from stages.split import run_stage_split

from .config import load_pipeline_config
from utils.run_context import start_run

STAGE_DIRS = {
    "canonicalize": "01_canonicalize",
    "minhash": "02_minhash",
    "pairs": "03_pairs",
    "cluster_map": "04_cluster_map",
    "snapshot": "05_snapshot",
    "split": "06_split",
}


def set_up(out_root: str) -> dict[str, str]:
    stage_paths = {k: os.path.join(out_root, v) for k, v in STAGE_DIRS.items()}
    for p in stage_paths.values():
        os.makedirs(p, exist_ok=True)
    return stage_paths


def get_will_run(cfg) -> set[str]:
    """
    Returns the set of stages to run this invocation.
    Expects cfg.run.stages to be a list like ["canonicalize","minhash","lsh","snapshot"].
    """
    stages = getattr(cfg.run, "stages", None)
    if not stages:
        raise ValueError("cfg.run.stages is empty. Set run.stages in your YAML.")
    if not isinstance(stages, (list, tuple)):
        raise TypeError(
            f"cfg.run.stages must be a list/tuple, got: {type(stages).__name__}"
        )
    return {str(s).strip() for s in stages if str(s).strip()}


def run_or_load(stage: str, will_run: set[str], stage_paths: dict[str, str], run_fn):
    """
    stage: stage name like 'canonicalize'
    will_run: set(cfg.run.stages)
    stage_paths: dict of stage -> output dir
    run_fn: callable that returns a Ray dataset (and should write inside)
    """
    if stage in will_run:
        return run_fn()
    else:
        require_stage_output(stage, stage_paths[stage])
        return ray.data.read_parquet(stage_paths[stage])


def data_preprocess(cfg_path=r"configs\preprocess.yaml"):
    cfg = load_pipeline_config(cfg_path)

    # run stamp (P0.3)
    ctx = start_run(
        phase="preprocess",
        config_path=cfg_path,
        dataset_version_id=cfg.run.version,
        extras={"stages": cfg.run.stages},
    )
    print("run_dir:", ctx.run_dir)

    stage_paths = set_up(os.path.join(ctx.run_dir, "stages"))
    preflight(cfg, stage_paths)

    ray.init(**(cfg.run.ray_init_kwargs or {}))
    will_run = get_will_run(cfg)

    ds_canon = run_or_load(
        "canonicalize",
        will_run,
        stage_paths,
        lambda: run_stage_canonicalize(cfg, stage_paths),
    )
    write_done(
        "canonicalize",
        stage_paths["canonicalize"],
        {"rows": ds_canon.count(), "run_id": ctx.run_id, "version": cfg.run.version},
    )

    ds_minihash = run_or_load(
        "minhash",
        will_run,
        stage_paths,
        lambda: run_stage_minhash(cfg, stage_paths, ds_canon=ds_canon),
    )
    write_done(
        "minhash",
        stage_paths["minhash"],
        {"rows": ds_minihash.count(), "run_id": ctx.run_id, "version": cfg.run.version},
    )

    pairs_ds = run_or_load(
        "pairs",
        will_run,
        stage_paths,
        lambda: run_stage_lsh(cfg, stage_paths, ds_minihash=ds_minihash),
    )
    write_done(
        "pairs",
        stage_paths["pairs"],
        {"rows": pairs_ds.count(), "run_id": ctx.run_id, "version": cfg.run.version},
    )

    ds_cluster_map = run_or_load(
        "cluster_map",
        will_run,
        stage_paths,
        lambda: run_stage_cluster_map(cfg, stage_paths, pairs_ds=pairs_ds),
    )
    write_done(
        "cluster_map",
        stage_paths["cluster_map"],
        {
            "rows": ds_cluster_map.count(),
            "run_id": ctx.run_id,
            "version": cfg.run.version,
        },
    )

    snapshot_ds = run_or_load(
        "snapshot",
        will_run,
        stage_paths,
        lambda: run_stage_snapshot(
            cfg, stage_paths, ds_minihash=ds_minihash, ds_cluster_map=ds_cluster_map
        ),
    )
    write_done(
        "snapshot",
        stage_paths["snapshot"],
        {"rows": snapshot_ds.count(), "run_id": ctx.run_id, "version": cfg.run.version},
    )

    run_or_load(
        "split",
        will_run,
        stage_paths,
        lambda: run_stage_split(cfg, stage_paths, snapshot_ds=snapshot_ds),
    )
    write_done(
        "split",
        stage_paths["split"],
        {"run_id": ctx.run_id, "version": cfg.run.version},
    )

    write_done(
        "preprocess_run",
        ctx.run_dir,
        {"run_id": ctx.run_id, "version": cfg.run.version, "stages": list(will_run)},
    )

    reports_dir = os.path.join(ctx.run_dir, "reports")
    run_data_quality_report(cfg, stage_paths, reports_dir)
    write_done(
        "data_quality", reports_dir, {"run_id": ctx.run_id, "version": cfg.run.version}
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Ray data preprocessing pipeline."
    )
    parser.add_argument("--config", type=str, default=r"configs\preprocess.yaml")
    args = parser.parse_args()

    data_preprocess(cfg_path=args.config)
