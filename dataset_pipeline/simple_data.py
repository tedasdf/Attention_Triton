import os
import ray

from stages.report_quality import run_data_quality_report
from stages.snapshot import run_stage_snapshot
from stages.cluster import run_stage_cluster_map
from stages.canonicalize import run_stage_canonicalize
from stages.minihash import run_stage_minhash
from stages.pairs import run_stage_lsh
from stages.split import run_stage_split

from data_utils import (
    _maybe_count_rows,
    require_stage_output,
    preflight,
    write_done,
)
from config import load_pipeline_config
from run_context import start_run


STAGE_DIRS = {
    "canonicalize": "01_canonicalize",
    "minhash": "02_minhash",
    "pairs": "03_pairs",
    "cluster_map": "04_cluster_map",
    "snapshot": "05_snapshot",
    "split": "06_split",
}

STAGE_ORDER = [
    "canonicalize",
    "minhash",
    "pairs",
    "cluster_map",
    "snapshot",
    "split",
]


def set_up(out_root: str) -> dict[str, str]:
    stage_paths = {
        name: os.path.join(out_root, folder) for name, folder in STAGE_DIRS.items()
    }
    for path in stage_paths.values():
        os.makedirs(path, exist_ok=True)
    return stage_paths


def get_will_run(cfg) -> set[str]:
    stages = getattr(cfg.run, "stages", None)
    if not stages:
        raise ValueError("cfg.run.stages is empty. Set run.stages in your YAML.")
    if not isinstance(stages, (list, tuple)):
        raise TypeError(
            f"cfg.run.stages must be a list/tuple, got: {type(stages).__name__}"
        )
    return {str(stage).strip() for stage in stages if str(stage).strip()}


def run_or_load_stage(
    stage: str,
    will_run: set[str],
    stage_paths: dict[str, str],
    run_fn,
):
    if stage in will_run:
        ds = run_fn()
    else:
        require_stage_output(stage, stage_paths[stage])
        ds = ray.data.read_parquet(stage_paths[stage])

    rows, _ = _maybe_count_rows(ds)
    return ds, rows


def data_preprocess(cfg_path: str):
    cfg = load_pipeline_config(cfg_path)

    ctx = start_run(
        phase="preprocess",
        config_path=cfg_path,
        dataset_version_id=cfg.run.version,
        extras={"stages": cfg.run.stages},
    )
    print("run_dir:", ctx.run_dir)

    stage_paths = set_up(os.path.join(cfg.run.input_dir, "stages"))
    preflight(cfg, stage_paths)

    reports_dir = os.path.join(ctx.run_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    ray_started = False
    try:
        ray.init(**(cfg.run.ray_init_kwargs or {}))
        ray_started = True

        will_run = get_will_run(cfg)
        datasets = {}

        stage_runners = {
            "canonicalize": lambda: run_stage_canonicalize(cfg, stage_paths),
            "minhash": lambda: run_stage_minhash(
                cfg,
                stage_paths,
                ds_canon=datasets["canonicalize"],
            ),
            "pairs": lambda: run_stage_lsh(
                cfg,
                stage_paths,
                ds_minihash=datasets["minhash"],
            ),
            "cluster_map": lambda: run_stage_cluster_map(
                cfg,
                stage_paths,
                pairs_ds=datasets["pairs"],
            ),
            "snapshot": lambda: run_stage_snapshot(
                cfg,
                stage_paths,
                ds_minihash=datasets["minhash"],
                ds_cluster_map=datasets["cluster_map"],
            ),
            "split": lambda: run_stage_split(
                cfg,
                stage_paths,
                snapshot_ds=datasets["snapshot"],
            ),
        }

        for stage in STAGE_ORDER:
            ds, rows = run_or_load_stage(
                stage=stage,
                will_run=will_run,
                stage_paths=stage_paths,
                run_fn=stage_runners[stage],
            )
            datasets[stage] = ds

            done_payload = {
                "run_id": ctx.run_id,
                "version": cfg.run.version,
            }
            if rows is not None:
                done_payload["rows"] = rows

            write_done(stage, stage_paths[stage], done_payload)

        run_data_quality_report(cfg, stage_paths, reports_dir)
        write_done(
            "data_quality",
            reports_dir,
            {
                "run_id": ctx.run_id,
                "version": cfg.run.version,
            },
        )

        write_done(
            "preprocess_run",
            ctx.run_dir,
            {
                "run_id": ctx.run_id,
                "version": cfg.run.version,
                "stages": list(will_run),
                "status": "success",
            },
        )

    finally:
        if ray_started:
            ray.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Ray data preprocessing pipeline."
    )
    parser.add_argument("--config", type=str, default=r"configs/preprocess.yaml")
    args = parser.parse_args()

    data_preprocess(cfg_path=args.config)
