import ray
import os

import time
import traceback

try:
    import psutil
except ImportError:
    psutil = None


from .stages.report_quality import run_data_quality_report
from .stages.snapshot import run_stage_snapshot
from .stages.cluster import run_stage_cluster_map
from .data_utils import (
    _maybe_count_rows,
    log_preprocess_metrics,
    require_stage_output,
    preflight,
    write_done,
)
from .stages.canonicalize import run_stage_canonicalize
from .stages.minihash import run_stage_minhash
from .stages.pairs import run_stage_lsh
from .stages.split import run_stage_split

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


def _process_rss_mb() -> float:
    if psutil is None:
        return 0.0
    try:
        return float(psutil.Process(os.getpid()).memory_info().rss / (1024**2))
    except Exception:
        return 0.0


def _ray_resource_snapshot() -> dict[str, float]:
    try:
        cluster = ray.cluster_resources()
        available = ray.available_resources()
        return {
            "system/cpu_total": float(cluster.get("CPU", 0.0)),
            "system/cpu_available": float(available.get("CPU", 0.0)),
            "system/gpu_total": float(cluster.get("GPU", 0.0)),
            "system/gpu_available": float(available.get("GPU", 0.0)),
        }
    except Exception:
        return {
            "system/cpu_total": 0.0,
            "system/cpu_available": 0.0,
            "system/gpu_total": 0.0,
            "system/gpu_available": 0.0,
        }


def run_or_load_observed(
    stage: str,
    will_run: set[str],
    stage_paths: dict[str, str],
    run_fn,
    metrics_path: str,
    run_t0: float,
):
    stage_t0 = time.perf_counter()
    mode = "run" if stage in will_run else "load_cached"

    if stage in will_run:
        ds = run_fn()
    else:
        require_stage_output(stage, stage_paths[stage])
        ds = ray.data.read_parquet(stage_paths[stage])

    rows, count_time = _maybe_count_rows(ds)
    stage_time = time.perf_counter() - stage_t0
    elapsed_time = time.perf_counter() - run_t0

    metrics = {
        "stage/name": stage,
        "stage/mode": mode,
        "preprocess/stage_time_sec": float(stage_time),
        "preprocess/elapsed_time_sec": float(elapsed_time),
        "preprocess/count_time_sec": float(count_time),
        "system/process_rss_mb": float(_process_rss_mb()),
    }
    metrics.update(_ray_resource_snapshot())

    if rows is not None:
        metrics["preprocess/rows"] = int(rows)
        metrics["preprocess/rows_per_sec"] = float(rows / max(stage_time, 1e-8))

    log_preprocess_metrics(metrics_path, metrics)
    return ds, rows, stage_time


def data_preprocess(cfg_path=r"configs/preprocess.yaml"):
    run_t0 = time.perf_counter()
    cfg = load_pipeline_config(cfg_path)

    ctx = start_run(
        phase="preprocess",
        config_path=cfg_path,
        dataset_version_id=cfg.run.version,
        extras={"stages": cfg.run.stages},
    )
    print("run_dir:", ctx.run_dir)

    stage_paths = set_up(os.path.join(ctx.run_dir, "stages"))
    preflight(cfg, stage_paths)

    reports_dir = os.path.join(ctx.run_dir, "reports")
    metrics_path = os.path.join(reports_dir, "preprocess_metrics.jsonl")

    log_preprocess_metrics(
        metrics_path,
        {
            "event": "run_start",
            "run_id": ctx.run_id,
            "run_dir": ctx.run_dir,
            "config_path": cfg_path,
            "dataset_version": cfg.run.version,
            "run/stages": list(cfg.run.stages),
            "preprocess/elapsed_time_sec": 0.0,
            "system/process_rss_mb": float(_process_rss_mb()),
        },
    )

    ray_started = False
    try:
        ray.init(**(cfg.run.ray_init_kwargs or {}))
        ray_started = True
        will_run = get_will_run(cfg)

        ds_canon, canon_rows, canon_time = run_or_load_observed(
            "canonicalize",
            will_run,
            stage_paths,
            lambda: run_stage_canonicalize(cfg, stage_paths),
            metrics_path,
            run_t0,
        )
        write_done(
            "canonicalize",
            stage_paths["canonicalize"],
            {
                "rows": canon_rows,
                "stage_time_sec": canon_time,
                "run_id": ctx.run_id,
                "version": cfg.run.version,
            },
        )

        ds_minihash, minhash_rows, minhash_time = run_or_load_observed(
            "minhash",
            will_run,
            stage_paths,
            lambda: run_stage_minhash(cfg, stage_paths, ds_canon=ds_canon),
            metrics_path,
            run_t0,
        )
        write_done(
            "minhash",
            stage_paths["minhash"],
            {
                "rows": minhash_rows,
                "stage_time_sec": minhash_time,
                "run_id": ctx.run_id,
                "version": cfg.run.version,
            },
        )

        pairs_ds, pairs_rows, pairs_time = run_or_load_observed(
            "pairs",
            will_run,
            stage_paths,
            lambda: run_stage_lsh(cfg, stage_paths, ds_minihash=ds_minihash),
            metrics_path,
            run_t0,
        )
        write_done(
            "pairs",
            stage_paths["pairs"],
            {
                "rows": pairs_rows,
                "stage_time_sec": pairs_time,
                "run_id": ctx.run_id,
                "version": cfg.run.version,
            },
        )

        ds_cluster_map, cluster_rows, cluster_time = run_or_load_observed(
            "cluster_map",
            will_run,
            stage_paths,
            lambda: run_stage_cluster_map(cfg, stage_paths, pairs_ds=pairs_ds),
            metrics_path,
            run_t0,
        )
        write_done(
            "cluster_map",
            stage_paths["cluster_map"],
            {
                "rows": cluster_rows,
                "stage_time_sec": cluster_time,
                "run_id": ctx.run_id,
                "version": cfg.run.version,
            },
        )

        snapshot_ds, snapshot_rows, snapshot_time = run_or_load_observed(
            "snapshot",
            will_run,
            stage_paths,
            lambda: run_stage_snapshot(
                cfg,
                stage_paths,
                ds_minihash=ds_minihash,
                ds_cluster_map=ds_cluster_map,
            ),
            metrics_path,
            run_t0,
        )
        write_done(
            "snapshot",
            stage_paths["snapshot"],
            {
                "rows": snapshot_rows,
                "stage_time_sec": snapshot_time,
                "run_id": ctx.run_id,
                "version": cfg.run.version,
            },
        )

        split_ds, split_rows, split_time = run_or_load_observed(
            "split",
            will_run,
            stage_paths,
            lambda: run_stage_split(cfg, stage_paths, snapshot_ds=snapshot_ds),
            metrics_path,
            run_t0,
        )
        split_done = {
            "stage_time_sec": split_time,
            "run_id": ctx.run_id,
            "version": cfg.run.version,
        }
        if split_rows is not None:
            split_done["rows"] = split_rows
        write_done("split", stage_paths["split"], split_done)

        dq_t0 = time.perf_counter()
        run_data_quality_report(cfg, stage_paths, reports_dir)
        dq_time = time.perf_counter() - dq_t0

        log_preprocess_metrics(
            metrics_path,
            {
                "stage/name": "data_quality",
                "stage/mode": "run",
                "preprocess/stage_time_sec": float(dq_time),
                "preprocess/elapsed_time_sec": float(time.perf_counter() - run_t0),
                "system/process_rss_mb": float(_process_rss_mb()),
                **_ray_resource_snapshot(),
            },
        )
        write_done(
            "data_quality",
            reports_dir,
            {
                "stage_time_sec": dq_time,
                "run_id": ctx.run_id,
                "version": cfg.run.version,
            },
        )

        total_elapsed = time.perf_counter() - run_t0

        write_done(
            "preprocess_run",
            ctx.run_dir,
            {
                "run_id": ctx.run_id,
                "version": cfg.run.version,
                "stages": list(will_run),
                "elapsed_time_sec": total_elapsed,
                "status": "success",
            },
        )

        log_preprocess_metrics(
            metrics_path,
            {
                "event": "run_end",
                "status": "success",
                "run_id": ctx.run_id,
                "dataset_version": cfg.run.version,
                "preprocess/elapsed_time_sec": float(total_elapsed),
                "system/process_rss_mb": float(_process_rss_mb()),
                **_ray_resource_snapshot(),
            },
        )

    except Exception as e:
        total_elapsed = time.perf_counter() - run_t0
        log_preprocess_metrics(
            metrics_path,
            {
                "event": "run_end",
                "status": "failed",
                "run_id": ctx.run_id,
                "dataset_version": cfg.run.version,
                "error/type": type(e).__name__,
                "error/message": str(e),
                "error/traceback": traceback.format_exc(),
                "preprocess/elapsed_time_sec": float(total_elapsed),
                "system/process_rss_mb": float(_process_rss_mb()),
            },
        )
        raise

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
