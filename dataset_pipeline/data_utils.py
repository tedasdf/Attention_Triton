import os
import glob
from typing import Any

import json
import time

PIPELINE_ORDER = ["canonicalize", "minhash", "pairs", "cluster_map", "snapshot"]

# -------------------------
# Stage dependency table
# -------------------------
# Meaning:
# - canonicalize needs raw input_dir
# - minhash needs canonicalize output
# - pairs needs minhash output
# - snapshot needs minhash + pairs outputs
STAGE_DEPS = {
    "canonicalize": {"needs_input_dir": True, "needs": []},
    "minhash": {"needs_input_dir": False, "needs": ["canonicalize"]},
    "pairs": {"needs_input_dir": False, "needs": ["minhash"]},
    "cluster_map": {"needs_input_dir": False, "needs": ["pairs"]},
    "snapshot": {"needs_input_dir": False, "needs": ["minhash", "cluster_map"]},
    "split": {"needs_input_dir": False, "needs": ["snapshot"]},
}


# -------------------------
# Safe getters (dict or attrs)
# -------------------------
def get_stage_enabled(cfg: Any, stage: str) -> bool:
    """
    Supports:
      cfg.stages["minhash"]  (dict-like)
      cfg.stages.minhash     (attr-like)
    """
    stages = getattr(cfg, "stages", None)
    if stages is None:
        raise AttributeError("cfg has no 'stages' field")

    # dict-like
    try:
        if isinstance(stages, dict):
            return bool(stages.get(stage, False))
        if hasattr(stages, "get"):  # e.g. pydantic dict-ish
            return bool(stages.get(stage, False))
    except Exception:
        pass

    # attr-like
    if hasattr(stages, stage):
        return bool(getattr(stages, stage))

    raise KeyError(f"Stage '{stage}' not found in cfg.stages")


# -------------------------
# Path checks
# -------------------------
def parquet_exists(dir_path: str) -> bool:
    """
    True if directory contains any parquet files (including nested).
    Ray writes directories with part files; we just need to detect presence.
    """
    if not dir_path or not os.path.exists(dir_path):
        return False
    # check for *.parquet anywhere under the directory
    return len(glob.glob(os.path.join(dir_path, "**", "*.parquet"), recursive=True)) > 0


def require_input_dir(input_dir: str):
    if not input_dir:
        raise ValueError(
            "input_dir is None/empty. Provide --input_dir or set cfg.paths.input_dir"
        )
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")
    if not parquet_exists(input_dir):
        raise FileNotFoundError(f"input_dir has no parquet files: {input_dir}")


def require_stage_output(stage: str, stage_out_dir: str):
    if not parquet_exists(stage_out_dir):
        raise FileNotFoundError(
            f"Stage '{stage}' output missing at: {stage_out_dir}\n"
            f"Enable stages.{stage}: true or run that stage first."
        )


def preflight(cfg, stage_paths: dict[str, str]):
    stages = cfg.run.stages
    if not stages or len(stages) == 0:
        raise ValueError("run.stages is empty. Set run.stages in YAML.")

    # validate names + order
    unknown = [s for s in stages if s not in STAGE_DEPS]
    if unknown:
        raise ValueError(f"Unknown stages in run.stages: {unknown}")

    will_run = set(stages)

    # check in dependency order (helps readability of errors)
    for stage in PIPELINE_ORDER:
        if stage not in will_run:
            continue

        dep = STAGE_DEPS[stage]

        if dep["needs_input_dir"]:
            require_input_dir(cfg.run.input_dir)

        for upstream in dep["needs"]:
            if upstream in will_run:
                continue
            require_stage_output(upstream, stage_paths[upstream])


def done_path(stage_dir: str) -> str:
    return os.path.join(stage_dir, "_DONE.json")


def write_done(stage: str, stage_dir: str, meta: dict | None = None):
    payload = {
        "stage": stage,
        "finished_at": time.time(),
        **(meta or {}),
    }
    with open(done_path(stage_dir), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def is_done(stage_dir: str) -> bool:
    return os.path.exists(done_path(stage_dir))
