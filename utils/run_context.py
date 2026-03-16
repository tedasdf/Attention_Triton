""" "
run_context.py

input:
    - phase: "preprcoess" | "train" | "eval" | "serve"
    - config_path: paht to the YAML used
    - artifact_root (optional): default "artifacts"
    - dataset_version_id (optional): a string like "v0001" or a DVC-dervied id
    - extra metadata (optional): anything you want in the manifest (gpu name, hostname, cmd used, etc)


return:
    - run id: unique id for this run
    - run_dir: artifacts/runs/<phase>/<run_id>/
    - paths to stamp files:
        - config_snapshot_path (inside run_dir)
        - manifest_path (inside run_dir)
    - log_path (optional)
    - resolved_output_dir (optional)
"""

import os
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class ContextResult:
    run_id: str
    run_dir: str
    config_snapshot_path: str
    manifest_path: str


def _safe_run(cmd: list[str]) -> Optional[str]:
    """Run a command and return stdout.strip(), or None if it fails."""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return None


def start_run(
    phase: str,
    config_path: str,
    dataset_version_id: Optional[str] = None,
    artifacts_root: str = "artifacts",
    extras: Optional[Dict[str, Any]] = None,
) -> ContextResult:
    # 1) Create a unique run_id (timestamp + short git hash if available)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_commit = _safe_run(["git", "rev-parse", "HEAD"])
    short = git_commit[:7] if git_commit else "nogit"
    run_id = f"{ts}_{short}"

    # 2) Create run directory
    run_dir = os.path.join(artifacts_root, "runs", phase, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # 3) Snapshot config
    config_snapshot_path = os.path.join(run_dir, "config.snapshot.yaml")
    shutil.copy2(config_path, config_snapshot_path)

    # 4) Git dirty flag (optional but very useful)
    dirty = False
    status = _safe_run(["git", "status", "--porcelain"])
    if status is not None and status != "":
        dirty = True

    # 5) Write manifest
    manifest_path = os.path.join(run_dir, "manifest.json")
    manifest = {
        "phase": phase,
        "run_id": run_id,
        "started_at": datetime.now().isoformat(),
        "config_path": config_path,
        "config_snapshot": os.path.basename(config_snapshot_path),
        "dataset_version_id": dataset_version_id,
        "git": {
            "commit": git_commit,
            "dirty": dirty,
        },
        "extras": extras or {},
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return ContextResult(
        run_id=run_id,
        run_dir=run_dir,
        config_snapshot_path=config_snapshot_path,
        manifest_path=manifest_path,
    )
