import yaml
from dataclasses import dataclass
from typing import Any

from stages.canonicalize import CanonicalizerConfig
from stages.minihash import MinHashConfig
from stages.lsh import LSHConfig
from stages.snapshot import DedupConfig


@dataclass
class RunConfig:
    version: str = "v0001"
    ray: dict[str, Any] = None  # init kwargs


@dataclass
class PipelineConfig:
    run: RunConfig
    canonicalize: CanonicalizerConfig
    minhash: MinHashConfig
    lsh: LSHConfig
    snapshot: DedupConfig
    snapshot_keep_cols: list[str] | None = None


def load_pipeline_config(path: str) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    run = RunConfig(**raw["run"])
    canon = CanonicalizerConfig(**raw["canonicalize"])
    mh = MinHashConfig(**raw["minhash"])
    lsh = LSHConfig(**raw["lsh"])
    snap = DedupConfig(**raw["snapshot"])

    keep_cols = raw.get("snapshot", {}).get("keep_cols")
    return PipelineConfig(
        run=run,
        canonicalize=canon,
        minhash=mh,
        lsh=lsh,
        snapshot=snap,
        snapshot_keep_cols=keep_cols,
    )
