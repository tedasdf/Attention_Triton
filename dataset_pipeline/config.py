import yaml
from dataclasses import dataclass
from typing import Any, Optional

from .stages.snapshot import DedupConfig
from .stages.cluster import ClusterConfig
from .stages.canonicalize import CanonicalizerConfig
from .stages.minihash import MinHashConfig
from .stages.pairs import LSHConfig
# from stages.snapshot import DedupConfig


@dataclass
class RunConfig:
    version: str = "v0001"
    ray_init_kwargs: Optional[dict[str, Any]] = None
    stages: Optional[list[str]] = None
    input_dir: Optional[str] = None
    output_dir: str = "artifacts/run/preprocess/"


@dataclass
class SplitConfig:
    train: float
    val: float
    test: float
    key: str
    hash: str


@dataclass
class PipelineConfig:
    run: RunConfig
    canonicalize: CanonicalizerConfig
    minhash: MinHashConfig
    lsh: LSHConfig
    cluster: ClusterConfig
    snapshot: DedupConfig
    split: Optional[SplitConfig] = None
    # snapshot_keep_cols: list[str] | None = None


def load_pipeline_config(path: str) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    run = RunConfig(**raw["run"])
    canon = CanonicalizerConfig(**raw["canonicalize"])
    mh = MinHashConfig(**raw["minhash"])
    lsh = LSHConfig(**raw["pairs"])
    cluster = ClusterConfig(**raw["cluster"])
    snap = DedupConfig(**raw["snapshot"])
    split = SplitConfig(**raw["split"]) if "split" in raw else None

    return PipelineConfig(
        run=run,
        canonicalize=canon,
        minhash=mh,
        lsh=lsh,
        cluster=cluster,
        snapshot=snap,
        split=split,
    )
