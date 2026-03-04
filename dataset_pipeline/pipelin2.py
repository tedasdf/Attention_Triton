import os
import hashlib
from dataclasses import dataclass
from typing import Literal

import ray
import pandas as pd


# ---------- configs ----------
@dataclass
class ClusterConfig:
    pairs_batch_rows: int = 200_000  # tune down if driver RAM is tight
    map_write_rows: int = 1_000_000  # write cluster_map in chunks


@dataclass
class SnapshotConfig:
    mode: Literal["cluster", "exact"] = "cluster"
    key_col: str | None = None


# ---------- helpers ----------
def stable_md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()


def stable_cluster_id(root: str) -> str:
    return stable_md5(root)


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        p = self.parent.get(x)
        if p is None:
            self.parent[x] = x
            return x
        # path compression
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def dedup_snapshot(ds, mode: str, key_col: str | None = None):
    if mode not in {"exact", "cluster"}:
        raise ValueError("mode must be 'exact' or 'cluster'")

    if key_col is None:
        key_col = "canon_hash" if mode == "exact" else "cluster_id"

    ds_k = ds.filter(lambda r: bool(r.get(key_col)))

    def pick_best_group(pdf: "pd.DataFrame") -> "pd.DataFrame":
        code_len = pdf["code_ref"].fillna("").str.len()
        instr_len = pdf["instruction"].fillna("").str.len()
        id_hash = pdf["id"].fillna("").map(stable_md5)

        pdf = pdf.assign(_code_len=code_len, _instr_len=instr_len, _id_hash=id_hash)
        pdf = pdf.sort_values(
            by=["_code_len", "_instr_len", "_id_hash"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        return pdf.head(1).drop(columns=["_code_len", "_instr_len", "_id_hash"])

    reps = ds_k.groupby(key_col).map_groups(pick_best_group, batch_format="pandas")
    return reps, key_col


# ---------- main stage ----------
def cluster_and_snapshot(
    minihash_out_dir: str,
    pairs_out_dir: str,
    out_dir: str,
    cluster_cfg: ClusterConfig = ClusterConfig(),
    snap_cfg: SnapshotConfig = SnapshotConfig(),
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Read pairs and build union-find STREAMING (no take_all)
    # pairs_ds = ray.data.read_parquet(pairs_out_dir).select_columns(["id1", "id2"])
    # uf = UnionFind()

    # # pandas batches are convenient here
    # for pdf in pairs_ds.iter_batches(
    #     batch_size=cluster_cfg.pairs_batch_rows,
    #     batch_format="pyarrow",
    # ):
    #     # pdf has columns id1,id2
    #     for a, b in zip(pdf["id1"].tolist(), pdf["id2"].tolist()):
    #         uf.union(a, b)

    # print("union-find built. unique ids seen in pairs:", len(uf.parent))

    # 2) Write cluster map to parquet in chunks: (id, cluster_id)
    #    IMPORTANT: cluster_map only includes ids that appear in pairs.
    cluster_map_dir = os.path.join(out_dir, "cluster_map")
    # os.makedirs(cluster_map_dir, exist_ok=True)

    # rows = []
    # wrote_parts = 0
    # for _id in uf.parent.keys():
    #     root = uf.find(_id)
    #     cid = stable_cluster_id(root)
    #     rows.append({"id": _id, "cluster_id": cid})

    #     if len(rows) >= cluster_cfg.map_write_rows:
    #         ray.data.from_items(rows).write_parquet(
    #             os.path.join(cluster_map_dir, f"part_{wrote_parts:05d}")
    #         )
    #         wrote_parts += 1
    #         rows = []

    # if rows:
    #     ray.data.from_items(rows).write_parquet(
    #         os.path.join(cluster_map_dir, f"part_{wrote_parts:05d}")
    #     )

    # print("wrote cluster_map:", cluster_map_dir)

    # 3) Join minihash dataset with cluster_map on "id"
    ds_minihash = ray.data.read_parquet(minihash_out_dir)

    ds_cluster_map = ray.data.read_parquet(cluster_map_dir)  # (id, cluster_id)
    local_map = {row["id"]: row["cluster_id"] for row in ds_cluster_map.take_all()}

    # 2. Use a map function to "lookup" the cluster_id
    def attach_cluster_id(row):
        # This mimics a left join + fill_singletons in one go
        row["cluster_id"] = local_map.get(row["id"], stable_md5(row["id"]))
        return row

    ds_with_cluster = ds_minihash.map(attach_cluster_id)

    # Fill singleton cluster_id for rows that never appeared in pairs
    def fill_singletons(row):
        if not row.get("cluster_id"):
            row["cluster_id"] = stable_md5(row["id"])
        return row

    ds_with_cluster = ds_with_cluster.map(fill_singletons)

    # 4) Snapshot dedup (cluster mode)
    reps, used_key = dedup_snapshot(
        ds_with_cluster, mode=snap_cfg.mode, key_col=snap_cfg.key_col
    )

    keep_cols = [
        "dataset",
        "id",
        "instruction",
        "code_ref",
        "language",
        "has_code",
        "cluster_id",
    ]
    reps_out = reps.select_columns([c for c in keep_cols if c in reps.schema().names])

    snapshot_dir = os.path.join(out_dir, "snapshot")
    reps_out.write_parquet(snapshot_dir)

    print("group_by:", used_key)
    print("wrote snapshot:", snapshot_dir)
    print("sample:", reps_out.take(3))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cluster + snapshot from stored minihash + pairs."
    )
    parser.add_argument("--minihash_dir", type=str, required=True)
    parser.add_argument("--pairs_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    ray.init()
    cluster_and_snapshot(
        minihash_out_dir=args.minihash_dir,
        pairs_out_dir=args.pairs_dir,
        out_dir=args.out_dir,
    )
