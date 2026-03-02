"""
cluster_ids.py

Reads:
  1) LSH candidate pairs parquet (id1,id2) from intermediate/lsh_pairs_v0001_single
  2) MinHash dataset parquet (must contain 'id') from intermediate/minhash_node_v0001

Builds connected-component clusters via Union-Find and writes:
  intermediate/clustered_v0001  (same rows as minhash dataset + 'cluster_id')
"""

import hashlib
from typing import Dict


def stable_cluster_id(s: str) -> str:
    """Stable ID string for clusters (and singletons)"""
    return hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()


class UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            return x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_id_to_cluster(pairs: list[dict]) -> Dict[str, str]:
    """pairs: list of {'id1':..., 'id2':...} -> returns id -> cluster_id"""
    uf = UnionFind()

    # union edges
    for p in pairs:
        a = p.get("id1", "")
        b = p.get("id2", "")
        if a and b:
            uf.union(a, b)

    # collect all ids that appeared in any edge
    all_ids = set()
    for p in pairs:
        a = p.get("id1", "")
        b = p.get("id2", "")
        if a:
            all_ids.add(a)
        if b:
            all_ids.add(b)

    # map each id -> root
    id_to_root = {i: uf.find(i) for i in all_ids}

    # root -> stable cluster id
    roots = set(id_to_root.values())
    root_to_cid = {r: stable_cluster_id(r) for r in roots}

    # id -> cluster_id
    return {i: root_to_cid[r] for i, r in id_to_root.items()}


def make_add_cluster_id(id_to_cluster: Dict[str, str]):
    """Closure so Ray can serialize the mapping cleanly."""

    def add_cluster_id(row: dict) -> dict:
        _id = row.get("id", "")
        # singletons get their own cluster_id = hash(id)
        row["cluster_id"] = id_to_cluster.get(_id, stable_cluster_id(_id))
        return row

    return add_cluster_id


if __name__ == "__main__":
    import ray

    ray.init()

    # ---- inputs ----
    pairs_dir = "intermediate/lsh_node_v0001"
    minihash_dir = "intermediate/minhash_node_v0001"

    # Read candidate pairs (small local test: OK to materialize)
    pairs_ds = ray.data.read_parquet(pairs_dir)
    pairs = pairs_ds.take_all()
    print("num pairs:", len(pairs))

    # Build mapping id -> cluster_id
    id_to_cluster = build_id_to_cluster(pairs)
    print("num clustered ids (appear in pairs):", len(id_to_cluster))
    print("num clusters (non-singleton components):", len(set(id_to_cluster.values())))

    # Read minihash dataset and attach cluster_id to every row
    ds_minihash = ray.data.read_parquet(minihash_dir)
    ds_with_cluster = ds_minihash.map(make_add_cluster_id(id_to_cluster))

    # quick sanity
    print(ds_with_cluster.select_columns(["id", "cluster_id"]).take(5))

    # ---- output ----
    out_dir = "intermediate/clustered_v0001"
    ds_with_cluster.write_parquet(out_dir)
    print("wrote:", out_dir)

    # optional: show top cluster sizes
    sizes = ds_with_cluster.groupby("cluster_id").count()
    print("top clusters:")
    print(sizes.sort("count()", descending=True).take(10))
