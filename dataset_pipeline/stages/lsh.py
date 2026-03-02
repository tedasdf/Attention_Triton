import hashlib
from dataclasses import dataclass
from itertools import combinations


@dataclass
class LSHConfig:
    batch_size: int = 512
    k: int = 128
    b: int = 32
    max_bucket_size: int = 2000


def band_key(sig, band_idx, r):
    start = band_idx * r
    bb = b"".join(
        int(v).to_bytes(8, "little", signed=True) for v in sig[start : start + r]
    )
    h = hashlib.blake2b(bb, digest_size=8).hexdigest()
    return f"{band_idx}:{h}"


def lsh_pairs_map_batches(batch, k=128, b=32, max_bucket_size=2000):
    assert k % b == 0
    ids = batch["id"]
    sigs = batch["sig"]
    ok = batch.get("sig_ok", [True] * len(ids))

    r = k // b
    buckets = {}

    for _id, sig, good in zip(ids, sigs, ok):
        if not good:
            continue
        if sig is None:
            continue
        # handle list/tuple/np.ndarray uniformly
        try:
            sig_len = len(sig)
        except Exception:
            continue
        if sig_len != k:
            continue

        for band_idx in range(b):
            key = band_key(sig, band_idx, r)
            buckets.setdefault(key, []).append(_id)

    out_id1, out_id2 = [], []
    for bucket_ids in buckets.values():
        bucket_ids = sorted(set(bucket_ids))
        if len(bucket_ids) < 2 or len(bucket_ids) > max_bucket_size:
            continue
        for a, c in combinations(bucket_ids, 2):
            out_id1.append(a)
            out_id2.append(c)

    return {"id1": out_id1, "id2": out_id2}


if __name__ == "__main__":
    import ray

    ray.init()

    cfg = LSHConfig(batch_size=512, k=128, b=32, max_bucket_size=2000)

    in_dir = "intermediate/minhash_node_v0001"
    ds_minihash = ray.data.read_parquet(in_dir).filter(lambda r: r.get("sig_ok", False))

    def lsh_pairs_map_batches_safe(batch, k=128, b=32, max_bucket_size=2000):
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
            batch_size=cfg.batch_size,
            fn_kwargs={"k": cfg.k, "b": cfg.b, "max_bucket_size": cfg.max_bucket_size},
        )
        .groupby(["id1", "id2"])
        .count()
        .drop_columns(["count()"])
    )

    pairs = pairs_ds.take_all()
    print("pairs:", len(pairs))
    print(pairs[:5])

    out_dir = "intermediate/lsh_node_v0001"
    pairs_ds_single = pairs_ds.repartition(1)
    pairs_ds_single.write_parquet(out_dir)

    print("wrote:", out_dir)

    print("num pairs:", pairs_ds.count())

    pairs = pairs_ds.take(5)
    print(pairs)

    # optional: join back to original rows for inspection (cheap way: collect ids)
    ids = set([p["id1"] for p in pairs] + [p["id2"] for p in pairs])
    sample = ds_minihash.filter(lambda r: r["id"] in ids).select_columns(
        ["id", "dataset", "language", "parse_ok"]
    )
    print(sample.take_all())
