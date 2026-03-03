from dataclasses import dataclass


@dataclass
class PairsJobConfig:
    # where to read minhash rows (must contain id, sig, sig_ok)
    minhash_dir: str
    out_dir: str
    cfg_path: str = "./config/pipeline.yaml"
    version: str = "v0001"

    # safety caps (these are the important ones)
    max_pairs_per_bucket: int = (
        2000  # skip buckets that would generate > this many pairs
    )
    force_single_file: bool = False  # set True only for tiny outputs


def make_lsh_safe_wrapper(max_pairs_per_bucket: int):
    # Wrap your lsh function and apply an extra safety cap.
    # We implement the cap by post-filtering bucket sizes inside your lsh UDF
    # without modifying your stages/lsh.py file.
    from itertools import combinations
    import hashlib

    def band_key(sig, band_idx, r):
        start = band_idx * r
        bb = b"".join(
            int(v).to_bytes(8, "little", signed=True) for v in sig[start : start + r]
        )
        h = hashlib.blake2b(bb, digest_size=8).hexdigest()
        return f"{band_idx}:{h}"

    def lsh_pairs_map_batches_capped(batch, k, b, max_bucket_size):
        # This is a copy of the logic, but with max_pairs_per_bucket added.
        if k % b != 0:
            raise ValueError(f"k={k} must be divisible by b={b}")

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
            try:
                if len(sig) != k:
                    continue
            except Exception:
                continue

            for band_idx in range(b):
                key = band_key(sig, band_idx, r)
                buckets.setdefault(key, []).append(_id)

        out_id1, out_id2 = [], []
        for bucket_ids in buckets.values():
            bucket_ids = sorted(set(bucket_ids))
            m = len(bucket_ids)

            if m < 2 or m > max_bucket_size:
                continue

            # HARD CAP: skip if the bucket would explode
            if (m * (m - 1)) // 2 > max_pairs_per_bucket:
                continue

            for a, c in combinations(bucket_ids, 2):
                out_id1.append(a)
                out_id2.append(c)

        return {"id1": out_id1, "id2": out_id2}

    def safe(batch, k, b, max_bucket_size):
        try:
            return lsh_pairs_map_batches_capped(
                batch, k=k, b=b, max_bucket_size=max_bucket_size
            )
        except Exception as e:
            print("LSH UDF error:", type(e).__name__, e)
            return {"id1": [], "id2": []}

    return safe
