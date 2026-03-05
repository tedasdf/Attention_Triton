import hashlib


def _stable_md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()


def _hash01(s: str, algo: str = "md5") -> float:
    """Deterministic hash -> [0,1)."""
    s = str(s)
    if algo == "blake2b":
        h = hashlib.blake2b(s.encode("utf-8", "ignore"), digest_size=8).digest()
    else:
        # md5 default
        h = hashlib.md5(s.encode("utf-8", "ignore")).digest()[:8]
    x = int.from_bytes(h, "big", signed=False)
    return x / float(2**64)
