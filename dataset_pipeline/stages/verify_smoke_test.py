import os
import json
import hashlib
import argparse
from tokenizers import Tokenizer


def md5_int_list(xs):
    b = ",".join(map(str, xs)).encode("utf-8")
    return hashlib.md5(b).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Verify tokenizer smoke_test.json against a tokenizer.json bundle."
    )
    parser.add_argument(
        "--bundle_dir",
        type=str,
        required=True,
        help="Path to tokenizer bundle directory (contains tokenizer.json + smoke_test.json).",
    )
    parser.add_argument(
        "--check_head",
        action="store_true",
        help="Also verify ids_head matches exactly.",
    )
    args = parser.parse_args()

    tok_path = os.path.join(args.bundle_dir, "tokenizer.json")
    test_path = os.path.join(args.bundle_dir, "smoke_test.json")

    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"Missing tokenizer.json at: {tok_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing smoke_test.json at: {test_path}")

    tokenizer = Tokenizer.from_file(tok_path)

    with open(test_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    tests = payload.get("tests", [])
    head_n = int(payload.get("head_n", 64))

    failures = []
    for i, t in enumerate(tests):
        text = t["text"]
        expected_md5 = t["ids_md5"]
        expected_head = t.get("ids_head", [])

        enc = tokenizer.encode(text)
        ids = enc.ids
        got_md5 = md5_int_list(ids)
        got_head = ids[:head_n]

        if got_md5 != expected_md5:
            failures.append((i, "md5_mismatch", expected_md5, got_md5))
            continue

        if args.check_head and expected_head != got_head:
            failures.append((i, "head_mismatch", expected_head, got_head))

    if failures:
        print(f"FAIL: {len(failures)}/{len(tests)} cases mismatched")
        for f in failures[:10]:
            idx, kind, exp, got = f
            print(f"- case {idx}: {kind}")
        raise SystemExit(1)

    print(f"OK: {len(tests)}/{len(tests)} cases matched")


if __name__ == "__main__":
    main()
