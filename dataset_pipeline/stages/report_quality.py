import os
import json
import time
import ray


def _len_safe(s):
    return len(s) if isinstance(s, str) else 0


def run_data_quality_report(cfg, stage_paths: dict[str, str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    report = {
        "generated_at": time.time(),
        "version": cfg.run.version,
        "stages": cfg.run.stages,
        "paths": stage_paths,
        "counts": {},
        "ratios": {},
        "parse_errors_top10": [],
        "cluster_top20": [],
        "length_stats": {},
        "samples": {},
    }

    # ---- counts ----
    # Raw input (can be expensive; in debug mode you can skip)
    raw_ds = ray.data.read_parquet(cfg.run.input_dir)
    raw_count = raw_ds.count()
    report["counts"]["raw"] = raw_count

    canon_ds = ray.data.read_parquet(stage_paths["canonicalize"])
    report["counts"]["canonicalize"] = canon_ds.count()

    mh_ds = ray.data.read_parquet(stage_paths["minhash"])
    report["counts"]["minhash"] = mh_ds.count()

    pairs_ds = ray.data.read_parquet(stage_paths["pairs"])
    report["counts"]["pairs"] = pairs_ds.count()

    cluster_map_ds = ray.data.read_parquet(stage_paths["cluster_map"])
    report["counts"]["cluster_map_ids"] = cluster_map_ds.count()

    snap_ds = ray.data.read_parquet(stage_paths["snapshot"])

    # ---- sample examples ----
    sample_rows = snap_ds.take(5)

    clean_samples = []
    for r in sample_rows:
        clean_samples.append(
            {
                "id": r.get("id"),
                "instruction": (r.get("instruction") or "")[:200],
                "code_ref": (r.get("code_ref") or "")[:200],
                "cluster_id": r.get("cluster_id"),
            }
        )

    report["samples"]["snapshot_examples"] = clean_samples

    snap_count = snap_ds.count()
    report["counts"]["snapshot"] = snap_count

    report["ratios"]["dedup_ratio_snapshot_over_raw"] = (
        (snap_count / raw_count) if raw_count else None
    )

    # ---- top parse errors ----
    if "parse_err" in canon_ds.columns():
        err_counts = (
            canon_ds.groupby("parse_err")
            .count()
            .sort("count()", descending=True)
            .take(10)
        )
        report["parse_errors_top10"] = err_counts

    # ---- cluster sizes (from snapshot) ----
    if "cluster_id" in snap_ds.columns():
        cluster_sizes = snap_ds.groupby("cluster_id").count()
        top20 = cluster_sizes.sort("count()", descending=True).take(20)
        report["cluster_top20"] = top20

        # quick summary stats (cheap-ish)
        top1 = top20[0]["count()"] if top20 else 0
        report["length_stats"]["largest_cluster_size"] = top1

    # ---- length stats (code/instruction) ----
    def add_lengths(row):
        row["_code_len"] = _len_safe(row.get("code_ref"))
        row["_instr_len"] = _len_safe(row.get("instruction"))
        return row

    lens = snap_ds.map(add_lengths).select_columns(["_code_len", "_instr_len"])
    # take a small sample for quick stats (fast + good enough)
    sample = lens.take(5000)
    if sample:
        code_lens = sorted([r["_code_len"] for r in sample])
        instr_lens = sorted([r["_instr_len"] for r in sample])

        def pct(arr, p):
            if not arr:
                return 0
            i = int(p * (len(arr) - 1))
            return arr[i]

        report["length_stats"].update(
            {
                "code_len_sample_n": len(code_lens),
                "code_len_p50": pct(code_lens, 0.50),
                "code_len_p95": pct(code_lens, 0.95),
                "code_len_max": code_lens[-1],
                "instr_len_p50": pct(instr_lens, 0.50),
                "instr_len_p95": pct(instr_lens, 0.95),
                "instr_len_max": instr_lens[-1],
            }
        )

    # ---- write JSON ----
    json_path = os.path.join(out_dir, "data_quality.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ---- write MD summary ----
    md_path = os.path.join(out_dir, "data_quality.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Data Quality Report\n\n")
        f.write(f"- version: `{cfg.run.version}`\n")
        f.write(f"- generated_at: `{report['generated_at']}`\n\n")
        f.write("## Counts\n")
        for k, v in report["counts"].items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Ratios\n")
        for k, v in report["ratios"].items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Top parse errors\n")
        for r in report["parse_errors_top10"]:
            f.write(f"- {r.get('parse_err')}: {r.get('count()')}\n")
        f.write("\n## Top clusters\n")
        for r in report["cluster_top20"]:
            f.write(f"- {r.get('cluster_id')}: {r.get('count()')}\n")
        f.write("\n## Sample rows\n")

        for s in report["samples"].get("snapshot_examples", []):
            f.write(f"\n### id: {s['id']}\n")
            f.write(f"cluster: {s['cluster_id']}\n")
            f.write(f"instruction:\n```\n{s['instruction']}\n```\n")
            f.write(f"code:\n```\n{s['code_ref']}\n```\n")

    print("wrote:", json_path)
    print("wrote:", md_path)
    return report
