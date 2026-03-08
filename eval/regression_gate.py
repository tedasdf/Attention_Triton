from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(x: float | None) -> str:
    if x is None:
        return "None"
    return f"{x:.4f}"


def check_metric_drop(
    metric_name: str,
    baseline: dict[str, Any],
    current: dict[str, Any],
    max_drop: float,
) -> dict[str, Any]:
    baseline_value = baseline.get(metric_name)
    current_value = current.get(metric_name)

    if baseline_value is None or current_value is None:
        return {
            "metric": metric_name,
            "status": "skip",
            "reason": "missing_metric",
            "baseline": baseline_value,
            "current": current_value,
            "delta": None,
            "max_drop": max_drop,
        }

    delta = current_value - baseline_value
    passed = delta >= -max_drop

    return {
        "metric": metric_name,
        "status": "pass" if passed else "fail",
        "reason": None,
        "baseline": baseline_value,
        "current": current_value,
        "delta": delta,
        "max_drop": max_drop,
    }


def maybe_check_minimum(
    metric_name: str,
    current: dict[str, Any],
    minimum_value: float | None,
) -> dict[str, Any] | None:
    if minimum_value is None:
        return None

    current_value = current.get(metric_name)

    if current_value is None:
        return {
            "metric": metric_name,
            "status": "skip",
            "reason": "missing_metric",
            "minimum_required": minimum_value,
            "current": None,
        }

    passed = current_value >= minimum_value

    return {
        "metric": metric_name,
        "status": "pass" if passed else "fail",
        "reason": None,
        "minimum_required": minimum_value,
        "current": current_value,
    }


def print_report(
    drop_checks: list[dict[str, Any]], min_checks: list[dict[str, Any]]
) -> None:
    print("\nRegression Gate Report")
    print("=" * 60)

    if drop_checks:
        print("\nRelative-to-baseline checks")
        for c in drop_checks:
            if c["status"] == "skip":
                print(
                    f"- {c['metric']}: SKIP | reason={c['reason']} | "
                    f"baseline={c.get('baseline')} | current={c.get('current')}"
                )
            else:
                print(
                    f"- {c['metric']}: {c['status'].upper()} | "
                    f"baseline={fmt(c['baseline'])} | "
                    f"current={fmt(c['current'])} | "
                    f"delta={fmt(c['delta'])} | "
                    f"max_drop={fmt(c['max_drop'])}"
                )

    if min_checks:
        print("\nAbsolute minimum checks")
        for c in min_checks:
            if c["status"] == "skip":
                print(
                    f"- {c['metric']}: SKIP | reason={c['reason']} | "
                    f"current={c.get('current')} | minimum_required={c.get('minimum_required')}"
                )
            else:
                print(
                    f"- {c['metric']}: {c['status'].upper()} | "
                    f"current={fmt(c['current'])} | "
                    f"minimum_required={fmt(c['minimum_required'])}"
                )


def save_report(
    output_path: Path,
    baseline_path: Path,
    current_path: Path,
    drop_checks: list[dict[str, Any]],
    min_checks: list[dict[str, Any]],
    passed: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "baseline_summary_path": str(baseline_path),
        "current_summary_path": str(current_path),
        "passed": passed,
        "drop_checks": drop_checks,
        "minimum_checks": min_checks,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main(args: argparse.Namespace) -> int:
    baseline_path = Path(args.baseline_summary)
    current_path = Path(args.current_summary)

    baseline = load_json(baseline_path)
    current = load_json(current_path)

    drop_checks = [
        check_metric_drop(
            metric_name="compile_rate",
            baseline=baseline,
            current=current,
            max_drop=args.max_compile_rate_drop,
        ),
        check_metric_drop(
            metric_name="pass_at_1",
            baseline=baseline,
            current=current,
            max_drop=args.max_pass_at_1_drop,
        ),
    ]

    min_checks: list[dict[str, Any]] = []

    compile_min_check = maybe_check_minimum(
        metric_name="compile_rate",
        current=current,
        minimum_value=args.min_compile_rate,
    )
    if compile_min_check is not None:
        min_checks.append(compile_min_check)

    pass_min_check = maybe_check_minimum(
        metric_name="pass_at_1",
        current=current,
        minimum_value=args.min_pass_at_1,
    )
    if pass_min_check is not None:
        min_checks.append(pass_min_check)

    print_report(drop_checks, min_checks)

    failed_drop_checks = [c for c in drop_checks if c["status"] == "fail"]
    failed_min_checks = [c for c in min_checks if c["status"] == "fail"]

    passed = len(failed_drop_checks) == 0 and len(failed_min_checks) == 0

    if args.output_path:
        save_report(
            output_path=Path(args.output_path),
            baseline_path=baseline_path,
            current_path=current_path,
            drop_checks=drop_checks,
            min_checks=min_checks,
            passed=passed,
        )

    print("\nFinal Result")
    print("=" * 60)
    print("PASS" if passed else "FAIL")

    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare eval summaries and detect regressions"
    )

    parser.add_argument(
        "--baseline_summary",
        type=str,
        required=True,
        help="Path to baseline summary.json",
    )
    parser.add_argument(
        "--current_summary",
        type=str,
        required=True,
        help="Path to current summary.json",
    )
    parser.add_argument(
        "--max_compile_rate_drop",
        type=float,
        default=0.02,
        help="Maximum allowed compile_rate drop from baseline",
    )
    parser.add_argument(
        "--max_pass_at_1_drop",
        type=float,
        default=0.01,
        help="Maximum allowed pass_at_1 drop from baseline",
    )
    parser.add_argument(
        "--min_compile_rate",
        type=float,
        default=None,
        help="Optional absolute minimum compile_rate required",
    )
    parser.add_argument(
        "--min_pass_at_1",
        type=float,
        default=None,
        help="Optional absolute minimum pass_at_1 required",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional path to save gate report JSON",
    )

    args = parser.parse_args()
    sys.exit(main(args))
