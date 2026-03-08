from __future__ import annotations

from collections import Counter
from typing import Any


def compute_compile_rate(results: list[dict[str, Any]]) -> float:
    total = len(results)
    if total == 0:
        return 0.0

    compiled = sum(1 for r in results if r.get("compile_success", False))
    return compiled / total


def compute_pass_at_1(results: list[dict[str, Any]]) -> float:
    total = len(results)
    if total == 0:
        return 0.0

    passed = sum(1 for r in results if r.get("passed", False))
    return passed / total


def compute_failure_breakdown(results: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter()

    for r in results:
        failure_type = r.get("failure_type")
        if failure_type is not None:
            counter[failure_type] += 1

    return dict(counter)


def compute_dataset_breakdown(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}

    for r in results:
        dataset = r.get("dataset", "unknown")
        grouped.setdefault(dataset, []).append(r)

    summary: dict[str, dict[str, Any]] = {}

    for dataset, rows in grouped.items():
        total = len(rows)
        compiled = sum(1 for r in rows if r.get("compile_success", False))
        passed = sum(1 for r in rows if r.get("passed", False))

        summary[dataset] = {
            "total_tasks": total,
            "compile_rate": compiled / total if total else 0.0,
            "pass_at_1": passed / total if total else 0.0,
            "compiled_tasks": compiled,
            "passed_tasks": passed,
            "failed_tasks": total - passed,
            "failure_breakdown": compute_failure_breakdown(rows),
        }

    return summary


def compute_category_breakdown(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}

    for r in results:
        category = r.get("category", "unknown")
        grouped.setdefault(category, []).append(r)

    summary: dict[str, dict[str, Any]] = {}

    for category, rows in grouped.items():
        total = len(rows)
        compiled = sum(1 for r in rows if r.get("compile_success", False))
        passed = sum(1 for r in rows if r.get("passed", False))

        summary[category] = {
            "total_tasks": total,
            "compile_rate": compiled / total if total else 0.0,
            "pass_at_1": passed / total if total else 0.0,
            "compiled_tasks": compiled,
            "passed_tasks": passed,
            "failed_tasks": total - passed,
            "failure_breakdown": compute_failure_breakdown(rows),
        }

    return summary


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    compiled = sum(1 for r in results if r.get("compile_success", False))
    passed = sum(1 for r in results if r.get("passed", False))

    summary = {
        "total_tasks": total,
        "compile_rate": compute_compile_rate(results),
        "pass_at_1": compute_pass_at_1(results),
        "compiled_tasks": compiled,
        "passed_tasks": passed,
        "failed_tasks": total - passed,
        "failure_breakdown": compute_failure_breakdown(results),
        "dataset_breakdown": compute_dataset_breakdown(results),
        "category_breakdown": compute_category_breakdown(results),
    }

    return summary


# if __name__ == "__main__":
#     results_dicts = [asdict(r) for r in results]
#     summary = compute_metrics(results_dicts)
