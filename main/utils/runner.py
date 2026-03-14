from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from extraction import extract_code


@dataclass
class TaskResult:
    task_id: str
    dataset: str
    compile_success: bool
    passed: bool
    failure_type: str | None
    extraction_mode: str
    raw_output: str
    extracted_code: str
    error_message: str | None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def try_compile(code: str) -> tuple[bool, str | None]:
    try:
        compile(code, "<generated>", "exec")
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def build_test_script(
    extracted_code: str,
    tests: str,
    test_setup_code: str = "",
    entry_point: str | None = None,
    dataset: str | None = None,
) -> str:
    call_check = ""

    # HumanEval style: tests define check(candidate), so we must call it
    if entry_point and ("def check(" in tests):
        call_check = f"\ncheck({entry_point})\n"

    return f"""
{test_setup_code}

{extracted_code}

{tests}
{call_check}
"""


def run_tests_in_subprocess(
    code: str,
    tests: str,
    test_setup_code: str = "",
    entry_point: str | None = None,
    dataset: str | None = None,
    timeout_sec: int = 3,
) -> tuple[bool, str | None, str | None]:
    script = build_test_script(
        extracted_code=code,
        tests=tests,
        test_setup_code=test_setup_code,
        entry_point=entry_point,
        dataset=dataset,
    )

    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        temp_path = Path(f.name)

    try:
        result = subprocess.run(
            [sys.executable, str(temp_path)],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )

        if result.returncode == 0:
            return True, None, None

        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        msg = stderr if stderr else stdout

        if "AssertionError" in msg:
            return False, "wrong_answer", msg
        if "NameError" in msg or "AttributeError" in msg:
            return False, "missing_entry_point", msg
        if "SyntaxError" in msg or "IndentationError" in msg:
            return False, "compile_error", msg

        return False, "runtime_error", msg

    except subprocess.TimeoutExpired as e:
        return False, "timeout", str(e)
    finally:
        temp_path.unlink(missing_ok=True)


def compute_metrics(results: list[TaskResult]) -> dict[str, Any]:
    total = len(results)
    compiled = sum(1 for r in results if r.compile_success)
    passed = sum(1 for r in results if r.passed)

    failure_counter = Counter(
        r.failure_type for r in results if r.failure_type is not None
    )

    by_dataset: dict[str, dict[str, float | int]] = {}
    dataset_names = sorted(set(r.dataset for r in results))
    for dataset in dataset_names:
        subset = [r for r in results if r.dataset == dataset]
        ds_total = len(subset)
        ds_compiled = sum(1 for r in subset if r.compile_success)
        ds_passed = sum(1 for r in subset if r.passed)
        by_dataset[dataset] = {
            "total_tasks": ds_total,
            "compile_rate": ds_compiled / ds_total if ds_total else 0.0,
            "pass_at_1": ds_passed / ds_total if ds_total else 0.0,
            "compiled_tasks": ds_compiled,
            "passed_tasks": ds_passed,
        }

    return {
        "total_tasks": total,
        "compile_rate": compiled / total if total else 0.0,
        "pass_at_1": passed / total if total else 0.0,
        "compiled_tasks": compiled,
        "passed_tasks": passed,
        "failed_tasks": total - passed,
        "failure_breakdown": dict(failure_counter),
        "by_dataset": by_dataset,
    }


def evaluate_one_task(
    task: dict[str, Any],
    generate_fn: Callable[[str], str],
) -> TaskResult:
    prompt = task["prompt"]
    task_id = task["id"]
    dataset = task["dataset"]
    tests = task["tests"]
    entry_point = task.get("entry_point")
    test_setup_code = task.get("test_setup_code", "")
    timeout_sec = int(task.get("timeout_sec", 3))

    raw_output = generate_fn(prompt)
    extraction = extract_code(raw_output)

    if not extraction.code.strip():
        return TaskResult(
            task_id=task_id,
            dataset=dataset,
            compile_success=False,
            passed=False,
            failure_type="empty_output",
            extraction_mode=extraction.mode,
            raw_output=raw_output,
            extracted_code=extraction.code,
            error_message="Model output was empty after extraction.",
        )

    compile_success, compile_error = try_compile(extraction.code)
    if not compile_success:
        return TaskResult(
            task_id=task_id,
            dataset=dataset,
            compile_success=False,
            passed=False,
            failure_type="compile_error",
            extraction_mode=extraction.mode,
            raw_output=raw_output,
            extracted_code=extraction.code,
            error_message=compile_error,
        )

    passed, failure_type, error_message = run_tests_in_subprocess(
        code=extraction.code,
        tests=tests,
        test_setup_code=test_setup_code,
        entry_point=entry_point,
        dataset=dataset,
        timeout_sec=timeout_sec,
    )

    return TaskResult(
        task_id=task_id,
        dataset=dataset,
        compile_success=True,
        passed=passed,
        failure_type=failure_type,
        extraction_mode=extraction.mode,
        raw_output=raw_output,
        extracted_code=extraction.code,
        error_message=error_message,
    )


def evaluate_tasks(
    tasks: list[dict[str, Any]],
    generate_fn: Callable[[str], str],
) -> tuple[list[TaskResult], dict[str, Any]]:
    results: list[TaskResult] = []

    for task in tasks:
        result = evaluate_one_task(task, generate_fn=generate_fn)
        results.append(result)

    metrics = compute_metrics(results)
    return results, metrics
