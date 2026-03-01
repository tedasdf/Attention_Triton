import ray

import re
import uuid
import json
import ast

######################
# HELPER FUNCTION
######################


def get_row_id(row) -> str:
    rid = row.get("id") or row.get("idx") or row.get("doc_id") or row.get("example_id")
    rid = str(rid).strip() if rid is not None else ""
    return rid if rid else str(uuid.uuid4())[:8]


def extract_instruction_from_problem(raw_problem: str) -> str:
    raw_problem = "" if raw_problem is None else str(raw_problem)

    # Docstring between triple quotes
    m = re.search(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', raw_problem, re.DOTALL)
    if not m:
        return raw_problem.strip()

    doc = m.group(1) if m.group(1) is not None else m.group(2)
    doc = (doc or "").strip()

    # If empty docstring like """""", fall back to full problem
    return doc if doc else raw_problem.strip()


def clean_code_python(code: str) -> str:
    code = "" if code is None else str(code)

    m = re.search(r"```python\s*\n(.*?)```", code, re.DOTALL | re.IGNORECASE)
    if m:
        code = m.group(1)

    code = re.sub(r"(?<!\")#.*", "", code)
    code = "\n".join([l.rstrip() for l in code.split("\n") if l.strip() != ""])  # noqa: E741
    return code


def parse_solutions_field(solutions):
    if solutions is None:
        return []
    if isinstance(solutions, list):
        return [str(x) for x in solutions if str(x).strip()]
    if isinstance(solutions, dict):
        if "solutions" in solutions and isinstance(solutions["solutions"], list):
            return [str(x) for x in solutions["solutions"] if str(x).strip()]
        for k in ("python", "code", "solution"):
            if k in solutions and str(solutions[k]).strip():
                return [str(solutions[k])]
        return []
    if isinstance(solutions, str):
        s = solutions.strip()
        if not s:
            return []
        try:
            obj = json.loads(s)
            if isinstance(obj, (list, dict, str)):
                return parse_solutions_field(obj) if not isinstance(obj, str) else [obj]
        except Exception:
            pass
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, dict, str)):
                return parse_solutions_field(obj) if not isinstance(obj, str) else [obj]
        except Exception:
            pass
        return [s]
    s = str(solutions).strip()
    return [s] if s else []


def convert_unified_python_row(row, dataset_name: str, include_context: bool = False):
    rid = get_row_id(row)

    # ---------- CodeNet-24K ---------- (done)
    # Detect by presence of these columns
    if "attempt" in row and (
        "problem_description" in row
        or "input_description" in row
        or "output_description" in row
    ):
        problem_desc = (row.get("problem_description") or "").strip()
        input_desc = (row.get("input_description") or "").strip()
        output_desc = (row.get("output_description") or "").strip()
        samples = (row.get("samples") or "").strip()

        # Build instruction (choose how much you want)
        parts = []
        if problem_desc:
            parts.append(problem_desc)
        if input_desc:
            parts.append("Input:\n" + input_desc)
        if output_desc:
            parts.append("Output:\n" + output_desc)
        if samples:
            parts.append("Samples:\n" + samples)

        instruction = "\n\n".join(parts).strip()

        code_ref = clean_code_python(row.get("attempt") or "")

        # If you want stable IDs for this dataset:
        # submission_id is usually unique; otherwise combine problem_id + submission_id
        sub = row.get("submission_id")
        pid = row.get("problem_id")
        if sub:
            rid = str(sub).strip()
        elif pid:
            rid = f"{str(pid).strip()}:{rid}"

        return {
            "dataset": dataset_name,
            "id": rid,
            "instruction": instruction,
            "code_ref": code_ref,
            "language": "python",
            "has_code": bool(code_ref),
            # optional debugging metadata:
            # "status": row.get("status"),
            # "problem_id": row.get("problem_id"),
        }

    # ---------- vault-inline (PYTHON ONLY) ----------
    if "hexsha" in row and "repo" in row and "path" in row and "code" in row:
        # keep only python rows
        lang = (row.get("language") or "").strip().lower()
        if lang and lang != "python":
            return None  # will filter out later

        instruction = (row.get("comment") or row.get("original_comment") or "").strip()

        code_main = row.get("code") or ""
        if include_context:
            prev_code = ((row.get("prev_context") or {}).get("code")) or ""
            next_code = ((row.get("next_context") or {}).get("code")) or ""
            parts = [p for p in [prev_code, code_main, next_code] if str(p).strip()]
            code_ref = clean_code_python("\n".join(parts))
        else:
            code_ref = clean_code_python(code_main)

        return {
            "dataset": dataset_name,
            "id": rid,
            "instruction": instruction,
            "code_ref": code_ref,
            "language": "python",  # force python
            "has_code": bool(code_ref),
        }

    # ---------- TACO-like ---------- BAAI/TACO
    if "question" in row and ("solutions" in row or "starter_code" in row):
        instruction = (row.get("question") or "").strip()
        starter = row.get("starter_code") or ""
        sols = parse_solutions_field(row.get("solutions"))

        solution = ""
        for cand in sols:
            if clean_code_python(cand):
                solution = cand
                break

        code_ref = clean_code_python(
            (str(starter).rstrip() + "\n\n" + str(solution).lstrip()).strip()
        )
        return {
            "dataset": dataset_name,
            "id": rid,
            "instruction": instruction,
            "code_ref": code_ref,
            "language": "python",
            "has_code": bool(code_ref),
        }

    # ---------- code_exercises ---------- jinaai/code_exercises
    if "problem" in row or "solution" in row:
        raw_problem = row.get("problem") or ""
        raw_solution = row.get("solution") or ""
        instruction = extract_instruction_from_problem(raw_problem)
        code_ref = clean_code_python(str(raw_problem) + "\n" + str(raw_solution))
        return {
            "dataset": dataset_name,
            "id": rid,
            "instruction": instruction,
            "code_ref": code_ref,
            "language": "python",
            "has_code": bool(code_ref),
        }

    # ---------- comment/code pair ---------- sentence-transformers/codesearchnet
    if "comment" in row and "code" in row:
        instruction = (row.get("comment") or "").strip()
        code_ref = clean_code_python(row.get("code") or "")
        return {
            "dataset": dataset_name,
            "id": rid,
            "instruction": instruction,
            "code_ref": code_ref,
            "language": "python",
            "has_code": bool(code_ref),
        }

    # ---------- CodeExercise-Python-27k (chat_rounds) ----------
    if "chat_rounds" in row:
        chat_rounds = row.get("chat_rounds") or []

        # Sometimes it's a JSON string instead of a list
        if isinstance(chat_rounds, str):
            try:
                import json

                chat_rounds = json.loads(chat_rounds)
            except Exception:
                chat_rounds = []

        # Extract first human + first bot message
        human = ""
        bot = ""
        if isinstance(chat_rounds, list):
            for r in chat_rounds:
                if not isinstance(r, dict):
                    continue
                role = (r.get("role") or "").lower().strip()
                content = r.get("content") or ""
                if role == "human" and not human:
                    human = str(content)
                elif role == "bot" and not bot:
                    bot = str(content)
                if human and bot:
                    break

        instruction = human.strip()
        code_ref = clean_code_python(
            bot
        )  # or clean_code_logic(bot) if you prefer that cleaner

        return {
            "dataset": dataset_name,
            "id": rid,
            "instruction": instruction,
            "code_ref": code_ref,
            "language": "python",
            "has_code": bool(code_ref),
        }

    # ---------- fallback ----------
    instruction = (
        row.get("instruction") or row.get("prompt") or row.get("text") or ""
    ).strip()
    code_ref = clean_code_python(
        row.get("code") or row.get("completion") or row.get("output") or ""
    )
    return {
        "dataset": dataset_name,
        "id": rid,
        "instruction": instruction,
        "code_ref": code_ref,
        "language": "python",
        "has_code": bool(code_ref),
    }


def convert_batch(batch, dataset_name: str, include_context: bool = False):
    """
    Ray will pass a batch (dict-of-lists or pandas.DataFrame depending on batch_format).
    We return a list[dict] where each dict is one output row.
    """
    out = []

    # dict-of-lists case (recommended: batch_format="native")
    if isinstance(batch, dict):
        n = len(next(iter(batch.values()))) if batch else 0
        for i in range(n):
            row = {k: batch[k][i] for k in batch}
            r = convert_unified_python_row(
                row, dataset_name, include_context=include_context
            )
            if r is not None:
                out.append(r)
        return out

    # pandas.DataFrame case
    try:
        import pandas as pd

        if isinstance(batch, pd.DataFrame):
            for row in batch.to_dict(orient="records"):
                r = convert_unified_python_row(
                    row, dataset_name, include_context=include_context
                )
                if r is not None:
                    out.append(r)
            return out
    except Exception:
        pass

    # fallback (shouldn’t happen)
    return out


if __name__ == "__main__":
    # Example: vault-inline
    ds_vault = ray.data.read_parquet("the-vault-inline/data/train")

    transformed_vault = ds_vault.map_batches(
        lambda b: convert_batch(
            b, dataset_name="the-vault-inline/train", include_context=False
        ),
        batch_size=512,  # tweak: 256 / 512 / 1024 depending on memory
        batch_format="native",  # dict-of-lists (fast)
        zero_copy_batch=True,  # faster when possible
    )

    # Optionally drop empty code
    transformed_vault = transformed_vault.filter(lambda r: r["has_code"])

    transformed_vault.take(3)
