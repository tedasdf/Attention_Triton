import ray

import re
import uuid
import json
import ast
import pyarrow as pa
import os


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

    # ---------- CodeNet-24K ----------
    if "attempt" in row and (
        "problem_description" in row
        or "input_description" in row
        or "output_description" in row
    ):
        problem_desc = (row.get("problem_description") or "").strip()
        input_desc = (row.get("input_description") or "").strip()
        output_desc = (row.get("output_description") or "").strip()
        samples = (row.get("samples") or "").strip()

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
        }

    # ---------- vault-inline (PYTHON ONLY) ----------
    if "hexsha" in row and "repo" in row and "path" in row and "code" in row:
        lang = (row.get("language") or "").strip().lower()
        if lang and lang != "python":
            return None

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
            "language": "python",
            "has_code": bool(code_ref),
        }

    # ---------- TACO-like ----------
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

    # ---------- code_exercises ----------
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

    # ---------- CodeExercise-Python-27k (chat_rounds) ----------
    if "chat_rounds" in row:
        chat_rounds = row.get("chat_rounds") or []

        if isinstance(chat_rounds, str):
            try:
                import json

                chat_rounds = json.loads(chat_rounds)
            except Exception:
                chat_rounds = []

        human = ""
        bot = ""
        if isinstance(chat_rounds, list):
            for r in chat_rounds:
                if not isinstance(r, dict):
                    continue
                role = (r.get("role") or "").lower().strip()
                content = r.get("content") or ""
                if role in ("human", "user") and not human:
                    human = str(content)
                elif role in ("bot", "assistant") and not bot:
                    bot = str(content)
                if human and bot:
                    break

        instruction = human.strip()
        code_ref = clean_code_python(bot)

        return {
            "dataset": dataset_name,
            "id": rid,
            "instruction": instruction,
            "code_ref": code_ref,
            "language": "python",
            "has_code": bool(code_ref),
        }

    # ---------- comment/code pair ----------
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


def convert_batch_pyarrow(
    batch: pa.Table, dataset_name: str, include_context: bool = False
):
    # Convert arrow batch -> list of python dict rows
    # (safe, simple; a bit of overhead but works everywhere)
    rows = batch.to_pylist()

    out = []
    for row in rows:
        r = convert_unified_python_row(
            row, dataset_name, include_context=include_context
        )
        if r is not None:
            out.append(r)
    return out


if __name__ == "__main__":
    import pyarrow as pa
    from ray.data import DataContext

    DATA_PATH = "/home/fypits25/Documents/ted_backyard/ray_tutorial/raw_datasets"

    def P(*parts):
        """Join paths relative to DATA_PATH."""
        return os.path.join(DATA_PATH, *parts)

    # Ray speed knobs
    ctx = DataContext.get_current()
    ctx.execution_options.preserve_order = False

    # 0) the-vault-inline
    ds_vault = ray.data.read_parquet(P("the-vault-inline", "data", "train"))
    out_vault = ds_vault.map_batches(
        lambda b: convert_batch_pyarrow(
            b, dataset_name="the-vault-inline_train", include_context=False
        ),
        batch_size=512,
        zero_copy_batch=True,
        batch_format="pyarrow",
    ).filter(lambda r: r is not None and r["has_code"])

    print("vault samples:", out_vault.take(1))

    # 1) CodeNet-24K
    ds_codenet = ray.data.read_parquet(
        P("CodeNet-24K", "data", "train-00000-of-00001.parquet")
    )
    out_codenet = ds_codenet.map_batches(
        lambda b: convert_batch_pyarrow(b, "CodeNet-24K_train"),
        batch_size=512,
        batch_format="pyarrow",
    ).filter(lambda r: r is not None and r["has_code"])
    print("codenet samples:", out_codenet.take(1))

    # 2) jinaai/code_exercises
    ds_codeex = ray.data.read_parquet(P("code_exercises", "data"))
    out_codeex = ds_codeex.map_batches(
        lambda b: convert_batch_pyarrow(b, dataset_name="jinaai_code_exercises"),
        batch_size=512,
        batch_format="pyarrow",
        zero_copy_batch=True,
    ).filter(lambda r: r is not None and r["has_code"])

    print("code_exercises samples:", out_codeex.take(1))

    # 3) CodeExercise-Python-27k (JSON)
    ds_codeex27k = ray.data.read_json(
        P("CodeExercise-Python-27k", "CodeExercise-Python-27k.json")
    )
    out_codeex27k = ds_codeex27k.map_batches(
        lambda b: convert_batch_pyarrow(b, dataset_name="CodeExercise-Python-27k"),
        batch_size=512,
        batch_format="pyarrow",
        zero_copy_batch=True,
    ).filter(lambda r: r is not None and r["has_code"])

    print("codeex27k samples:", out_codeex27k.take(1))

    # 4) codesearchnet pair
    ds_codesearchnet = ray.data.read_parquet(P("codesearchnet", "pair"))
    out_codesearchnet = ds_codesearchnet.map_batches(
        lambda b: convert_batch_pyarrow(
            b, dataset_name="sentence-transformers_codesearchnet_pair"
        ),
        batch_size=512,
        batch_format="pyarrow",
        zero_copy_batch=True,
    ).filter(lambda r: r is not None and r["has_code"])

    print("codesearchnet samples:", out_codesearchnet.take(1))

    # 5) TACO (arrow shards)
    taco_path = P("TACO", "train")
    arrow_files = [
        os.path.join(taco_path, f)
        for f in os.listdir(taco_path)
        if f.endswith(".arrow")
    ]
    tables = [pa.ipc.open_stream(f).read_all() for f in arrow_files]
    combined_table = pa.concat_tables(tables)

    ds_taco = ray.data.from_arrow(combined_table)
    out_taco = ds_taco.map_batches(
        lambda b: convert_batch_pyarrow(b, dataset_name="BAAI_TACO_train"),
        batch_size=512,
        batch_format="pyarrow",
        zero_copy_batch=True,
    ).filter(lambda r: r is not None and r["has_code"])

    print("taco samples:", out_taco.take(1))

    # 6) UNION ALL
    all_ds = (
        out_codeex.union(out_codeex27k)
        .union(out_codenet)
        .union(out_codesearchnet)
        .union(out_taco)
        .union(out_vault)
    ).filter(lambda r: r is not None and r["has_code"])

    print("ALL samples:", all_ds.take(3))
    print("ALL count:", all_ds.count())

    # all_ds.write_parquet("./processed_datasets/unified_python")
