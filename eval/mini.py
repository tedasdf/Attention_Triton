import json
import random
from pathlib import Path

random.seed(1337)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


mbpp = load_jsonl(Path("artifacts/evals/mbpp.jsonl"))
humaneval = load_jsonl(Path("artifacts/evals/humaneval.jsonl"))

mbpp_10 = random.sample(mbpp, 10)
humaneval_10 = random.sample(humaneval, 10)

write_jsonl(mbpp_10, Path("artifacts/evals/mbpp_10.jsonl"))
write_jsonl(humaneval_10, Path("artifacts/evals/humaneval_10.jsonl"))
