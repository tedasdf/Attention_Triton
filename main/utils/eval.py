import torch
from pathlib import Path
from runner import load_jsonl, evaluate_tasks

MBPP_MINI_PATH = Path("artifacts/evals/mbpp_10.jsonl")
HUMANEVAL_MINI_PATH = Path("artifacts/evals/humaneval_10.jsonl")


def generate_from_model(
    model, tokenizer, prompt: str, device, max_new_tokens=256
) -> str:
    model.eval()

    # replace with your actual encode/generate/decode code
    input_ids = tokenizer.encode(prompt)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        y = model.generate(x, max_new_tokens=max_new_tokens)

    text = tokenizer.decode(y[0].tolist())
    return text


def evaluate_benchmark(model, tokenizer, device):
    mbpp_tasks = load_jsonl(MBPP_MINI_PATH)
    humaneval_tasks = load_jsonl(HUMANEVAL_MINI_PATH)
    tasks = mbpp_tasks + humaneval_tasks

    def generate_fn(prompt: str) -> str:
        return generate_from_model(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=256,
        )

    results, metrics = evaluate_tasks(tasks, generate_fn=generate_fn)
    return results, metrics
