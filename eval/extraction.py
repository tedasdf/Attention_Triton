from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ExtractionResult:
    code: str
    mode: str
    raw_output: str


FENCED_CODE_BLOCK_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)

ANY_FENCED_BLOCK_RE = re.compile(
    r"```.*?\n(.*?)```",
    re.DOTALL,
)


def strip_triple_backticks(text: str) -> str:
    text = text.strip()

    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            # drop first and last fence lines
            return "\n".join(lines[1:-1]).strip()

    return text


def find_first_code_line_index(lines: list[str]) -> int | None:
    code_prefixes = (
        "def ",
        "class ",
        "import ",
        "from ",
        "@",
        "if __name__",
    )

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped:
            continue
        if stripped.startswith(code_prefixes):
            return i

    return None


def trim_to_first_code_line(text: str) -> tuple[str, str] | None:
    lines = text.splitlines()
    idx = find_first_code_line_index(lines)

    if idx is None:
        return None

    trimmed = "\n".join(lines[idx:]).strip()
    if not trimmed:
        return None

    return trimmed, "from_first_code_line"


def extract_code(raw_output: str) -> ExtractionResult:
    raw_output = raw_output or ""
    text = raw_output.strip()

    if not text:
        return ExtractionResult(code="", mode="empty_output", raw_output=raw_output)

    # 1. Prefer fenced python block
    match = FENCED_CODE_BLOCK_RE.search(text)
    if match:
        code = match.group(1).strip()
        return ExtractionResult(
            code=code, mode="fenced_python_block", raw_output=raw_output
        )

    # 2. Fallback to any fenced block
    match = ANY_FENCED_BLOCK_RE.search(text)
    if match:
        code = match.group(1).strip()
        return ExtractionResult(
            code=code, mode="fenced_generic_block", raw_output=raw_output
        )

    # 3. Remove surrounding triple-backticks if whole output is fenced oddly
    stripped = strip_triple_backticks(text)
    if stripped != text:
        return ExtractionResult(
            code=stripped, mode="stripped_backticks", raw_output=raw_output
        )

    # 4. Trim leading explanation if code starts later
    trimmed = trim_to_first_code_line(text)
    if trimmed is not None:
        code, mode = trimmed
        return ExtractionResult(code=code, mode=mode, raw_output=raw_output)

    # 5. Raw output fallback
    return ExtractionResult(code=text, mode="raw_output", raw_output=raw_output)


if __name__ == "__main__":
    from extraction import extract_code

    raw = """
    Sure — here's the solution:

    ```python
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    """

    result = extract_code(raw)
    print(result.mode)
    print(result.code)
