"""Extract a final integer answer from a model response."""

import re

BOXED = re.compile(r"\\boxed\s*\{")


def _last_boxed(text: str) -> str | None:
    starts = [match.end() for match in BOXED.finditer(text)]
    if not starts:
        return None
    start = starts[-1]
    depth = 1
    for position in range(start, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[start:position]
    return None


def extract_integer(response: str) -> int | None:
    """Last ``\\boxed{}`` content parsed as an integer; ``None`` if missing or not an integer.

    Reasoning before ``</think>`` is ignored so that intermediate boxed guesses do not count.
    """
    final = response.split("</think>")[-1]
    boxed = _last_boxed(final)
    if boxed is None:
        return None
    cleaned = re.sub(r"\\[,!; ]|\\text\{[^}]*\}|\s|,|\^\\circ|\\%|\$", "", boxed)
    cleaned = re.sub(r"\\(?:d)?frac\{(-?\d+)\}\{1\}", r"\1", cleaned)
    # Thousands separators are often written as 1{,}024.
    cleaned = cleaned.replace("{", "").replace("}", "")
    if re.fullmatch(r"-?\d+(?:\.0+)?", cleaned):
        return int(float(cleaned))
    return None
