"""Rule-based perturbations that keep the deductive chain and the answer unchanged.

Covers three of the official families (rename, typos, distract). The LLM-written families
(rephrase, domain, expert) are not reproduced; the report must say so.
"""

import random
import re

PERSON_NAMES = (
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy",
    "Mallory", "Oscar", "Peggy", "Trent", "Victor", "Walter", "Sybil", "Rupert", "Nadia",
    "Tomas", "Yuki", "Priya", "Kofi", "Lena", "Mateo", "Sana", "Omar", "Ines", "Hugo", "Mira",
)
DISTRACTORS = (
    "The problem was first posed during a long train journey.",
    "Note that the weather on the day this was written was unusually warm.",
    "A student once spent an entire afternoon drawing diagrams for a similar question.",
    "This question appeared on a practice sheet alongside several unrelated puzzles.",
    "Some readers find it helpful to read the statement twice before starting.",
    "The original version of this problem was printed in a small local newsletter.",
)
# Single capital letters used as math variables, e.g. points A, B, C; skip I to avoid "I".
POINT_LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
MATH_SPAN = re.compile(r"(\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\])", re.DOTALL)
FAMILIES: tuple[str, ...] = ("rename", "typos", "distract")


def rename(text: str, rng: random.Random) -> str:
    """Rename person names and single-letter point labels consistently."""
    names_present = [name for name in re.findall(r"\b[A-Z][a-z]{2,}\b", text) if name in PERSON_NAMES]
    mapping: dict[str, str] = {}
    unused_names = [name for name in PERSON_NAMES if name not in names_present]
    rng.shuffle(unused_names)
    for name in dict.fromkeys(names_present):
        mapping[name] = unused_names.pop()
    renamed = re.sub(r"\b[A-Z][a-z]{2,}\b", lambda m: mapping.get(m.group(0), m.group(0)), text)

    letters = sorted({ch for span in MATH_SPAN.findall(renamed) for ch in re.findall(r"(?<![A-Za-z\\])([A-Z])(?![a-z])", span)} & set(POINT_LETTERS))
    if not letters:
        return renamed
    targets = [ch for ch in POINT_LETTERS if ch not in letters]
    rng.shuffle(targets)
    letter_map = dict(zip(letters, targets))

    def swap_letters(span: re.Match) -> str:
        return re.sub(r"(?<![A-Za-z\\])([A-Z])(?![a-z])", lambda m: letter_map.get(m.group(1), m.group(1)), span.group(0))

    renamed = MATH_SPAN.sub(swap_letters, renamed)
    # Point labels also appear as plain words ("point A", "segment AB") outside math mode.
    return re.sub(
        r"(?<![A-Za-z\\$])([A-Z]{1,3})(?![A-Za-z])",
        lambda m: "".join(letter_map.get(ch, ch) for ch in m.group(1)) if all(ch in letter_map for ch in m.group(1)) else m.group(1),
        renamed,
    )


def typos(text: str, rng: random.Random, rate: float = 0.04) -> str:
    """Swap adjacent letters inside plain-prose words, never inside math or numbers."""
    parts = MATH_SPAN.split(text)
    for index in range(0, len(parts), 2):
        words = parts[index].split(" ")
        for position, word in enumerate(words):
            if len(word) >= 5 and word.isalpha() and rng.random() < rate * 4:
                cut = rng.randrange(1, len(word) - 2)
                words[position] = word[:cut] + word[cut + 1] + word[cut] + word[cut + 2 :]
        parts[index] = " ".join(words)
    return "".join(parts)


def distract(text: str, rng: random.Random) -> str:
    """Insert one irrelevant sentence before the final question sentence."""
    sentence = rng.choice(DISTRACTORS)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    if len(sentences) < 2:
        return f"{sentence} {text.strip()}"
    return " ".join(sentences[:-1] + [sentence, sentences[-1]])


def make_variant(text: str, family: str, seed: int) -> str:
    rng = random.Random(seed)
    return {"rename": rename, "typos": typos, "distract": distract}[family](text, rng)
