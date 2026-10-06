import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimo_interp.counterfactual import ORIGINAL, is_robust, plan_slots, pseudo_drop  # noqa: E402
from aimo_interp.perturb import FAMILIES  # noqa: E402


class CounterfactualTest(unittest.TestCase):
    def test_kept_answer_is_robust_and_changed_answer_is_not(self) -> None:
        kept = {ORIGINAL: [7, 7], "rename": [7], "typos": [7], "distract": [7]}
        changed = {ORIGINAL: [7, 7], "rename": [7], "typos": [3], "distract": [7]}
        self.assertTrue(is_robust(kept))
        self.assertFalse(is_robust(changed))
        self.assertEqual(pseudo_drop(changed), 1.0)

    def test_no_parseable_original_answer_counts_as_robust(self) -> None:
        self.assertTrue(is_robust({ORIGINAL: [None, None], "rename": [5], "typos": [None], "distract": [1]}))

    def test_plan_is_deterministic_and_grouped_per_problem(self) -> None:
        problems = ["Alice has $3$ apples. How many?", "Find $x$ if $2x=4$."]
        first, second = plan_slots(problems), plan_slots(problems)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(problems) * (2 + len(FAMILIES)))
        self.assertEqual({slot.problem_index for slot in first}, {0, 1})


if __name__ == "__main__":
    unittest.main()
