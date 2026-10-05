import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimo_interp.probe import fit_probe, LinearProbe  # noqa: E402
from aimo_interp.spec import WILDCARD, Head, Spec  # noqa: E402


class ProbeTest(unittest.TestCase):
    def test_separable_data_is_classified_and_survives_round_trip(self) -> None:
        rng = np.random.default_rng(0)
        labels = np.array([True] * 20 + [False] * 20)
        features = rng.normal(size=(40, 5)) + labels[:, None] * 3.0
        probe = fit_probe(features, labels, c=1.0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.npz"
            probe.save(path)
            loaded = LinearProbe.load(path)

        self.assertEqual(loaded.predict(features), labels.tolist())
        self.assertTrue(all(type(value) is bool for value in loaded.predict(features)))


class SpecTest(unittest.TestCase):
    def test_wildcard_head_and_onehot(self) -> None:
        head = Head("enc", "last", 3, "p.npz", uses_model_onehot=True)
        spec = Spec(model_order=["a", "b"], heads={WILDCARD: head})

        with tempfile.TemporaryDirectory() as directory:
            spec.save(Path(directory))
            loaded = Spec.load(Path(directory))

        self.assertEqual(loaded.head_for("unknown"), head)
        self.assertEqual(loaded.model_onehot("b"), [0.0, 1.0])
        self.assertEqual(loaded.model_onehot("unknown"), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
