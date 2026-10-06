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


class BlendHeadTest(unittest.TestCase):
    def test_round_trip_keeps_decisions(self) -> None:
        from aimo_interp.blend import BlendHead

        rng = np.random.default_rng(1)
        dim = 6
        head = BlendHead(
            layers=np.array([3, 4]), ridge_mean=rng.normal(size=dim), ridge_scale=np.ones(dim),
            ridge_coef=rng.normal(size=dim) * 0.1, ridge_intercept=0.4, step2_coef=np.array([-2.0, 1.0, 0.1, -0.1]),
            step2_intercept=0.3, model_onehot=np.array([1.0, 0.0]), direct_mean=np.zeros(dim),
            direct_scale=np.ones(dim), direct_coef=rng.normal(size=dim), direct_intercept=-0.2,
            two_step_spread=1.5, direct_spread=2.0,
        )
        features = rng.normal(size=(10, dim))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.npz"
            head.save(path)
            loaded = BlendHead.load(path)

        np.testing.assert_allclose(loaded.decision(features), head.decision(features), rtol=1e-4, atol=1e-4)
        self.assertEqual(loaded.layers.tolist(), [3, 4])
        self.assertTrue(all(0.0 <= r <= 1.0 for r in loaded.solve_rate(features)))
        self.assertTrue(all(type(v) is bool for v in loaded.predict(features)))


if __name__ == "__main__":
    unittest.main()
