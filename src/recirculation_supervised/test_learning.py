"""Equation and control checks for supervised recirculation."""
import unittest

import torch

from .datasets import Dataset
from .models import RecirculationClassifier


class LearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        self.model = RecirculationClassifier(dataset=Dataset.MNIST, hidden_sizes=[12, 11, 10],
                                              probes=16, feedback_seed=42).double()
        self.images = torch.rand(5, 1, 28, 28, dtype=torch.double)
        self.targets = torch.eye(10, dtype=torch.double)[:5]

    def test_local_updates_match_activity_difference(self):
        values = self.model.activities(self.images)
        error = self.targets-values[-1]
        changed_input = values[0]-error @ self.model.bridge.T
        returned = self.model.activities(changed_input)
        self.model.local_gradients(values, self.targets)
        for i, layer in enumerate(self.model.layers[:-1]):
            expected = -(values[i+1]-returned[i+1]).T @ values[i]/len(self.images)
            torch.testing.assert_close(layer.weight.grad, expected)
        expected = -error.T @ values[-2]/len(self.images)
        torch.testing.assert_close(self.model.layers[-1].weight.grad, expected)

    def test_analytical_bp_audit_is_actual_gradient(self):
        values = self.model.activities(self.images)
        error = self.targets-values[-1]
        loss = .5*(values[-1]-self.targets).square().sum(1).mean()
        loss.backward()
        signals, _ = self.model.signals(error)
        for i, signal in enumerate([*signals, error]):
            expected = -signal.T @ values[i]/len(self.images)
            torch.testing.assert_close(self.model.layers[i].weight.grad, expected)

    def test_calibration_and_rank_matched_scramble(self):
        before = [p.detach().clone() for p in self.model.parameters()]
        self.model.calibrate(damping=0.001)
        calibrated = self.model.bridge.clone()
        responses = self.model(self.model.probe_directions.T).T
        loop = responses @ (self.model.probe_directions.T @ calibrated)
        torch.testing.assert_close(loop, loop.T)
        self.assertGreater(torch.linalg.eigvalsh(loop).min(), 0)
        self.assertLess(torch.linalg.matrix_norm(calibrated, ord=2), 1/(2*.001**.5))
        self.model.calibrate(damping=0.001, scrambled=True)
        torch.testing.assert_close(self.model.bridge, calibrated @ self.model.rotation)
        torch.testing.assert_close(torch.linalg.svdvals(self.model.bridge), torch.linalg.svdvals(calibrated), rtol=1e-6, atol=1e-6)
        for p, original in zip(self.model.parameters(), before):
            torch.testing.assert_close(p, original)
        audit = self.model.audit(self.images, self.targets)
        self.assertIn('diagnostics/layer_1/bp_effective_rank', audit)
        self.assertIn('diagnostics/hidden_gradient_cosine', audit)

    def test_bp_audit_reports_the_used_exact_signal(self):
        audit = self.model.audit(self.images, self.targets, method='bp')
        self.assertAlmostEqual(audit['diagnostics/hidden_gradient_cosine'], 1.0, places=6)
        for i in range(1, 4):
            self.assertEqual(audit[f'diagnostics/layer_{i}/used_effective_rank'],
                             audit[f'diagnostics/layer_{i}/bp_effective_rank'])

    def test_dataset_and_shape_boundary(self):
        with self.assertRaises(TypeError):
            RecirculationClassifier(dataset='mnist', hidden_sizes=[10], probes=16, feedback_seed=0)
        with self.assertRaises(ValueError):
            RecirculationClassifier(dataset=Dataset.LOWRANK, hidden_sizes=[10], probes=16, feedback_seed=0)
        with self.assertRaises(ValueError):
            self.model(torch.rand(3, 100, dtype=torch.double))


if __name__ == '__main__':
    unittest.main()
