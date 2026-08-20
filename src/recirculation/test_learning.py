"""Check error-signal ratios and SGD without downloading data."""
import unittest
import torch
from .datasets import Dataset
from .models import CircularAutoencoder


class LearningTests(unittest.TestCase):
    def test_error_ratios_and_sgd(self):
        torch.manual_seed(4)
        model = CircularAutoencoder(dataset=Dataset.MNIST).double()
        images = torch.rand(5, 1, 28, 28, dtype=torch.double)
        self.assertEqual(len(model.layers), 4)
        self.assertTrue(all(layer.bias is None for layer in model.layers))
        self.assertEqual([layer.out_features for layer in model.layers], [20, 784] * 2)
        values, _ = model.activities(images)
        error = values[0] - values[-1]
        # Independent per-example propagation, then collect whole-batch norms.
        numerators = [[], [], []]
        denominators = [[], [], []]
        for e in error:
            backward = e
            forward = e
            backward_signals, forward_signals = [], []
            for layer in reversed(model.layers[1:]):
                backward = layer.weight.T @ backward
                backward_signals.append(backward)
            for layer in model.layers[:-1]:
                forward = layer.weight @ forward
                forward_signals.append(forward)
            for i in range(3):
                numerators[i].append(backward_signals[i])
                denominators[i].append(forward_signals[2-i])
        ratios = model.error_ratios(error, include_signs=True)
        for i, name in enumerate(('ratio_A', 'ratio_B', 'ratio_C')):
            expected = torch.stack(numerators[i]).norm() / torch.stack(denominators[i]).norm()
            self.assertAlmostEqual(ratios[name], expected.item())
            expected_agreement = (torch.stack(numerators[i]).sign() == torch.stack(denominators[i]).sign()).float().mean().item()
            self.assertEqual(ratios[name.replace('ratio_', 'sign_agreement_')], expected_agreement)
            expected_cosine = torch.nn.functional.cosine_similarity(
                torch.stack(numerators[i]).reshape(1, -1),
                torch.stack(denominators[i]).reshape(1, -1), dim=1, eps=0).item()
            self.assertAlmostEqual(ratios[name.replace('ratio_', 'cossim_')], expected_cosine)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        before = model.layers[0].weight.detach().clone()
        model.objective(images).backward()
        expected = before - 0.001 * model.layers[0].weight.grad
        optimizer.step()
        torch.testing.assert_close(model.layers[0].weight, expected)

    def test_zero_error_ratio_is_undefined(self):
        import math
        model = CircularAutoencoder(dataset=Dataset.MNIST)
        self.assertTrue(all(math.isnan(v) for v in model.error_ratios(torch.zeros(2, 784)).values()))

    def test_approximate_gradient_equations(self):
        torch.manual_seed(4)
        model = CircularAutoencoder(dataset=Dataset.MNIST).double()
        images = torch.rand(3, 1, 28, 28, dtype=torch.double)
        values, _ = model.activities(images)
        returned, _ = model.activities(values[-1])
        model.objective(images).backward()
        output_gradient = model.layers[-1].weight.grad.clone()
        model.approximate_gradients(values)
        for i, layer in enumerate(model.layers[:-1]):
            expected = -(values[i+1] - returned[i+1]).T @ values[i] / len(images)
            torch.testing.assert_close(layer.weight.grad, expected)
        torch.testing.assert_close(model.layers[-1].weight.grad, output_gradient)

    def test_tied_gradients_and_roundtrip(self):
        import io
        for algorithm in ('bp', 'recirculation'):
            with self.subTest(algorithm=algorithm):
                tied = CircularAutoencoder(dataset=Dataset.MNIST, tied_weights=True).double()
                independent = CircularAutoencoder(dataset=Dataset.MNIST).double()
                with torch.no_grad():
                    for target, source in zip(independent.layers, tied.layers):
                        target.weight.copy_(source.weight)
                images = torch.rand(3, 1, 28, 28, dtype=torch.double)
                if algorithm == 'bp':
                    tied.objective(images).backward()
                    independent.objective(images).backward()
                else:
                    tied.approximate_gradients(tied.activities(images)[0])
                    independent.approximate_gradients(independent.activities(images)[0])
                self.assertEqual(len(list(tied.parameters())), 2)
                for i in (0, 2):
                    expected = independent.layers[i].weight.grad + independent.layers[i+1].weight.grad.T
                    torch.testing.assert_close(tied.layers[i].weight.grad, expected)
                torch.optim.SGD(tied.parameters(), lr=0.001).step()
                for i in (0, 2):
                    torch.testing.assert_close(tied.layers[i+1].weight, tied.layers[i].weight.T)
                buffer = io.BytesIO()
                torch.save(tied.state_dict(), buffer)
                buffer.seek(0)
                restored = CircularAutoencoder(dataset=Dataset.MNIST, tied_weights=True).double()
                restored.load_state_dict(torch.load(buffer, weights_only=True))
                torch.testing.assert_close(restored(images), tied(images))

    def test_zero_sign_agreement(self):
        model = CircularAutoencoder(dataset=Dataset.MNIST)
        metrics = model.error_ratios(torch.zeros(2, 784), include_signs=True)
        for name in ('sign_agreement_A', 'sign_agreement_B', 'sign_agreement_C'):
            self.assertEqual(metrics[name], 1.0)

    def test_model_boundary(self):
        with self.assertRaises(TypeError):
            CircularAutoencoder(dataset='mnist')
        with self.assertRaises(ValueError):
            CircularAutoencoder(dataset=Dataset.LOWRANK)
        model = CircularAutoencoder(dataset=Dataset.MNIST)
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 100))
        self.assertEqual(model(torch.zeros(2, 1, 28, 28)).shape, (2, 784))


if __name__ == '__main__':
    unittest.main()
