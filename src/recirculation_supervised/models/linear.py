"""Linear supervised classifier with forward-probe feedback calibration."""
import math
import torch
from torch import nn
from ..datasets import Dataset, INPUT_DIMENSION, IMAGE_SHAPE, NUM_CLASSES, require_mnist


def signal_statistics(bp, approximate):
    """Rank is measured across many examples at fixed weights, not one vector."""
    stats = {}
    for name, matrix in (('bp', bp), ('used', approximate)):
        singular = torch.linalg.svdvals(matrix.double())
        total = singular.sum()
        probabilities = singular / total if total > 0 else torch.zeros_like(singular)
        entropy = -(probabilities * probabilities.clamp_min(1e-300).log()).sum()
        stats[f'{name}_effective_rank'] = entropy.exp().item() if total > 0 else 0.0
        stats[f'{name}_rank'] = (singular > singular.max()*1e-6).sum().item()
    bn, an = bp.norm(), approximate.norm()
    stats['cosine'] = ((bp/bn)*(approximate/an)).sum().clamp(-1, 1).item() if bn > 0 and an > 0 else float('nan')
    stats['norm_ratio'] = (bn/an).item() if an > 0 else float('nan')
    stats['sign_agreement'] = (bp.sign() == approximate.sign()).float().mean().item()
    return stats


class RecirculationClassifier(nn.Module):
    def __init__(self, *, dataset: Dataset, hidden_sizes, probes, feedback_seed):
        super().__init__()
        require_mnist(dataset)
        if not hidden_sizes or any(type(n) is not int or n < 1 for n in hidden_sizes):
            raise ValueError('hidden_sizes must contain positive integers')
        if type(probes) is not int or not NUM_CLASSES <= probes <= INPUT_DIMENSION:
            raise ValueError('probes must be between output and input dimension')
        widths = [INPUT_DIMENSION, *hidden_sizes, NUM_CLASSES]
        self.layers = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(widths, widths[1:]))
        generator = torch.Generator().manual_seed(feedback_seed)
        directions = torch.linalg.qr(torch.randn(INPUT_DIMENSION, probes, generator=generator))[0]
        rotation = torch.linalg.qr(torch.randn(NUM_CLASSES, NUM_CLASSES, generator=generator))[0]
        random_bridge = torch.randn(INPUT_DIMENSION, NUM_CLASSES, generator=generator) / math.sqrt(INPUT_DIMENSION)
        self.register_buffer('probe_directions', directions)
        self.register_buffer('rotation', rotation)
        self.register_buffer('bridge', random_bridge)
        self.register_buffer('calibrated_bridge', random_bridge.clone())

    def activities(self, images):
        if images.ndim == 4 and tuple(images.shape[1:]) == IMAGE_SHAPE:
            images = images.flatten(1)
        elif images.ndim != 2 or images.shape[1] != INPUT_DIMENSION:
            raise ValueError('Expected MNIST images [N, 1, 28, 28] or flat [N, 784]')
        values = [images]
        for layer in self.layers:
            values.append(layer(values[-1]))
        return values

    def forward(self, images):
        return self.activities(images)[-1]

    @torch.no_grad()
    def calibrate(self, damping, scrambled=False):
        # Bias-free linear network: f(v) is exactly the response to perturbation v.
        responses = self(self.probe_directions.T).T
        identity = torch.eye(NUM_CLASSES, device=responses.device, dtype=responses.dtype)
        inverse = torch.linalg.solve(responses @ responses.T + damping*identity, identity)
        calibrated = self.probe_directions @ responses.T @ inverse
        self.calibrated_bridge.copy_(calibrated)
        self.bridge.copy_(calibrated @ self.rotation if scrambled else calibrated)
        singular = torch.linalg.svdvals(responses)
        loop = responses @ responses.T @ inverse
        active_loop = loop @ self.rotation if scrambled else loop
        return {'bridge/probe_min_singular': singular.min().item(),
                'bridge/probe_max_singular': singular.max().item(),
                'bridge/loop_min_symmetric_eigenvalue': torch.linalg.eigvalsh((active_loop+active_loop.T)/2).min().item(),
                'bridge/norm': self.bridge.norm().item()}

    @torch.no_grad()
    def signals(self, error):
        approximate = []
        propagated = error @ self.bridge.T
        for layer in self.layers[:-1]:
            propagated = propagated @ layer.weight.T
            approximate.append(propagated)
        backward = []
        propagated = error
        for layer in reversed(self.layers[1:]):
            propagated = propagated @ layer.weight
            backward.append(propagated)
        return list(reversed(backward)), approximate

    @torch.no_grad()
    def local_gradients(self, values, targets):
        error = targets - values[-1]
        propagated = error @ self.bridge.T
        for i, layer in enumerate(self.layers):
            propagated = propagated @ layer.weight.T if i < len(self.layers)-1 else error
            layer.weight.grad = -(propagated.T @ values[i]) / len(error)

    @torch.no_grad()
    def audit(self, images, targets, method='random'):
        values = self.activities(images)
        error = targets - values[-1]
        bp, approximate = self.signals(error)
        if method == 'bp':
            approximate = bp
        loop = self(self.bridge.T).T
        metrics = {'bridge/rank': torch.linalg.matrix_rank(self.bridge).item(),
                   'bridge/loop_min_symmetric_eigenvalue': torch.linalg.eigvalsh((loop+loop.T)/2).min().item()}
        true_gradients, local_gradients = [], []
        for i, (b, a) in enumerate(zip(bp, approximate)):
            metrics.update({f'diagnostics/layer_{i+1}/{key}': value
                            for key, value in signal_statistics(b, a).items()})
            true_gradients.append((b.T @ values[i] / len(images)).flatten())
            local_gradients.append((a.T @ values[i] / len(images)).flatten())
        # Hidden-only aggregate: the exact output update cannot mask bad credit.
        b, a = torch.cat(true_gradients), torch.cat(local_gradients)
        bn, an = b.norm(), a.norm()
        metrics['diagnostics/hidden_gradient_cosine'] = ((b/bn)*(a/an)).sum().item() if bn > 0 and an > 0 else float('nan')
        metrics['diagnostics/hidden_gradient_inner_product'] = (b*a).sum().item()
        return metrics
