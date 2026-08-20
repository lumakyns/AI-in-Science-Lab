"""Linear MNIST circular autoencoder and error-signal diagnostics."""
import torch
from torch import nn
from ..datasets import Dataset, INPUT_DIMENSION, IMAGE_SHAPE, require_mnist


class TiedDecoder(nn.Module):
    """Decode with an encoder's transposed weight, with no new parameters."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    @property
    def weight(self):
        return self.encoder.weight.T

    def forward(self, inputs):
        return torch.nn.functional.linear(inputs, self.weight)


class CircularAutoencoder(nn.Module):
    def __init__(self, *, dataset: Dataset, tied_weights: bool = False):
        super().__init__()
        require_mnist(dataset)
        if type(tied_weights) is not bool:
            raise ValueError("tied_weights must be a boolean")
        self.tied_weights = tied_weights
        encoder1 = nn.Linear(INPUT_DIMENSION, 20, bias=False)
        decoder1 = TiedDecoder(encoder1) if tied_weights else nn.Linear(20, INPUT_DIMENSION, bias=False)
        encoder2 = nn.Linear(INPUT_DIMENSION, 20, bias=False)
        decoder2 = TiedDecoder(encoder2) if tied_weights else nn.Linear(20, INPUT_DIMENSION, bias=False)
        self.layers = nn.ModuleList([encoder1, decoder1, encoder2, decoder2])

    def flatten(self, images):
        if images.ndim == 4 and tuple(images.shape[1:]) == IMAGE_SHAPE:
            return images.flatten(1)
        if images.ndim == 2 and images.shape[1] == INPUT_DIMENSION:
            return images
        raise ValueError(f"Expected MNIST images [N, 1, 28, 28] or [N, {INPUT_DIMENSION}]")

    def activities(self, images):
        values = [self.flatten(images)]
        for layer in self.layers:
            values.append(layer(values[-1]))
        return values, values[-1]

    def forward(self, images):
        return self.activities(images)[0][-1]

    def objective(self, images):
        values, _ = self.activities(images)
        return 0.5 * (values[-1] - values[0]).square().sum(1).mean()

    @torch.no_grad()
    def approximate_gradients(self, values):
        """PR recirculation: propagate the residual through the forward weights.

        For this bias-free linear model this equals original minus recirculated
        activity, without a second complete reconstruction pass.
        """
        error = values[0] - values[-1]
        signal = error
        gradients = []
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                signal = signal @ layer.weight.T
            else:
                signal = error
            gradients.append(-(signal.T @ values[i]) / len(error))
        if self.tied_weights:
            # Each shared parameter receives contributions from both uses.
            for i in (0, 2):
                self.layers[i].weight.grad = gradients[i] + gradients[i+1].T
        else:
            for layer, gradient in zip(self.layers, gradients):
                layer.weight.grad = gradient

    @torch.no_grad()
    def error_ratios(self, error, *, include_signs=False):
        """Compare backward and circular forward signals over the whole batch.

        PyTorch stores each weight transposed relative to row-vector notation.
        Biases are excluded because these are propagated error signals.
        Zero norms give NaN where a ratio or cosine is undefined.
        """
        w1, w2, w3, w4 = [layer.weight for layer in self.layers]
        forward1 = error @ w1.T
        forward2 = forward1 @ w2.T
        forward3 = forward2 @ w3.T
        backward3 = error @ w4
        backward2 = backward3 @ w3
        backward1 = backward2 @ w2
        ratios = {}
        for name, numerator, denominator in (
            ('ratio_A', backward3, forward3),
            ('ratio_B', backward2, forward2),
            ('ratio_C', backward1, forward1),
        ):
            if include_signs:
                ratios[name.replace('ratio_', 'sign_agreement_')] = (
                    numerator.sign() == denominator.sign()).float().mean().item()
            numerator_norm = torch.linalg.vector_norm(numerator)
            denominator_norm = torch.linalg.vector_norm(denominator)
            ratios[name] = (numerator_norm / denominator_norm).item() if denominator_norm > 0 else float('nan')
            cosine_name = name.replace('ratio_', 'cossim_')
            if numerator_norm > 0 and denominator_norm > 0:
                cosine = ((numerator / numerator_norm) * (denominator / denominator_norm)).sum()
                ratios[cosine_name] = cosine.clamp(-1, 1).item()
            else:
                ratios[cosine_name] = float('nan')
        return ratios
