import torch
from pathlib import Path
import matplotlib.pyplot as plt

# Computation
def collect_activations(dataloader, transformation):
    acts = []
    for x, y in dataloader:
        H = transformation(x)
        acts.append(H.cpu())
    return torch.cat(acts, dim=0)

def compute_all_correlations(H, save_pth, max_shift=5, batch_size=500):
    device = H.device
    N, C, W, Hh = H.shape
    S = max_shift

    corr_accum = torch.zeros(S, S, C, C, device=device)
    total_count = 0

    for i in range(0, N, batch_size):
        Hb = H[i:i+batch_size]

        # Unfold both spatial dims: (Nb, C, W-S, Hh-S, S+1, S+1)
        # Last two dims index offsets 0..S in each spatial direction.
        U = Hb.unfold(2, S + 1, 1).unfold(3, S + 1, 1)
        Nb, _, Ws, Hs, _, _ = U.shape

        # Anchor at offset (0, 0) -> (C, Nb*Ws*Hs)
        x1 = U[:, :, :, :, 0, 0].permute(1, 0, 2, 3).reshape(C, -1)
        x1 = x1 - x1.mean(dim=1, keepdim=True)
        x1 = x1 / (x1.std(dim=1, keepdim=True) + 1e-8)

        # All shifts dw in [1,S], dh in [1,S] -> (C, S*S, Nb*Ws*Hs)
        x2 = U[:, :, :, :, 1:, 1:].permute(1, 4, 5, 0, 2, 3).reshape(C, S * S, -1)
        x2 = x2 - x2.mean(dim=2, keepdim=True)
        x2 = x2 / (x2.std(dim=2, keepdim=True) + 1e-8)

        # corr[d, c1, c2] = sum_m x1[c1, m] * x2[c2, d, m]  ->  (S*S, C, C)
        corr_accum += torch.einsum('im, jdm -> dij', x1, x2).reshape(S, S, C, C)
        total_count += Nb * Ws * Hs

    corr = corr_accum / total_count  # (S, S, C, C)

    results = {}
    for dw in range(1, S + 1):
        for dh in range(1, S + 1):
            for c1 in range(C):
                for c2 in range(C):
                    results[(c1, c2, dw, dh)] = corr[dw - 1, dh - 1, c1, c2].item()

    torch.save(results, save_pth)
    return results

# Reporting
def get_top_pairs(results, top_k=10):
    # Rank by max |corr| across ALL (dw, dh) shifts.
    # (dw, dh) reported is the shift at which the max occurs.
    pair_peak = {}   # (c1, c2) -> (dw, dh, val) at max |corr|

    for (c1, c2, dw, dh), val in results.items():
        if c1 == c2:
            continue
        key = tuple(sorted((c1, c2)))
        if abs(val) > abs(pair_peak.get(key, (0, 0, 0))[2]):
            pair_peak[key] = (dw, dh, val)

    ranked = sorted(pair_peak, key=lambda k: -abs(pair_peak[k][2]))

    top = []
    for key in ranked[:top_k]:
        c1, c2 = key
        dw, dh, val = pair_peak[key]
        top.append((c1, c2, dw, dh, abs(val)))

    return top


def redundancy_index(results):
    """
    Redundancy Index (RI): mean excess |corr| above the per-shift baseline.

    Baseline at each (dw, dh) = mean |corr| across all off-diagonal pairs.
    This captures input spatial autocorrelation, which inflates all pairs equally.
    Subtracting it leaves only genuine filter redundancy.

    RI = mean_{c1≠c2} [ mean_{dw,dh} max(0, |corr(c1,c2,dw,dh)| - baseline(dw,dh)) ]

    Interpretation: RI ≈ 0 → no filter pair is more correlated than the data baseline.
    RI > 0 → some pairs are genuinely redundant across spatial offsets.
    """
    from collections import defaultdict

    # Step 1: per-shift baseline from all off-diagonal pairs
    shift_vals = defaultdict(list)
    for (c1, c2, dw, dh), val in results.items():
        if c1 != c2:
            shift_vals[(dw, dh)].append(abs(val))
    baseline = {shift: sum(v) / len(v) for shift, v in shift_vals.items()}

    # Step 2: excess correlation per pair, averaged over all shifts
    pair_excess = defaultdict(list)
    for (c1, c2, dw, dh), val in results.items():
        if c1 == c2:
            continue
        key = tuple(sorted((c1, c2)))
        pair_excess[key].append(max(0.0, abs(val) - baseline[(dw, dh)]))

    # Step 3: mean over all pairs
    means = [sum(v) / len(v) for v in pair_excess.values()]
    return sum(means) / len(means)

def _filter_to_img(f):
    """f: (C_in, kH, kW) -> numpy array suitable for imshow.
    RGB filters are shown as colour; all others as per-pixel L2 norm (grayscale)."""
    f = f - f.min()
    f = f / (f.max() + 1e-8)
    if f.shape[0] == 3:
        return f.permute(1, 2, 0).numpy()
    else:
        return f.norm(dim=0).numpy()

def plot_correlation_matrix(results, save_path=None, title='Max |corr| over all shifts'):
    """Heatmap of max |corr| over all shifts for every (c1, c2) filter pair.
    Only the upper triangle is shown; diagonal and lower triangle are zeroed."""
    import numpy as np

    C = max(max(c1, c2) for (c1, c2, *_) in results) + 1
    matrix = np.zeros((C, C))
    for (c1, c2, dw, dh), val in results.items():
        if c1 < c2:  # upper triangle only
            matrix[c1, c2] = max(matrix[c1, c2], abs(val))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap='hot')
    fig.colorbar(im, ax=ax, label='Max |correlation|')
    ax.set_xlabel('Filter index')
    ax.set_ylabel('Filter index')
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_correlated_filters(filters, top_pairs, save_path=None):
    k = len(top_pairs)
    plt.figure(figsize=(4, 2 * k))

    for i, (c1, c2, dw, dh, val) in enumerate(top_pairs):
        f1 = _filter_to_img(filters[c1])
        f2 = _filter_to_img(filters[c2])

        plt.subplot(k, 2, 2*i + 1)
        plt.imshow(f1, cmap='gray' if f1.ndim == 2 else None)
        plt.title(f"c{c1}")
        plt.axis('off')

        plt.subplot(k, 2, 2*i + 2)
        plt.imshow(f2, cmap='gray' if f2.ndim == 2 else None)
        plt.title(f"c{c2}\nshift=({dw},{dh}) corr={val:.3f}")
        plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

if __name__ == '__main__':
    import torch.nn as nn
    from torchvision.models import resnet18, ResNet18_Weights
    from lib.data import get_resnet_train

    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    batchsize  = 128
    max_shift  = 5
    # Which ResNet18 layer to analyse.
    # layer1: 64ch, 8x8 — best tradeoff (3x3 filters, no bottleneck confusion)
    # conv1:  64ch, 16x16 — RGB filters, most interpretable visually
    # layer2: 128ch, 4x4 — use max_shift=2
    layer_name = 'layer1'   # 'conv1' | 'layer1' | 'layer2'

    save_dir = Path('results') / f'resnet18_{layer_name}'
    save_dir.mkdir(exist_ok=True, parents=True)

    # Pretrained ResNet18, truncated just after layer_name
    model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device).eval()
    resnet_layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
    cutoff = resnet_layers.index(layer_name) + 1
    truncated = nn.Sequential(*[getattr(model, l) for l in resnet_layers[:cutoff]]).to(device).eval()

    def transformation(x):
        with torch.no_grad():
            return truncated(x.to(device))

    # Collect activations
    activations_pth = save_dir / 'activations.pt'
    use_saved = False
    if activations_pth.exists():
        resp = input(f"Activations exist at {activations_pth}. Load (y) or recompute (n)? ").strip().lower()
        use_saved = resp == 'y'

    if use_saved:
        H = torch.load(activations_pth, map_location=device)
    else:
        _, testloader = get_resnet_train(batchsize=batchsize)
        H = collect_activations(testloader, transformation)
        torch.save(H, activations_pth)

    H = H.to(device)
    print(f"Activations: {H.shape}")  # (N, C, W, Hh)

    # Compute correlations over all spatial shifts
    results = compute_all_correlations(H, save_dir / 'correlations.pt', max_shift=max_shift)
    top_pairs = get_top_pairs(results, top_k=10)

    # Extract filters that produce the target layer's output
    target = getattr(model, layer_name)
    if hasattr(target, 'weight'):
        filters = target.weight.data.clone().cpu()           # conv1: (64, 3, 7, 7)
    else:
        filters = target[-1].conv2.weight.data.clone().cpu() # layer1/2: last conv in last block

    plot_correlated_filters(filters, top_pairs, save_dir / 'top_pairs.png')
