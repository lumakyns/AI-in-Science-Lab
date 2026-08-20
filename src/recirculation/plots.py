"""Export run figures without requiring a graphical display."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_plots(history, originals, reconstructions, *, directory, title):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    epochs = [row['epoch'] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for key, label in [('train/epoch_loss', 'Training'), ('validation/loss', 'Validation')]:
        axes[0].plot(epochs, [row[key] for row in history], label=label)
    axes[0].set(title='Reconstruction objective', xlabel='Epoch', ylabel='Loss (sum per example)')
    axes[0].legend()
    axes[1].plot(epochs, [row['validation/mse'] for row in history], color='tab:green')
    axes[1].set(title='Validation reconstruction error', xlabel='Epoch', ylabel='MSE per coordinate')
    for ax in axes:
        ax.grid(alpha=.25)
    fig.suptitle(title)
    curves = directory / 'learning_curves.png'
    fig.savefig(curves, dpi=180)
    plt.close(fig)

    count = len(originals)
    fig, axes = plt.subplots(2, count, figsize=(2.4*count, 5), squeeze=False, constrained_layout=True)
    for column in range(count):
        for row, values in enumerate((originals, reconstructions)):
            ax = axes[row, column]
            sample = values[column]
            ax.imshow(sample[0], cmap='gray', vmin=0, vmax=1)
            ax.set_axis_off()
            ax.set_title(f'{"Original" if row == 0 else "Reconstruction"} {column+1}')
    fig.suptitle(title)
    examples = directory / 'reconstructions.png'
    fig.savefig(examples, dpi=180)
    plt.close(fig)
    return curves, examples
