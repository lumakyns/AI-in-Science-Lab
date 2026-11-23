"""Visualize saved CNN2 weights without training or contacting W&B."""
import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


def render_frame(maps, epoch, limit):
    fig, axes = plt.subplots(1, len(maps), figsize=(6 * len(maps), 5),
                             squeeze=False, layout='constrained')
    for ax, (label, values) in zip(axes.flat, maps):
        plot = ax.imshow(values, cmap='RdBu_r', vmin=-limit, vmax=limit,
                         interpolation='nearest')
        ax.set(title=f'{label} | epoch {epoch}', xlabel='Kernel x', ylabel='Kernel y')
    fig.suptitle('20×20 convolution: mean across input and output channels')
    fig.colorbar(plot, ax=list(axes.flat), label='Mean signed weight')
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120)
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.convert('RGB')


def render_grid(values, titles, title, path):
    columns = 4 if len(values) <= 16 else 8
    rows = (len(values) + columns - 1) // columns
    limit = max(float(np.abs(values).max()), np.finfo(np.float32).eps)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.5, rows * 2.5),
                             squeeze=False, layout='constrained')
    for ax, value, caption in zip(axes.flat, values, titles):
        plot = ax.imshow(value, cmap='RdBu_r', vmin=-limit, vmax=limit,
                         interpolation='nearest')
        ax.set_title(caption, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in list(axes.flat)[len(values):]:
        ax.set_visible(False)
    fig.suptitle(title)
    fig.colorbar(plot, ax=list(axes.flat), label='Signed weight; shared scale')
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return limit


def generate(run_dirs, output, seed=0, samples=16):
    output = output.resolve()
    output.relative_to(Path('checkpoints/cnn2').resolve())
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    records = []
    for directory in run_dirs:
        checkpoint_path = directory / 'checkpoints/last.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        label = f"{checkpoint['config']['dataset']} / seed {checkpoint['config']['seed']}"
        layers = {key.removesuffix('.weight'): value for key, value in checkpoint['model'].items()
                  if key.endswith('.weight') and value.ndim == 4 and tuple(value.shape[-2:]) == (20, 20)}
        if not layers:
            raise ValueError(f'No 20×20 convolution in {checkpoint_path}')
        for layer, weight in layers.items():
            paths = sorted((directory / 'images').glob(f'{layer}_epoch_*.npy'))
            epochs = [int(path.stem.rsplit('_', 1)[1]) for path in paths]
            if epochs != list(range(checkpoint['epoch'] + 1)):
                raise ValueError(f'Missing or unexpected epoch maps for {directory}: {epochs}')
            maps = [np.load(path, allow_pickle=False) for path in paths]
            if any(values.shape != (20, 20) or not np.isfinite(values).all() for values in maps):
                raise ValueError(f'Invalid saved maps in {directory}')
            weights = weight.float().numpy()
            if not np.isfinite(weights).all():
                raise ValueError(f'Nonfinite weights in {checkpoint_path}')
            np.testing.assert_allclose(maps[-1], weight.float().mean(dim=(0, 1)).numpy(),
                                       rtol=1e-5, atol=1e-8)
            slug = f"{checkpoint['config']['dataset']}_seed{checkpoint['config']['seed']}_{layer}"
            rng = np.random.default_rng(seed)
            indices = rng.choice(weights.shape[0] * weights.shape[1],
                                 size=min(samples, weights.shape[0] * weights.shape[1]), replace=False)
            pairs = [(int(i // weights.shape[1]), int(i % weights.shape[1])) for i in indices]
            sampled = np.stack([weights[out_channel, in_channel] for out_channel, in_channel in pairs])
            sample_path = output / f'{slug}_final_samples.png'
            sample_limit = render_grid(sampled, [f'out {o} ← in {i}' for o, i in pairs],
                                      f'{label} | {layer} | epoch {checkpoint["epoch"]}\n'
                                      f'Random individual 20×20 channel kernels (sample seed {seed})', sample_path)
            means_path = output / f'{slug}_final_output_means.png'
            render_grid(weights.mean(axis=1), [f'out {o}: mean over inputs' for o in range(len(weights))],
                        f'{label} | {layer} | epoch {checkpoint["epoch"]}\n'
                        'Each output filter averaged over its input channels', means_path)
            np.save(output / f'{slug}_epoch_means.npy', np.stack(maps))
            np.save(output / f'{slug}_final_samples.npy', sampled)
            records.append(dict(label=label, layer=layer, source=str(directory.resolve()),
                                checkpoint=str(checkpoint_path.resolve()), epochs=epochs,
                                shape=list(weights.shape), sample_pairs=pairs,
                                sample_color_limit=sample_limit, samples=sample_path.name,
                                output_means=means_path.name, maps=maps))
    if len({tuple(record['epochs']) for record in records}) != 1:
        raise ValueError('Runs must have matching epochs for a synchronized timelapse')
    limit = max(float(np.abs(values).max()) for record in records for values in record['maps'])
    limit = max(limit, np.finfo(np.float32).eps)
    frames_dir = output / 'frames'
    frames_dir.mkdir()
    frames = []
    for index, epoch in enumerate(records[0]['epochs']):
        frame = render_frame([(f"{r['label']} / {r['layer']}", r['maps'][index]) for r in records], epoch, limit)
        frame.save(frames_dir / f'epoch_{epoch:03d}.png')
        frames.append(frame)
    frames[0].save(output / 'mean_weights_timelapse.gif', save_all=True,
                   append_images=frames[1:], duration=[700] * (len(frames) - 1) + [2000], loop=0)
    with Image.open(output / 'mean_weights_timelapse.gif') as gif:
        if gif.n_frames != len(frames):
            raise ValueError('GIF frame count does not match saved epochs')
    metadata = dict(sample_seed=seed, timelapse_color_limit=limit,
                    averaging='Signed arithmetic mean over input and output channels',
                    runs=[{key: value for key, value in record.items() if key != 'maps'} for record in records])
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    sections = ''.join(f'<h2>{r["label"]}: {r["layer"]}</h2>'
                       f'<h3>Final epoch: random individual channel kernels</h3><img src="{r["samples"]}">'
                       f'<h3>Final epoch: each output filter averaged over input channels</h3>'
                       f'<img src="{r["output_means"]}">' for r in records)
    (output / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>CNN2 convolution weights</title>'
        '<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 24px;'
        'color:#182333;background:#f8fafc}img{max-width:100%;background:white}p{max-width:900px}</style>'
        '<h1>CNN2 convolution weights</h1><p>Existing checkpoints only. Epoch 0 is initialization; '
        'epochs 1–10 follow training. The timelapse averages all input and output channels into one '
        '20×20 map per layer. Its symmetric scale is fixed across both runs and all epochs, with no clipping. '
        'Blue is negative, white is zero, red is positive. Individual kernels use their own shared grid scale. '
        'Channel indices are zero-based.</p><img src="mean_weights_timelapse.gif">' + sections)
    print(json.dumps(dict(output=str(output), frames=len(frames), runs=len(records), verified=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', help='Run directories containing checkpoints/ and images/')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sample-seed', type=int, default=0)
    parser.add_argument('--samples', type=int, default=16)
    args = parser.parse_args()
    if args.samples < 1 or args.sample_seed < 0:
        parser.error('samples must be positive and sample-seed nonnegative')
    runs = args.runs or sorted(path.parent.parent for path in Path('checkpoints/cnn2/runs').glob('*/*/checkpoints/last.pt'))
    if not runs:
        parser.error('No existing checkpoints found')
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    generate(runs, args.output or Path('checkpoints/cnn2/visualizations') / timestamp,
             args.sample_seed, args.samples)


if __name__ == '__main__':
    main()
