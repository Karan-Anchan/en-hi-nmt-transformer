"""Render training-loss and validation-metric figures from CSV logs.

Aesthetic via ``theme.py`` (light + dark). Lines are colored by value with
a ``LineCollection`` gradient so peaks light up — same visual language as
the attention heatmaps.

    python plot_metrics.py          # light theme
    python plot_metrics.py --dark   # dark theme
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from theme import apply_rc, metric_cmap


def _format_step(value, _):
    return f'{value/1000:.0f}k' if value >= 1000 else f'{value:.0f}'


def _strip(ax, p, *, x_label: str | None = None, y_label: str | None = None):
    ax.spines['left'].set_color(p['hairline'])
    ax.spines['bottom'].set_color(p['hairline'])
    ax.yaxis.grid(True, color=p['hairline'], linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    if x_label:
        ax.set_xlabel(x_label, fontsize=9.5, color=p['soft_ink'], loc='left')
    if y_label:
        ax.set_ylabel(y_label, fontsize=9.5, color=p['soft_ink'], loc='top',
                      rotation=0, labelpad=-12)


def _gradient_line(ax, x, y, *, cmap, value_range, linewidth=2.4, zorder=3):
    """Draw a line whose color along its length is sampled from ``cmap``.

    ``value_range = (vmin, vmax)`` controls the normalization, so we can
    color by absolute value rather than per-line min/max — useful when
    multiple series share an axis and we want bright = "high score".
    Returns the colour the cmap assigns to the final point so callers
    can match annotation text colour to the line endpoint.
    """
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    vmin, vmax = value_range
    norm = plt.Normalize(vmin, vmax)
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=linewidth,
                        capstyle='round', joinstyle='round', zorder=zorder)
    lc.set_array(np.asarray(y[:-1]))
    ax.add_collection(lc)
    return cmap(norm(y[-1]))


def plot_train_loss(logdir: str, out_path: str, p: dict, cmap):
    df = pd.read_csv(Path(logdir) / 'train_loss.csv')
    fig, ax = plt.subplots(figsize=(10, 4.8))

    # Raw loss whispered underneath
    ax.plot(df['step'], df['loss'], color=p['whisper'], linewidth=1.0, alpha=0.7)

    # EMA trend: gradient coloured by step, so the line warms up as training
    # converges (cool/dark at the start, bright/champagne at the end).
    span = max(1, len(df) // 25)
    ema = df['loss'].ewm(span=span).mean().to_numpy()
    steps = df['step'].to_numpy()
    end_color = _gradient_line(
        ax, steps, ema,
        cmap=cmap.reversed(),  # high loss → cool/dark, low loss → bright
        value_range=(float(ema.min()), float(ema.max())),
        linewidth=2.4,
    )

    ax.annotate(f'loss  {ema[-1]:.2f}', xy=(steps[-1], ema[-1]),
                xytext=(8, 0), textcoords='offset points',
                color=end_color, fontsize=10.5, va='center', fontweight='medium')

    if 'lr' in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df['step'], df['lr'], color=p['whisper'], linewidth=1.0)
        ax2.set_yticks([])
        for s in ax2.spines.values(): s.set_visible(False)
        peak_idx = df['lr'].idxmax()
        ax2.annotate('learning rate (Noam schedule)',
                     xy=(df['step'].iloc[peak_idx], df['lr'].iloc[peak_idx]),
                     xytext=(8, 6), textcoords='offset points',
                     color=p['soft_ink'], fontsize=8.5, style='italic')

    ax.set_title('Training loss')
    _strip(ax, p, x_label='step', y_label='cross-entropy')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_format_step))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax.margins(x=0.02); ax.set_xlim(left=0)
    ax.set_ylim(bottom=ema.min() - 0.2, top=df['loss'].max() + 0.3)
    plt.subplots_adjust(left=0.08, right=0.92, top=0.85, bottom=0.16)
    plt.savefig(out_path)
    plt.close()
    print(f'Wrote {out_path}')


def plot_val_metrics(logdir: str, out_path: str, p: dict, cmap):
    df = pd.read_csv(Path(logdir) / 'val_metrics.csv')
    fig, ax = plt.subplots(figsize=(10, 4.8))

    steps = df['step'].to_numpy()
    chrf = df['chrf'].to_numpy()
    bleu = df['bleu'].to_numpy()

    # Both lines share the same value range so high scores look "hot" and
    # low scores look "cool" across the figure — chrF++ ends up mostly warm,
    # BLEU mostly cool, and the visual rank reflects the actual rank.
    value_range = (0.0, max(chrf.max(), 50.0))

    end_color_chrf = _gradient_line(ax, steps, chrf, cmap=cmap,
                                    value_range=value_range, linewidth=2.6)
    end_color_bleu = _gradient_line(ax, steps, bleu, cmap=cmap,
                                    value_range=value_range, linewidth=2.0)

    # Tiny end-of-line dot for each series so the inline label has an anchor
    ax.scatter([steps[-1]], [chrf[-1]], s=18, color=end_color_chrf, zorder=4)
    ax.scatter([steps[-1]], [bleu[-1]], s=14, color=end_color_bleu, zorder=4)

    ax.annotate(f'chrF++  {chrf[-1]:.1f}',
                xy=(steps[-1], chrf[-1]),
                xytext=(10, 0), textcoords='offset points',
                color=end_color_chrf, fontsize=11, va='center', fontweight='medium')
    ax.annotate(f'BLEU  {bleu[-1]:.1f}',
                xy=(steps[-1], bleu[-1]),
                xytext=(10, 0), textcoords='offset points',
                color=end_color_bleu, fontsize=11, va='center', fontweight='medium')

    # Subtle highlight ring on the best chrF++
    best_idx = int(np.argmax(chrf))
    best_color = cmap(plt.Normalize(*value_range)(chrf[best_idx]))
    ax.scatter([steps[best_idx]], [chrf[best_idx]],
               s=160, facecolor='none', edgecolor=best_color,
               linewidth=1.0, alpha=0.5, zorder=4)

    ax.set_title('Validation quality')
    _strip(ax, p, x_label='step', y_label='score')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_format_step))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax.margins(x=0.02); ax.set_xlim(left=0)
    ax.set_ylim(bottom=-1, top=max(chrf.max(), bleu.max()) + 5)
    plt.subplots_adjust(left=0.08, right=0.88, top=0.85, bottom=0.16)
    plt.savefig(out_path)
    plt.close()
    print(f'Wrote {out_path}')


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--logdir', default='runs/tmodel')
    a.add_argument('--dark', action='store_true')
    a.add_argument('--both', action='store_true', help='Emit light and dark')
    a.add_argument('--loss-out', default=None)
    a.add_argument('--metrics-out', default=None)
    args = a.parse_args()

    themes = [False, True] if args.both else [args.dark]
    for dark in themes:
        p = apply_rc(dark=dark)
        cmap = metric_cmap(dark)
        suffix = '_dark' if dark else ''
        loss_out = args.loss_out if (args.loss_out and not args.both) \
                   else f'beautiful_train_loss{suffix}.png'
        metrics_out = args.metrics_out if (args.metrics_out and not args.both) \
                      else f'beautiful_metrics{suffix}.png'

        if Path(args.logdir, 'train_loss.csv').exists():
            plot_train_loss(args.logdir, loss_out, p, cmap)
        else:
            print(f'No train_loss.csv in {args.logdir} — skipping')
        if Path(args.logdir, 'val_metrics.csv').exists():
            plot_val_metrics(args.logdir, metrics_out, p, cmap)
        else:
            print(f'No val_metrics.csv in {args.logdir} — skipping')


if __name__ == '__main__':
    main()
