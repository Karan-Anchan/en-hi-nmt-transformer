"""Render high-quality attention heatmaps as PNGs.

Aesthetic matches ``plot_metrics.py``: minimalist-luxury via ``theme.py``.
Both light and dark themes are supported via the ``dark`` flag on the
plotting functions; ``make_attention_pngs.py`` exposes this on the CLI.

The figures show how each generated Hindi token (rows) attends to source
English tokens (columns). Brighter cells mean higher attention weight.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from theme import attention_cmap, palette

# Devanagari-capable font fallback so Hindi tokens render on the axes.
mpl.rcParams['font.family'] = ['Inter', 'Nirmala UI', 'Mangal',
                               'Noto Sans Devanagari', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False


def _style_axis(ax, p, *, x_labels, y_labels, show_y=True, show_x=True):
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    if show_x:
        ax.set_xticklabels(x_labels, rotation=60, fontsize=7,
                           ha='right', color=p['soft_ink'])
    else:
        ax.set_xticklabels([])
    if show_y:
        ax.set_yticklabels(y_labels, fontsize=7, color=p['soft_ink'])
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_color(p['hairline'])
        spine.set_linewidth(0.6)


def plot_attention_grid(attn_layers: dict[int, np.ndarray],
                        src_tokens: list[str], tgt_tokens: list[str],
                        title: str, out_path: str | None = None,
                        dark: bool = False) -> None:
    """One subplot per (layer, head). ``attn_layers[layer]`` is ``(heads, tgt, src)``."""
    p = palette(dark)
    cmap = attention_cmap(dark)
    layers = sorted(attn_layers.keys())
    n_heads = next(iter(attn_layers.values())).shape[0]

    cell_w, cell_h = 2.2, 2.4
    fig, axes = plt.subplots(
        len(layers), n_heads,
        figsize=(cell_w * n_heads + 1.0, cell_h * len(layers) + 1.3),
        dpi=130, squeeze=False, facecolor=p['canvas'],
        gridspec_kw={'hspace': 0.45, 'wspace': 0.35},
    )
    fig.patch.set_facecolor(p['canvas'])

    for ri, layer in enumerate(layers):
        attn = attn_layers[layer]
        for ci in range(n_heads):
            ax = axes[ri][ci]
            ax.set_facecolor(p['canvas'])
            data = attn[ci, :len(tgt_tokens), :len(src_tokens)]
            ax.imshow(data, cmap=cmap, aspect='auto',
                      vmin=0.0, vmax=max(0.4, float(data.max())),
                      interpolation='nearest')

            show_y = (ci == 0)
            show_x = (ri == len(layers) - 1)
            _style_axis(ax, p, x_labels=src_tokens, y_labels=tgt_tokens,
                        show_y=show_y, show_x=show_x)

            if ri == 0:
                ax.set_title(f'Head {ci}', fontsize=9.5, color=p['ink'],
                             fontweight='medium', pad=10, loc='center')
            if ci == 0:
                ax.text(-0.32, 0.5, f'Layer {layer}',
                        transform=ax.transAxes,
                        rotation=0, fontsize=10, color=p['ink'],
                        fontweight='medium', va='center', ha='right')

    fig.suptitle(title, fontsize=12.5, color=p['ink'], x=0.02, y=0.985,
                 ha='left', fontweight='medium')

    # Slim attention ramp legend, top-right of the figure (clear of axes)
    cbar_ax = fig.add_axes([0.78, 0.97, 0.16, 0.012])
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    cbar_ax.imshow(grad, cmap=cmap, aspect='auto')
    cbar_ax.set_xticks([]); cbar_ax.set_yticks([])
    for s in cbar_ax.spines.values(): s.set_visible(False)
    fig.text(0.78, 0.955, 'low', fontsize=8, color=p['soft_ink'], ha='left')
    fig.text(0.94, 0.955, 'high', fontsize=8, color=p['soft_ink'], ha='right')
    fig.text(0.86, 0.99, 'attention', fontsize=8, color=p['soft_ink'],
             ha='center', style='italic')

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches='tight', facecolor=p['canvas'])
        print(f'Saved {out_path}')
    plt.close(fig)


def plot_attention_single(attn: np.ndarray,
                          src_tokens: list[str], tgt_tokens: list[str],
                          title: str = 'Attention', out_path: str | None = None,
                          dark: bool = False) -> None:
    """Single (tgt, src) attention map — used in the Gradio app."""
    p = palette(dark)
    cmap = attention_cmap(dark)
    data = attn[:len(tgt_tokens), :len(src_tokens)]
    fig_w = max(7, 0.45 * len(src_tokens) + 2.5)
    fig_h = max(4.5, 0.45 * len(tgt_tokens) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140, facecolor=p['canvas'])
    fig.patch.set_facecolor(p['canvas'])
    ax.set_facecolor(p['canvas'])
    im = ax.imshow(data, cmap=cmap, aspect='auto',
                   vmin=0.0, vmax=max(0.4, float(data.max())),
                   interpolation='nearest')
    _style_axis(ax, p, x_labels=src_tokens, y_labels=tgt_tokens)
    ax.set_title(title, fontsize=12, color=p['ink'], fontweight='medium',
                 loc='left', pad=18)

    cb = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02, aspect=18)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, color=p['hairline'],
                      labelcolor=p['soft_ink'], length=0)
    cb.set_label('attention weight', color=p['soft_ink'], fontsize=9, labelpad=8)

    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches='tight', facecolor=p['canvas'])
        print(f'Saved {out_path}')
    plt.close(fig)
