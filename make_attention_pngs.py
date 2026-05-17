"""Render cross-attention + decoder-self-attention PNGs from the trained model.

Refreshes the figures referenced by README.md after a retrain.

    python make_attention_pngs.py
    python make_attention_pngs.py --dark
    python make_attention_pngs.py --both
    python make_attention_pngs.py --sentence "Custom sentence to visualize."
"""
from __future__ import annotations

import argparse

import torch

from attention_heatmap import plot_attention_grid
from config import get_best_weights_path, get_config
from decode import build_inputs, detok, greedy_decode, load_inference_model


DEFAULT_SENTENCE = 'Artificial intelligence is changing the world.'


def _readable(tok, idx, specials):
    if idx in specials:
        return specials[idx]
    s = tok.decode([idx])
    return s if s.strip() else (tok.id_to_token(idx) or '?')


def _specials(tok):
    return {
        tok.token_to_id('[SOS]'): '[SOS]',
        tok.token_to_id('[EOS]'): '[EOS]',
        tok.token_to_id('[PAD]'): '',
    }


def _stack(kind: str, layers):
    """Return ``{layer_idx: ndarray (heads, T_out, T_in)}`` for the chosen attn family."""
    out = {}
    for li, layer in enumerate(layers):
        block = (layer.cross_attention_block if kind == 'cross'
                 else layer.self_attention_block)
        out[li] = block.attention_scores[0].detach().float().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dark', action='store_true', help='Dark theme only')
    ap.add_argument('--both', action='store_true', help='Emit both light and dark')
    ap.add_argument('--sentence', default=DEFAULT_SENTENCE,
                    help='Source sentence to visualize')
    args = ap.parse_args()

    cfg = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tok_src, tok_tgt, _ = load_inference_model(cfg, get_best_weights_path(cfg), device)

    src, src_mask, src_ids = build_inputs(args.sentence, tok_src, cfg['max_seq_len'], device)
    with torch.no_grad():
        out_ids = greedy_decode(model, src, src_mask, tok_src, tok_tgt,
                                cfg['max_seq_len'], device).tolist()

    src_labels = [_readable(tok_src, i, _specials(tok_src)) for i in src_ids]
    tgt_specials = _specials(tok_tgt)
    tgt_labels = [_readable(tok_tgt, i, tgt_specials) for i in out_ids][:len(src_labels) + 4]

    print(f'Source : {args.sentence}')
    print(f'Hindi  : {detok(out_ids, tok_tgt)}')

    cross = _stack('cross', model.decoder.layers)
    dec_self = _stack('self', model.decoder.layers)
    last = cfg['n_layers'] - 1
    themes = [False, True] if args.both else [args.dark]

    for dark in themes:
        suffix = '_dark' if dark else ''
        plot_attention_grid(
            {0: cross[0], last: cross[last]},
            src_tokens=src_labels, tgt_tokens=tgt_labels,
            title=f'Cross-attention — "{args.sentence}"',
            out_path=f'results/visualizations/encoder-decoder{suffix}.png',
            dark=dark,
        )
        plot_attention_grid(
            {0: dec_self[0], last: dec_self[last]},
            src_tokens=tgt_labels, tgt_tokens=tgt_labels,
            title='Decoder self-attention (causal)',
            out_path=f'results/visualizations/decoder{suffix}.png',
            dark=dark,
        )


if __name__ == '__main__':
    main()
