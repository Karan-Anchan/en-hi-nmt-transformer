"""Quick command-line translator.

    python translate.py "Good morning, how are you?"
    python translate.py --beam 1 "I love mangoes."
    python translate.py --ckpt weights/tmodel_05.pt "..."
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import get_best_weights_path, get_config
from decode import load_inference_model, translate_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument('text', help='English sentence to translate')
    p.add_argument('--ckpt', default=None)
    p.add_argument('--beam', type=int, default=4)
    p.add_argument('--alpha', type=float, default=0.6)
    args = p.parse_args()

    cfg = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = args.ckpt or get_best_weights_path(cfg)
    if not Path(ckpt).exists():
        raise SystemExit(f'No checkpoint at {ckpt}. Train first.')
    model, tok_src, tok_tgt, _ = load_inference_model(cfg, ckpt, device)
    print(translate_text(model, args.text, tok_src, tok_tgt, device,
                         max_len=cfg['max_seq_len'], beam_size=args.beam,
                         length_penalty=args.alpha))


if __name__ == '__main__':
    main()
