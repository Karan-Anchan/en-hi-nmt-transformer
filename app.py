"""Gradio demo for the English → Hindi Transformer.

Run:
    python app.py
and open http://127.0.0.1:7860 .

Sliders control beam size + length penalty; the cross-attention heatmap
(final decoder layer, averaged over heads) is rendered with the same
minimalist palette as the README figures. Pass ``--dark`` for the dark
theme heatmap.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import torch

from attention_heatmap import plot_attention_single
from config import get_best_weights_path, get_config
from decode import (beam_search_decode, build_inputs, detok, greedy_decode,
                    load_inference_model)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIG = get_config()
MODEL = None
TOK_SRC = None
TOK_TGT = None
DARK = False


def _ensure_model_loaded(ckpt_path: str):
    global MODEL, TOK_SRC, TOK_TGT
    MODEL, TOK_SRC, TOK_TGT, _ = load_inference_model(CONFIG, ckpt_path, DEVICE)
    print(f'Loaded {ckpt_path} on {DEVICE}')


def _readable(tok, idx):
    sos = tok.token_to_id('[SOS]')
    eos = tok.token_to_id('[EOS]')
    pad = tok.token_to_id('[PAD]')
    if idx == sos: return '[SOS]'
    if idx == eos: return '[EOS]'
    if idx == pad: return ''
    s = tok.decode([idx])
    return s if s.strip() else (tok.id_to_token(idx) or '?')


@torch.no_grad()
def translate(text: str, beam_size: int, length_penalty: float):
    if MODEL is None:
        return 'Model is not loaded — check the console for the checkpoint path.', None, ''
    text = (text or '').strip()
    if not text:
        return '', None, ''

    src, src_mask, src_ids = build_inputs(text, TOK_SRC, CONFIG['max_seq_len'], DEVICE)
    if beam_size <= 1:
        out = greedy_decode(MODEL, src, src_mask, TOK_SRC, TOK_TGT,
                            CONFIG['max_seq_len'], DEVICE)
    else:
        out = beam_search_decode(MODEL, src, src_mask, TOK_SRC, TOK_TGT,
                                 CONFIG['max_seq_len'], DEVICE,
                                 beam_size=beam_size, length_penalty=length_penalty)
    out_ids = out.tolist()
    hindi = detok(out_ids, TOK_TGT)

    # Head-averaged cross-attention from the final decoder layer
    attn = MODEL.decoder.layers[-1].cross_attention_block.attention_scores
    attn = attn[0].mean(dim=0).detach().float().cpu().numpy()

    src_labels = [_readable(TOK_SRC, i) for i in src_ids]
    tgt_labels = [_readable(TOK_TGT, i) for i in out_ids if i != TOK_TGT.token_to_id('[PAD]')]

    import matplotlib.pyplot as plt
    plt.close('all')
    plot_attention_single(attn, src_labels, tgt_labels,
                          title='Cross-attention — final decoder layer (head-avg)',
                          out_path=None, dark=DARK)
    fig = plt.gcf()

    info = (f'src tokens: {len(src_labels)} · tgt tokens: {len(tgt_labels)} · '
            f'beam={beam_size} · α={length_penalty}')
    return hindi, fig, info


EXAMPLES = [
    ['Good morning, how are you today?', 4, 0.6],
    ['Artificial intelligence is changing the world.', 4, 0.6],
    ['She is reading a book in the library.', 4, 0.6],
    ['India won the cricket match yesterday.', 4, 0.6],
    ['The government should think about the future of the students.', 4, 0.6],
]


def build_ui():
    with gr.Blocks(title='English → Hindi Transformer', theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            '# English → Hindi Neural Machine Translation\n'
            'A 6-layer Transformer trained from scratch in PyTorch on '
            '[AI4Bharat Samanantar](https://huggingface.co/datasets/ai4bharat/samanantar). '
            'Try a sentence — the cross-attention map shows which source tokens the '
            'model looked at while generating each Hindi token.'
        )
        with gr.Row():
            with gr.Column(scale=1):
                src = gr.Textbox(label='English', lines=3,
                                 placeholder='Type something to translate…')
                beam = gr.Slider(1, 8, value=4, step=1, label='Beam size (1 = greedy)')
                alpha = gr.Slider(0.0, 1.5, value=0.6, step=0.05, label='Length penalty α')
                btn = gr.Button('Translate', variant='primary')
            with gr.Column(scale=1):
                tgt = gr.Textbox(label='Hindi', lines=3)
                info = gr.Markdown('')
        attn = gr.Plot(label='Cross-attention heatmap')
        gr.Examples(EXAMPLES, inputs=[src, beam, alpha], label='Examples')
        btn.click(translate, [src, beam, alpha], [tgt, attn, info])
        src.submit(translate, [src, beam, alpha], [tgt, attn, info])
    return demo


def main():
    global DARK
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default=get_best_weights_path(CONFIG))
    p.add_argument('--share', action='store_true')
    p.add_argument('--dark', action='store_true', help='Dark-theme attention heatmap')
    args = p.parse_args()
    DARK = args.dark
    if not Path(args.ckpt).exists():
        raise SystemExit(f'No checkpoint at {args.ckpt} — run train.py first.')
    _ensure_model_loaded(args.ckpt)
    build_ui().launch(share=args.share, inbrowser=True)


if __name__ == '__main__':
    main()
