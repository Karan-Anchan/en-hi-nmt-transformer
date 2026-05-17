"""Decoders and inference helpers for the Transformer.

* ``greedy_decode`` / ``beam_search_decode`` — token-id generators with a
  shared calling convention so they're drop-in swappable.
* ``translate_text`` — end-to-end raw string → string.
* ``build_inputs`` / ``detok`` — tokenize-then-pad and detokenize-then-strip
  used by ``eval.py``, ``app.py``, and ``translate.py`` (kept in one place
  so behaviour stays consistent).
* ``load_inference_model`` — checkpoint → ready-to-use model + tokenizers
  triple. Encapsulates the boilerplate that every entry point needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from tokenizers import Tokenizer

from dataset import causal_mask


# --------------------------- inference helpers ---------------------------

def build_inputs(text: str, tok_src: Tokenizer, max_len: int, device) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Return ``(src_tensor, src_mask, raw_ids_with_sos_eos)`` for one sentence."""
    sos = tok_src.token_to_id('[SOS]')
    eos = tok_src.token_to_id('[EOS]')
    pad = tok_src.token_to_id('[PAD]')
    ids = tok_src.encode(text).ids[: max_len - 2]
    tokens = [sos] + ids + [eos] + [pad] * (max_len - len(ids) - 2)
    src = torch.tensor(tokens, dtype=torch.int64, device=device).unsqueeze(0)
    src_mask = (src != pad).unsqueeze(0).unsqueeze(0).int()
    return src, src_mask, [sos] + ids + [eos]


def detok(ids: Iterable[int], tok_tgt: Tokenizer) -> str:
    """Drop SOS/EOS/PAD specials and decode."""
    sos = tok_tgt.token_to_id('[SOS]')
    eos = tok_tgt.token_to_id('[EOS]')
    pad = tok_tgt.token_to_id('[PAD]')
    return tok_tgt.decode([t for t in ids if t not in (sos, eos, pad)])


def load_inference_model(config: dict, ckpt_path: str, device: torch.device):
    """Return ``(model, tok_src, tok_tgt, state)`` ready for inference."""
    from model import build_transformer  # local import to keep eval/app startup snappy

    tok_src = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_src']))
    tok_tgt = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_tgt']))
    model = build_transformer(
        tok_src.get_vocab_size(), tok_tgt.get_vocab_size(),
        config['max_seq_len'], config['max_seq_len'],
        d_model=config['d_model'], N=config['n_layers'], h=config['n_heads'],
        dropout=config['dropout'], d_ff=config['d_ff'],
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model, tok_src, tok_tgt, state


# ------------------------------- decoders -------------------------------

@torch.no_grad()
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.full((1, 1), sos_idx, dtype=source.dtype, device=device)

    for _ in range(max_len - 1):
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        logits = model.project(out[:, -1])
        next_token = torch.argmax(logits, dim=-1)
        decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)
        if next_token.item() == eos_idx:
            break
    return decoder_input.squeeze(0)


@torch.no_grad()
def beam_search_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt,
                       max_len, device, beam_size: int = 4, length_penalty: float = 0.6):
    """Length-normalized beam search.

    Each step expands every live beam by its top ``beam_size`` next tokens,
    then keeps the global top ``beam_size``. Finished hypotheses (those that
    emitted ``[EOS]``) get the GNMT length penalty ``((5 + len) / 6) ** α``
    and join a separate pool; the best-scoring finished hypothesis wins. If
    nothing terminates within ``max_len`` we fall back to the best live beam
    after the same length normalization.
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    encoder_output = model.encode(source, source_mask)

    beams = [(torch.tensor([[sos_idx]], dtype=source.dtype, device=device), 0.0)]
    finished: list[tuple[torch.Tensor, float]] = []

    for _ in range(max_len - 1):
        candidates: list[tuple[torch.Tensor, float]] = []
        for seq, score in beams:
            dec_mask = causal_mask(seq.size(1)).type_as(source_mask).to(device)
            out = model.decode(encoder_output, source_mask, seq, dec_mask)
            log_probs = torch.log_softmax(model.project(out[:, -1]), dim=-1).squeeze(0)
            top_lp, top_ids = torch.topk(log_probs, beam_size)
            for lp, tid in zip(top_lp.tolist(), top_ids.tolist()):
                new_seq = torch.cat([seq, torch.tensor([[tid]], dtype=seq.dtype, device=device)], dim=1)
                candidates.append((new_seq, score + lp))

        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = []
        for seq, score in candidates:
            if seq[0, -1].item() == eos_idx:
                length = seq.size(1) - 1
                finished.append((seq, score / (((5.0 + length) / 6.0) ** length_penalty)))
            else:
                beams.append((seq, score))
            if len(beams) >= beam_size:
                break
        if not beams:
            break

    if not finished:
        finished = [(seq, score / (((5.0 + max(1, seq.size(1) - 1)) / 6.0) ** length_penalty))
                    for seq, score in beams]
    finished.sort(key=lambda x: x[1], reverse=True)
    return finished[0][0].squeeze(0)


def translate_text(model, text: str, tok_src: Tokenizer, tok_tgt: Tokenizer, device,
                   max_len: int = 96, beam_size: int = 4, length_penalty: float = 0.6) -> str:
    """End-to-end: raw English string → Hindi string."""
    src, src_mask, _ = build_inputs(text, tok_src, max_len, device)
    model.eval()
    if beam_size <= 1:
        ids = greedy_decode(model, src, src_mask, tok_src, tok_tgt, max_len, device)
    else:
        ids = beam_search_decode(model, src, src_mask, tok_src, tok_tgt, max_len, device,
                                 beam_size=beam_size, length_penalty=length_penalty)
    return detok(ids.tolist(), tok_tgt)
