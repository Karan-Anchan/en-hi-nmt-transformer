"""Bilingual dataset for the IIT Bombay En-Hi corpus.

This module is intentionally small: it tokenizes a pre-loaded sentence pair,
adds [SOS]/[EOS] and pads to a fixed length, and returns the masks the
Transformer expects.

Sentences that exceed ``seq_len`` (in tokens, after BPE) are silently
truncated so that we never raise mid-batch.
"""
import torch
from torch.utils.data import Dataset
from typing import Any


def causal_mask(size: int) -> torch.Tensor:
    """Lower-triangular boolean mask: position i can attend to positions <= i."""
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id('[SOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id('[PAD]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id('[EOS]')], dtype=torch.int64)
        # source uses its own PAD id (same special-tokens layout, but separate vocabs)
        self.src_pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        pair = self.ds[index]
        src_text = pair['translation'][self.src_lang]
        tgt_text = pair['translation'][self.tgt_lang]

        # Truncate so SOS + tokens + EOS fits in seq_len. -2 leaves room for both.
        enc_ids = self.tokenizer_src.encode(src_text).ids[: self.seq_len - 2]
        dec_ids = self.tokenizer_tgt.encode(tgt_text).ids[: self.seq_len - 2]

        enc_pad = self.seq_len - len(enc_ids) - 2
        dec_pad = self.seq_len - len(dec_ids) - 1  # decoder input has SOS + tokens

        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_ids, dtype=torch.int64),
            self.eos_token,
            self.src_pad_token.repeat(enc_pad),
        ])
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_ids, dtype=torch.int64),
            self.pad_token.repeat(dec_pad),
        ])
        label = torch.cat([
            torch.tensor(dec_ids, dtype=torch.int64),
            self.eos_token,
            self.pad_token.repeat(dec_pad),
        ])

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.src_pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
