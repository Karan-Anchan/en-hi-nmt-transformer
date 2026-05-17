"""Transformer architecture from *Attention Is All You Need* (Vaswani 2017).

Each block is a small, self-contained ``nn.Module``. Read top-to-bottom for
a faithful walk-through of the paper:

    InputEmbeddings           — token-id → ``d_model`` vector, scaled by √d_model
    PositionalEncoding        — fixed sin/cos position vectors added to the input
    LayerNormalization        — gain/bias LN; pre-norm is applied inside ResidualConnection
    FeedForwardBlock          — two-layer position-wise FFN with ReLU
    MultiHeadAttention        — scaled dot-product attention, batched over h heads
    ResidualConnection        — x + dropout(sublayer(norm(x)))  (pre-norm variant)
    EncoderBlock / Encoder    — N stacked encoder blocks + final LN
    DecoderBlock / Decoder    — same, plus a cross-attention sublayer per block
    ProjectionLayer           — final linear to vocab logits
    Transformer               — assembles encoder, decoder, embeddings, projection
    build_transformer         — factory with Xavier init
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):
    """Token-id → ``d_model`` vector, scaled by ``√d_model`` (paper §3.4)."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Add fixed sin/cos positional vectors to the embeddings (paper §3.5).

    Pre-computed once into a non-trainable buffer; sliced at forward time so
    sequences shorter than ``seq_len`` cost nothing extra.
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    """LayerNorm with learnable gain/bias. Identical to ``nn.LayerNorm`` but
    spelt out so the paper-to-code mapping is obvious."""

    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """Position-wise FFN: linear → ReLU → dropout → linear (paper §3.3)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention, batched over ``h`` parallel heads (paper §3.2.2).

    Stores the last-computed attention matrix on ``self.attention_scores`` so
    visualization code (the heatmaps in ``attention_heatmap.py``) can pull
    it without re-running the forward pass.
    """

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        assert d_model % h == 0, 'd_model must be divisible by the number of heads'
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attention_scores: torch.Tensor | None = None

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout | None):
        d_k = query.shape[-1]
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e4)
        weights = scores.softmax(dim=-1)
        if dropout is not None:
            weights = dropout(weights)
        return weights @ value, weights

    def forward(self, q, k, v, mask):
        # Project then split into heads: (B, T, d_model) → (B, h, T, d_k)
        def split(x, lin):
            x = lin(x)
            return x.view(x.shape[0], x.shape[1], self.h, self.d_k).transpose(1, 2)

        query = split(q, self.w_q)
        key = split(k, self.w_k)
        value = split(v, self.w_v)
        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)
        # Merge heads back: (B, h, T, d_k) → (B, T, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    """Pre-norm residual: ``x + dropout(sublayer(LN(x)))``.

    Pre-norm trains more stably than post-norm for deep Transformers
    (Xiong et al., 2020) and removes the need for an LR warmup spike
    relative to the original paper.
    """

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """Self-attention → FFN, each wrapped in a residual connection."""

    def __init__(self, features: int, self_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    """Stack of ``N`` encoder blocks + final LayerNorm."""

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    """Self-attention → cross-attention → FFN. The cross-attention is what
    lets the decoder peek at the encoded source sentence."""

    def __init__(self, features: int, self_attention_block: MultiHeadAttention,
                 cross_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    """Stack of ``N`` decoder blocks + final LayerNorm."""

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    """Final linear from ``d_model`` → ``vocab_size`` (returns logits, not softmax)."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.proj(x)


class Transformer(nn.Module):
    """Full encoder–decoder Transformer.

    ``encode`` / ``decode`` / ``project`` are exposed separately so that
    inference loops (greedy + beam search) can call the encoder once and
    reuse its output across decoder steps.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        return self.encoder(self.src_pos(self.src_embed(src)), src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor,
               tgt: torch.Tensor, tgt_mask: torch.Tensor):
        return self.decoder(self.tgt_pos(self.tgt_embed(tgt)),
                            encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int,
                      src_seq_len: int, tgt_seq_len: int,
                      d_model: int = 512, N: int = 6, h: int = 8,
                      dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    """Factory: build a Transformer with Xavier-initialized weights."""
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_blocks = nn.ModuleList([
        EncoderBlock(d_model,
                     MultiHeadAttention(d_model, h, dropout),
                     FeedForwardBlock(d_model, d_ff, dropout),
                     dropout)
        for _ in range(N)
    ])
    decoder_blocks = nn.ModuleList([
        DecoderBlock(d_model,
                     MultiHeadAttention(d_model, h, dropout),  # self
                     MultiHeadAttention(d_model, h, dropout),  # cross
                     FeedForwardBlock(d_model, d_ff, dropout),
                     dropout)
        for _ in range(N)
    ])

    model = Transformer(
        Encoder(d_model, encoder_blocks),
        Decoder(d_model, decoder_blocks),
        src_embed, tgt_embed,
        src_pos, tgt_pos,
        ProjectionLayer(d_model, tgt_vocab_size),
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
