"""Train (or load) BPE tokenizers for the source and target language.

We use **byte-level BPE** (GPT-2 / RoBERTa style) — spaces are encoded as
part of the byte sequence with a special `Ġ` prefix on word-initial
tokens, so the encode → decode roundtrip preserves word boundaries
exactly. This was a real bug in the first iteration of this project: with
a Whitespace pre-tokenizer + the default BPEDecoder, translations decoded
without spaces (`मौजूदाछोड़ें` instead of `मौजूदा छोड़ें`), which tanked BLEU
even though the model itself was producing the right tokens.
"""
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
from tokenizers.decoders import ByteLevel as ByteLevelDec

SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]


def _build_bpe() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDec()
    return tokenizer


def get_or_build_tokenizer(config, sentences: Iterable[str], lang: str) -> Tokenizer:
    """Return a tokenizer for ``lang``, training a fresh BPE if none is cached.

    ``sentences`` is consumed only when training; once a tokenizer JSON exists
    on disk, the iterable is ignored.
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if tokenizer_path.exists():
        return Tokenizer.from_file(str(tokenizer_path))

    vocab_size = config['vocab_size_src'] if lang == config['lang_src'] else config['vocab_size_tgt']
    tokenizer = _build_bpe()
    trainer = BpeTrainer(
        special_tokens=SPECIAL_TOKENS,
        vocab_size=vocab_size,
        min_frequency=2,
        initial_alphabet=ByteLevelPre.alphabet(),  # full byte alphabet so no UNKs
        show_progress=True,
    )
    tokenizer.train_from_iterator(sentences, trainer=trainer)
    tokenizer.save(str(tokenizer_path))
    return tokenizer
