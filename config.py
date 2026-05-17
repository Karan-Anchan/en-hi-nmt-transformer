"""Hyperparameters and file paths.

A single source of truth for the training/inference pipeline. Edit the dict
returned by ``get_config`` to change any aspect of the run.
"""
from pathlib import Path


def get_config():
    return {
        # --- Data ---
        # Samanantar (AI4Bharat) — 10M en-hi pairs, much cleaner and more
        # natural-English-heavy than the older IIT-B corpus.
        "dataset_name": "ai4bharat/samanantar",
        "dataset_config": "hi",
        "src_field": "src",          # Samanantar schema: {idx, src, tgt}
        "tgt_field": "tgt",
        "lang_src": "en",
        "lang_tgt": "hi",
        "max_seq_len": 128,          # BPE tokens; Samanantar p95 source ≈ 42 words
        "max_train_examples": 500_000,  # subset of the full 10M for ~3h training
        "split_seed": 42,
        "val_fraction": 0.005,       # 0.5% val (~5k pairs)
        "test_fraction": 0.005,      # 0.5% held-out test (~5k pairs)

        # --- Tokenizer (byte-level BPE) ---
        "vocab_size_src": 16000,
        "vocab_size_tgt": 16000,
        "tokenizer_file": "tokenizer_{0}.json",

        # --- Model ---
        # d_model=384/n_heads=6 keeps step time ~5/s on RTX 5070 while still
        # giving Transformer-Base style depth. Bumping to d_model=512 ~doubled
        # training time without a clear quality bump on this size of corpus.
        "d_model": 384,
        "n_layers": 6,
        "n_heads": 6,
        "d_ff": 1536,
        "dropout": 0.1,

        # --- Training ---
        "batch_size": 48,            # comfortably fits at seq_len=128 in fp16
        "num_epochs": 8,
        "warmup_steps": 4000,
        "label_smoothing": 0.1,
        "grad_clip": 1.0,
        "amp": True,
        "log_every_steps": 200,
        "val_every_steps": 4000,
        "val_max_examples": 400,

        # --- Decoding ---
        "beam_size": 4,
        "length_penalty": 0.6,

        # --- Paths ---
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "experiment_name": "runs/tmodel",
        "results_dir": "results",
    }


def get_weights_file_path(config, epoch: str) -> str:
    return str(Path('.') / config['model_folder'] / f"{config['model_basename']}{epoch}.pt")


def get_best_weights_path(config) -> str:
    return get_weights_file_path(config, "best")
