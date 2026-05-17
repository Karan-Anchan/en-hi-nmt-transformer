"""Training entry point for the English → Hindi Transformer.

Highlights of this recipe (vs. a vanilla Adam loop):
  * Byte-level BPE tokenization (spaces survive the encode/decode round-trip).
  * Noam learning-rate schedule (warmup + 1/√step) per the original paper.
  * Mixed-precision (fp16) with grad clipping and label smoothing.
  * Deterministic train/val/test split — the test set is materialized to
    ``runs/tmodel/test.jsonl`` so ``eval.py`` always scores on the same data.
  * Tracks the best checkpoint by validation chrF++.
  * Logs to TensorBoard *and* a flat CSV so plots can be regenerated.
"""
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import sacrebleu
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_best_weights_path, get_config, get_weights_file_path
from dataset import BilingualDataset
from decode import greedy_decode
from model import build_transformer
from tokenizer_train import get_or_build_tokenizer


# ----------------------------- utilities --------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def noam_lr(step: int, d_model: int, warmup_steps: int) -> float:
    """Original Transformer LR schedule: warmup linearly then decay 1/sqrt(step)."""
    step = max(1, step)
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)


def split_dataset(ds, val_fraction: float, test_fraction: float, seed: int):
    """Deterministic 3-way split. Returns (train, val, test) Hugging Face Datasets."""
    n = len(ds)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    return ds.select(train_idx), ds.select(val_idx), ds.select(test_idx)


def _clean_pair(en: str, hi: str) -> bool:
    """Cheap quality filter: drop empties, template placeholders, length-ratio outliers."""
    if not en or not hi:
        return False
    en, hi = en.strip(), hi.strip()
    if not en or not hi:
        return False
    if '_' in en or '_' in hi:  # template placeholders like "Set as _ Desktop"
        return False
    # very-long ratio outliers are usually garbage
    ratio = len(hi) / max(1, len(en))
    if ratio > 4.0 or ratio < 0.25:
        return False
    return True


def _extract_pair(row, config):
    """Return (en, hi) regardless of dataset schema."""
    if 'translation' in row:
        return row['translation'].get(config['lang_src']), row['translation'].get(config['lang_tgt'])
    return row.get(config.get('src_field', 'src')), row.get(config.get('tgt_field', 'tgt'))


def load_and_filter(config) -> Dataset:
    cfg_name = config.get('dataset_config')
    if cfg_name:
        raw = load_dataset(config['dataset_name'], cfg_name, split='train')
    else:
        raw = load_dataset(config['dataset_name'], split='train')
    cleaned = {'translation': []}
    cap = config.get('max_train_examples')
    # If we're capping, sample uniformly from a much larger pool so we filter
    # *first*, then take the first N good pairs — avoids burning the cap on
    # bad rows at the head of the file.
    iterator = raw
    if cap:
        iterator = raw.select(range(min(len(raw), cap * 3)))
    kept = 0
    for row in tqdm(iterator, desc='Filtering corpus'):
        en, hi = _extract_pair(row, config)
        if _clean_pair(en, hi):
            cleaned['translation'].append({config['lang_src']: en.strip(),
                                           config['lang_tgt']: hi.strip()})
            kept += 1
            if cap and kept >= cap:
                break
    print(f"Kept {kept:,} pairs after filtering")
    return Dataset.from_dict(cleaned)


# ----------------------------- validation --------------------------------

@torch.no_grad()
def quick_validate(model, val_loader, tokenizer_src, tokenizer_tgt, max_len, device,
                   writer, global_step, csv_path, num_examples: int = 200):
    model.eval()
    expected, predicted = [], []
    for i, batch in enumerate(val_loader):
        if i >= num_examples:
            break
        out_ids = greedy_decode(
            model, batch['encoder_input'].to(device), batch['encoder_mask'].to(device),
            tokenizer_src, tokenizer_tgt, max_len, device,
        )
        sos = tokenizer_tgt.token_to_id('[SOS]')
        eos = tokenizer_tgt.token_to_id('[EOS]')
        ids = [t for t in out_ids.tolist() if t not in (sos, eos)]
        predicted.append(tokenizer_tgt.decode(ids))
        expected.append(batch['tgt_text'][0])

    bleu = sacrebleu.corpus_bleu(predicted, [expected]).score
    chrf = sacrebleu.corpus_chrf(predicted, [expected]).score

    if writer:
        writer.add_scalar('Val/SacreBLEU', bleu, global_step)
        writer.add_scalar('Val/chrF++', chrf, global_step)
        writer.flush()

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    write_header = not Path(csv_path).exists()
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['step', 'bleu', 'chrf'])
        w.writerow([global_step, f"{bleu:.4f}", f"{chrf:.4f}"])
    return bleu, chrf


# ----------------------------- main loop --------------------------------

def train_model(config):
    set_seed(config['split_seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    Path(config['experiment_name']).mkdir(parents=True, exist_ok=True)

    # ----- data -----
    print('Loading dataset...')
    full_ds = load_and_filter(config)
    train_ds_raw, val_ds_raw, test_ds_raw = split_dataset(
        full_ds, config['val_fraction'], config['test_fraction'], config['split_seed']
    )
    print(f"Train: {len(train_ds_raw):,} | Val: {len(val_ds_raw):,} | Test: {len(test_ds_raw):,}")

    # Persist the test split paths for the eval script — we save indices implicitly
    # by saving the test set itself as JSONL for reproducibility.
    test_jsonl = Path(config['experiment_name']) / 'test.jsonl'
    with open(test_jsonl, 'w', encoding='utf-8') as f:
        for row in test_ds_raw:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'Saved held-out test set to {test_jsonl}')

    # ----- tokenizers -----
    tok_src = get_or_build_tokenizer(
        config, (r['translation']['en'] for r in train_ds_raw), config['lang_src']
    )
    tok_tgt = get_or_build_tokenizer(
        config, (r['translation']['hi'] for r in train_ds_raw), config['lang_tgt']
    )
    print(f"Vocab: en={tok_src.get_vocab_size()} hi={tok_tgt.get_vocab_size()}")

    train_dl = DataLoader(
        BilingualDataset(train_ds_raw, tok_src, tok_tgt,
                         config['lang_src'], config['lang_tgt'], config['max_seq_len']),
        batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True,
    )
    val_dl = DataLoader(
        BilingualDataset(val_ds_raw, tok_src, tok_tgt,
                         config['lang_src'], config['lang_tgt'], config['max_seq_len']),
        batch_size=1, shuffle=False, num_workers=0,
    )

    # ----- model -----
    model = build_transformer(
        tok_src.get_vocab_size(), tok_tgt.get_vocab_size(),
        config['max_seq_len'], config['max_seq_len'],
        d_model=config['d_model'], N=config['n_layers'], h=config['n_heads'],
        dropout=config['dropout'], d_ff=config['d_ff'],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {n_params/1e6:.2f}M')

    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and config['amp']))

    pad_id = tok_tgt.token_to_id('[PAD]')
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=config['label_smoothing']).to(device)

    initial_epoch, global_step = 0, 0
    best_chrf = -1.0

    if config.get('preload'):
        ckpt_path = get_weights_file_path(config, config['preload'])
        if Path(ckpt_path).exists():
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
            initial_epoch = state['epoch'] + 1
            global_step = state['global_step']
            best_chrf = state.get('best_chrf', -1.0)
            print(f'Resumed from {ckpt_path} (epoch {initial_epoch}, step {global_step})')

    train_loss_csv = Path(config['experiment_name']) / 'train_loss.csv'
    metrics_csv = Path(config['experiment_name']) / 'val_metrics.csv'

    def save_ckpt(path, epoch, global_step, best_chrf):
        torch.save({
            'epoch': epoch, 'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_chrf': best_chrf, 'config': config,
        }, path)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        running = 0.0
        last_log_step = global_step
        t0 = time.time()
        pbar = tqdm(train_dl, desc=f'Epoch {epoch:02d}')
        for batch in pbar:
            global_step += 1
            lr = noam_lr(global_step, config['d_model'], config['warmup_steps'])
            for g in optimizer.param_groups:
                g['lr'] = lr

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda' and config['amp'])):
                enc_in = batch['encoder_input'].to(device, non_blocking=True)
                dec_in = batch['decoder_input'].to(device, non_blocking=True)
                enc_mask = batch['encoder_mask'].to(device, non_blocking=True)
                dec_mask = batch['decoder_mask'].to(device, non_blocking=True)
                label = batch['label'].to(device, non_blocking=True)

                enc_out = model.encode(enc_in, enc_mask)
                dec_out = model.decode(enc_out, enc_mask, dec_in, dec_mask)
                logits = model.project(dec_out)
                loss = loss_fn(logits.view(-1, tok_tgt.get_vocab_size()), label.view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.3f}', lr=f'{lr:.2e}')

            if global_step % config['log_every_steps'] == 0:
                avg = running / max(1, (global_step - last_log_step))
                running, last_log_step = 0.0, global_step
                writer.add_scalar('Train/Loss', avg, global_step)
                writer.add_scalar('Train/LR', lr, global_step)
                with open(train_loss_csv, 'a', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    if f.tell() == 0:
                        w.writerow(['step', 'loss', 'lr'])
                    w.writerow([global_step, f'{avg:.6f}', f'{lr:.8f}'])

            if global_step % config['val_every_steps'] == 0:
                bleu, chrf = quick_validate(
                    model, val_dl, tok_src, tok_tgt, config['max_seq_len'], device,
                    writer, global_step, metrics_csv,
                    num_examples=config['val_max_examples'],
                )
                model.train()
                tqdm.write(f'[step {global_step}] val BLEU={bleu:.2f} chrF++={chrf:.2f}')
                if chrf > best_chrf:
                    best_chrf = chrf
                    save_ckpt(get_best_weights_path(config), epoch, global_step, best_chrf)
                    tqdm.write(f'  ↳ new best chrF++={chrf:.2f}, saved to weights/tmodel_best.pt')

        dt = time.time() - t0
        bleu, chrf = quick_validate(
            model, val_dl, tok_src, tok_tgt, config['max_seq_len'], device,
            writer, global_step, metrics_csv, num_examples=config['val_max_examples'],
        )
        print(f'Epoch {epoch} done in {dt/60:.1f} min | val BLEU={bleu:.2f} chrF++={chrf:.2f}')
        if chrf > best_chrf:
            best_chrf = chrf
            save_ckpt(get_best_weights_path(config), epoch, global_step, best_chrf)

        # ``tmodel_last.pt`` is overwritten each epoch — used to resume.
        # Per-epoch snapshots are 400+ MB each; uncomment if you want them.
        save_ckpt(get_weights_file_path(config, 'last'), epoch, global_step, best_chrf)
        # save_ckpt(get_weights_file_path(config, f'{epoch:02d}'), epoch, global_step, best_chrf)

    writer.close()


if __name__ == '__main__':
    train_model(get_config())
