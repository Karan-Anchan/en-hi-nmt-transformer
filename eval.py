"""End-to-end evaluation on the held-out test set.

What this reports (and why):
  * **SacreBLEU** — standard MT metric. Word-overlap based.
  * **chrF++** — character n-gram F-score; tokenization-free and more
    reliable for Indian languages.
  * **TER** — translation edit rate (lower is better).
  * **Latency / throughput** — wall-clock ms per sentence and tokens/sec
    on the current GPU, both for greedy and beam search.

Outputs:
  * ``results/eval_report.json`` — machine-readable summary.
  * ``results/eval_report.md``   — human-readable table for the README.
  * ``results/predictions.tsv``  — source / target / greedy / beam columns.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import sacrebleu
import torch
from tqdm import tqdm

from config import get_best_weights_path, get_config
from decode import beam_search_decode, build_inputs, detok, greedy_decode, load_inference_model


def _load_test_set(experiment_name: str) -> list[tuple[str, str]]:
    test_path = Path(experiment_name) / 'test.jsonl'
    if not test_path.exists():
        raise FileNotFoundError(
            f'Held-out test set not found at {test_path}. '
            'Run train.py first so the split is materialized.'
        )
    with open(test_path, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    return [(r['translation']['en'], r['translation']['hi']) for r in rows]


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def _time(fn, device) -> tuple:
    _sync(device); t0 = time.perf_counter()
    out = fn()
    _sync(device)
    return out, time.perf_counter() - t0


def _metrics(preds, refs):
    return {
        'sacrebleu': round(sacrebleu.corpus_bleu(preds, [refs]).score, 2),
        'chrf_pp':   round(sacrebleu.corpus_chrf(preds, [refs]).score, 2),
        'ter':       round(sacrebleu.corpus_ter(preds, [refs]).score, 2),
    }


def _perf(times, token_count):
    total = sum(times)
    return {
        'mean_latency_ms':       round(1000.0 * total / len(times), 2),
        'p95_latency_ms':        round(1000.0 * sorted(times)[int(0.95 * len(times)) - 1], 2),
        'throughput_tok_per_s':  round(token_count / max(total, 1e-9), 1),
        'sentences_per_s':       round(len(times) / max(total, 1e-9), 2),
    }


def _write_markdown(report: dict, out_path: Path) -> None:
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('# Evaluation report\n\n')
        f.write(f'- Checkpoint: `{report["checkpoint"]}`\n')
        f.write(f'- Test sentences: {report["n_test"]}\n')
        f.write(f'- Device: {report["gpu"]}\n')
        f.write(f'- Beam size: {report["beam_size"]} | length penalty α = {report["length_penalty"]}\n\n')
        f.write('| Decoder | SacreBLEU ↑ | chrF++ ↑ | TER ↓ | latency (ms) | p95 (ms) | tokens/s | sent/s |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|---:|\n')
        for name, key in [('Greedy', 'greedy'), (f'Beam ({report["beam_size"]})', 'beam')]:
            r = report[key]
            f.write(f"| {name} | {r['sacrebleu']} | {r['chrf_pp']} | {r['ter']} | "
                    f"{r['mean_latency_ms']} | {r['p95_latency_ms']} | "
                    f"{r['throughput_tok_per_s']} | {r['sentences_per_s']} |\n")


def evaluate(config, ckpt_path: str | None = None, max_examples: int | None = None,
             beam_size: int = 4, length_penalty: float = 0.6, output_dir: str = 'results'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    ckpt_path = ckpt_path or get_best_weights_path(config)
    model, tok_src, tok_tgt, state = load_inference_model(config, ckpt_path, device)
    print(f'Loaded checkpoint {ckpt_path} (step={state.get("global_step", "?")})')

    pairs = _load_test_set(config['experiment_name'])
    if max_examples:
        pairs = pairs[:max_examples]
    print(f'Evaluating on {len(pairs)} test pairs')

    refs, greedy_preds, beam_preds = [], [], []
    greedy_times, beam_times = [], []
    greedy_tokens, beam_tokens = 0, 0

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tsv_path = Path(output_dir) / 'predictions.tsv'
    with open(tsv_path, 'w', encoding='utf-8', newline='') as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['source', 'reference', 'greedy', 'beam'])

        for src_text, tgt_text in tqdm(pairs, desc='Translating'):
            refs.append(tgt_text)
            src, src_mask, _ = build_inputs(src_text, tok_src, config['max_seq_len'], device)

            g_ids, g_dt = _time(
                lambda: greedy_decode(model, src, src_mask, tok_src, tok_tgt,
                                      config['max_seq_len'], device),
                device,
            )
            greedy_times.append(g_dt); greedy_tokens += len(g_ids)
            g_text = detok(g_ids.tolist(), tok_tgt); greedy_preds.append(g_text)

            b_ids, b_dt = _time(
                lambda: beam_search_decode(model, src, src_mask, tok_src, tok_tgt,
                                           config['max_seq_len'], device,
                                           beam_size=beam_size, length_penalty=length_penalty),
                device,
            )
            beam_times.append(b_dt); beam_tokens += len(b_ids)
            b_text = detok(b_ids.tolist(), tok_tgt); beam_preds.append(b_text)

            writer.writerow([src_text, tgt_text, g_text, b_text])

    report = {
        'checkpoint': ckpt_path,
        'n_test': len(pairs),
        'device': str(device),
        'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
        'beam_size': beam_size,
        'length_penalty': length_penalty,
        'greedy': {**_metrics(greedy_preds, refs), **_perf(greedy_times, greedy_tokens)},
        'beam':   {**_metrics(beam_preds,   refs), **_perf(beam_times,   beam_tokens)},
    }

    out_json = Path(output_dir) / 'eval_report.json'
    out_md = Path(output_dir) / 'eval_report.md'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    _write_markdown(report, out_md)

    print('\n=== Results ===')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'\nWrote: {out_json}, {out_md}, {tsv_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default=None, help='Path to .pt checkpoint (default: weights/tmodel_best.pt)')
    p.add_argument('--max', type=int, default=None, help='Cap number of test examples')
    p.add_argument('--beam', type=int, default=4)
    p.add_argument('--alpha', type=float, default=0.6, help='Length penalty')
    p.add_argument('--out', default='results')
    args = p.parse_args()
    evaluate(get_config(), args.ckpt, args.max, args.beam, args.alpha, args.out)


if __name__ == '__main__':
    main()
