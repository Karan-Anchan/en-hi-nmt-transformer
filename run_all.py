"""End-to-end recipe: train -> evaluate -> regenerate plots.

Useful when you want to reproduce every artifact in the repo in one shot.
Each step skips itself if its expected output already exists, so re-running
this script is idempotent.

    python run_all.py             # full pipeline, picks up wherever it left off
    python run_all.py --eval-only # skip training, just re-run eval + plots
    python run_all.py --force     # force re-run every step
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PY = sys.executable


def run(label: str, cmd: list[str]) -> None:
    print(f'\n=== {label} ===\n$ {" ".join(cmd)}')
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f'{label} failed (exit {proc.returncode})')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    weights = Path('weights/tmodel_best.pt')

    if not args.eval_only:
        if args.force or not weights.exists():
            run('Train model', [PY, 'train.py'])
        else:
            print(f'Skipping training — {weights} already exists. Use --force to retrain.')

    if not weights.exists():
        raise SystemExit('No trained checkpoint found — cannot evaluate.')

    run('Evaluate on test set', [PY, 'eval.py'])
    run('Regenerate plots', [PY, 'plot_metrics.py'])

    print('\nDone. Artifacts:')
    print(' • weights/tmodel_best.pt')
    print(' • results/eval_report.json, eval_report.md, predictions.tsv')
    print(' • beautiful_train_loss.png, beautiful_metrics.png')


if __name__ == '__main__':
    main()
