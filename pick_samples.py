"""Pick interesting qualitative examples from results/predictions.tsv.

Used by the README to surface a handful of strong / weak translations
without manual cherry-picking. Selects:

  * 5 highest-chrF beam translations across length buckets
  * 2 lowest-chrF beam translations (failure cases to be honest about)

Run after `python eval.py`.
"""
from __future__ import annotations

import csv
from pathlib import Path

import sacrebleu


def main():
    src = Path('results/predictions.tsv')
    if not src.exists():
        raise SystemExit(f'No {src} — run eval.py first.')

    rows = []
    with open(src, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            length = len(row['source'].split())
            chrf = sacrebleu.sentence_chrf(row['beam'], [row['reference']]).score
            rows.append({**row, 'length': length, 'chrf': chrf})

    rows.sort(key=lambda r: r['chrf'], reverse=True)

    short = next((r for r in rows if 3 <= r['length'] <= 6 and r['chrf'] > 50), None)
    med = next((r for r in rows if 7 <= r['length'] <= 12 and r['chrf'] > 50), None)
    longr = next((r for r in rows if r['length'] >= 13 and r['chrf'] > 50), None)
    top = rows[:5]
    bot = sorted(rows, key=lambda r: r['chrf'])[:2]

    out = Path('results/qualitative_samples.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('# Qualitative samples\n\n')
        f.write('## Strong translations\n\n')
        f.write('| length | chrF++ | Source (EN) | Reference (HI) | Beam (HI) |\n')
        f.write('|---:|---:|---|---|---|\n')
        for r in [short, med, longr] + top:
            if r is None:
                continue
            f.write(f"| {r['length']} | {r['chrf']:.1f} | {r['source']} | {r['reference']} | {r['beam']} |\n")
        f.write('\n## Failure modes (lowest chrF++)\n\n')
        f.write('| length | chrF++ | Source (EN) | Reference (HI) | Beam (HI) |\n')
        f.write('|---:|---:|---|---|---|\n')
        for r in bot:
            f.write(f"| {r['length']} | {r['chrf']:.1f} | {r['source']} | {r['reference']} | {r['beam']} |\n")
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
