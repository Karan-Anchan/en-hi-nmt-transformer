# Evaluation report

- Checkpoint: `weights\tmodel_best.pt`
- Test sentences: 500
- Device: NVIDIA GeForce RTX 5070
- Beam size: 4 | length penalty α = 0.6

| Decoder | SacreBLEU ↑ | chrF++ ↑ | TER ↓ | latency (ms) | p95 (ms) | tokens/s | sent/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Greedy | 16.18 | 41.36 | 74.38 | 426.2 | 951.93 | 134.3 | 2.35 |
| Beam (4) | 16.93 | 41.58 | 71.41 | 3960.99 | 4275.27 | 13.8 | 0.25 |
