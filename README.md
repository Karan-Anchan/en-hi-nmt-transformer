## 📈 Enhanced Training & Visualization
- Training and validation metrics (loss, BLEU, CHRF++, WER, CER) are now logged to TensorBoard and CSV for easy analysis.
- Qualitative translation outputs are logged as CSV tables per evaluation epoch.
- Run TensorBoard: tensorboard --logdir runs
- For publication-quality plots, use plot_metrics.py on generated CSVs.
- See attention_heatmap.py (provided in repo) for model attention visualization when available.

*Note: Please install sacrebleu and matplotlib for the dependencies.*