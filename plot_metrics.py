import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

def plot_metric(filepath, metric_name, ylabel):
    data = pd.read_csv(filepath)
    plt.figure(figsize=(8, 4))
    plt.plot(data['step'], data['value'], marker='o')
    plt.title(f'{metric_name} over Training Steps/Epochs')
    plt.xlabel('Step/Epoch')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_all_metrics(logdir='runs/tmodel'):
    metric_files = glob.glob(os.path.join(logdir, '*.csv'))
    for metric_file in metric_files:
        metric_name = os.path.basename(metric_file).replace('.csv', '').replace('_', ' ')
        ylabel = metric_name
        plot_metric(metric_file, metric_name, ylabel)

# Usage example:
# plot_all_metrics('runs/tmodel')