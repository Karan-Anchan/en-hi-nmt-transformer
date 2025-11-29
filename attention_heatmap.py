import matplotlib.pyplot as plt
import numpy as np

def plot_attention(attn_weights, src_tokens, tgt_tokens, title="Attention Heatmap", out_path=None):
    """
    attn_weights: 2D numpy array of attention weights (target_len x source_len)
    src_tokens: List of source token strings
    tgt_tokens: List of target token strings
    title: Optional title for the plot
    out_path: If provided, saves plot to this path
    """
    plt.figure(figsize=(min(18, 1+len(src_tokens)//2), min(12, 1+len(tgt_tokens)//2)))
    plt.imshow(attn_weights, aspect='auto', interpolation='nearest', cmap='viridis')
    plt.colorbar(label='Attention Score')
    plt.xticks(ticks=np.arange(len(src_tokens)), labels=src_tokens, rotation=45, ha='right', fontsize=7)
    plt.yticks(ticks=np.arange(len(tgt_tokens)), labels=tgt_tokens, fontsize=7)
    plt.xlabel('Source Tokens')
    plt.ylabel('Target Tokens')
    plt.title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
        print(f"Saved attention heatmap to {out_path}")
    else:
        plt.show()
    plt.close()