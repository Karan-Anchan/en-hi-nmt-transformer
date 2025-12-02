
-----

# English-to-Hindi Neural Machine Translation (Transformer)

  

A complete implementation of the **Transformer** architecture ("Attention Is All You Need") built from scratch in PyTorch for translating English text to Hindi. Trained on the IIT Bombay English-Hindi Corpus.

## 📌 Overview

This project implements a Sequence-to-Sequence Transformer model with Multi-Head Self-Attention and Cross-Attention mechanisms. It features a custom training loop, dynamic learning rate scheduling, and real-time validation metrics.

**Key Features:**

  * **Architecture:** 6-layer Encoder/Decoder, 512 embedding dimension, 8 attention heads.
  * **Dataset:** [cfilt/iitb-english-hindi](https://huggingface.co/datasets/cfilt/iitb-english-hindi).
  * **Tokenizer:** Word-level tokenization trained specifically on the corpus.
  * **Visualization:** Integration with Altair for attention map visualization.

-----

## 📊 Performance & Results

The model was trained for over **900,000 steps**. The best performance was observed at checkpoint **Step 651,791**.

### Key Metrics

| Metric | Score | Description |
| :--- | :--- | :--- |
| **SacreBLEU** | **62.48** | High correspondence with reference translations. |
| **CHRF++** | **71.78** | Strong character-level n-gram overlap. |
| **CER** | **0.28** | Low Character Error Rate (\~28%). |
| **Training Loss**| **2.95** | Converged from initial \~10.24. |

### Training Progression

**Loss Convergence:**
The model shows a consistent downward trend in loss, stabilizing effectively after step 600k.

*(beautiful_train_loss.png)*

**Metric Evolution:**
Both SacreBLEU and CHRF++ show strong correlation with training steps, peaking around the selected checkpoint.

*(beautiful_metrics.png)*

-----

## 🧠 Qualitative Analysis

The model demonstrates robust handling of formal sentence structures and proper noun transliteration.

| Source (English) | Prediction (Hindi) | Note |
| :--- | :--- | :--- |
| *(a) the manufacture or processing of goods;* | *( क ) माल का विनिर्माण या प्रसंस्करण* | Perfect translation of legal/formal text. |
| *Daniel Molkentin* | *डेनियल मॉल्केनटिन* | Accurate transliteration of foreign names. |
| *Last Quarter Moon* | *पिछले चौथाई चन्द्रमा* | Correct semantic understanding. |

### Attention Visualization

[Image of Encoder-Decoder Attention]

Understanding how the Decoder focuses on specific Encoder tokens during generation (Cross-Attention):

*(decoder.png)*

*(encoder-decoder.png)*

-----

## 🛠️ Installation & Setup

1.  **Clone the repository**

    ```bash
    git clone https://github.com/KaranAnchan/en-hi-nmt-transformer.git
    cd en-hi-nmt-transformer
    ```

2.  **Install dependencies**

    ```bash
    pip install requirements.txt
    ```

## 🚀 Usage

### Inference (Translation)

To translate a sentence using the trained weights:

```python
import torch
from model import build_transformer
from config import get_config
from train import greedy_decode, get_ds

# 1. Load Config and Weights
config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-build the model structure (ensure vocab sizes match your training)
# You might need to load the tokenizer first to get exact vocab sizes
model = build_transformer(vocab_src_len=..., vocab_tgt_len=..., 
                          seq_len=config['seq_len'], d_model=config['d_model'])

checkpoint = torch.load("weights/tmodel_06.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

# 2. Translate
src_text = "The manufacture of goods."
# ... (Load tokenizers and preprocess text) ...
# output = greedy_decode(model, src_text, ...)
print(output)
# Output: "माल का विनिर्माण।"
```

### Training

To train the model from scratch:

```bash
python train.py
```

*Modify `config.py` to adjust batch size, learning rate, or number of epochs.*

-----

## 📂 Project Structure

  * `model.py`: Complete Transformer architecture (Embeddings, Positional Encoding, Multi-Head Attention, Encoder/Decoder blocks).
  * `train.py`: Training loop, validation, and checkpoint saving.
  * `dataset.py`: Custom PyTorch Dataset class for bilingual text.
  * `config.py`: Hyperparameters and file path configurations.
  * `attention_visual.ipynb`: Notebook for generating attention heatmaps.

-----

## 📜 Acknowledgements

  * Dataset provided by [IIT Bombay](https://www.cfilt.iitb.ac.in/).
  * Architecture based on the paper *[Attention Is All You Need](https://arxiv.org/abs/1706.03762)* (Vaswani et al., 2017).