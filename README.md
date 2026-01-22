
---

# GPT-2 Layer Interpretability: POS-Tag Residual Analysis

This project explores the internal dynamics of the GPT-2 Transformer model. By intercepting internal activations using PyTorch hooks, we measure the **L2 Magnitude** of updates made by Attention and MLP sub-layers, aggregated by **Part-of-Speech (POS)** tags.

## 🎯 Project Goals

* **Quantify "Work":** Identify which layers contribute most to the hidden state for specific grammatical categories (e.g., Verbs vs. Punctuation).
* **Linguistic Probing:** Observe how the model's focus shifts from syntactic processing (early layers) to semantic/predictive processing (late layers).
* **Alignment Research:** Demonstrate a heuristic for aligning Byte-Pair Encoding (BPE) sub-tokens with spaCy's linguistic tokens.

## 🏗️ Directory Structure

```text
gpt2_interpretability/
├── src/
│   ├── alignment.py    # Aligns spaCy POS tags with GPT-2 BPE tokens
│   ├── hooks.py        # Manages PyTorch forward hooks and state
│   ├── model_utils.py  # Utilities for loading GPT-2 and Tokenizers
│   └── processor.py    # Main aggregation and L2 norm calculation logic
├── results/            # Output directory for generated visualizations
├── main.py             # Entry point for the analysis
└── requirements.txt    # Project dependencies

```

## 🛠️ Methodology

### 1. Extraction via Hooks

We register forward hooks on every `Block` in the GPT-2 architecture. Specifically, we capture:

* **Attention Residuals:** The output of `block.attn` before the residual add.
* **MLP Residuals:** The output of `block.mlp` before the residual add.

### 2. Token Alignment

Because GPT-2 uses BPE, a single word like "unbelievable" may be split into multiple tokens. We use a matching heuristic to map the POS tag from a spaCy `Doc` object back to every corresponding sub-token in the Transformer's input.

### 3. Metric Calculation

For every token  at layer , we calculate the L2 Norm:



These magnitudes are then averaged across the entire corpus for each POS tag.

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* CUDA-capable GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gpt2-interpretability.git
cd gpt2-interpretability

```


2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

```



### Running the Analysis

Execute the main script to process the WikiText-2 dataset and generate plots:

```bash
python main.py

```

## 📊 Expected Outputs

The script generates two primary visualizations in the `results/` folder:

1. **Attention Magnitude by POS:** Shows which layers' attention heads are most active for specific word types.
2. **MLP Magnitude by POS:** Highlights where the feed-forward networks provide the most refinement to the residual stream.

---

