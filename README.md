# 🧩 Sudoku Transformer — Constraint-Aware Sequence Models & Mechanistic Interpretability

> A from-scratch Transformer architecture that solves Sudoku puzzles as a **parallel, constraint-satisfaction problem** rather than a left-to-right sequence prediction task — paired with a full mechanistic interpretability investigation into *how* the model reasons over the grid.

This project explores two central questions:
1. Can a Transformer learn to solve a classic combinatorial constraint-satisfaction puzzle *without* being told any of Sudoku's rules explicitly?
2. If it can, what is actually happening inside the model as it "reasons" — and does a **recurrent, weight-shared** architecture reason differently than a **deep, stacked** one?

---

## 📌 Overview

Sudoku is a compelling testbed for Transformer reasoning research: it's not a sequential language task, it has **rigid, well-defined logical constraints** (each digit 1–9 must appear exactly once per row, column, and 3×3 box), and success/failure is unambiguous and easy to grade per-cell. That makes it an ideal sandbox for studying whether attention-based architectures can internalize structured, relational reasoning — and for probing what they've actually learned once they do.

The project consists of two complementary threads:

| Component | What it does |
|---|---|
| **Model Architectures** | Two Transformer variants — a **Recurrent/Looped Transformer** (single shared encoder block applied iteratively) and a **Stacked Transformer** (multiple distinct encoder layers) — trained to predict the full solved grid in parallel from a partially-filled puzzle |
| **Interpretability Suite** | Built with [`nnsight`](https://nnsight.net/), including per-cell **entropy/uncertainty heatmaps**, a **Logit Lens** across recurrent reasoning steps, and **manual attention-head visualization** to see which cells the model attends to when resolving a given digit |

---

## Why This Is a Nontrivial Modeling Problem

Naively, Sudoku looks like a per-cell classification task — predict a digit 1–9 (or blank) for each of the 81 cells. But treating it that way ignores the entire point of the puzzle: **cells are not independent.** A correct prediction for one cell depends on the values (and non-values) of every other cell in its row, column, and box. This project addresses that head-on:

- **Structural positional encoding** — instead of a standard 1D positional embedding, cells are encoded with **row, column, and 3×3-box identity embeddings** (`Sudoku3DPositionalEncoding`), directly injecting Sudoku's grid topology into the model rather than hoping attention discovers it from scratch.
- **Constraint-aware auxiliary loss** — alongside standard cross-entropy on the target digits, a custom **Cardinality Constraint Loss (CSP Loss)** penalizes the model whenever the *summed probability mass* for any digit within a row, column, or box deviates from 1.0 — explicitly encoding the "exactly once per unit" rule of Sudoku into the training signal across all 27 constraint units (9 rows + 9 columns + 9 boxes).
- **Iterative refinement via recurrence** — rather than predicting the answer in one shot, the Recurrent Transformer applies the *same* encoder block repeatedly (e.g., 6 times) to the same grid representation, allowing the model to progressively refine its guesses the way a human iteratively narrows down candidates.

---

## Architectures

### 1. Recurrent (Looped) Transformer
```
Input Grid (81 cells, 0=blank / 1-9=digit)
   │
   ▼
Token Embedding + 3D Structural Positional Encoding
   │
   ▼
┌─────────────────────────────┐
│  Shared Encoder Block        │◄──┐
│  (self-attention + FFN)      │   │  Looped N times
└─────────────────────────────┘───┘
   │
   ▼
Linear Output Head → 81 × 10 logits (per-cell digit distribution)
```
A **single encoder block's weights are reused across every reasoning step**, similar in spirit to Universal Transformers / looped-architecture reasoning models — the model must learn one general-purpose "refinement operator" rather than a different specialized function per depth.

### 2. Stacked Transformer
Identical embedding and positional encoding scheme, but with **N distinct encoder layers** stacked sequentially (standard deep Transformer encoder), giving each layer its own independent weights.

Both models share the same **3D positional encoding** and **combined CE + CSP loss objective**, isolating architecture (shared-weight recurrence vs. depth with independent weights) as the true experimental variable.

---

## Training Results

Both models were trained for 100 epochs on a Kaggle Sudoku puzzle/solution dataset (10,000 training puzzles / 2,000 validation puzzles), using AdamW and a combined `CE + 0.5 × CSP` loss, evaluated on **per-cell digit accuracy**.

| Model | Final Train Digit Acc | Final Val Digit Acc |
|---|---|---|
| **Recurrent Transformer** (6 shared-weight steps, d_model=128) | 93.49% | **95.48%** |
| **Stacked Transformer** (6 independent layers, d_model=128) | 91.89% | 92.96% |

**Key finding:** the recurrent, weight-shared model *outperformed* the deeper stacked model with an equivalent number of reasoning steps — despite using a fraction of the parameters (one encoder block vs. six independent ones). This suggests that for a highly structured, rule-based task like Sudoku, forcing the model to reuse a single general "constraint-propagation operator" is a more effective inductive bias than giving it more unconstrained capacity.

> **Note:** accuracy above is measured **per-cell**, not per-fully-solved-puzzle — a meaningful next step (see Future Work) is tracking exact full-grid solve rate, which is a substantially harder bar.

---

## Interpretability Analysis

The second half of the project asks: now that we have a working model, *how* does it actually solve the puzzle? Using [`nnsight`](https://nnsight.net/) to instrument the trained Recurrent Transformer, three complementary lenses were built:

### 1. Output Entropy Heatmaps
For every cell, entropy is computed over the model's final softmax distribution and rendered as a 9×9 heatmap. This exposes exactly where the model is *confident* vs. *uncertain* — high-entropy cells correspond to positions the model finds ambiguous given the current grid state, giving a direct visual proxy for "how hard is this cell" from the model's own perspective.

### 2. Recurrent Logit Lens
Because the Recurrent Transformer applies the *same* block repeatedly, its intermediate hidden states between each reasoning step can be intercepted with forward hooks and projected through the final unembedding layer (`fc_out`) at *every* step — not just the last. This produces an entropy heatmap **per recurrent iteration**, effectively visualizing the model's evolving confidence as it iteratively refines the board, analogous to a Logit Lens analysis across depth in a standard deep Transformer, but applied across *recurrent time* instead.

### 3. Manual Attention-Head Visualization
Rather than relying on a black-box attention visualizer, Q/K/V projections are manually recomputed from the captured self-attention input (`in_proj_weight` / `in_proj_bias`) to reconstruct raw attention probabilities per head. This allows querying, for any specific cell and head, *exactly which of the other 80 cells it attends to most** — a direct test of whether the model has implicitly learned to attend along Sudoku's row/column/box constraint structure without ever being told those relationships exist as anything other than positional embeddings.

---

## Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch (`nn.TransformerEncoder`, custom encoder blocks) |
| Interpretability | [`nnsight`](https://nnsight.net/) (model wrapping, forward hooks) |
| Data | Pandas, NumPy, Kaggle Sudoku puzzle/solution dataset |
| Visualization | Matplotlib, Seaborn (heatmaps), Plotly (training curves) |
| Environment | Google Colab (GPU runtime) |

---

## Getting Started

### Prerequisites
```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn plotly
pip install nnsight
```

### Dataset
The notebook expects a Kaggle-style Sudoku CSV with `puzzle` and `solution` columns (each an 81-character string, `0` = blank), e.g., the [1M Sudoku Puzzles dataset](https://www.kaggle.com/datasets/bryanpark/sudoku). Update the `csv_path` in `train_pipeline()` to point to your local/Drive copy.

### Running
Open `sudoku_transformer.ipynb` and run cells top to bottom:
1. **Recurrent Transformer** — architecture definition + `train_pipeline()` training loop
2. **Interpretability tooling** — load the trained model, wrap with `nnsight`
3. **Entropy heatmaps, Logit Lens, attention visualization** — run on a sample puzzle
4. **Stacked Transformer** — alternate architecture for direct comparison

---


## License

This project is available under the MIT License. See `LICENSE` for details.

---


