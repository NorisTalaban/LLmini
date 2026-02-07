# LLmini — Scaling and Attention Dynamics in a Minimal Decoder-Only Transformer

A GPT-style language model built from scratch in PyTorch, trained on ~14M tokens from 11 public English datasets. The project centers on an **ablation study across three model scales** and a **systematic attention analysis** (entropy and sparsity per layer and head), designed to observe how architectural width and depth affect both convergence and the internal structure of attention.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Architecture](#architecture)
3. [Training Data](#training-data)
4. [Training Setup](#training-setup)
5. [Ablation Study](#ablation-study)
6. [Attention Analysis](#attention-analysis)
7. [Discussion](#discussion)
8. [Future Work](#future-work)
9. [Reproducibility](#reproducibility)
10. [How to Run](#how-to-run)
11. [Project Structure](#project-structure)
12. [References](#references)

---

## Motivation

Most LLM papers report results at a single scale. Here, everything is fixed (data, optimizer, schedule, seed) except three tightly coupled hyperparameters: embedding dimension (`d_model`), number of attention heads, and number of layers. Every difference in the results can be traced back to those architectural choices alone.

The second focus of the project is **attention behavior**. For each trained model, attention weights are extracted and analyzed through two complementary lenses: **entropy** (how diffuse or concentrated each head's attention distribution is) and **sparsity** (the fraction of near-zero weights). This analysis is not an afterthought — it is built into the evaluation pipeline from the start, to study how the internal structure of attention changes with model scale.

The entire pipeline runs on a single Colab GPU (T4/L4) in under two hours per variant.

---

## Architecture

LLmini is a **decoder-only autoregressive Transformer** with pre-norm residual blocks. Each block follows the sequence: LayerNorm → Multi-Head Self-Attention → Residual → LayerNorm → FFN → Residual. The output layer projects to vocabulary logits via a linear head (no weight tying).

```
Input Tokens
     │
     ▼
┌─────────────────────┐
│  Token Embedding     │  (vocab_size × d_model)
│  + Position Embed    │  (block_size × d_model)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Transformer Block   │  × num_layers
│  ┌─────────────────┐ │
│  │ LN → MHA → Drop │ │  + residual
│  │ LN → FFN → Drop │ │  + residual
│  └─────────────────┘ │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  LayerNorm (final)   │
│  Linear → Logits     │  (d_model × vocab_size)
└─────────────────────┘
```

The FFN expands to `4 × d_model` with GELU activation, matching the GPT-2 design. Dropout is 0.1 on attention outputs and 0.3 on FFN outputs.

### Ablation Variants

The three models differ **only** in `d_model`, `num_heads`, and `num_layers`. Every head operates on a fixed dimension of 64 (`d_model / num_heads = 64`), so adding a head always adds exactly one 64-dimensional attention subspace rather than changing per-head capacity.

| | Model A | Model B | Model C |
|---|---------|---------|---------|
| **d_model** | 384 | 320 | 256 |
| **num_heads** | 6 | 5 | 4 |
| **num_layers** | 10 | 9 | 8 |
| **d_head** | 64 | 64 | 64 |
| **FFN dim** | 1536 | 1280 | 1024 |

Going from C → B → A simultaneously widens the representation, adds one attention subspace, and stacks one more layer. This makes the three dimensions co-vary, which reflects how models are typically scaled in practice but means the individual contributions of width, heads, and depth cannot be disentangled from these experiments alone.

---

## Training Data

The corpus is assembled from **11 public English datasets** spanning narrative text, encyclopedic knowledge, reading comprehension, commonsense reasoning, and math.

| Dataset | Samples | Tokens | % Tokens | Avg Tok/Sample |
|---|---:|---:|---:|---:|
| TinyStories | 19,077 | 4,137,714 | 29.6% | 216.9 |
| SQuAD | 20,000 | 2,691,776 | 19.3% | 134.6 |
| WikiText-2 | 19,827 | 1,962,696 | 14.1% | 99.0 |
| BoolQ | 9,413 | 1,219,124 | 8.7% | 129.5 |
| SciQ | 11,625 | 1,172,914 | 8.4% | 100.9 |
| AG News | 20,000 | 1,060,001 | 7.6% | 53.0 |
| HellaSwag | 19,952 | 704,757 | 5.0% | 35.3 |
| Winogrande | 20,000 | 433,315 | 3.1% | 21.7 |
| GSM8K | 7,473 | 412,695 | 3.0% | 55.2 |
| CommonsenseQA | 9,741 | 150,470 | 1.1% | 15.4 |
| ARC-Challenge | 1,119 | 28,217 | 0.2% | 25.2 |
| **Total** | **158,227** | **13,973,679** | **100%** | **88.3** |

TinyStories and SQuAD dominate the token budget because their samples are longer (217 and 135 tokens on average), providing exposure to multi-sentence coherence. The token distribution is imbalanced (29.6% to 0.2%) — this is a known limitation, not a design choice.

### Preprocessing

Tokenization uses the **GPT-2 BPE tokenizer** (50,257 tokens). Texts shorter than the 256-token context window are right-padded; longer texts are split into overlapping chunks (stride = 128). The dataset covers 92.3% of the GPT-2 vocabulary.

| Metric | Value |
|---|---|
| Context window | 256 tokens |
| Total chunks | 160,745 |
| Padded (short texts) | 150,971 (95.4%) |
| Chunked (long texts) | 7,256 (4.6%) |
| Train / Val split | 95% / 5% |
| Token utilization | 34.0% |

The 66% padding overhead is a direct consequence of the right-skewed length distribution (median 56, mean 88.3 tokens). Most samples are much shorter than the 256-token window.

---

## Training Setup

All three models are trained with **identical hyperparameters and data**, ensuring differences in the results come only from the architecture.

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (β₁=0.9, β₂=0.999, ε=1e-8) |
| Learning rate | 3e-4 |
| Weight decay | 0.05 |
| LR schedule | Cosine annealing + 10% linear warmup |
| Physical batch size | 32 |
| Gradient accumulation | 2 steps (effective batch = 64) |
| Mixed precision | FP16 via `torch.cuda.amp` |
| Gradient clipping | max_norm = 1.0 |
| Epochs | 3 |
| Dropout (attention) | 0.1 |
| Dropout (FFN + residual) | 0.3 |
| Seed | 42 (fixed everywhere) |

---

## Ablation Study

### Loss and Perplexity

| Epoch | | Model A (384/6/10) | Model B (320/5/9) | Model C (256/4/8) |
|:---:|---|:---:|:---:|:---:|
| **1** | Train Loss | 5.873 | 6.029 | 6.220 |
| | Val Loss | 4.728 | 4.902 | 5.104 |
| | Val Perplexity | **113.05** | 134.61 | 164.66 |
| **2** | Train Loss | 4.447 | 4.647 | 4.879 |
| | Val Loss | 4.180 | 4.391 | 4.634 |
| | Val Perplexity | **65.38** | 80.68 | 102.90 |
| **3** | Train Loss | 4.076 | 4.320 | 4.594 |
| | Val Loss | 4.080 | 4.301 | 4.553 |
| | Val Perplexity | **59.12** | 73.77 | 94.88 |

Model A reaches a final validation perplexity of 59.12 versus 94.88 for Model C. The larger model converges faster from the first epoch, which is consistent with the observation by Kaplan et al. (2020) that larger models are more sample-efficient.

### Overfitting Gap (Val Loss − Train Loss)

| Epoch | Model A | Model B | Model C |
|:---:|:---:|:---:|:---:|
| 1 | −1.145 | −1.127 | −1.116 |
| 2 | −0.266 | −0.256 | −0.246 |
| 3 | **+0.003** | −0.019 | −0.041 |

Model A is the only variant to cross zero at epoch 3, suggesting it is beginning to memorize. Models B and C remain in the underfitting regime throughout. This is expected given the data budget: Hoffmann et al. (2022) established that compute-optimal training requires roughly 20 tokens per parameter. Model A would need on the order of a billion tokens by that estimate — the corpus provides ~14M, roughly 80× below the Chinchilla-optimal ratio. Under this lens, **all three models are severely data-starved**, and the fact that even Model A barely overfits confirms the dataset is the bottleneck, not the architecture.

### Compute

| | Model A | Model B | Model C |
|---|:---:|:---:|:---:|
| Training time | 69.8 min | 57.8 min | 44.5 min |
| Peak VRAM | 7.57 GB | 6.85 GB | 6.21 GB |
| Final Val PPL | 59.12 | 73.77 | 94.88 |

Model C trains 36% faster and uses 18% less VRAM than Model A. For rapid iteration on data pipelines or loss functions, a smaller model gives faster feedback at the cost of higher perplexity.

---

## Attention Analysis

After training, attention weights are extracted from a single forward pass (seed=42) and analyzed per layer and per head. Two metrics are computed:

- **Entropy**: `H = −Σ p·log(p)` over the attention distribution. High entropy means the head attends broadly; low entropy means it focuses on specific tokens.
- **Sparsity**: fraction of attention weights below 1e-3. High sparsity means most positions receive near-zero attention.

These two metrics are strongly negatively correlated (approximately −0.9 in all three models): heads that attend broadly are never sparse, and vice versa.

### Per-Layer Entropy and Sparsity

| Layer | Entropy (A) | Sparsity (A) | Entropy (B) | Sparsity (B) | Entropy (C) | Sparsity (C) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 4.27 | 0.509 | 4.18 | 0.512 | 4.32 | 0.505 |
| 2 | 4.11 | 0.522 | 4.22 | 0.516 | 4.22 | 0.515 |
| 3 | 4.24 | 0.515 | 4.21 | 0.541 | 4.02 | 0.592 |
| 4 | 4.17 | 0.530 | 4.01 | 0.576 | 3.72 | 0.652 |
| 5 | 4.12 | 0.557 | 3.58 | 0.651 | 3.60 | 0.684 |
| 6 | 3.76 | 0.641 | 3.36 | 0.727 | 3.52 | 0.713 |
| 7 | 3.14 | 0.764 | 3.28 | 0.764 | 3.57 | 0.711 |
| 8 | 3.25 | 0.777 | 3.42 | 0.749 | 3.78 | 0.666 |
| 9 | 3.77 | 0.648 | 3.63 | 0.656 | — | — |
| 10 | 4.03 | 0.573 | — | — | — | — |

### Aggregate Metrics

| | Model A | Model B | Model C |
|---|:---:|:---:|:---:|
| Mean entropy | 3.887 | 3.766 | 3.843 |
| Mean sparsity | 0.604 | 0.633 | 0.630 |
| Sparsity range | 0.509 – 0.777 | 0.512 – 0.764 | 0.505 – 0.713 |
| Total head×layer | 60 | 45 | 32 |

### What the Attention Data Shows

**Early layers are diffuse, middle layers become selective.** All three models show the same qualitative trend: entropy is high and sparsity is low in the first few layers, then entropy drops and sparsity rises in the middle layers. This is consistent with findings from PyramidKV (Cai et al., 2024), which observes that attention entropy decreases with layer depth in Transformers, and with observations from the Attention Condensation literature (2025) that initial layers exhibit dense, globally distributed attention.

**The final layers partially relax.** In Models A and B, the last 1–2 layers show a reversal: entropy increases and sparsity decreases relative to the middle layers. Model C (8 layers) shows a weaker version of this pattern. This is noted as an empirical observation in these models and does not claim to generalize beyond this scale.

**Deeper models develop a wider dynamic range.** Model A (10 layers) reaches a sparsity peak of 0.777 (layers 7–8), compared to 0.764 for Model B and 0.713 for Model C. The extra layers give Model A more room to develop differentiated attention patterns. Whether this directly causes the lower perplexity or is merely a correlate of having more capacity is not something these experiments can determine.

---

## Discussion

### What These Results Support

The observation that larger models converge faster on the same data aligns with Kaplan et al. (2020), who found that loss scales as a power-law with model size and that larger models are more sample-efficient. The fact that Model A achieves lower perplexity than C with identical training confirms this effect holds even at small scale.

The overfitting analysis is consistent with the Chinchilla framework (Hoffmann et al., 2022): with a tokens-to-parameters ratio far below the recommended ~20:1, all three models are data-limited. More training data would likely improve all variants, with the larger models benefiting the most.

### What These Results Do Not Show

Since `d_model`, `num_heads`, and `num_layers` all change together, the individual contribution of each factor to the perplexity improvement **cannot be isolated**. Kaplan et al. (2020) found that at constant total parameters, specific choices of width versus depth have "minimal effects within a wide range" — this design does not test this because total parameters also change across variants.

The attention analysis is descriptive, not causal. The observation that deeper models develop sharper attention patterns could be a consequence of lower loss rather than a cause of it. Establishing causality would require interventional experiments (e.g., pruning specific heads and measuring the impact on loss).

---

## Future Work

This project is a starting point. Several directions are planned to address its current limitations and push the model toward more meaningful capabilities.

### Scaling the Data

The most immediate bottleneck is the corpus size. At ~14M tokens, all three models are severely data-starved relative to their capacity (the Chinchilla ratio suggests ~1B tokens even for the smallest variant). The next step is to scale the training data by one or two orders of magnitude — targeting 500M–1B tokens — by expanding the existing sources (full WikiText-103 instead of WikiText-2, full SQuAD and BoolQ, larger slices of TinyStories) and adding new high-quality corpora: OpenWebText, BookCorpus, RedPajama subsets, arXiv abstracts for technical language, and GitHub code for structured reasoning patterns. This would also resolve the 66% padding overhead: with more long-form text, the length distribution would shift and the packing strategy (concatenating multiple short texts into a single chunk instead of padding) becomes both feasible and necessary. A larger corpus would allow observation of whether the attention entropy/sparsity patterns change when the models are no longer data-limited, and whether Model A's slight overfitting at epoch 3 turns into genuine generalization improvement with sufficient data.

### Exploring Reasoning

The current model is a pure next-token predictor with no explicit reasoning capability. A natural evolution is to explore whether structured reasoning can emerge or be induced at this scale. Concretely, this means training on chain-of-thought traces from GSM8K and ARC (not just the questions, but the step-by-step solutions), adding synthetic reasoning datasets (e.g., logical deductions, simple arithmetic sequences), and evaluating on tasks that require multi-step inference. The question is not whether a 50M-parameter model can compete with GPT-4 on reasoning — it cannot — but whether the *onset* of reasoning-like behavior can be observed: does the model learn to produce intermediate steps before a final answer? Does the attention pattern change when the input contains a chain-of-thought versus a flat question? This connects directly to the existing attention analysis: if reasoning relies on specific attention structures (e.g., heads that attend to previous reasoning steps), the entropy/sparsity framework is already in place to detect it.

### Scaling the Model with Serious Compute

The current ablation runs on a single T4 GPU in under two hours. The architecture and training loop are designed to scale beyond this. With access to multi-GPU infrastructure (A100 or H100 nodes), the plan is to scale along three axes simultaneously: model size (d_model=768, 12 heads, 16+ layers — approaching GPT-2 small), context window (1024 or 2048 tokens), and training duration (10–20 epochs on the larger corpus). This would require implementing distributed training (FSDP or DeepSpeed ZeRO), gradient checkpointing to fit larger models in memory, and a more sophisticated data pipeline with streaming and dynamic batching. The ablation framework would extend naturally: instead of three variants at ~30–60M parameters, the comparison would span from 50M to 300M+, which is the range where Kaplan et al. (2020) observed the clearest power-law scaling behavior. The goal is to test whether the attention patterns documented in this project (diffuse early → sparse middle → relaxed final) hold at larger scale, or whether new structures emerge that are invisible at the current size.

### Other Improvements

- **Packing instead of padding**: concatenate short texts into full-length chunks, eliminating the 66% padding overhead and making every training token count.
- **Isolated ablations**: change one axis at a time (e.g., same d_model and heads, vary only depth) to disentangle the contributions of width, head count, and layer count.
- **Downstream evaluation**: add zero-shot benchmarks (HellaSwag, BoolQ, ARC) to measure capability beyond perplexity.
- **Weight tying**: share the embedding and output projection weights to reduce parameter count and potentially improve generalization, as done in GPT-2.
- **Rotary positional embeddings (RoPE)**: replace learned positional embeddings with RoPE for better extrapolation to longer sequences.

---

## Reproducibility

Every source of randomness is controlled:

| Component | Method |
|---|---|
| Python `random` | `random.seed(42)` |
| NumPy | `np.random.seed(42)` |
| PyTorch CPU | `torch.manual_seed(42)` |
| PyTorch GPU | `torch.cuda.manual_seed_all(42)` |
| CuDNN | `deterministic=True`, `benchmark=False` |
| DataLoader | seeded generator, `num_workers=0` |
| Train/Val split | `random_split` with seeded generator |
| Text generation | explicit seed per call |

The data pipeline produces identical outputs across runs: 158,227 texts → 160,745 chunks → 152,708 train / 8,037 val.

---

## How to Run

### Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA
- Transformers, Datasets (HuggingFace)
- GPU with ≥ 8 GB VRAM (T4, L4, or equivalent)

```bash
pip install torch transformers datasets pandas numpy scipy matplotlib seaborn tqdm psutil
```

### Training

The entire pipeline is in a single notebook. To select a variant, change the model constructor:

```python
# Model A
model = LLmini(vocab_size=50257, d_model=384, num_heads=6, num_layers=10, ...)

# Model B
model = LLmini(vocab_size=50257, d_model=320, num_heads=5, num_layers=9, ...)

# Model C
model = LLmini(vocab_size=50257, d_model=256, num_heads=4, num_layers=8, ...)
```

### Generation

```python
output = generate(model, "The movie was", max_new_tokens=50, temperature=0.7, top_k=20)
```

---

## Project Structure

```
LLmini/
├── 384_6_10.ipynb             # Model A (d=384, h=6, L=10)
├── 320_5_9.ipynb              # Model B (d=320, h=5, L=9)
├── 256_4_8.ipynb              # Model C (d=256, h=4, L=8)
└── README.md
```

**Note**: The `LLmini_STEM_model/` directory is created automatically when running the notebooks in Google Colab and contains:
- `pytorch_model.bin` — Final trained weights
- `best_model.pt` — Best checkpoint during training
- `model_config.json` — Architecture configuration
- Tokenizer files (GPT-2 BPE)

---

## References

1. **Vaswani, A., Shazeer, N., Parmar, N., et al. (2017)**  
   *Attention Is All You Need.*  
   NeurIPS.

2. **Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020)**  
   *Scaling Laws for Neural Language Models.*  
   arXiv:2001.08361.

3. **Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022)**  
   *Training Compute-Optimal Large Language Models.*  
   arXiv:2203.15556.

4. **Cai, Z., et al. (2024)**  
   *PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling.*  
   arXiv:2406.02069.

---

## License

Released for educational and research purposes.  
Datasets are loaded from HuggingFace Hub under their respective licenses.  
GPT-2 tokenizer used under the MIT License.
