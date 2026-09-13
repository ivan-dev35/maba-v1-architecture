---
language:
- en
license: mit
library_name: transformers
tags:
- maba
- maba-v1
- maba-v1.1
- architecture
- recurrent
- gated-deltanet
- gdn
- gdn-2
- linear-attention
- linear-recurrence
- state-space-model
- ssm
- gqa
- grouped-query-attention
- swiglu
- rmsnorm
- rope
- speculative-decoding
- mtp
- multi-token-prediction
- efficient-llm
- lightweight-llm
- 100m
- pytorch
- safetensors
- cpp
- avx2
---

<p align="center">
  <img src="https://huggingface.co/AndrewThompson1233/maba-v1-architecture/resolve/main/assets/logo.svg" width="160" alt="Maba Logo" />
</p>

# Maba Architecture: Sub-Quadratic Hybrid Linear-Recurrent Attention

Official specification and reference implementation of the Maba neural network architecture. Maba combines Gated DeltaNet linear recurrence (GDN-2) with Grouped-Query Attention (GQA), 2-pass physical block recycling, and native multi-token prediction (MTP) speculative decoding.

> [!NOTE]
> **Pretrained Weights and Evaluation Benchmarks**
> For trained model checkpoints, Safetensors weights, and empirical evaluation results on standard benchmarks (ARC, HellaSwag, Story Cloze), see the model repository:
> **[AndrewThompson1233/maba-101m](https://huggingface.co/AndrewThompson1233/maba-101m)**

---

<p align="center">
  <img src="assets/architecture_comparison.svg" width="900" alt="Maba Architecture Feature Comparison" />
</p>

---

## Architectural Principles

Standard transformers scale at O(N^2) memory and compute with sequence length. Pure linear RNNs and state-space models scale at O(1) state memory but struggle with associative recall across long token horizons.

Maba resolves this trade-off through a 3:1 macro-interleaved block structure:
* **75% Linear Recurrent Blocks (GDN-2)**: updates an input-dependent recurrent state matrix in O(1) memory per step with dynamic gating.
* **25% Grouped-Query Attention Blocks (GQA)**: provides exact associative retrieval and cross-context routing at low KV-cache overhead (4:1 query-to-KV compression).
* **2-Pass Physical Block Recycling**: passes token representations through 20 physical blocks twice with layer-index positional conditioning, producing 40 effective layers from a 20-block parameter budget.
* **Integrated Multi-Token Prediction (MTP)**: built-in speculative heads (k=2) verify subsequent tokens during generation without requiring external companion models.

---

## Architectural Specifications

### Parameter Allocation (101M Reference Configuration)

| Dimension | Specification | Notes |
| :--- | :--- | :--- |
| Total Parameters | 101,177,984 | Exact parameter count |
| Core Computation Parameters | 96,327,040 | 95.21% of total parameter budget |
| Vocabulary Tax (Embeddings) | 4,850,944 | 4.79% of total parameter budget |
| Vocabulary Size (V) | 32,768 | Byte-level BPE |
| Embedding Rank (d_emb) | 128 | Factorized input/output projections |
| Model Dimension (dim) | 640 | Hidden state width |
| Physical Blocks | 20 | 15 GDN-2 + 5 GQA |
| Number of Passes | 2 | Forward recurrence across physical blocks |
| Effective Depth | 40 layers | 2 passes x 20 physical blocks |
| Attention Query Heads | 10 heads | d_head = 64 |
| Attention KV Heads | 2 heads | 4:1 query-to-KV compression |
| FFN Intermediate Dimension | 1,728 | SwiGLU activation (8/3 x dim) |
| Recurrent Conv Kernel | 4 | 1D depthwise causal convolution |
| Max Context Window | 4,096 tokens | Extendable via RoPE theta scaling |
| Speculative Horizon | k=2 | Integrated auxiliary prediction heads |

### 4-Way Macro Architecture Comparison

| Metric | Maba v1.1 | Qwen 3.8 | Qwen 3.8 Flash Next | MiniCPM5 |
| :--- | :--- | :--- | :--- | :--- |
| Parameter Budget | ~101M | ~101M | ~101M | ~101M |
| Core Computation Ratio | **95.21%** | 75.00% | 74.99% | 100.0% |
| Recurrence Share | **75% (GDN-2)** | 75% (GDN) | 75% (GDN) | 0% (Pure Attention) |
| Quadratic Attention Share | **25% (GQA)** | 25% (GQA) | 25% (QSA Sparse) | 100% (GQA) |
| Physical Blocks | 20 blocks | 20 blocks | 20 blocks | 28 blocks |
| Effective Layer Depth | **40 layers** | 20 layers | 20 layers | 28 layers |
| KV-Cache Footprint (4k) | **10.0 MB (-76.2%)** | 10.0 MB (-76.2%) | 2.5 MB (-94.0%) | 42.0 MB (Baseline) |
| Speculative Heads | **Built-in MTP (k=2)** | Built-in MTP (k=2) | Built-in MTP (k=2) | None |

---

## Block Architecture

### 1. Factorized Token Embeddings
To avoid vocabulary parameters consuming core computation capacity, Maba factorizes the embedding matrix:
* W_emb: V x d_emb (32,768 x 128)
* W_proj_in: d_emb x dim (128 x 640)
* W_proj_out: dim x d_emb (640 x 128)
This reduces embedding parameters to 4.85M (4.79% of budget), leaving 95.21% of weights dedicated to sequence modeling.

### 2. GDN-2 Recurrence Block (75% of Layers)
The Gated DeltaNet layer computes an input-dependent recurrent update over state matrix S of size (d_head x d_head):
* 1D depthwise causal convolution over projected inputs (kernel size 4).
* Data-dependent decay gate beta_t = sigmoid(W_beta x_t + b_beta).
* State update: S_t = S_{t-1} * beta_t + v_t (x) k_t^T.
* State readout: o_t = S_t q_t.

### 3. GQA Quadratic Attention Block (25% of Layers)
Every fourth block (blocks 3, 7, 11, 15, 19) is a Grouped-Query Attention block:
* 10 query heads and 2 key-value heads.
* Per-head RMSNorm applied to query and key vectors before dot-product attention.
* Rotary Position Embedding (RoPE) with theta = 500,000.

### 4. Feed-Forward Network (SwiGLU)
Each block contains an intermediate SwiGLU FFN:
* FFN(x) = (SiLU(x W_gate) * x W_up) W_down
* dim = 640, d_ffn = 1,728.

### 5. Gated Residual Connections
Instead of fixed addition, residual streams use a learned gating mechanism:
* y = norm(x) + sigmoid(w_gate) * block(norm(x))
* Initialized with bias = 2.0 (sigmoid approx 0.88), ensuring stable gradient flow at initialization while allowing layers to dynamically regulate residual contribution.

---

## Native C++ Inference Engine

The repository includes a standalone C++ inference implementation in `cpp/`:
* Cache-aligned unit-stride row-major loop order for GDN-2 state updates.
* AVX2 / FMA vectorization with zero heap allocations during autoregressive generation.
* Step latency: 10.20 us per head update on x86_64.
* Numerical parity with PyTorch: maximum logit discrepancy strictly below 7.62e-5.

Build instructions:
```bash
cd cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
./test_numerical
```

---

## Verification Suite

The repository contains 105 automated unit and end-to-end tests:
* `tests/test_components.py`: Layer-level unit tests (RMSNorm, RoPE, SwiGLU, GDN-2, GQA, GatedRes, MTP, Newton-Schulz).
* `tests/test_e2e_suite.py`: Multi-tier verification (numerical stability, autograd continuity across all 366 tensors, state isolation, boundary sequence lengths).
* `tests/test_scaling.py`: Preset verification (50M, 100M, 300M, 1B, 3B, 7B, 30B).
* `tests/test_speculative_generation.py`: Speculative decoding cache invariance.
* `tests/verify_params.py`: Exact parameter budget accounting (101,177,984 total, 96,327,040 core).

Run tests:
```bash
pytest tests/
```

---

## Quickstart (Python)

### Installation
```bash
git clone https://github.com/ivan-dev35/maba-v1-architecture.git
cd maba-v1-architecture
pip install -e .
```

### Initializing the Architecture
```python
import torch
from maba.config import Config
from maba.model import Model

# Initialize 101M reference architecture
cfg = Config.from_preset("100M")
model = Model(cfg)

# Forward pass
tokens = torch.randint(0, cfg.vocab_size, (1, 64))
outputs = model(tokens)
logits = outputs["logits"]
print(f"Logits shape: {logits.shape}")  # [1, 64, 32768]
```

### High-Speed Speculative Generation
```python
from maba.generate import spec_gen
from maba.tokenizer import Tokenizer

tok = Tokenizer()
output_text, acceptance_rate, steps = spec_gen(
    model,
    tok,
    prompt="Artificial intelligence architecture design",
    max_new_tokens=64
)
```
