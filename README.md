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
- scaling
- 100m
- 1b
- 3b
- 7b
- 30b
- pytorch
- safetensors
- cpp
- avx2
---

<p align="center">
  <img src="https://huggingface.co/AndrewThompson1233/maba-v1-architecture/resolve/main/assets/logo.svg" width="160" alt="Maba Logo" />
</p>

# Maba Architecture: Sub-Quadratic Hybrid Linear-Recurrent Attention

Official specification, scaling topology, and reference implementation of the Maba neural network architecture. Maba combines Gated DeltaNet linear recurrence (GDN-2) with Grouped-Query Attention (GQA), 2-pass physical block recycling, and native multi-token prediction (MTP) speculative decoding.

> [!NOTE]
> **Pretrained Weights and Evaluation Benchmarks**
> For trained model checkpoints, Safetensors weights, and empirical evaluation results on standard benchmarks (ARC, HellaSwag, Story Cloze), see the model repository:
> **[AndrewThompson1233/maba-101m](https://huggingface.co/AndrewThompson1233/maba-101m)**

---

<p align="center">
  <img src="https://huggingface.co/AndrewThompson1233/maba-v1-architecture/resolve/main/assets/architecture_comparison.svg" width="900" alt="Maba Architecture Feature Comparison" />
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

## Exact Parameter & Memory Breakdown (101M Reference Model)

### 1. Parameter Accounting

| Component | Sub-Layers | Exact Parameters | % of Total | Function |
| :--- | :--- | :---: | :---: | :--- |
| **Factorized Embedding** | W_emb (32,768 x 128) | 4,194,304 | 4.15% | Token lookup table |
| **Embedding Projections** | W_proj_in + W_proj_out | 163,840 | 0.16% | Rank 128 <-> Dim 640 |
| **Embedding Subtotal** | **Vocab Tax** | **4,358,144** | **4.31%** | **Static parameter overhead** |
| **15 GDN-2 Blocks** | Recurrence + SwiGLU FFN | 74,803,200 | 73.93% | Linear O(1) recurrence |
| **5 GQA Blocks** | Attention + SwiGLU FFN | 21,523,840 | 21.27% | Quadratic routing |
| **Computation Core** | **All 20 Physical Blocks** | **96,327,040** | **95.21%** | **Core sequence modeling** |
| **Final RMSNorm** | Layer normalization gain | 640 | <0.01% | Final feature variance scale |
| **MTP Auxiliary Head** | k=2 projection and norm | 492,160 | 0.49% | Native speculative decoding |
| **Total Architecture** | **Full Model Parameters** | **101,177,984** | **100.00%** | **Exact parameter count** |

### 2. Weight Memory Footprint by Precision

| Precision | Bytes per Parameter | Model Weights VRAM | Memory Footprint Notes |
| :--- | :---: | :---: | :--- |
| **FP32 (Full Precision)** | 4 bytes | **385.96 MB** | Default PyTorch weights |
| **BF16 / FP16 (Half Precision)** | 2 bytes | **192.98 MB** | Standard inference and training |
| **INT8 (Quantized)** | 1 byte | **96.49 MB** | Edge devices and embedded systems |
| **INT4 (GPTQ / AWQ)** | 0.5 bytes | **48.25 MB** | Microcontroller and mobile inference |

### 3. KV-Cache and Recurrent State Scaling

Maba separates state memory into constant recurrent state (GDN-2) and compressed quadratic attention cache (GQA 4:1):

| Context Length (Tokens) | Maba v1.1 GQA Cache | Maba v1.1 GDN-2 State | Maba v1.1 Total Cache | Pure Attention Baseline | Memory Reduction |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1,024 (1k)** | 2.50 MB | 1.17 MB (Fixed) | **3.67 MB** | 10.50 MB | **-65.0%** |
| **2,048 (2k)** | 5.00 MB | 1.17 MB (Fixed) | **6.17 MB** | 21.00 MB | **-70.6%** |
| **4,096 (4k)** | 10.00 MB | 1.17 MB (Fixed) | **11.17 MB** | 42.00 MB | **-73.4%** |
| **8,192 (8k)** | 20.00 MB | 1.17 MB (Fixed) | **21.17 MB** | 84.00 MB | **-74.8%** |
| **16,384 (16k)** | 40.00 MB | 1.17 MB (Fixed) | **41.17 MB** | 168.00 MB | **-75.5%** |
| **32,768 (32k)** | 80.00 MB | 1.17 MB (Fixed) | **81.17 MB** | 336.00 MB | **-75.8%** |
| **65,536 (64k)** | 160.00 MB | 1.17 MB (Fixed) | **161.17 MB** | 672.00 MB | **-76.0%** |
| **131,072 (128k)** | 320.00 MB | 1.17 MB (Fixed) | **321.17 MB** | 1,344.00 MB | **-76.1%** |

* Note: GDN-2 recurrent state is strictly O(1) constant: 15 blocks x 10 heads x (64 x 64 state) x 2 bytes = 1.17 MB. It never grows, regardless of sequence length.

---

## 4-Way Macro Architecture Comparison (~101M Parameters)

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

## Scaling Specifications (100M to 30B)

The Maba architecture scales systematically from on-device 100M to large-scale 30B parameters, supporting context horizons up to 131k tokens.

For complete multi-scale topology configurations, closed-form parameter derivation equations, and architectural audits against 2026 foundation models (Qwen3.5, Muse-30B, Gemma4), see the dedicated scaling specification:

**[SCALING.md](SCALING.md)**

---

## Block Architecture

### 1. Factorized Token Embeddings
To avoid vocabulary parameters consuming core computation capacity, Maba factorizes the embedding matrix:
* W_emb: V x d_emb (32,768 x 128)
* W_proj_in: d_emb x dim (128 x 640)
* W_proj_out: dim x d_emb (640 x 128)
This reduces embedding parameters to 4.36M (4.31% of budget), leaving 95.21% of weights dedicated to sequence modeling.

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

## Hardware Acceleration & Distributed Training

The architecture includes automated device detection and distributed execution in `maba/hardware.py`:
* **NVIDIA CUDA**: Multi-GPU training via PyTorch Distributed Data Parallel (DDP) with NCCL all-reduce.
* **Apple Silicon**: Metal Performance Shaders (MPS) auto-detection and acceleration.
* **Google Cloud TPU**: PyTorch/XLA auto-detection and execution.
* **x86_64 AVX2**: Fast CPU fallback with OpenMP multi-threading.
* **Hybrid Optimizer**: Built-in Muon (matrix parameters via Newton-Schulz orthogonalization) and AdamW (vectors and embeddings).

Check hardware status:
```bash
python3 -m maba.cli hardware
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
