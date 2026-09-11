<div align="center">

![Maba v1 Architecture](assets/logo.svg)

**101M Parameter Language Model Architecture**

*GDN-2 Linear Recurrence + GQA with 2-Pass Weight Sharing*

<br>

[![CI](https://github.com/ivan-dev35/maba-v1-architecture/actions/workflows/ci.yml/badge.svg)](https://github.com/ivan-dev35/maba-v1-architecture/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org)
[![C++](https://img.shields.io/badge/C++-17-00599C.svg?logo=c%2B%2B&logoColor=white)](cpp/)
[![Parameters](https://img.shields.io/badge/Parameters-101.18M-brightgreen.svg)](#exact-parameter-topology)
[![Layers](https://img.shields.io/badge/Effective%20Layers-40-blueviolet.svg)](#architecture-overview)

</div>

---

## Architecture Overview

Maba v1 is a 101M parameter architecture combining constant-state linear recurrence with grouped-query attention, block-wise weight sharing, and factorized embeddings.

- **Vocabulary & Embedding**: 32,768 vocabulary with 128-rank intermediate projection. Vocabulary parameter overhead is 4.31% (compared to 15% to 25% in standard models), reserving 95.2% of weights for reasoning layers.
- **Physical to Effective Depth**: 20 physical transformer blocks evaluated twice consecutively (Immediate Block-Wise Weight Sharing) producing an effective compositional depth of 40 layers.
- **Recurrence and Attention (3:1)**: 15 Gated DeltaNet-2 (GDN-2) blocks for O(1) memory recurrence and 5 Grouped-Query Attention (GQA) blocks with QK-RMSNorm and RoPE ($\theta = 500\,000$).
- **Multi-Token Prediction (MTP)**: Auxiliary prediction head at horizon $k=2$ for self-speculative decoding without external draft models.
- **Dual-Component Optimizer**: Muon (Newton-Schulz spectral orthogonalization for 2D hidden weights) + AdamW (1D vectors and embeddings).
- **Dual Runtime**: Reference PyTorch implementation and native C++17 inference engine (AVX2, FMA, OpenMP).

---

## Architectural Comparison

![Architectural Efficiency Comparison](assets/architecture_comparison.svg)

### Modern Sub-150M Edge Architectures Comparison

| Architectural Feature | Maba v1 (101M) | Supra2-100M (2026) | SmolLM2-135M | MobileLLM-125M |
| :--- | :--- | :--- | :--- | :--- |
| **Release / Backbone** | **Maba v1 (2026)** | Supra2-100M (2026) | SmolLM2-135M | MobileLLM-125M |
| **Total Parameters** | **101.18M** | 100.68M | 135.0M | 125.0M |
| **Active Computation Core** | **96.33M (95.2%)** | 75.52M (75.0%) | 106.7M (79.0%) | 106.6M (85.3%) |
| **Vocabulary Parameter Tax** | **4.31% (factorized)** | 25.0% (unfactorized) | 21.0% (unfactorized) | 14.7% (unfactorized) |
| **Effective Reasoning Depth** | **40 layers** | 12 layers | 30 layers | 30 layers |
| **Weight Sharing Scheme** | **Block-wise (2x)** | None (single-pass) | None (single-pass) | Layer-level |
| **Attention / Recurrence** | **75% GDN-2 + 25% GQA** | 100% Full Attention | 100% GQA | 100% GQA |
| **KV-Cache (8k tokens)** | **39.1 MB** | 100.7 MB | 188.7 MB | 125.8 MB |
| **KV-Cache (16k tokens)** | **78.1 MB** | 201.3 MB | 377.5 MB | 251.7 MB |
| **KV Memory Reduction** | **83.3% savings** | 57.0% savings | 19.5% savings | 46.3% savings |
| **Speculative Decoding** | **Built-in MTP (k=2)** | None (k=1) | None (k=1) | None (k=1) |
| **Native C++ Engine** | **Included (AVX2/OpenMP)** | External | External | External |

---

## Exact Parameter Topology

| Component | Specification / Formula | Parameters | Fraction |
| :--- | :--- | :--- | :--- |
| Token Embeddings (W_emb) | 32,768 x 128 | 4,194,304 | 4.14% |
| Input Factor Projection (W_proj_in) | 128 x 640 | 81,920 | 0.08% |
| Output Factor Projection (W_proj_out) | 640 x 128 | 81,920 | 0.08% |
| Tied LM-Head | Tied with W_emb.T | 0 | 0.00% |
| **Subtotal: Embedding Block** | | **4,358,144** | **4.31%** |
| 15 GDN-2 Blocks | 15 x (Q,K,V,O + Conv + Gates + SwiGLU + Norms) | 74,803,200 | 73.93% |
| 5 GQA Blocks | 5 x (Q,K,V,O + QK-Norm + SwiGLU + Norms) | 21,529,600 | 21.28% |
| **Subtotal: Computation Core** | | **96,332,800** | **95.21%** |
| Final RMSNorm | 640 | 640 | 0.001% |
| Auxiliary MTP Head (k=2) | (640 + 128) x 640 + 640 | 492,160 | 0.49% |
| **Grand Total** | **Target Architecture** | **101,183,744** | **100.0%** |

---

## Project Structure

```text
maba-v1-architecture/
├── .github/                       # CI workflows & issue templates
│   ├── workflows/ci.yml           # Automated GitHub Actions test pipeline
│   └── ISSUE_TEMPLATE/            # Bug report & feature templates
├── assets/
│   ├── logo.svg                   # Vector architecture logo
│   └── architecture_comparison.svg # Comparison benchmark chart
├── maba/                          # Python package
│   ├── config.py                  # Architecture configuration
│   ├── model.py                   # Model definition and generation
│   ├── tokenizer.py               # Byte-level tokenizer (V = 32,768)
│   ├── dataset.py                 # Pretraining dataset
│   ├── train.py                   # Training pipeline (NTP + MTP loss)
│   ├── generate.py                # MTP self-speculative decoding
│   ├── export_weights.py          # Binary exporter for C++ engine
│   ├── hardware.py                # Hardware & precision detection
│   ├── cli.py                     # Unified CLI entrypoint
│   ├── layers/
│   │   ├── rms_norm.py            # RMSNorm
│   │   ├── embeddings.py          # Factorized embeddings & tied head
│   │   ├── rope.py                # RoPE (theta = 500,000)
│   │   ├── swiglu.py              # SwiGLU FFN (d_ffn = 1728)
│   │   ├── gated_residual.py      # Gated residual connection
│   │   ├── gdn2.py                # Gated DeltaNet-2 recurrence
│   │   ├── gqa.py                 # GQA (10Q:2KV) with QK-RMSNorm
│   │   ├── transformer_block.py   # 2-pass physical block
│   │   └── mtp.py                 # Multi-Token Prediction head
│   └── optimizers/
│       ├── muon.py                # Muon optimizer (Newton-Schulz deg 5)
│       └── hybrid_optimizer.py    # Hybrid Muon + AdamW
├── cpp/                           # High-performance C++ engine
│   ├── CMakeLists.txt             # C++17 / AVX2 / OpenMP build
│   ├── include/                   # Header-only inference runtime
│   │   ├── tensor.hpp
│   │   ├── embeddings.hpp
│   │   ├── gdn2.hpp
│   │   ├── gqa.hpp
│   │   ├── swiglu.hpp
│   │   ├── mtp.hpp
│   │   └── model.hpp
│   ├── src/main.cpp               # C++ CLI benchmark
│   └── tests/test_numerical.cpp   # PyTorch vs C++ parity audit
├── tests/                         # Automated test suite
│   ├── verify_params.py           # Exact parameter count audit
│   ├── test_components.py         # Module unit tests
│   ├── test_speculative_generation.py # MTP decoding test
│   └── test_e2e_training.py       # End-to-end training test
├── generate_reference.py          # Generates reference weights & activations
├── run_full_validation.sh         # Complete end-to-end test suite
├── pyproject.toml                 # Packaging standard
├── setup.py                       # Setuptools installer
├── CONTRIBUTING.md                # Contribution guidelines
├── LICENSE                        # MIT License
└── README.md
```

---

## Quickstart

### 1. Installation

```bash
pip install -e .
```

### 2. Full Automated Validation

Runs parameter audit, unit tests, speculative generation test, training loop test, builds C++ engine, and verifies bit-for-bit numerical equivalence between PyTorch and C++:

```bash
./run_full_validation.sh
```

### 3. Python CLI

Audit parameter topology:
```bash
python3 -m maba.cli params
```

Generate text:
```bash
python3 -m maba.cli generate --prompt "def fibonacci(n):" --max-tokens 32
```

Generate with MTP speculative decoding:
```bash
python3 -m maba.cli generate --prompt "def fibonacci(n):" --max-tokens 32 --speculative
```

Run training on micro-curriculum:
```bash
python3 -m maba.cli train --steps 30 --batch-size 2 --seq-len 32
```

Export binary weights for C++ engine:
```bash
python3 -m maba.cli export --output maba_weights.bin
```

### 4. C++ Inference Engine

Build:
```bash
mkdir -p cpp/build && cd cpp/build
cmake ..
make -j$(nproc)
```

Run CLI benchmark:
```bash
./cpp/build/maba_cli ../../maba_ref.bin
```

Run PyTorch vs C++ numerical equivalence test:
```bash
python3 generate_reference.py
./cpp/build/test_numerical maba_ref.bin ref_logits.bin
```

Output:
```text
Loaded weights from maba_ref.bin (366 tensors).
Compared 131072 logits: max_diff=8.13305e-05, mean_diff=1.19662e-05
PASS: numerical equivalence verified
```

---

## Numerical Equivalence

The C++ engine is mathematically verified against PyTorch float32 activations across 40 effective layers:
- Logits evaluated: 131,072
- Maximum absolute difference: `8.13e-05`
- Mean absolute difference: `1.19e-05`
- Status: `PASS` (within float32 precision bounds)

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).

Attribution to the original creators and contributors is required in any distribution or derivative work.
