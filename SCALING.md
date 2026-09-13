# Maba Architecture: Scaling and 2026 Architectural Comparison

Technical specification and architectural comparison for scaling the Maba architecture from 100M to 1B, 3B, 7B, and 30B parameters against 2026 frontier and edge architectures (Qwen3.5, Muse-Glimmer-30B, Gemma4).

---

## 1. Visual Comparison

<p align="center">
  <img src="https://huggingface.co/AndrewThompson1233/maba-v1-architecture/resolve/main/assets/scaling_comparison.svg" width="900" alt="Maba Multi-Scale Comparison Against 2026 Architectures" />
</p>

---

## 2. Scaling Topology Presets

| Metric | Maba-100M | Maba-1B | Maba-3B | Maba-7B | Maba-30B |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Parameters** | 101,177,984 (101.2M) | 1,004,729,600 (1.00B) | 2,977,156,608 (2.98B) | 7,127,820,544 (7.13B) | 29,039,812,864 (29.04B) |
| **Core Parameters** | 96,327,040 (95.21%) | 982,508,800 (97.79%) | 2,941,302,784 (98.80%) | 7,071,844,608 (99.21%) | 28,930,813,184 (99.62%) |
| **Embedding Parameters** | 4,358,144 (4.31%) | 17,498,112 (1.74%) | 26,836,992 (0.90%) | 37,093,376 (0.52%) | 59,572,224 (0.21%) |
| **Vocab Tax** | 4.31% | 1.74% | 0.90% | 0.52% | 0.21% |
| **Vocabulary Size (V)** | 32,768 | 64,256 | 64,256 | 64,256 | 64,256 |
| **Embedding Rank (d_emb)** | 128 | 256 | 384 | 512 | 768 |
| **Model Dimension (dim)** | 640 | 2048 | 2816 | 4096 | 6656 |
| **Physical Blocks** | 20 | 20 | 32 | 36 | 52 |
| **Effective Depth (2-pass)** | 40 layers | 40 layers | 64 layers | 72 layers | 104 layers |
| **Block Ratio (GDN:GQA)** | 3:1 (15 GDN + 5 GQA) | 3:1 (15 GDN + 5 GQA) | 3:1 (24 GDN + 8 GQA) | 3:1 (27 GDN + 9 GQA) | 3:1 (39 GDN + 13 GQA) |
| **Query Heads (n_heads)** | 10 | 16 | 22 | 32 | 52 |
| **KV Heads (n_kv_heads)** | 2 | 4 | 4 | 8 | 4 |
| **Head Dimension (d_head)** | 64 | 128 | 128 | 128 | 128 |
| **FFN Dimension (d_ffn)** | 1728 | 5504 | 7488 | 11008 | 19968 |
| **Conv Kernel (k_size)** | 4 | 4 | 4 | 4 | 4 |
| **Context Length (max_len)** | 4096 | 8192 | 16384 | 32768 | 131072 |
| **Speculative Horizon** | k=2 (Built-in MTP) | k=2 (Built-in MTP) | k=2 (Built-in MTP) | k=2 (Built-in MTP) | k=2 (Built-in MTP) |

---

## 3. Comparison Against 2026 Architectures

### Architectural Convergence: Maba vs Qwen3.5 Series (Alibaba 2026)

Both Maba and the Qwen3.5 foundation series converged on the identical 3:1 macro layout:
`3 x (Gated DeltaNet -> SwiGLU) -> 1 x (GQA Attention -> SwiGLU)`.

The decisive divergence lies in embedding topology and parameter allocation:

| Feature | Maba-1B (2026) | Qwen3.5-0.8B (2026) | Qwen3.5-2B (2026) | Maba-3B (2026) | Qwen3.5-4B (2026) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Backbone** | GDN-2 + GQA (3:1) | GDN + Attention (3:1) | GDN + Attention (3:1) | GDN-2 + GQA (3:1) | GDN + Attention (3:1) |
| **Total Parameters** | 1,004.7M (1.00B) | 800.0M (0.80B) | 2,000.0M (2.00B) | 2,977.2M (2.98B) | 4,000.0M (4.00B) |
| **Core Parameters** | 982.5M (97.79%) | 545.7M (68.21%) | 1,491.4M (74.57%) | 2,941.3M (98.80%) | 3,364.3M (84.11%) |
| **Embedding Parameters** | 17.5M (Rank 256) | 254.3M (Monolithic) | 508.6M (Monolithic) | 26.8M (Rank 384) | 635.7M (Monolithic) |
| **Vocab Parameter Tax** | **1.74%** | **31.79%** | **25.43%** | **0.90%** | **15.89%** |
| **Vocabulary Size** | 64,256 | 248,320 | 248,320 | 64,256 | 248,320 |
| **Physical Layers** | 20 (15 GDN + 5 GQA) | 24 (18 GDN + 6 GQA) | 24 (18 GDN + 6 GQA) | 32 (24 GDN + 8 GQA) | 32 (24 GDN + 8 GQA) |
| **Effective Depth** | 40 layers (2-pass) | 24 layers | 24 layers | 64 layers (2-pass) | 32 layers |
| **KV Cache (131k FP16)** | **1,280.0 MB** | 768.0 MB | 1,536.0 MB | **2,048.0 MB** | 4,096.0 MB |
| **Speculative Decoding** | Built-in MTP (k=2) | Built-in MTP | Built-in MTP | Built-in MTP (k=2) | Built-in MTP |

### 30B Tier (27B - 31B)

| Feature | Maba-30B (2026) | Muse-Glimmer-30B (Meta 2026) | Qwen3.8-27B (2026) | Gemma4-31B (Google 2026) |
| :--- | :--- | :--- | :--- | :--- |
| **Backbone Architecture** | GDN-2 Recurrence + GQA | Dense Transformer + Local Window | Linear Attention + Full Attention | Dense Transformer |
| **Total Parameters** | 29.04B | 29.6B (incl. 1.8B ViT-G) | 27.2B | 30.7B |
| **Core Computation Parameters** | 28.93B (99.62%) | 26.9B (90.88%) | 24.6B (90.30%) | 27.1B (88.35%) |
| **Vocab Parameter Tax** | 0.21% (Rank 768) | 9.12% (Untied 202k) | 9.70% (Untied 248k) | 11.65% (Direct 256k) |
| **Layer Mixture** | 75% GDN-2 + 25% GQA | 75% Local Window (2k) + 25% Global | 75% Linear Attn + 25% Full Attn | 100% Full Attention |
| **Attention Layers With State Growth** | 13 layers (25%) | 13 global + 39 local window | 16 layers (25%) | 54 layers (100%) |
| **KV Cache Footprint (131k FP16)** | 3.5 GB | 4.5 GB (1.8 GB + 2.7 GB drafter) | 8.6 GB | 28.3 GB |
| **Speculative Decoding Mechanism** | Built-in MTP (k=2, 0 extra VRAM) | DFlash block-diffusion (5-layer companion) | MTP auxiliary layer (k=2) | External drafter / k=1 |
| **Effective Layer Depth** | 104 layers (2-pass block) | 52 layers | 64 layers | 54 layers |

### Mid Tier (7B - 9B)

| Feature | Maba-7B (2026) | Qwen3.5-9B (2026) | Qwen3-8B (2026) | IFM/K2-Horizon-7B (2026) |
| :--- | :--- | :--- | :--- | :--- |
| **Backbone Architecture** | GDN-2 Recurrence + GQA | GDN + GQA Hybrid (3:1) | Dense Transformer | Dense Transformer |
| **Total Parameters** | 7,127.9M (7.13B) | 8,800.0M (8.80B) | 8,280.4M (8.28B) | 7,974.7M (7.97B) |
| **Core Computation Parameters** | 7,071.9M (99.21%) | 7,782.9M (88.44%) | 7,035.8M (84.97%) | 6,948.1M (87.13%) |
| **Vocab Parameter Tax** | 0.52% (Rank 512) | 11.56% (248k Vocab) | 15.03% (Untied 152k Vocab) | 12.87% (Tied 250k Vocab) |
| **Layer Mixture** | 75% GDN-2 + 25% GQA | 75% GDN + 25% GQA | 100% Full Attention | 100% Full Attention |
| **KV Cache Layers** | 9 layers (25%) | 8 layers (25%) | 36 layers (100%) | 36 layers (100%) |
| **KV Cache Footprint (131k FP16)** | 4,608.0 MB (4.6 GB) | 4,096.0 MB (4.1 GB) | 19,327.4 MB (19.3 GB) | 18,432.0 MB (18.4 GB) |
| **Speculative Decoding** | Built-in MTP (k=2) | Built-in MTP | k=1 | k=1 |
| **Effective Layer Depth** | 72 layers | 32 layers | 36 layers | 36 layers |

---

## 4. KV-Cache Memory Footprint

In quadratic attention architectures, every layer allocates key-value cache with memory complexity O(N).
In Maba, 75% of layers are GDN-2 linear recurrence blocks with a constant hidden state (dim x dim) that does not grow with context length. Only the 25% GQA layers store token history.

The table below reports single-stream FP16 KV-cache memory in megabytes (MB) across sequence lengths:

$$\mathrm{KV~Cache~Memory} = 4 \times N_{\mathrm{gqa}} \times n_{\mathrm{kv}} \times d_{\mathrm{head}} \times S$$

| Model Scale | Model | 32,768 Tokens | 65,536 Tokens | 131,072 Tokens | Reduction vs Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **30B Tier** | **Maba-30B** | **873.8 MB** | **1,747.6 MB** | **3,495.3 MB (3.5 GB)** | **Reference** |
| 30B Tier | Muse-Glimmer-30B | 1,126.0 MB | 2,252.0 MB | 4,504.0 MB (4.5 GB) | -22.4% |
| 30B Tier | Qwen3.8-27B | 2,147.5 MB | 4,295.0 MB | 8,589.9 MB (8.6 GB) | -59.3% |
| 30B Tier | Gemma4-31B | 7,077.9 MB | 14,155.8 MB | 28,311.6 MB (28.3 GB) | -87.7% |
| **7B Tier** | **Maba-7B** | **1,152.0 MB** | **2,304.0 MB** | **4,608.0 MB (4.6 GB)** | **Reference** |
| 7B Tier | IFM/K2-Horizon-7B | 4,608.0 MB | 9,216.0 MB | 18,432.0 MB (18.4 GB) | -75.0% |
| 7B Tier | Qwen3-8B | 4,831.8 MB | 9,663.7 MB | 19,327.4 MB (19.3 GB) | -76.2% |
| **3B Tier** | **Maba-3B** | **512.0 MB** | **1,024.0 MB** | **2,048.0 MB (2.0 GB)** | **Reference** |
| 3B Tier | MiniCPM5-2B | 1,344.0 MB | 2,688.0 MB | 5,376.0 MB (5.4 GB) | -61.9% |
| 3B Tier | Spark-X2.5-4B | 2,880.0 MB | 5,760.0 MB | 11,520.0 MB (11.5 GB) | -82.2% |
| **1B Tier** | **Maba-1B** | **320.0 MB** | **640.0 MB** | **1,280.0 MB (1.3 GB)** | **Reference** |
| 1B Tier | Qwen3.5-2B | 384.0 MB | 768.0 MB | 1,536.0 MB (1.5 GB) | -16.7% |
| 1B Tier | MiniCPM5-1B | 576.0 MB | 1,152.0 MB | 2,304.0 MB (2.3 GB) | -44.4% |
| 1B Tier | K2-Horizon-0.9B | 1,344.0 MB | 2,688.0 MB | 5,376.0 MB (5.4 GB) | -76.2% |

---

## 5. Mathematical Parameter Formulations

The exact parameter counts for all Maba configurations are defined by the following closed-form expressions (with zero KaTeX mode errors):

### Factorized Embeddings
$$W_{\mathrm{emb}} = V \cdot d_{\mathrm{emb}} + 2 \cdot d_{\mathrm{emb}} \cdot d_{\mathrm{model}}$$

### GDN-2 Recurrent Block
$$P_{\mathrm{gdn2}} = 4 d^2 + 3 d \cdot d_{\mathrm{ffn}} + 3 d \cdot k_{\mathrm{conv}} + 3 d \cdot n_{\mathrm{heads}} + 6 d$$

### GQA Attention Block
$$P_{\mathrm{gqa}} = 2 d^2 + 2 d (n_{\mathrm{kv}} \cdot d_{\mathrm{head}}) + 3 d \cdot d_{\mathrm{ffn}} + 8 d$$

### Auxiliary Multi-Token Prediction Head (k=2)
$$P_{\mathrm{mtp}} = d (d + d_{\mathrm{emb}}) + d$$

### Total Architecture Parameters
$$P_{\mathrm{total}} = W_{\mathrm{emb}} + N_{\mathrm{gdn}} \cdot P_{\mathrm{gdn2}} + N_{\mathrm{gqa}} \cdot P_{\mathrm{gqa}} + d_{\mathrm{model}} + P_{\mathrm{mtp}}$$

---

## 6. Instantiation and Zero-RAM Verification

### Python API

```python
from maba.config import Config
from maba.model import Model
import torch

# Load configuration presets
cfg_1b = Config.from_preset("1B")
cfg_3b = Config.from_preset("3B")
cfg_7b = Config.from_preset("7B")
cfg_30b = Config.from_preset("30B")

# Zero-RAM algebraic parameter audit
audit = cfg_30b.compute_param_count()
print(f"Total: {audit['total']:,} | Vocab tax: {audit['vocab_tax_pct']}% | Core: {audit['core_pct']}%")

# Memory-safe PyTorch instantiation on meta device (0 bytes allocated in RAM)
with torch.device("meta"):
    model_30b = Model(cfg_30b)
    meta_params = sum(p.numel() for p in model_30b.parameters())
    print(f"Instantiated 30B parameters on meta device: {meta_params:,}")
```

### CLI Command

```bash
# Audit 100M configuration
python3 -m maba.cli params --scale 100M

# Audit 1B configuration
python3 -m maba.cli params --scale 1B

# Audit 3B configuration
python3 -m maba.cli params --scale 3B

# Audit 7B configuration
python3 -m maba.cli params --scale 7B

# Audit 30B configuration
python3 -m maba.cli params --scale 30B
```
