# Maba Architecture: Scaling Specification (100M - 7B)

Technical specification and architectural comparison for scaling the Maba architecture from 100M to 1B, 3B, and 7B parameters.

---

## 1. Scaling Topology Presets

| Metric | Maba-100M | Maba-1B | Maba-3B | Maba-7B |
| :--- | :--- | :--- | :--- | :--- |
| **Total Parameters** | 101,183,744 (101.2M) | 1,004,748,800 (1.00B) | 2,977,199,616 (2.98B) | 7,127,891,968 (7.13B) |
| **Core Parameters** | 96,332,800 (95.21%) | 982,528,000 (97.79%) | 2,941,345,792 (98.80%) | 7,071,916,032 (99.21%) |
| **Embedding Parameters** | 4,358,144 (4.31%) | 17,498,112 (1.74%) | 26,836,992 (0.90%) | 37,093,376 (0.52%) |
| **Vocab Tax** | 4.31% | 1.74% | 0.90% | 0.52% |
| **Vocabulary Size** | 32,768 | 64,256 | 64,256 | 64,256 |
| **Embedding Rank (d_emb)** | 128 | 256 | 384 | 512 |
| **Model Dimension (dim)** | 640 | 2048 | 2816 | 4096 |
| **Physical Blocks** | 20 | 20 | 32 | 36 |
| **Effective Depth (2-pass)** | 40 layers | 40 layers | 64 layers | 72 layers |
| **Block Ratio (GDN:GQA)** | 3:1 (15 GDN + 5 GQA) | 3:1 (15 GDN + 5 GQA) | 3:1 (24 GDN + 8 GQA) | 3:1 (27 GDN + 9 GQA) |
| **Query Heads (n_heads)** | 10 | 16 | 22 | 32 |
| **KV Heads (n_kv_heads)** | 2 | 4 | 4 | 8 |
| **Head Dimension (d_head)** | 64 | 128 | 128 | 128 |
| **FFN Dimension (d_ffn)** | 1728 | 5504 | 7488 | 11008 |
| **Conv Kernel (k_size)** | 4 | 4 | 4 | 4 |
| **Context Length (max_len)** | 4096 | 8192 | 16384 | 32768 |
| **Speculative Horizon** | k=2 (MTP) | k=2 (MTP) | k=2 (MTP) | k=2 (MTP) |

---

## 2. Comparison Against 2026 Small and Mid Architectures

### 100M Tier

| Feature | Maba-100M (2026) | Supra2-100M (2026) | SmolLM2-135M | MobileLLM-125M |
| :--- | :--- | :--- | :--- | :--- |
| **Backbone** | Maba (GDN-2 + GQA) | Qwen3 (Transformer) | Transformer | Transformer |
| **Total Parameters** | 101.18M | 100.68M | 135.0M | 125.0M |
| **Core Parameters** | 96.33M (95.2%) | 75.52M (75.0%) | 106.7M (79.0%) | 106.6M (85.3%) |
| **Vocab Tax** | 4.31% | 25.0% | 21.0% | 14.7% |
| **Attention / Recurrence** | 75% GDN-2 + 25% GQA | 100% Full Attention | 100% GQA | 100% GQA |
| **KV Cache Layers** | 5 layers (25%) | 12 layers (100%) | 30 layers (100%) | 30 layers (100%) |
| **Effective Depth** | 40 layers | 12 layers | 30 layers | 30 layers |
| **Speculative Horizon** | k=2 (native MTP) | k=1 | k=1 | k=1 |

### 1B Tier

| Feature | Maba-1B (2026) | IFM/K2-Horizon-0.9B (2026) | openbmb/MiniCPM5-1B (2026) | Llama-3.2-1B |
| :--- | :--- | :--- | :--- | :--- |
| **Backbone** | Maba (GDN-2 + GQA) | Transformer | Transformer | Transformer |
| **Total Parameters** | 1,004.7M (1.00B) | 920.8M (0.92B) | 840.4M (0.84B) | 1,235.8M (1.24B) |
| **Core Parameters** | 982.5M (97.8%) | 822.1M (89.3%) | 639.8M (76.1%) | 973.2M (78.8%) |
| **Vocab Tax** | 1.74% | 10.7% | 23.9% | 21.2% |
| **Vocabulary Size** | 64,256 (Rank 256) | 64,256 (Direct) | 130,560 (Direct) | 128,256 (Direct) |
| **Attention / Recurrence** | 75% GDN-2 + 25% GQA | 100% Full Attention | 100% Full Attention | 100% Full Attention |
| **KV Cache Layers** | 5 layers (25%) | 28 layers (100%) | 24 layers (100%) | 16 layers (100%) |
| **Effective Depth** | 40 layers | 28 layers | 24 layers | 16 layers |
| **Speculative Horizon** | k=2 (native MTP) | k=1 | k=1 | k=1 |

### 3B Tier

| Feature | Maba-3B (2026) | openbmb/MiniCPM5-2B (2026) | XHToken/Spark-X2.5-4B (2026) | Llama-3.2-3B |
| :--- | :--- | :--- | :--- | :--- |
| **Backbone** | Maba (GDN-2 + GQA) | Transformer | Spark Transformer | Transformer |
| **Total Parameters** | 2,977.2M (2.98B) | 2,246.5M (2.25B) | 3,758.1M (3.76B) | 3,212.7M (3.21B) |
| **Core Parameters** | 2,941.3M (98.8%) | 1,979.4M (88.1%) | 3,422.5M (91.1%) | 2,819.3M (87.8%) |
| **Vocab Tax** | 0.90% | 11.9% | 8.9% | 12.2% |
| **Vocabulary Size** | 64,256 (Rank 384) | 130,560 (Direct) | 131,072 (Direct) | 128,256 (Direct) |
| **Attention / Recurrence** | 75% GDN-2 + 25% GQA | 100% Full Attention | 100% Full Attention | 100% Full Attention |
| **KV Cache Layers** | 8 layers (25%) | 42 layers (100%) | 36 layers (100%) | 28 layers (100%) |
| **Effective Depth** | 64 layers | 42 layers | 36 layers | 28 layers |
| **Speculative Horizon** | k=2 (native MTP) | k=1 | k=1 | k=1 |

### 7B Tier

| Feature | Maba-7B (2026) | IFM/K2-Horizon-7B (2026) | Llama-3.1-8B | Qwen2.5-7B |
| :--- | :--- | :--- | :--- | :--- |
| **Backbone** | Maba (GDN-2 + GQA) | Transformer | Transformer | Transformer |
| **Total Parameters** | 7,127.9M (7.13B) | 7,974.7M (7.97B) | 8,030.3M (8.03B) | 7,615.6M (7.62B) |
| **Core Parameters** | 7,071.9M (99.2%) | 6,948.1M (87.1%) | 6,979.6M (86.9%) | 7,070.7M (92.8%) |
| **Vocab Tax** | 0.52% | 12.9% | 13.1% | 7.2% |
| **Vocabulary Size** | 64,256 (Rank 512) | 250,624 (Direct) | 128,256 (Direct) | 152,064 (Direct) |
| **Attention / Recurrence** | 75% GDN-2 + 25% GQA | 100% Full Attention | 100% Full Attention | 100% Full Attention |
| **KV Cache Layers** | 9 layers (25%) | 36 layers (100%) | 32 layers (100%) | 28 layers (100%) |
| **Effective Depth** | 72 layers | 36 layers | 32 layers | 28 layers |
| **Speculative Horizon** | k=2 (native MTP) | k=1 | k=1 | k=1 |

---

## 3. KV-Cache Memory Footprint

In standard Transformer architectures, every layer allocates key-value cache with memory complexity O(N).
In Maba, 75% of layers are GDN-2 recurrence blocks with fixed O(1) hidden states (dim x dim) that do not grow with sequence length. Only the 25% GQA layers store token history.

The table below reports single-stream FP16 KV-cache memory in megabytes (MB) across sequence lengths:

$$\text{KV Cache Size (bytes)} = 4 \times N_{\text{gqa}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times S$$

| Model Scale | Model | 32,768 Tokens | 65,536 Tokens | 131,072 Tokens | Reduction vs Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1B Tier** | **Maba-1B** | **320.0 MB** | **640.0 MB** | **1,280.0 MB** | **Reference** |
| 1B Tier | MiniCPM5-1B | 576.0 MB | 1,152.0 MB | 2,304.0 MB | -44.4% |
| 1B Tier | Llama-3.2-1B | 1,024.0 MB | 2,048.0 MB | 4,096.0 MB | -68.8% |
| 1B Tier | K2-Horizon-0.9B | 1,344.0 MB | 2,688.0 MB | 5,376.0 MB | -76.2% |
| **3B Tier** | **Maba-3B** | **512.0 MB** | **1,024.0 MB** | **2,048.0 MB** | **Reference** |
| 3B Tier | MiniCPM5-2B | 1,344.0 MB | 2,688.0 MB | 5,376.0 MB | -61.9% |
| 3B Tier | Spark-X2.5-4B | 2,880.0 MB | 5,760.0 MB | 11,520.0 MB | -82.2% |
| 3B Tier | Llama-3.2-3B | 3,584.0 MB | 7,168.0 MB | 14,336.0 MB | -85.7% |
| **7B Tier** | **Maba-7B** | **1,152.0 MB** | **2,304.0 MB** | **4,608.0 MB** | **Reference** |
| 7B Tier | Qwen2.5-7B | 1,792.0 MB | 3,584.0 MB | 7,168.0 MB | -35.7% |
| 7B Tier | Llama-3.1-8B | 4,096.0 MB | 8,192.0 MB | 16,384.0 MB | -71.9% |
| 7B Tier | K2-Horizon-7B | 4,608.0 MB | 9,216.0 MB | 18,432.0 MB | -75.0% |

---

## 4. Parameter Formulations

The exact parameter counts for all Maba configurations are defined by the following closed-form expressions:

### Factorized Embeddings
$$W_{\text{emb\_total}} = V \cdot d_{\text{emb}} + 2 \cdot d_{\text{emb}} \cdot d_{\text{model}}$$

### GDN-2 Recurrent Layer
$$P_{\text{gdn2}} = 4 d^2 + 3 d \cdot d_{\text{ffn}} + 3 d \cdot k_{\text{conv}} + 3 d \cdot n_{\text{heads}} + 6 d$$

### GQA Attention Layer
$$P_{\text{gqa}} = 2 d^2 + 2 d (n_{\text{kv\_heads}} \cdot d_{\text{head}}) + 3 d \cdot d_{\text{ffn}} + 8 d$$

### Auxiliary MTP Head (k=2)
$$P_{\text{mtp}} = d (d + d_{\text{emb}}) + d$$

### Total Model Parameters
$$P_{\text{total}} = W_{\text{emb\_total}} + N_{\text{gdn}} \cdot P_{\text{gdn2}} + N_{\text{gqa}} \cdot P_{\text{gqa}} + d_{\text{model}} + P_{\text{mtp}}$$

---

## 5. Instantiation and Zero-RAM Verification

### Python API

```python
from maba.config import Config
from maba.model import Model
import torch

# Load configuration preset
cfg_1b = Config.from_preset("1B")
cfg_3b = Config.from_preset("3B")
cfg_7b = Config.from_preset("7B")

# Zero-RAM algebraic parameter audit
audit = cfg_1b.compute_param_count()
print(f"Total: {audit['total']:,} | Vocab tax: {audit['vocab_tax_pct']}%")

# Memory-safe PyTorch instantiation on meta device
with torch.device("meta"):
    model_7b = Model(cfg_7b)
    meta_params = sum(p.numel() for p in model_7b.parameters())
    print(f"Instantiated 7B parameters on meta device: {meta_params:,}")
```

### CLI Command

```bash
# Audit 1B configuration
python3 -m maba.cli params --scale 1B

# Audit 3B configuration
python3 -m maba.cli params --scale 3B

# Audit 7B configuration
python3 -m maba.cli params --scale 7B
```
