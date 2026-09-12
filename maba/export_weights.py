import struct
import os
import torch
from .config import Config
from .model import Model

MAGIC = 0x4D414241

def export_bin(model: Model, out_path: str):
    cfg = model.config
    sd = model.state_dict()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    with open(out_path, "wb") as f:
        hdr = struct.pack(
            "<IIIIIIIIII",
            MAGIC,
            cfg.vocab_size,
            cfg.dim,
            cfg.d_emb,
            cfg.d_head,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.n_layers,
            cfg.d_ffn,
            len(sd)
        )
        f.write(hdr)

        for name, param in sd.items():
            nb = name.encode("utf-8")
            t = param.detach().cpu().float().contiguous()
            shape = list(t.shape)

            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", len(shape)))
            for d in shape:
                f.write(struct.pack("<I", d))
            f.write(t.numpy().tobytes())

    print(f"Exported {len(sd)} tensors ({os.path.getsize(out_path) / (1024*1024):.2f} MB) -> {out_path}")

export_weights_binary = export_bin

if __name__ == "__main__":
    m = Model(Config())
    export_bin(m, "maba_weights.bin")
