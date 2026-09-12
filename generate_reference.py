import struct
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from maba.config import Config
from maba.model import Model
from maba.export_weights import export_bin

def generate_reference(
    bin_path: str = "maba_ref.bin",
    ref_logits_path: str = "ref_logits.bin"
):
    torch.manual_seed(42)
    cfg = Config()
    model = Model(cfg)
    model.eval()

    export_bin(model, bin_path)

    tokens = [1, 100, 200, 300]
    inp = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        out = model(inp)
        logits = out["logits"][0]

    L, V = logits.shape
    with open(ref_logits_path, "wb") as f:
        f.write(struct.pack("<II", L, V))
        f.write(logits.cpu().float().numpy().tobytes())

    print(f"Exported reference: {bin_path} and {ref_logits_path}")

if __name__ == "__main__":
    generate_reference()
