import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maba.config import Config
from maba.model import Model

def verify():
    cfg = Config()
    m = Model(cfg)
    c = m.count_params()

    expected = [
        ("W_emb", 32768 * 128, c["w_emb"]),
        ("W_proj_in", 128 * 640, c["w_proj_in"]),
        ("W_proj_out", 640 * 128, c["w_proj_out"]),
        ("Embedding Subtotal", 4358144, c["embedding_total"]),
        ("15 GDN-2 Blocks", 74803200, c["gdn2_blocks"]),
        ("5 GQA Blocks", 21523840, c["gqa_blocks"]),
        ("Computation Core", 96327040, c["core_total"]),
        ("Final RMSNorm", 640, c["final_norm"]),
        ("MTP Head (k=2)", 492160, c["mtp_head"]),
    ]

    ok = True
    for name, exp, act in expected:
        match = (exp == act)
        if not match:
            ok = False
        print(f"{name:<22} | exp: {exp:>10,} | act: {act:>10,} | {'OK' if match else 'FAIL'}")

    tot_exp = 101177984
    tot_act = c["total"]
    print("-" * 55)
    print(f"{'TOTAL':<22} | exp: {tot_exp:>10,} | act: {tot_act:>10,} | {'OK' if tot_exp == tot_act else 'FAIL'}")
    assert ok and tot_exp == tot_act, "Parameter count mismatch"

if __name__ == "__main__":
    verify()
