import time
import torch
from typing import Tuple
from .model import Model
from .tokenizer import Tokenizer

@torch.no_grad()
def spec_gen(
    model: Model,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 40,
    device: str = "cpu"
) -> Tuple[str, float, int]:
    if max_new_tokens <= 0:
        return prompt, 0.0, 0

    model.eval()
    inp = torch.tensor([tokenizer.encode(prompt, add_bos=True)], device=device)
    cur = inp.clone()

    n_acc = 0
    steps = 0
    t0 = time.time()

    out = model(cur, return_states=True)
    states = out["states"]
    lg = out["logits"]
    nh = out["hidden_states"]

    t1 = torch.argmax(lg[:, -1, :], dim=-1, keepdim=True)
    last_h = nh[:, -1:, :]
    emb_t1 = model.embeddings.factor_emb(t1)
    mtp_lg = model.mtp_head(last_h, emb_t1, model.embeddings)
    d2 = torch.argmax(mtp_lg[:, -1, :], dim=-1, keepdim=True)

    while cur.shape[1] - inp.shape[1] < max_new_tokens:
        steps += 1
        pos = cur.shape[1]
        out1 = model(t1, states=states, return_states=True, start_pos=pos)
        lg1 = out1["logits"]
        nh1 = out1["hidden_states"]
        st1 = out1["states"]

        v_tok = torch.argmax(lg1[:, -1, :], dim=-1, keepdim=True)

        if v_tok.item() == d2.item():
            n_acc += 1
            cur = torch.cat([cur, t1, d2], dim=1)
            out2 = model(d2, states=st1, return_states=True, start_pos=pos + 1)
            states = out2["states"]
            nh2 = out2["hidden_states"]
            t1 = torch.argmax(out2["logits"][:, -1, :], dim=-1, keepdim=True)
            last_h = nh2[:, -1:, :]
            emb_t1 = model.embeddings.factor_emb(t1)
            mtp_lg = model.mtp_head(last_h, emb_t1, model.embeddings)
            d2 = torch.argmax(mtp_lg[:, -1, :], dim=-1, keepdim=True)
        else:
            cur = torch.cat([cur, t1], dim=1)
            states = st1
            t1 = v_tok
            last_h = nh1[:, -1:, :]
            emb_t1 = model.embeddings.factor_emb(t1)
            mtp_lg = model.mtp_head(last_h, emb_t1, model.embeddings)
            d2 = torch.argmax(mtp_lg[:, -1, :], dim=-1, keepdim=True)

    el = time.time() - t0
    txt = tokenizer.decode(cur[0].tolist())
    n_tok = cur.shape[1] - inp.shape[1]
    acc = n_acc / max(1, steps)

    print(f"Speculative: {n_tok} toks | {steps} steps | ratio {n_tok/max(1,steps):.2f}x | acc {acc*100:.1f}% | {el:.2f}s ({n_tok/max(1e-5, el):.1f} tok/s)")
    return txt, acc, steps

speculative_generate = spec_gen
