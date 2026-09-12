import os
import time
import math
from typing import Optional, Union, Dict, Any
import torch
from torch.utils.data import DataLoader, DistributedSampler
from .config import Config
from .model import Model
from .optimizers import HybridOpt
from .dataset import Dataset
from .tokenizer import Tokenizer
from .hardware import (
    get_device,
    get_dtype,
    is_tpu_available,
    mark_step,
    optimizer_step,
    clip_grad_norm,
    wrap_loader,
    is_master_process,
    master_print,
    rendezvous,
    reduce_gradients
)

def train(
    n_steps: int = 50,
    batch_size: int = 4,
    seq_len: int = 64,
    ckpt_path: str = "maba_checkpoint.pt",
    device: Union[str, torch.device] = "auto",
    dtype: Optional[Union[str, torch.dtype]] = None,
    scale: str = "100M"
):
    if isinstance(device, str):
        dev = get_device(device)
    else:
        dev = device

    target_dtype = get_dtype(dev, requested=dtype if isinstance(dtype, str) else None)
    if isinstance(dtype, torch.dtype):
        target_dtype = dtype

    dev_str = str(dev).upper()
    master_print(f"Training Maba ({scale.upper()}) on {dev_str} [dtype={str(target_dtype).replace('torch.', '')}]")

    cfg = Config.from_preset(scale)
    model = Model(cfg).to(device=dev, dtype=target_dtype)

    counts = model.count_params()
    master_print(f"Params: {counts['total']:,} (emb: {counts['embedding_total']:,}, core: {counts['core_total']:,})")

    tok = Tokenizer()
    ds = Dataset(tokenizer=tok, seq_len=seq_len, repeat=20)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    device_loader = wrap_loader(dl, dev)

    opt = HybridOpt(model, lr_muon=0.02, wd_muon=0.01, lr_adamw=1.5e-3, wd_adamw=0.1)

    model.train()
    step = 0
    t0 = time.time()
    data_iter = iter(device_loader)
    init_loss = None
    last_loss = None

    while step < n_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(device_loader)
            batch = next(data_iter)

        x = batch["input_ids"].to(dev)
        y = batch["labels"].to(dev)

        opt.zero_grad()
        out = model(x, labels=y)
        loss_dict = out["loss"]
        tot_loss = loss_dict["total_loss"]
        main_loss = loss_dict["main_loss"]
        mtp_loss = loss_dict["mtp_loss"]

        tot_loss.backward()
        reduce_gradients(opt)
        clip_grad_norm(model.parameters(), 1.0)
        opt.step(barrier=True)
        mark_step()

        del out, loss_dict
        opt.zero_grad(set_to_none=True)

        step += 1
        if step % 10 == 0 or step == 1 or step == n_steps:
            t_val = tot_loss.item()
            m_val = main_loss.item()
            mtp_val = mtp_loss.item()
            if init_loss is None:
                init_loss = t_val
            last_loss = t_val
            el = time.time() - t0
            ppl = math.exp(min(m_val, 20.0))
            master_print(f"step {step:3d}/{n_steps:3d} | loss {t_val:.4f} | ntp {m_val:.4f} | mtp {mtp_val:.4f} | ppl {ppl:.2f} | {el:.1f}s")
        del tot_loss, main_loss, mtp_loss

    master_print(f"Loss: {init_loss:.4f} -> {last_loss:.4f}")

    if is_master_process():
        os.makedirs(os.path.dirname(ckpt_path) if os.path.dirname(ckpt_path) else ".", exist_ok=True)
        cpu_sd = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save({
            "config": cfg,
            "model_state_dict": cpu_sd,
            "step": step,
            "loss": last_loss
        }, ckpt_path)

        prompt = "class Tensor"
        p_ids = torch.tensor([tok.encode(prompt, add_bos=True)], device=dev)
        out_ids = model.generate(p_ids, max_new_tokens=15, temperature=0.7)
        master_print(f"Sample generation: {tok.decode(out_ids[0].tolist())}")

    rendezvous("training_end")
    return model, last_loss

def _mp_tpu_fn(index: int, args_dict: Dict[str, Any]):
    import torch_xla.core.xla_model as xm
    dev = xm.xla_device()

    scale = args_dict.get("scale", "100M")
    cfg = Config.from_preset(scale)
    target_dtype = torch.bfloat16
    model = Model(cfg).to(device=dev, dtype=target_dtype)

    seq_len = args_dict.get("seq_len", 64)
    batch_size = args_dict.get("batch_size", 4)
    n_steps = args_dict.get("n_steps", 50)
    ckpt_path = args_dict.get("ckpt_path", "maba_checkpoint.pt")

    tok = Tokenizer()
    ds = Dataset(tokenizer=tok, seq_len=seq_len, repeat=20)
    sampler = DistributedSampler(
        ds,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    dl = DataLoader(ds, batch_size=batch_size, sampler=sampler)
    device_loader = wrap_loader(dl, dev)

    opt = HybridOpt(model, lr_muon=0.02, wd_muon=0.01, lr_adamw=1.5e-3, wd_adamw=0.1)

    model.train()
    step = 0
    t0 = time.time()
    data_iter = iter(device_loader)
    init_loss = None
    last_loss = None

    while step < n_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(device_loader)
            batch = next(data_iter)

        x = batch["input_ids"].to(dev)
        y = batch["labels"].to(dev)

        opt.zero_grad()
        out = model(x, labels=y)
        loss_dict = out["loss"]
        tot_loss = loss_dict["total_loss"]
        main_loss = loss_dict["main_loss"]
        mtp_loss = loss_dict["mtp_loss"]

        tot_loss.backward()
        reduce_gradients(opt)
        clip_grad_norm(model.parameters(), 1.0)
        opt.step(barrier=True)
        mark_step()

        del out, loss_dict
        opt.zero_grad(set_to_none=True)

        step += 1
        if step % 10 == 0 or step == 1 or step == n_steps:
            t_val = tot_loss.item()
            m_val = main_loss.item()
            mtp_val = mtp_loss.item()
            if init_loss is None:
                init_loss = t_val
            last_loss = t_val
            if xm.is_master_ordinal():
                el = time.time() - t0
                ppl = math.exp(min(m_val, 20.0))
                print(f"TPU core {index} | step {step:3d}/{n_steps:3d} | loss {t_val:.4f} | ntp {m_val:.4f} | mtp {mtp_val:.4f} | ppl {ppl:.2f} | {el:.1f}s")
        del tot_loss, main_loss, mtp_loss

    if xm.is_master_ordinal():
        print(f"TPU Multi-Core Loss: {init_loss:.4f} -> {last_loss:.4f}")
        os.makedirs(os.path.dirname(ckpt_path) if os.path.dirname(ckpt_path) else ".", exist_ok=True)
        cpu_sd = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save({
            "config": cfg,
            "model_state_dict": cpu_sd,
            "step": step,
            "loss": last_loss
        }, ckpt_path)

    xm.rendezvous("mp_tpu_training_end")

def train_tpu_multicore(
    num_cores: int = 8,
    n_steps: int = 50,
    batch_size: int = 4,
    seq_len: int = 64,
    ckpt_path: str = "maba_checkpoint.pt",
    scale: str = "100M"
):
    """Launches parallel distributed training across multiple TPU cores."""
    if not is_tpu_available():
        raise RuntimeError(
            "Multi-core TPU training requested, but torch_xla is not installed or no TPU hardware is accessible."
        )
    import torch_xla.distributed.xla_multiprocessing as xmp
    args_dict = {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "ckpt_path": ckpt_path,
        "scale": scale
    }
    print(f"Spawning distributed training across {num_cores} TPU cores...")
    xmp.spawn(_mp_tpu_fn, args=(args_dict,), nprocs=num_cores)

if __name__ == "__main__":
    dev = "tpu" if is_tpu_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    train(n_steps=30, batch_size=2, seq_len=32, device=dev)
