import os
import sys
import torch
from typing import Optional, Dict, Any

_TPU_CHECKED = False
_TPU_AVAILABLE = False

def is_tpu_available() -> bool:
    global _TPU_CHECKED, _TPU_AVAILABLE
    if _TPU_CHECKED:
        return _TPU_AVAILABLE

    _TPU_CHECKED = True
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        # Probe device availability
        dev = xm.xla_device()
        _TPU_AVAILABLE = True
    except Exception:
        _TPU_AVAILABLE = False
    return _TPU_AVAILABLE

def get_tpu_device(index: Optional[int] = None) -> torch.device:
    if not is_tpu_available():
        raise RuntimeError(
            "TPU accelerator requested, but torch_xla is not installed or no TPU hardware is accessible. "
            "Ensure you are running on a Google Cloud TPU VM, Kaggle TPU, or Colab TPU with PyTorch/XLA installed."
        )
    import torch_xla.core.xla_model as xm
    return xm.xla_device(index)

def get_device(dev_name: str = "auto") -> torch.device:
    name = dev_name.strip().lower()

    if name in ("tpu", "xla"):
        return get_tpu_device()

    if name == "auto":
        if is_tpu_available():
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return dev
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            dev = torch.device("cpu")
            threads = min(8, os.cpu_count() or 4)
            torch.set_num_threads(threads)
            return dev

    return torch.device(dev_name)

def get_dtype(device: torch.device, requested: Optional[str] = None) -> torch.dtype:
    if requested:
        r = requested.strip().lower()
        if r in ("bf16", "bfloat16"):
            return torch.bfloat16
        elif r in ("fp16", "float16"):
            return torch.float16
        elif r in ("fp32", "float32"):
            return torch.float32

    if device.type == "xla":
        return torch.bfloat16
    elif device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device.type == "mps":
        return torch.float16
    return torch.float32

def mark_step():
    """Forces execution of queued XLA computation graph on TPU."""
    if is_tpu_available():
        import torch_xla.core.xla_model as xm
        xm.mark_step()

def optimizer_step(optimizer, barrier: bool = True):
    """Executes optimizer step with cross-replica reduction on TPU."""
    if is_tpu_available():
        import torch_xla.core.xla_model as xm
        xm.optimizer_step(optimizer, barrier=barrier)
    else:
        optimizer.step()

def clip_grad_norm(parameters, max_norm: float = 1.0) -> torch.Tensor:
    """Clips gradient norms across parameters, supporting TPU and CPU/GPU."""
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

def wrap_loader(dataloader, device: torch.device):
    """Wraps PyTorch DataLoader with MpDeviceLoader for TPU asynchronous prefetching."""
    if device.type == "xla":
        try:
            from torch_xla.distributed.parallel_loader import MpDeviceLoader
            return MpDeviceLoader(dataloader, device)
        except Exception:
            return dataloader
    return dataloader

def is_master_process() -> bool:
    """Returns True if current process is rank 0 or not running in TPU distributed mode."""
    if is_tpu_available():
        try:
            import torch_xla.core.xla_model as xm
            return xm.is_master_ordinal(local=False)
        except Exception:
            return True
    return True

def master_print(*args, **kwargs):
    """Prints message only on rank 0."""
    if is_master_process():
        print(*args, **kwargs)

def rendezvous(tag: str):
    """Performs barrier synchronization across all TPU workers."""
    if is_tpu_available():
        try:
            import torch_xla.core.xla_model as xm
            xm.rendezvous(tag)
        except Exception:
            pass

def get_tpu_info() -> Dict[str, Any]:
    """Returns detailed diagnostic dictionary of TPU environment."""
    info = {
        "available": is_tpu_available(),
        "device": None,
        "world_size": 1,
        "ordinal": 0,
        "is_master": True,
        "xla_version": None,
        "pjrt_device": os.environ.get("PJRT_DEVICE", None)
    }
    if info["available"]:
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            info["device"] = str(xm.xla_device())
            info["world_size"] = xm.xrt_world_size()
            info["ordinal"] = xm.get_ordinal()
            info["is_master"] = xm.is_master_ordinal()
            info["xla_version"] = getattr(torch_xla, "__version__", "unknown")
        except Exception as e:
            info["error"] = str(e)
    return info

def get_hardware_status() -> Dict[str, Any]:
    """Returns full system hardware accelerator status."""
    dev = get_device("auto")
    opt_dtype = get_dtype(dev)
    tpu_info = get_tpu_info()
    return {
        "selected_device": str(dev),
        "optimal_dtype": str(opt_dtype).replace("torch.", ""),
        "tpu": tpu_info,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "cpu_threads": torch.get_num_threads()
    }

configure_hardware = get_device
get_optimal_dtype = get_dtype
