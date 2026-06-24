import os
import torch


def setup_gpu(device_id: int = 0) -> torch.device:
    """
    Returns a CUDA device. Aborts if CUDA is not available.
    Always call this at the top of every script and notebook.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. Check GPU drivers and PyTorch installation. "
            "This project requires an NVIDIA GPU (RTX 4060 confirmed)."
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    device = torch.device(f"cuda:{device_id}")

    name = torch.cuda.get_device_name(device_id)
    vram_gb = torch.cuda.get_device_properties(device_id).total_memory / 1e9
    print(f"[GPU] Using {name} | {vram_gb:.1f} GB VRAM | device=cuda:{device_id}")

    return device
