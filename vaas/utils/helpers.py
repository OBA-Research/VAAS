import json
import os


def require_torch():
    try:
        import torch
        import torchvision.transforms as T

        return torch, T
    except ImportError as err:
        raise SystemExit(
            "PyTorch is not installed.\n"
            "VAAS requires both PyTorch and torchvision.\n\n"
            "Install the correct PyTorch build for your system (CPU, CUDA, or ROCm):\n"
            "  https://pytorch.org/get-started/locally/\n\n"
            "Once PyTorch is installed, re-run your VAAS code."
        ) from err


# def require_torch():
#     try:
#         import torch
#         import torchvision.transforms as T
#     except ImportError as err:
#         if "torch" not in str(err):
#             raise
#         raise ImportError(
#             "PyTorch is not installed.\n"
#             "VAAS requires both PyTorch and torchvision.\n\n"
#             "Install PyTorch from:\n"
#             "  https://pytorch.org/get-started/locally/"
#         ) from err
#     return torch, T


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def check_CUDA_available():
    torch, _ = require_torch()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: cuda")
        print(f" - GPU 0: {torch.cuda.get_device_name(0)}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")
    return device
