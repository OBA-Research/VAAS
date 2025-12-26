import os

from vaas.utils.helpers import require_torch


def load_px_checkpoint(model, checkpoint_dir):
    torch, _ = require_torch()

    ckpt_path = os.path.join(checkpoint_dir, "best_model_px.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])


def load_ref_stats(checkpoint_dir):
    torch, _ = require_torch()

    ref_path = os.path.join(checkpoint_dir, "ref_stats.pth")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Missing reference stats: {ref_path}")

    stats = torch.load(ref_path, map_location="cpu")
    return stats["mu_ref"], stats["sigma_ref"]
