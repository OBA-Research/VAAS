import random

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


@torch.no_grad()
def compute_reference_stats(dataset, vit_model, device, transform, max_samples=200):
    """
    Computes reference attention statistics (mu_ref, sigma_ref) from authentic
    samples if available. Falls back to using tampered images
    or default constants if necessary.
    """
    print("Computing reference attention statistics...")
    authentic_indices = [i for i, m in enumerate(dataset.mask_paths) if m == "blank"]

    # Case 1: Authentic samples available
    if len(authentic_indices) > 0:
        print(
            f"Found {len(authentic_indices)}"
            "authentic samples for reference statistics."
        )
        selected_indices = random.sample(
            authentic_indices, min(len(authentic_indices), max_samples)
        )

    # Case 2: No authentic samples (e.g., DF2023)
    elif len(authentic_indices) == 0:
        print(
            "No authentic samples found — "
            "falling back to tampered samples for reference stats."
        )
        selected_indices = random.sample(
            range(len(dataset)), min(len(dataset), max_samples)
        )

    attn_means = []
    for idx in tqdm(selected_indices, total=len(selected_indices)):
        img_path = dataset.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)

        try:
            outputs = vit_model(
                img_t, output_attentions=True, interpolate_pos_encoding=True
            )
            attn_maps = outputs.attentions
            attn_mean_layers = torch.stack(
                [a.mean(dim=1)[:, 0, 1:] for a in attn_maps]
            ).mean(dim=0)
            mu = attn_mean_layers.mean().item()
            attn_means.append(mu)
        except Exception as e:
            print(f"Skipping sample {idx} due to error: {e}")

    # Case 3: No usable samples at all
    if len(attn_means) == 0:
        print(
            "Could not compute valid reference stats, "
            "using defaults (mu_ref=0.5, sigma_ref=0.1)"
        )
        mu_ref, sigma_ref = 0.5, 0.1
        return mu_ref, sigma_ref

    mu_ref = float(np.mean(attn_means))
    sigma_ref = float(np.std(attn_means))

    print(f"Computed reference stats: mu_ref={mu_ref:.4f}, sigma_ref={sigma_ref:.4f}")
    return mu_ref, sigma_ref
