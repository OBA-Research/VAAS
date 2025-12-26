from __future__ import annotations

import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge
from PIL import Image

from vaas.utils.helpers import require_torch


def visualize_inference(
    img: Image.Image,
    anomaly_map: np.ndarray,
    vit_model,
    fx_transform,
    s_h: float,
    save_path: str,
    threshold: float = 0.5,
    mode: str = "all",
):
    torch, _ = require_torch()

    img_resized = img.resize((224, 224))
    img_np = np.array(img_resized)

    pred = cv2.resize(anomaly_map, (224, 224))
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

    heat_px = cv2.applyColorMap(np.uint8(255 * pred), cv2.COLORMAP_INFERNO)
    heat_px = cv2.cvtColor(heat_px, cv2.COLOR_BGR2RGB)
    overlay_px = cv2.addWeighted(img_np, 0.4, heat_px, 0.8, 0)

    px_binary = (pred > threshold).astype(np.uint8) * 255
    heat_bin = cv2.applyColorMap(px_binary, cv2.COLORMAP_COOL)
    heat_bin = cv2.cvtColor(heat_bin, cv2.COLOR_BGR2RGB)
    overlay_bin = cv2.addWeighted(img_np, 0.2, heat_bin, 0.8, 0)

    img_t = fx_transform(img).unsqueeze(0)
    img_t = img_t.to(next(vit_model.parameters()).device)
    vit_model.eval()

    with torch.no_grad():
        out = vit_model(img_t, output_attentions=True)
        attn = out.attentions

    attn_mean = torch.stack([a.mean(dim=1)[:, 0, 1:] for a in attn]).mean(dim=0)

    n = int(math.sqrt(attn_mean.shape[-1]))
    attn_map = attn_mean.squeeze().cpu().numpy().reshape(n, n)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    attn_resized = cv2.resize(attn_map, (224, 224))
    heat_fx = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
    heat_fx = cv2.cvtColor(heat_fx, cv2.COLOR_BGR2RGB)
    overlay_fx = cv2.addWeighted(img_np, 0.7, heat_fx, 0.5, 0)

    panels = [img_np]
    titles = ["Image"]

    if mode in {"px", "all"}:
        panels.append(overlay_px)
        titles.append("Px Heatmap")

    if mode in {"binary", "all"}:
        panels.append(overlay_bin)
        titles.append("Px Binary")

    if mode in {"fx", "all"}:
        panels.append(overlay_fx)
        titles.append("Fx Attention")

    panels.append(None)
    titles.append("Hybrid Score (S_H)")

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    plt.subplots_adjust(wspace=0.05)

    for ax, title, panel in zip(axes, titles, panels, strict=False):
        ax.axis("off")
        ax.set_title(title, fontsize=11)
        if panel is not None:
            ax.imshow(panel)

    ax = axes[-1]
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.3, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")

    base = Wedge((0, 0), 1.0, 0, 180, facecolor="lightgray")
    fill = Wedge(
        (0, 0),
        1.0,
        180 * (1 - s_h),
        180,
        facecolor="#ffa552",
    )

    ax.add_patch(base)
    ax.add_patch(fill)

    theta = math.radians(180 * (1 - s_h))
    ax.plot(
        [0, 0.8 * math.cos(theta)],
        [0, 0.8 * math.sin(theta)],
        lw=2,
        color="black",
    )

    ax.text(
        0,
        -0.35,
        f"{s_h:.3f}",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#ff7b00",
    )

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
