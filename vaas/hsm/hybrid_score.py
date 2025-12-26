import math

import numpy as np
import torch
import torch.nn.functional as F


def compute_scores(
    img,
    mask,
    model_px,
    vit_model,
    mu_ref,
    sigma_ref,
    transform,
    alpha=0.5,
):
    px_device = next(model_px.parameters()).device
    fx_device = next(vit_model.parameters()).device

    img_t_px = transform(img).unsqueeze(0).to(px_device)

    with torch.no_grad():
        logits = model_px(img_t_px)
        logits = F.interpolate(
            logits,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        pred_sig = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
        S_P = 1.0 - float(pred_sig.mean())

    img_t_fx = transform(img).unsqueeze(0).to(fx_device)

    with torch.no_grad():
        vit_out = vit_model(img_t_fx, output_attentions=True)
        attn_maps = vit_out.attentions

    if attn_maps is None:
        raise RuntimeError("ViT model did not return attentions")

    attn_mean_layers = torch.stack([a.mean(dim=1)[:, 0, 1:] for a in attn_maps]).mean(
        dim=0
    )

    attn_values = attn_mean_layers.squeeze().detach().cpu().numpy()
    mu = float(np.mean(attn_values))

    mu_ref_f = (
        float(mu_ref)
        if not torch.is_tensor(mu_ref)
        else float(mu_ref.detach().cpu().item())
    )
    sigma_ref_f = (
        float(sigma_ref)
        if not torch.is_tensor(sigma_ref)
        else float(sigma_ref.detach().cpu().item())
    )

    delta = abs(mu - mu_ref_f)
    S_F = math.exp(-delta / (sigma_ref_f + 1e-8))

    S_H = alpha * S_F + (1.0 - alpha) * S_P

    return S_F, S_P, S_H, pred_sig
