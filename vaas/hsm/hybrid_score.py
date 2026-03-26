import math

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

    img_t = transform(img).unsqueeze(0)

    img_t_px = img_t.to(px_device)
    img_t_fx = img_t.to(fx_device)

    # Fx branch
    with torch.no_grad():
        vit_out = vit_model(img_t_fx, output_attentions=True)

    attn_maps = vit_out.attentions
    fx_tokens = vit_out.last_hidden_state

    if attn_maps is None:
        raise RuntimeError("ViT model did not return attentions")

    # Px branch (with cross-attention)
    with torch.no_grad():
        fx_tokens_px = fx_tokens.to(px_device)

        out_px = model_px(img_t_px, fx_tokens_px)
        logits = out_px["logits"]

        logits = F.interpolate(
            logits,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        pred_sig = torch.sigmoid(logits).squeeze()

        S_P = 1.0 - float(pred_sig.mean().item())

        pred_sig = pred_sig.detach().cpu().numpy()

    # Fx score
    attn_layers = []

    for a in attn_maps:
        head_mean = a.mean(dim=1)
        cls_to_patch = head_mean[:, 0, 1:]
        attn_layers.append(cls_to_patch)

    attn_mean_layers = torch.stack(attn_layers).mean(dim=0)

    mu = float(attn_mean_layers.mean().item())

    if torch.is_tensor(mu_ref):
        mu_ref = mu_ref.item()

    if torch.is_tensor(sigma_ref):
        sigma_ref = sigma_ref.item()

    delta = abs(mu - mu_ref)

    S_F = math.exp(-delta / (sigma_ref + 1e-8))

    # Hybrid
    S_H = alpha * S_F + (1.0 - alpha) * S_P

    return S_F, S_P, S_H, pred_sig
