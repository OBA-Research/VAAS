from __future__ import annotations

import warnings

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from PIL import Image
from transformers.utils import logging as hf_logging

from vaas.utils.helpers import require_torch

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")


torch = None
T = None


def _ensure_torch():
    global torch, T
    if torch is None or T is None:
        torch, T = require_torch()
    return torch, T


class VAASPipeline:
    def __init__(
        self,
        model_px,
        model_fx,
        mu_ref,
        sigma_ref,
        device,
        transform,
        alpha: float = 0.5,
    ):
        torch, _ = _ensure_torch()

        self.device = device
        self.model_px = model_px.to(device)
        self.model_fx = model_fx.to(device)

        self.mu_ref = (
            mu_ref.to(device)
            if torch.is_tensor(mu_ref)
            else torch.tensor(mu_ref, device=device)
        )
        self.sigma_ref = (
            sigma_ref.to(device)
            if torch.is_tensor(sigma_ref)
            else torch.tensor(sigma_ref, device=device)
        )

        self.transform = transform
        self.alpha = alpha

        self.model_px.eval()
        self.model_fx.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        device: str = "cpu",
        alpha: float = 0.5,
    ):
        torch, T = _ensure_torch()

        if isinstance(device, str):
            device = torch.device(device)

        from vaas.fx.fx_model import FxViT
        from vaas.inference.utils import load_px_checkpoint, load_ref_stats
        from vaas.px.px_model import PatchConsistencySegformer

        model_px = PatchConsistencySegformer()
        model_fx = FxViT()

        load_px_checkpoint(model_px, checkpoint_dir)
        mu_ref, sigma_ref = load_ref_stats(checkpoint_dir)

        transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        return cls(
            model_px=model_px,
            model_fx=model_fx,
            mu_ref=mu_ref,
            sigma_ref=sigma_ref,
            device=device,
            transform=transform,
            alpha=alpha,
        )

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        device: str = "cpu",
        alpha: float = 0.5,
    ):
        torch, T = _ensure_torch()

        try:
            px_path = hf_hub_download(repo_id, "model/px_model.pth")
            ref_path = hf_hub_download(repo_id, "model/ref_stats.pth")
        except (RepositoryNotFoundError, HfHubHTTPError) as e:
            raise SystemExit(
                "Failed to load VAAS model from Hugging Face.\n\n"
                f"Repository: {repo_id}\n"
                f"Reason: {e.__class__.__name__}\n\n"
                "If this is a private repository, ensure you are logged in:\n"
                "  huggingface-cli login\n"
            ) from e

        from vaas.fx.fx_model import FxViT
        from vaas.px.px_model import PatchConsistencySegformer

        model_px = PatchConsistencySegformer()
        model_px.load_state_dict(torch.load(px_path, map_location="cpu"))

        ref = torch.load(ref_path, map_location="cpu")
        mu_ref = ref["mu_ref"]
        sigma_ref = ref["sigma_ref"]

        model_fx = FxViT()

        transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        return cls(
            model_px=model_px,
            model_fx=model_fx,
            mu_ref=mu_ref,
            sigma_ref=sigma_ref,
            device=device,
            transform=transform,
            alpha=alpha,
        )

    def visualize(
        self,
        image,
        save_path="vaas_visualization.png",
        mode="all",
        threshold=0.5,
    ):
        _ensure_torch()

        from vaas.inference.visualize import visualize_inference

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        out = self(image)

        visualize_inference(
            img=image,
            anomaly_map=out["anomaly_map"],
            vit_model=self.model_fx,
            fx_transform=self.transform,
            s_h=out["S_H"],
            save_path=save_path,
            threshold=threshold,
            mode=mode,
        )

        return out

    def __call__(self, image: str | Image.Image) -> dict[str, float | np.ndarray]:
        torch, _ = _ensure_torch()

        from vaas.hsm.hybrid_score import compute_scores

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        with torch.no_grad():
            s_f, s_p, s_h, anomaly_map = compute_scores(
                img=image,
                mask=None,
                model_px=self.model_px,
                vit_model=self.model_fx,
                mu_ref=self.mu_ref,
                sigma_ref=self.sigma_ref,
                transform=self.transform,
                alpha=self.alpha,
            )

        if torch.is_tensor(anomaly_map):
            anomaly_map = anomaly_map.detach().cpu().numpy()

        return {
            "S_F": float(s_f),
            "S_P": float(s_p),
            "S_H": float(s_h),
            "anomaly_map": anomaly_map,
        }
