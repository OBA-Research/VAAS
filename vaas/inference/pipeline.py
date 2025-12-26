import warnings
from typing import Union

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers.utils import logging as hf_logging

from vaas.fx.fx_model import FxViT
from vaas.hsm.hybrid_score import compute_scores
from vaas.inference.utils import load_px_checkpoint, load_ref_stats
from vaas.inference.visualize import visualize_inference
from vaas.px.px_model import PatchConsistencySegformer
from vaas.utils.helpers import require_torch

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")
torch, T = require_torch()


class VAASPipeline:
    def __init__(
        self,
        model_px,
        model_fx,
        mu_ref,
        sigma_ref,
        device,
        transform,
        alpha=0.5,
    ):
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
        device: str | torch.device = "cpu",
        alpha: float = 0.5,
    ):
        if isinstance(device, str):
            device = torch.device(device)

        model_px = PatchConsistencySegformer()
        model_fx = FxViT()

        model_fx.eval()
        model_px.eval()

        load_px_checkpoint(model_px, checkpoint_dir)
        model_px = model_px.to(device)
        model_fx = model_fx.to(device)

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
        px_path = hf_hub_download(
            repo_id=repo_id,
            filename="model/px_model.pth",
        )
        ref_path = hf_hub_download(
            repo_id=repo_id,
            filename="model/ref_stats.pth",
        )

        model_px = PatchConsistencySegformer()
        state = torch.load(px_path, map_location="cpu")
        model_px.load_state_dict(state)

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
        """
        Run inference and save a qualitative visualization.

        Args:
        image: PIL Image or image path
        save_path: output image path
        mode: 'all', 'px', 'fx', or 'binary'
        threshold: binarization threshold for Px map
        """

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

    @torch.no_grad()
    def __call__(
        self, image: str | Image.Image
    ) -> dict[str, Union[float, "np.ndarray"]]:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

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
