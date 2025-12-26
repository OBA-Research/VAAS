import torch.nn as nn
from transformers import ViTConfig, ViTModel


class FxViT(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224"):
        super().__init__()
        config = ViTConfig.from_pretrained(model_name)
        config.output_attentions = True
        self.model = ViTModel.from_pretrained(model_name, config=config)
        self.model.set_attn_implementation("eager")

    def forward(self, x, **kwargs):
        if "output_attentions" not in kwargs:
            kwargs["output_attentions"] = True
        return self.model(x, **kwargs)
