import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

from vaas.fusion.fxpx_crossattention import FxPxCrossAttention


class PatchConsistencySegformer(nn.Module):
    def __init__(
        self,
        backbone_name="nvidia/segformer-b1-finetuned-ade-512-512",
        num_labels=1,
    ):
        super().__init__()

        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            backbone_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

        self.cross_attention = FxPxCrossAttention(px_dim=512, fx_dim=768, gamma=0.1)

    def forward(self, x, fx_tokens=None):
        outputs = self.segformer.segformer(x, output_hidden_states=True)

        hidden_states = list(outputs.hidden_states)

        feat = hidden_states[-1]

        B, C, H, W = feat.shape

        px_tokens = feat.flatten(2).transpose(1, 2)

        if fx_tokens is not None:
            px_tokens = self.cross_attention(px_tokens, fx_tokens)

        feat = px_tokens.transpose(1, 2).reshape(B, C, H, W)

        hidden_states[-1] = feat

        logits = self.segformer.decode_head(hidden_states)

        return {
            "logits": logits,
            "px_tokens": px_tokens,
        }
