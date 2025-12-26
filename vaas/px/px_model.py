import torch.nn as nn
from transformers import SegformerForSemanticSegmentation


class PatchConsistencySegformer(nn.Module):
    def __init__(
        self, backbone="nvidia/segformer-b1-finetuned-ade-512-512", num_labels=1
    ):
        super().__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            backbone,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        out = self.segformer(x)
        return out.logits
