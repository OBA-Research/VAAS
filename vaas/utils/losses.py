import torch
import torch.nn as nn


def dice_loss_from_logits(logits, targets, eps=1e-7):
    """
    Dice loss that takes raw logits as input.
    """
    probs = torch.sigmoid(logits)
    targets = targets.float()

    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def hybrid_seg_loss(logits, targets, bce_loss_fn=None, dice_weight=0.5):
    """
    Hybrid BCE + Dice loss for segmentation.
    Expects logits and targets to have the same spatial size.
    """
    if bce_loss_fn is None:
        bce_loss_fn = nn.BCEWithLogitsLoss()

    targets = targets.float()
    loss_bce = bce_loss_fn(logits, targets)
    loss_dice = dice_loss_from_logits(logits, targets)

    return loss_bce + dice_weight * loss_dice
