import torch


def compute_segmentation_metrics(logits, targets, threshold=0.5, eps=1e-7):
    """
    Computes IoU, F1, Precision, Recall for a batch of segmentation logits and masks.

    Args:
        logits: [B, 1, H, W] raw model outputs.
        targets: [B, 1, H, W] binary masks in {0, 1}.
    """

    with torch.no_grad():
        preds = (torch.sigmoid(logits) > threshold).float()
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        tp = (preds * targets).sum(dim=1)
        fp = (preds * (1 - targets)).sum(dim=1)
        fn = ((1 - preds) * targets).sum(dim=1)

        iou = tp / (tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        return (
            iou.mean().item(),
            f1.mean().item(),
            precision.mean().item(),
            recall.mean().item(),
        )
