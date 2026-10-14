"""Flood segmentation model, loss and metrics.

Input : 5 channels (post VV/VH, pre VV/VH, VV change) - see src/features.py. Same-track pre/post is what the
        rest of the pipeline already produces, so the model is change-aware instead of looking at one image.
Output: one logit per pixel for "new flood water" (Kuro Siwo class 2). Permanent water is class 'not flood'.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .features import CHANNELS

IGNORE = 255


def build_model(arch: str = "unet", encoder: str = "resnet34", in_channels: int = len(CHANNELS),
                encoder_weights: str | None = "imagenet"):
    import segmentation_models_pytorch as smp

    if arch == "unet":
        return smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=in_channels, classes=1)
    if arch == "segformer":
        enc = encoder if encoder.startswith("mit_") else "mit_b2"
        return smp.Segformer(encoder_name=enc, encoder_weights=encoder_weights, in_channels=in_channels, classes=1)
    raise ValueError(f"unknown arch {arch!r}")


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """BCE + (1 - Dice), both ignoring target == 255. logits (B,1,H,W); target (B,H,W) in {0,1,255}.

    Dice is computed over the whole batch (not per image), so flood-free patches don't produce 0/0 terms.
    """
    logits = logits[:, 0]
    valid = (target != IGNORE).float()
    t = (target == 1).float()
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp(min=1.0)
    p = torch.sigmoid(logits) * valid
    inter = (p * t).sum()
    dice = 1.0 - (2.0 * inter + eps) / (p.sum() + (t * valid).sum() + eps)
    return bce + dice


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        v = target != IGNORE
        p, t = pred[v].astype(bool), target[v] == 1
        self.tp += int((p & t).sum()); self.fp += int((p & ~t).sum())
        self.fn += int((~p & t).sum()); self.tn += int((~p & ~t).sum())

    def metrics(self) -> dict:
        tp, fp, fn = self.tp, self.fp, self.fn
        d = lambda a, b: float(a / b) if b else 0.0  # noqa: E731
        return {"iou": d(tp, tp + fp + fn), "f1": d(2 * tp, 2 * tp + fp + fn),
                "precision": d(tp, tp + fp), "recall": d(tp, tp + fn),
                "flood_px": tp + fn, "n_px": tp + fp + fn + self.tn}


@torch.no_grad()
def predict_probs(model, X: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(model(X)[:, 0])


def save_checkpoint(path, model, arch, encoder, threshold, extra: dict | None = None) -> None:
    torch.save({"state_dict": model.state_dict(), "arch": arch, "encoder": encoder,
                "in_channels": len(CHANNELS), "channels": CHANNELS, "threshold": float(threshold),
                **(extra or {})}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["arch"], ck["encoder"], ck["in_channels"], encoder_weights=None)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck
