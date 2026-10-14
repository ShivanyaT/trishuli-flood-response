"""PyTorch dataset over the Kuro Siwo cache: pre-image sampling, flips/rotations, speckle noise."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import build_features
from .train_data import INVALID


def speckle_db(db: np.ndarray, rng: np.random.Generator, looks=(8.0, 30.0)) -> np.ndarray:
    """Multiplicative gamma speckle applied in the dB domain (std ~ 4.34/sqrt(L) dB)."""
    L = rng.uniform(*looks)
    g = rng.gamma(L, 1.0 / L, size=db.shape)
    return db + 10.0 * np.log10(np.maximum(g, 1e-3)).astype(np.float32)


class PatchDataset(Dataset):
    """X: (N,6,H,W) dB [post_vv post_vh sec1_vv sec1_vh sec2_vv sec2_vh]; y: (N,H,W) uint8 (255 = ignore).

    Training picks sec1 or sec2 as 'the' pre-event image at random (a free augmentation, and at inference
    we only ever have one pre image). Evaluation always uses sec1.
    """

    def __init__(self, X, y, indices=None, train: bool = True, seed: int = 0):
        self.X, self.y, self.train, self.seed, self.epoch = X, y, train, seed, 0
        self.idx = np.arange(len(X)) if indices is None else np.asarray(indices)

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        j = int(self.idx[i])
        x = np.asarray(self.X[j], dtype=np.float32)
        y = np.asarray(self.y[j]).astype(np.int64)
        rng = np.random.default_rng((self.seed * 1_000_003 + self.epoch) * 1_000_003 + j)
        pre = 2 if (not self.train or rng.random() < 0.5) else 4
        post_vv, post_vh, pre_vv, pre_vh = x[0], x[1], x[pre], x[pre + 1]
        if self.train:
            if rng.random() < 0.5:  # speckle noise, independently per acquisition
                post_vv, post_vh = speckle_db(post_vv, rng), speckle_db(post_vh, rng)
            if rng.random() < 0.5:
                pre_vv, pre_vh = speckle_db(pre_vv, rng), speckle_db(pre_vh, rng)
        f = build_features(post_vv, post_vh, pre_vv, pre_vh)
        if self.train:
            k = int(rng.integers(0, 4))
            f, y = np.rot90(f, k, axes=(1, 2)), np.rot90(y, k)
            if rng.random() < 0.5:
                f, y = f[:, :, ::-1], y[:, ::-1]
        y = np.where(y == INVALID, 255, y)
        return torch.from_numpy(np.ascontiguousarray(f)), torch.from_numpy(np.ascontiguousarray(y))
