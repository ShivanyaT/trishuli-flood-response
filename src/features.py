"""Model input features, shared by training, evaluation and inference so they cannot drift apart.

Inputs are dB backscatter. Scaling uses fixed constants (not dataset statistics) so a model trained on
Kuro Siwo sees identically scaled data when run on a Trishuli GeoTIFF.
"""
from __future__ import annotations

import numpy as np

DB_LO, DB_HI, DB_CENTER, DB_SCALE = -35.0, 10.0, -15.0, 10.0
DELTA_CLIP, DELTA_SCALE = 15.0, 5.0

CHANNELS = ["post_vv", "post_vh", "pre_vv", "pre_vh", "delta_vv"]


def _norm(db: np.ndarray) -> np.ndarray:
    out = (np.clip(db, DB_LO, DB_HI) - DB_CENTER) / DB_SCALE
    return np.nan_to_num(out, nan=0.0)  # missing data -> the "typical" value 0


def build_features(post_vv, post_vh, pre_vv, pre_vh) -> np.ndarray:
    """(H,W) dB arrays -> (C,H,W) float32. NaN -> 0 after scaling."""
    delta = np.clip(post_vv - pre_vv, -DELTA_CLIP, DELTA_CLIP) / DELTA_SCALE
    chans = [_norm(post_vv), _norm(post_vh), _norm(pre_vv), _norm(pre_vh), np.nan_to_num(delta, nan=0.0)]
    return np.stack(chans).astype(np.float32)
