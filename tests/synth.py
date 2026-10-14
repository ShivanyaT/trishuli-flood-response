"""Synthetic Himalayan-style valley scene with known flood/debris/shadow regions (for offline tests and demos)."""
from __future__ import annotations

import numpy as np

RES = 10.0


def make_scene(h=300, w=300, seed=0, noise_db=1.0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.abs(yy - h // 2)
    # flat-ish valley floor (+-15 px), steep sides (tan 31 deg = 0.6) beyond it
    dem = 1000 + 0.2 * xx + np.where(dist <= 15, 0, (dist - 15) * RES * 0.6)
    dem = dem.astype(np.float32)

    river = dist <= 3
    flood = (dist <= 12) & (xx >= 100) & (xx < 250)             # includes the river
    debris_zone = (dist > 12) & (dist <= 15) & (xx >= 100) & (xx < 250)
    shadow = (yy < 60) & (xx > 20) & (xx < 120)                  # steep hillside radar shadow (both dates)

    def noise():
        return rng.normal(0, noise_db, (h, w)).astype(np.float32)

    vv_pre = -8 + noise()
    vv_pre[river] = -20 + noise()[river]
    vv_pre[shadow] = -22 + noise()[shadow]
    vv_post = vv_pre.copy()
    vv_post[flood & ~river] = -21 + noise()[flood & ~river]
    vv_post[debris_zone] = -3 + noise()[debris_zone]            # +5 dB brightening
    vh_pre, vh_post = vv_pre - 7, vv_post - 7
    s1_pre, s1_post = np.stack([vv_pre, vh_pre]), np.stack([vv_post, vh_post])

    def s2(post):
        b = np.zeros((6, h, w), np.float32)
        b[0], b[1], b[2], b[3], b[4], b[5] = 0.05, 0.08, 0.05, 0.40, 0.20, 1.0   # vegetated, not water
        if post:
            sel = flood & ~river
            b[1][sel], b[4][sel] = 0.10, 0.02                    # MNDWI ~ 0.67
            b[2][debris_zone], b[3][debris_zone] = 0.20, 0.22   # NDVI ~ 0.05 (stripped)
        else:
            b[1][river], b[4][river] = 0.10, 0.02               # river already water before
        return b

    return {"dem": dem, "s1_pre": s1_pre, "s1_post": s1_post, "s2_pre": s2(False), "s2_post": s2(True),
            "truth": {"flood_new": flood & ~river, "river": river, "debris": debris_zone, "shadow": shadow}}
