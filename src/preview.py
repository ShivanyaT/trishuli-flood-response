"""Quick-look figure: pre / post VV, change, and the flood + debris masks (used by notebooks/02_flood_map.ipynb)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_flood_overview(pre_vv, post_vv, flood, debris, out_png: Path | None = None, step: int = 4):
    """All inputs (H,W). Downsampled by `step` for display only."""
    s = (slice(None, None, step), slice(None, None, step))
    dvv = (post_vv - pre_vv)[s]
    fig, ax = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for a, img, title, kw in [
        (ax[0, 0], pre_vv[s], "Pre-event VV (dB)", dict(cmap="gray", vmin=-25, vmax=0)),
        (ax[0, 1], post_vv[s], "Post-event VV (dB)", dict(cmap="gray", vmin=-25, vmax=0)),
        (ax[0, 2], dvv, "Change: post - pre (dB)", dict(cmap="RdBu", vmin=-10, vmax=10)),
    ]:
        im = a.imshow(img, **kw)
        a.set_title(title)
        fig.colorbar(im, ax=a, shrink=0.7)
    base = np.clip((post_vv[s] + 25) / 25, 0, 1)
    over = np.dstack([base, base, base])
    over[flood[s] > 0] = [0.1, 0.4, 1.0]
    over[(debris[s] & 1) > 0] = [1.0, 0.5, 0.0]
    over[(debris[s] & 2) > 0] = [0.8, 0.1, 0.1]
    ax[1, 0].imshow(over)
    ax[1, 0].set_title("Flood (blue) | debris: S1 (orange), S2 (red)")
    ax[1, 1].imshow(flood[s], cmap="Blues")
    ax[1, 1].set_title("flood_mask")
    ax[1, 2].imshow(debris[s] > 0, cmap="Reds")
    ax[1, 2].set_title("debris_mask (any sensor)")
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=110)
    return fig
