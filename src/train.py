"""Train the flood segmentation model on a Kuro Siwo cache (see src/train_data.py).

    python -m src.train --cache data/kuro_siwo --epochs 30 --batch-size 16          # GPU (Kaggle/Colab)
    python -m src.train --cache data/kuro_siwo --epochs 1 --max-steps 20 --no-pretrained   # quick CPU check

Validation is held out BY EVENT (patches from one flood are spatially correlated, so a random patch split
would leak and overstate the score). The decision threshold is picked on that validation set.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import PatchDataset
from .model import Confusion, bce_dice_loss, build_model, predict_probs, save_checkpoint
from .train_data import load_cache

log = logging.getLogger(__name__)


def split_by_event(events: np.ndarray, val_frac: float = 0.15, seed: int = 0):
    """(train_idx, val_idx), holding out whole events. Falls back to a random split if events are unknown."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(events.tolist()) - {"unknown"})
    if len(uniq) >= 4:
        n_val = max(1, int(round(val_frac * len(uniq))))
        val_ev = set(rng.choice(uniq, n_val, replace=False).tolist())
        is_val = np.array([e in val_ev for e in events])
        return np.where(~is_val)[0], np.where(is_val)[0], "event"
    log.warning("event ids unavailable/too few (%d); using a RANDOM validation split (optimistic)", len(uniq))
    perm = rng.permutation(len(events))
    k = max(1, int(val_frac * len(events)))
    return np.sort(perm[k:]), np.sort(perm[:k]), "random"


@torch.no_grad()
def validate(model, loader, device, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
    model.eval()
    conf = {t: Confusion() for t in thresholds}
    for x, y in loader:
        p = predict_probs(model, x.to(device)).cpu().numpy()
        yy = y.numpy()
        for t in thresholds:
            conf[t].update(p >= t, yy)
    res = {t: c.metrics() for t, c in conf.items()}
    best = max(res, key=lambda t: res[t]["f1"])
    return best, res[best], res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="data/kuro_siwo")
    ap.add_argument("--out", default="models/flood_unet.pt")
    ap.add_argument("--arch", default="unet", choices=["unet", "segformer"])
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimiser steps (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, ev = load_cache(Path(a.cache), "train")
    tr_idx, va_idx, how = split_by_event(ev, seed=a.seed)
    log.info("device=%s  train=%d val=%d (%s split)  events=%d", device, len(tr_idx), len(va_idx), how, len(set(ev)))
    tr = PatchDataset(X, y, tr_idx, train=True, seed=a.seed)
    va = PatchDataset(X, y, va_idx, train=False)
    tl = DataLoader(tr, a.batch_size, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=device == "cuda")
    vl = DataLoader(va, a.batch_size, shuffle=False, num_workers=a.workers)

    model = build_model(a.arch, a.encoder, encoder_weights=None if a.no_pretrained else "imagenet").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    total = a.epochs * len(tl) if not a.max_steps else min(a.max_steps, a.epochs * len(tl))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    history, best_f1, step, t0 = [], -1.0, 0, time.time()
    for epoch in range(a.epochs):
        tr.set_epoch(epoch)
        model.train()
        run = []
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            with torch.autocast(device_type=device, enabled=use_amp):
                loss = bce_dice_loss(model(xb).float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run.append(loss.item())
            step += 1
            if a.max_steps and step >= a.max_steps:
                break
        thr, m, _ = validate(model, vl, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(run)), "val_threshold": thr, **m})
        log.info("epoch %2d  loss %.4f  val IoU %.3f F1 %.3f P %.3f R %.3f (thr %.1f)  %.0fs", epoch,
                 np.mean(run), m["iou"], m["f1"], m["precision"], m["recall"], thr, time.time() - t0)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(a.out, model, a.arch, a.encoder, thr,
                            {"val": m, "split": how, "epoch": epoch, "seed": a.seed})
        if a.max_steps and step >= a.max_steps:
            break
    Path("reports").mkdir(exist_ok=True)
    Path("reports/train_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    log.info("best val F1 %.3f -> %s", best_f1, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
