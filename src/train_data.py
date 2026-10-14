"""Kuro Siwo (GRD) -> a compact on-disk training cache.

Kuro Siwo is 1-2 TB, far too large to download for a 7-day project, so this streams WebDataset shards from
Hugging Face (`orion-ai-lab/Kuro-Siwo-Webdataset`, folders `train_GRD/` and `test_GRD/`), keeps only a
sample, and writes memory-mappable arrays:

    <cache>/<split>_X.npy       float16 (N, 6, H, W) dB: post_vv post_vh sec1_vv sec1_vh sec2_vv sec2_vh
    <cache>/<split>_y.npy       uint8   (N, H, W)    0 = not flood, 1 = flood, 255 = invalid (ignored)
    <cache>/<split>_events.npy  str     (N,)         event id from info.json (best effort)

Everything about the sample layout below is taken from the dataset card (field names flood_vv, sec1_vv,
mask, valid_mask, info.json; labels 0 no water / 1 permanent water / 2 flood). Array shapes and the units
(linear vs dB) were NOT documented, so both are detected at read time; check the first samples with
`python -m src.train_data inspect`.

    python -m src.train_data build --split train --max-patches 8000 --cache data/kuro_siwo
    python -m src.train_data build --split test  --max-patches 3000 --cache data/kuro_siwo
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import random
import sys
import tarfile
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np

log = logging.getLogger(__name__)
REPO = "orion-ai-lab/Kuro-Siwo-Webdataset"
EVENT_KEYS = ("actid", "act_id", "event", "event_id", "aoiid", "aoi_id", "id")
FLOOD, INVALID = 2, 255


def iter_tar_samples(fileobj: BinaryIO) -> Iterator[dict]:
    """Stream a WebDataset tar: consecutive members sharing a key prefix form one sample.

    Member `0001.flood_vv.npy` -> sample '0001', field 'flood_vv'. Works on a non-seekable stream, so only the
    bytes actually consumed are downloaded.
    """
    key, cur = None, {}
    with tarfile.open(fileobj=fileobj, mode="r|*") as tf:
        for m in tf:
            if not m.isfile():
                continue
            base = Path(m.name).name
            k, _, rest = base.partition(".")
            if not rest:
                continue
            data = tf.extractfile(m).read()
            if rest.endswith(".npy"):
                val = np.load(io.BytesIO(data), allow_pickle=False)
            elif rest.endswith(".json"):
                val = json.loads(data)
            else:
                continue
            field = rest.rsplit(".", 1)[0]
            if k != key and cur:
                yield cur
                cur = {}
            key = k
            cur[field] = val
    if cur:
        yield cur


def to_db_auto(x: np.ndarray) -> np.ndarray:
    """Linear intensity -> dB; leave dB untouched. Heuristic: dB backscatter is negative, linear is positive."""
    x = np.squeeze(x).astype(np.float32)
    finite = x[np.isfinite(x)]
    if finite.size and np.median(finite) > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.where(x > 0, 10.0 * np.log10(np.maximum(x, 1e-6)), np.nan)
    return x.astype(np.float32)


def event_id(info: dict | None) -> str:
    if not isinstance(info, dict):
        return "unknown"
    parts = [str(info[k]) for k in EVENT_KEYS if k in info]
    return "_".join(parts[:2]) if parts else "unknown"


def parse_sample(s: dict):
    """Raw sample dict -> (X (6,H,W) float32 dB, y (H,W) uint8, event) or None if unusable."""
    try:
        bands = [to_db_auto(s[k]) for k in ("flood_vv", "flood_vh", "sec1_vv", "sec1_vh", "sec2_vv", "sec2_vh")]
        mask = np.squeeze(s["mask"]).astype(np.int16)
    except KeyError:
        return None
    valid = np.squeeze(s["valid_mask"]).astype(bool) if "valid_mask" in s else np.ones_like(mask, bool)
    if bands[0].shape != mask.shape:
        return None
    y = np.where(mask == FLOOD, 1, 0).astype(np.uint8)
    y[(mask > FLOOD) | ~valid] = INVALID
    return np.stack(bands), y, event_id(s.get("info"))


def _shards(split: str, token: str | None):
    from huggingface_hub import HfApi
    files = HfApi(token=token).list_repo_files(REPO, repo_type="dataset")
    folder = f"{split}_GRD/"
    return sorted(f for f in files if f.startswith(folder) and f.endswith(".tar"))


def build_cache(split: str, cache_dir: Path, max_patches: int, per_shard: int = 60, neg_frac: float = 0.3,
                min_flood_frac: float = 0.002, seed: int = 0, token: str | None = None, shards=None) -> dict:
    """Stream shards in random order, take up to `per_shard` samples from each (diversity across events),
    keep patches with flood plus a fraction of flood-free ones."""
    from huggingface_hub import HfFileSystem

    rng = random.Random(seed)
    shards = shards or _shards(split, token)
    if not shards:
        raise RuntimeError(f"no .tar shards found under {split}_GRD/ in {REPO}")
    rng.shuffle(shards)
    fs = HfFileSystem(token=token)
    X, Y, E, n_neg = [], [], [], 0
    for sh in shards:
        taken = 0
        with fs.open(f"datasets/{REPO}/{sh}", "rb") as f:
            for raw in iter_tar_samples(f):
                parsed = parse_sample(raw)
                if parsed is None:
                    continue
                x, y, ev = parsed
                valid = y != INVALID
                frac = float((y == 1).sum()) / max(1, int(valid.sum()))
                if frac < min_flood_frac:
                    if n_neg >= neg_frac * max(1, len(X)):
                        continue
                    n_neg += 1
                X.append(x.astype(np.float16)); Y.append(y); E.append(ev)
                taken += 1
                if taken >= per_shard or len(X) >= max_patches:
                    break
        log.info("%s: %d patches (%d shards read)", split, len(X), shards.index(sh) + 1)
        if len(X) >= max_patches:
            break
    return save_cache(cache_dir, split, X, Y, E)


def save_cache(cache_dir: Path, split: str, X, Y, E) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{split}_X.npy", np.stack(X))
    np.save(cache_dir / f"{split}_y.npy", np.stack(Y))
    np.save(cache_dir / f"{split}_events.npy", np.array(E))
    info = {"split": split, "n": len(X), "events": len(set(E)),
            "flood_pixel_fraction": float(np.mean([(y == 1).sum() / max(1, (y != INVALID).sum()) for y in Y]))}
    (cache_dir / f"{split}_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log.info("cache %s", info)
    return info


def load_cache(cache_dir: Path, split: str):
    X = np.load(cache_dir / f"{split}_X.npy", mmap_mode="r")
    y = np.load(cache_dir / f"{split}_y.npy", mmap_mode="r")
    ev = np.load(cache_dir / f"{split}_events.npy")
    return X, y, ev


def inspect(split: str, token: str | None, n: int = 3) -> None:
    from huggingface_hub import HfFileSystem
    sh = _shards(split, token)
    print(f"{len(sh)} shards under {split}_GRD/; first: {sh[:3]}")
    with HfFileSystem(token=token).open(f"datasets/{REPO}/{sh[0]}", "rb") as f:
        for i, s in enumerate(iter_tar_samples(f)):
            print({k: (v.shape, str(v.dtype), float(np.nanmin(v)), float(np.nanmedian(v)), float(np.nanmax(v)))
                   if isinstance(v, np.ndarray) else v for k, v in s.items()})
            if i + 1 >= n:
                break


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "inspect"])
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--cache", default="data/kuro_siwo")
    ap.add_argument("--max-patches", type=int, default=8000)
    ap.add_argument("--per-shard", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf-token", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.cmd == "inspect":
        inspect(a.split, a.hf_token)
    else:
        build_cache(a.split, Path(a.cache), a.max_patches, a.per_shard, seed=a.seed, token=a.hf_token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
