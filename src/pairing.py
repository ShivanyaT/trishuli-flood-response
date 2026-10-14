"""Scene selection: same-orbit-track Sentinel-1 pre/post pairs and best Sentinel-2 pre/post per tile.

Pure functions (no network) so they can be unit-tested offline.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from shapely.ops import unary_union

from .cdse import Product
from .config import AOIConfig


# ----------------------------------------------------------------------------- Sentinel-1
@dataclass
class Acquisition:
    """All S1 frames of one satellite pass over one relative-orbit track on one UTC date."""
    rel_orbit: int
    direction: str
    day: str
    products: list[Product]
    coverage: float

    @property
    def start(self) -> datetime:
        return min(p.start for p in self.products)

    @property
    def track(self) -> tuple[int, str]:
        return (self.rel_orbit, self.direction)


@dataclass
class S1Pair:
    pre: Acquisition
    post: Acquisition
    score_hours: float

    @property
    def track(self) -> tuple[int, str]:
        return self.pre.track

    @property
    def products(self) -> list[Product]:
        return self.pre.products + self.post.products


def _is_dual_pol_grd(p: Product) -> bool:
    # name pattern S1A_IW_GRDH_1SDV_... : IW mode, GRD high-res, dual-pol VV+VH
    return "_IW_GRDH_1SDV_" in p.name


def group_acquisitions(products: list[Product], cfg: AOIConfig) -> list[Acquisition]:
    groups: dict[tuple, list[Product]] = defaultdict(list)
    for p in products:
        if not _is_dual_pol_grd(p):
            continue
        ro = p.attrs.get("relativeOrbitNumber")
        od = p.attrs.get("orbitDirection")
        if ro is None or od is None:
            continue
        groups[(int(ro), str(od).upper(), p.start.strftime("%Y-%m-%d"))].append(p)
    aoi = cfg.aoi
    acqs = []
    for (ro, od, day), plist in groups.items():
        cov = unary_union([p.geom for p in plist]).intersection(aoi).area / aoi.area
        acqs.append(Acquisition(ro, od, day, plist, cov))
    return acqs


def rank_s1_pairs(products: list[Product], cfg: AOIConfig) -> tuple[list[S1Pair], list[Acquisition]]:
    """Return viable pairs (best first) and all acquisitions (for diagnostics).

    A pair = same relative orbit AND same direction, pre before the event, post after it,
    both covering >= cfg.s1_min_coverage of the AOI. Preference: post as soon as possible
    after the event (flood/debris signal fades), then pre as close as possible.
    """
    acqs = group_acquisitions(products, cfg)
    ev = cfg.event_ts
    by_track: dict[tuple, list[Acquisition]] = defaultdict(list)
    for a in acqs:
        if a.coverage >= cfg.s1_min_coverage:
            by_track[a.track].append(a)
    pairs = []
    for lst in by_track.values():
        pre = [a for a in lst if a.start < ev and cfg.event_day_start - a.start <= timedelta(days=cfg.pre_window_days)]
        post = [a for a in lst if a.start > ev and a.start - ev <= timedelta(days=cfg.post_window_days)]
        if not pre or not post:
            continue
        b_pre = max(pre, key=lambda a: a.start)
        b_post = min(post, key=lambda a: a.start)
        post_h = (b_post.start - ev).total_seconds() / 3600
        pre_h = (ev - b_pre.start).total_seconds() / 3600
        pairs.append(S1Pair(b_pre, b_post, post_h + 0.5 * pre_h))
    pairs.sort(key=lambda p: p.score_hours)
    return pairs, acqs


def diagnose_s1(acqs: list[Acquisition], cfg: AOIConfig) -> str:
    """Human-readable explanation for when no same-track pair exists."""
    if not acqs:
        return "No dual-pol IW GRD scenes found in the search window; widen the windows."
    lines = [f"No same-track pre/post pair with >= {cfg.s1_min_coverage:.0%} AOI coverage. Acquisitions found:"]
    for a in sorted(acqs, key=lambda a: (a.track, a.day)):
        side = "pre " if a.start < cfg.event_ts else "post"
        lines.append(f"  track {a.rel_orbit:>3} {a.direction:<10} {a.day} {side} coverage={a.coverage:.0%}")
    lines.append("Try: larger post_window_days / pre_window_days, lower s1_min_coverage (then mosaic two tracks "
                 "ONLY if each track still has its own pre/post), or a smaller AOI.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- Sentinel-2
@dataclass
class S2Choice:
    tile: str
    role: str  # "pre" | "post"
    product: Product
    cloud: float
    cloudy: bool = False  # True if nothing met the cloud threshold (best available used)
    notes: list[str] = field(default_factory=list)


_TILE_RE = re.compile(r"_T(\d{2}[A-Z]{3})_")


def tile_of(name: str) -> str | None:
    m = _TILE_RE.search(name)
    return m.group(1) if m else None


def pick_s2(products: list[Product], cfg: AOIConfig) -> list[S2Choice]:
    ev, aoi = cfg.event_ts, cfg.aoi
    dedup: dict[tuple, Product] = {}
    for p in products:
        t = tile_of(p.name)
        if t and "MSIL2A" in p.name:
            k = (t, p.start.isoformat())
            if k not in dedup or p.name > dedup[k].name:  # keep latest processing baseline
                dedup[k] = p
    by_tile: dict[str, list[Product]] = defaultdict(list)
    for (t, _), p in dedup.items():
        if p.geom.intersection(aoi).area / aoi.area >= cfg.s2_min_tile_overlap:
            by_tile[t].append(p)

    out: list[S2Choice] = []
    for tile, plist in sorted(by_tile.items()):
        for role in ("pre", "post"):
            if role == "pre":
                cands = [p for p in plist if p.start < ev and cfg.event_day_start - p.start <= timedelta(days=cfg.pre_window_days)]
                gap = lambda p: (ev - p.start)  # noqa: E731
            else:
                cands = [p for p in plist if p.start > ev and p.start - ev <= timedelta(days=cfg.post_window_days)]
                gap = lambda p: (p.start - ev)  # noqa: E731
            if not cands:
                continue
            cloud = lambda p: float(p.attrs.get("cloudCover", 100.0))  # noqa: E731
            clear = [p for p in cands if cloud(p) <= cfg.s2_max_cloud]
            if clear:
                best, cloudy = min(clear, key=gap), False
            else:
                best, cloudy = min(cands, key=cloud), True
            c = S2Choice(tile, role, best, cloud(best), cloudy)
            if cloudy:
                c.notes.append(f"no scene <= {cfg.s2_max_cloud}% cloud; using lowest ({cloud(best):.0f}%)")
            out.append(c)
    return out
