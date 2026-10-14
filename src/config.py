"""AOI / event configuration, loaded from YAML and validated."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import yaml
from shapely.geometry import Polygon, box


def _to_date(v) -> date:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    return date.fromisoformat(str(v))


def _to_dt(v) -> datetime | None:
    if v in (None, ""):
        return None
    if not isinstance(v, datetime):
        v = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    return v if v.tzinfo else v.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class AOIConfig:
    name: str
    bbox: tuple[float, float, float, float]
    event_date: date
    osm_snapshot: date
    event_time_utc: datetime | None = None
    pre_window_days: int = 14
    post_window_days: int = 14
    s1_min_coverage: float = 0.90
    s2_max_cloud: float = 20.0
    s2_min_tile_overlap: float = 0.02
    osm_tile_deg: float = 0.25
    grid_res: float = 10.0          # metres, common analysis grid
    flood: dict = field(default_factory=dict)   # overrides for src.flood_map.FloodParams
    flow_source: tuple[float, float] | None = None   # (lon, lat) where the flood enters/starts, for src.flowpath

    @property
    def utm_epsg(self) -> int:
        """UTM zone of the AOI centre (auto, so any AOI works; the plan hard-coded 32645)."""
        lon = (self.bbox[0] + self.bbox[2]) / 2
        lat = (self.bbox[1] + self.bbox[3]) / 2
        zone = int((lon + 180) // 6) + 1
        return (32600 if lat >= 0 else 32700) + zone

    @property
    def event_ts(self) -> datetime:
        """Instant a scene must follow/precede to count as post/pre event."""
        if self.event_time_utc:
            return self.event_time_utc
        return datetime.combine(self.event_date, time(23, 59, 59), tzinfo=timezone.utc)

    @property
    def event_day_start(self) -> datetime:
        """00:00 UTC on event_date; the pre-window is measured back from here, the post-window from event_ts."""
        return datetime.combine(self.event_date, time(0, 0), tzinfo=timezone.utc)

    @property
    def search_start(self) -> datetime:
        return self.event_day_start - timedelta(days=self.pre_window_days + 1)

    @property
    def search_end(self) -> datetime:
        return self.event_ts + timedelta(days=self.post_window_days + 1)

    @property
    def aoi(self) -> Polygon:
        return box(*self.bbox)

    @property
    def aoi_wkt(self) -> str:
        return self.aoi.wkt


def load_config(path: str | Path) -> AOIConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    bbox = tuple(float(x) for x in raw["bbox"])
    cfg = AOIConfig(
        name=str(raw["name"]),
        bbox=bbox,  # type: ignore[arg-type]
        event_date=_to_date(raw["event_date"]),
        osm_snapshot=_to_date(raw["osm_snapshot"]),
        event_time_utc=_to_dt(raw.get("event_time_utc")),
        pre_window_days=int(raw.get("pre_window_days", 14)),
        post_window_days=int(raw.get("post_window_days", 14)),
        s1_min_coverage=float(raw.get("s1_min_coverage", 0.90)),
        s2_max_cloud=float(raw.get("s2_max_cloud", 20.0)),
        s2_min_tile_overlap=float(raw.get("s2_min_tile_overlap", 0.02)),
        osm_tile_deg=float(raw.get("osm_tile_deg", 0.25)),
        grid_res=float(raw.get("grid_res", 10.0)),
        flood=dict(raw.get("flood") or {}),
        flow_source=tuple(float(x) for x in raw["flow_source"]) if raw.get("flow_source") else None,
    )
    validate(cfg)
    return cfg


def validate(cfg: AOIConfig) -> None:
    w, s, e, n = cfg.bbox
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise ValueError(f"bbox must be [min_lon, min_lat, max_lon, max_lat], got {cfg.bbox}")
    if cfg.osm_snapshot >= cfg.event_date:
        raise ValueError(
            f"osm_snapshot ({cfg.osm_snapshot}) must be BEFORE event_date ({cfg.event_date}); "
            "post-event OSM edits may not be used as input."
        )
    if cfg.pre_window_days <= 0 or cfg.post_window_days <= 0:
        raise ValueError("pre/post windows must be positive")
