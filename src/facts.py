"""The copilot's ONLY source of numbers: deterministic tools over the pipeline outputs.

Nothing here calls a language model. Each tool returns plain JSON-able data plus the files it came from, so any
number in an answer can be traced to `data/outputs/*`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@dataclass
class Facts:
    event: dict
    flood: dict | None
    damage: dict | None
    cutoff: dict | None
    flowpath: dict | None
    out_dir: Path

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, out_dir: str | Path, event_name: str = "Trishuli 2026", event_date: str = "2026-08-26",
             bbox=None) -> "Facts":
        out = Path(out_dir)
        flood = _load(out / "flood_report.json")
        synthetic = bool(flood and flood.get("synthetic")) or (out.parent / "SYNTHETIC_DEMO.txt").exists()
        return cls(event={"name": event_name, "date": event_date, "bbox": bbox, "synthetic": synthetic},
                   flood=flood, damage=_load(out / "damage_summary.json"), cutoff=_load(out / "cutoff_summary.json"),
                   flowpath=_load(out / "flowpath_summary.json"), out_dir=out)

    def missing(self) -> list[str]:
        return [n for n, v in (("flood_report.json", self.flood), ("damage_summary.json", self.damage),
                               ("cutoff_summary.json", self.cutoff)) if v is None]

    # ------------------------------------------------------------------ tools (pure, deterministic)
    def flood_extent(self) -> dict:
        f = self.flood or {}
        return {"event": self.event["name"], "date": self.event["date"], "synthetic": self.event["synthetic"],
                "flood_area_km2": round(f.get("flood_area_km2", 0.0), 2),
                "debris_area_km2": round(f.get("debris_area_km2_s1", 0.0) + f.get("debris_area_km2_s2", 0.0), 2),
                "radar_valid_pct": round(100 * f.get("s1_valid_fraction", 0.0)),
                "radar_unobservable_pct": round(100 * f.get("s1_unobservable_fraction", 0.0)),
                "sources": ["flood_report.json"]}

    def summarize_damage(self) -> dict:
        d = self.damage or {}
        sev = d.get("buildings_by_severity", {})
        return {"buildings_hit": d.get("buildings_hit", 0), "buildings_high_overlap": sev.get("high", 0),
                "buildings_moderate_overlap": sev.get("moderate", 0), "buildings_low_overlap": sev.get("low", 0),
                "road_km_in_hazard": d.get("road_km_in_hazard", 0.0),
                "bridges_total": d.get("bridges_total_osm", 0), "bridges_exposed": d.get("bridges_exposed", 0),
                "bridges_exposed_names": d.get("bridges_exposed_names", []),
                "note": "exposed means inside the hazard footprint, not collapsed",
                "sources": ["damage_summary.json", "damage.gpkg"]}

    def count_flooded_buildings(self) -> dict:
        d = self.summarize_damage()
        return {k: d[k] for k in ("buildings_hit", "buildings_high_overlap", "buildings_moderate_overlap",
                                  "buildings_low_overlap")} | {"sources": d["sources"]}

    def list_cutoff_settlements(self) -> dict:
        c = self.cutoff or {}
        cut, cond = c.get("cut_off", []), c.get("cut_off_if_bridges_fail", [])
        qa = c.get("qa", {})
        return {"cut_off": cut, "cut_off_count": len(cut), "cut_off_if_bridges_fail": cond,
                "cut_off_if_bridges_fail_count": len(cond),
                "buildings_near_cut_off": c.get("buildings_near_cut_off", 0),
                "buildings_near_cut_off_if_bridges_fail": c.get("buildings_near_cut_off_if_bridges_fail", 0),
                "settlements_analysed": qa.get("settlements", 0),
                "settlements_without_mapped_route": (qa.get("settlements_without_road_access", 0)
                                                     + qa.get("settlements_unreachable_pre", 0)),
                "sources": ["cutoff_summary.json", "cutoff_settlements.csv"]}

    def settlements_along_flowpath(self) -> dict:
        f = self.flowpath
        if not f:
            return {"available": False, "sources": []}
        return {"available": True, "length_km": f["length_km"], "settlements": f["settlements_along_path"],
                "settlement_count": len(f["settlements_along_path"]), "sources": ["flowpath_summary.json"]}

    def limitations(self) -> dict:
        return {"items": [
            "Sentinel-1 revisits every 6 to 12 days; this is a post-event map, not a warning.",
            "Same-orbit pre/post pairing can miss the event by days.",
            "Radar shadow and layover in steep terrain cause missed flooding.",
            "Monsoon cloud often makes Sentinel-2 unusable.",
            "OSM completeness varies and rural Nepal is under-mapped, so counts are lower bounds.",
            "Exposed or cut off is inferred from 10 m masks; structural damage and bridge survival are unknown.",
            "No ground truth is used as input; validation against EMSR927 is separate and EMSR927 is not ground truth.",
            "Walking paths are not in the cut-off analysis (road access only).",
        ], "sources": []}

    def overview(self) -> dict:
        return {"extent": self.flood_extent(), "damage": self.summarize_damage(),
                "cutoff": self.list_cutoff_settlements(), "flowpath": self.settlements_along_flowpath(),
                "sources": ["flood_report.json", "damage_summary.json", "cutoff_summary.json"]}
