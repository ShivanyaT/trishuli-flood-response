"""Cut-off settlement analysis on the PRE-event OSM road snapshot.

Why not `osmnx.graph_from_bbox` as the plan says: that queries *current* OSM, which may contain post-event
edits (not allowed as input). The graph here is built from the ohsome pre-event GeoPackage instead.

Method
  1. Build a road graph (nodes at junctions/endpoints, edges split into <= `piece_m` pieces so a flood that
     crosses part of a road only blocks the pieces it touches).
  2. Block every piece whose length inside flood/debris is >= `min_blocked_m`. Two scenarios, because a 10 m
     radar mask cannot tell whether a bridge survived:
       optimistic   - bridges are never blocked (the span passes over the water)
       conservative - a bridge with new flood/debris within `bridge_buffer_m` is blocked
  3. Destinations = hospitals/clinics and towns/cities that are themselves outside the hazard.
  4. A settlement is CUT OFF only if it could reach a destination BEFORE the event and cannot AFTER. Settlements
     that never had a route in the OSM snapshot are reported as `unreachable_pre` / `no_road_access`, not as
     victims of this flood (OSM is incomplete in rural Nepal, so that would be an artefact).
  5. Roads are the motorable network by default; walking paths are excluded (set include_paths to change).

    python -m src.cutoff --config config/aoi_trishuli.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import substring, unary_union

from .config import load_config
from .damage import load_hazard, load_layer

log = logging.getLogger(__name__)

MOTORABLE = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "service",
             "track", "living_street", "motorway_link", "trunk_link", "primary_link", "secondary_link",
             "tertiary_link"}
PATHS = {"path", "footway", "steps", "cycleway", "pedestrian", "bridleway"}
SETTLEMENT_PLACES = {"village", "hamlet", "town", "city", "suburb", "neighbourhood"}
TOWN_PLACES = {"town", "city"}
BRIDGE_VALUES = {"yes", "viaduct", "movable", "cantilever", "aqueduct"}


@dataclass
class CutoffParams:
    piece_m: float = 100.0
    min_blocked_m: float = 15.0        # ~1.5 pixels of hazard across the road
    bridge_buffer_m: float = 10.0
    snap_m: float = 400.0              # settlement -> nearest road node
    building_radius_m: float = 300.0   # population proxy: OSM buildings within this radius
    cluster_link_m: float = 100.0      # unnamed-cluster detection
    cluster_min_buildings: int = 10
    cluster_near_place_m: float = 400.0
    degraded_ratio: float = 1.5        # post/pre distance to nearest destination
    degraded_extra_km: float = 2.0
    include_paths: bool = False


# ----------------------------------------------------------------------------- graph
def _is_bridge(v) -> bool:
    return str(v).lower() in BRIDGE_VALUES


def _pieces(line: LineString, piece_m: float):
    n = max(1, int(np.ceil(line.length / piece_m)))
    if n == 1:
        yield line
        return
    for i in range(n):
        yield substring(line, i * line.length / n, (i + 1) * line.length / n)


def build_road_graph(roads: gpd.GeoDataFrame, p: CutoffParams) -> nx.MultiGraph:
    """roads: lines in a projected CRS (metres). Node ids are (x, y) rounded to 0.1 m."""
    allowed = MOTORABLE | (PATHS if p.include_paths else set())
    hw = roads["highway"].astype(str) if "highway" in roads.columns else pd.Series("unclassified", index=roads.index)
    roads = roads[hw.isin(allowed)].explode(index_parts=False)
    roads = roads[roads.geometry.geom_type == "LineString"]
    key = lambda xy: (round(xy[0], 1), round(xy[1], 1))  # noqa: E731
    lines = list(roads.geometry.values)
    counts = Counter(key(c) for ln in lines for c in ln.coords)
    G = nx.MultiGraph()
    hws = roads["highway"].astype(str).values if "highway" in roads.columns else ["unclassified"] * len(lines)
    brs = roads["bridge"].map(_is_bridge).values if "bridge" in roads.columns else np.zeros(len(lines), bool)
    ids = roads["@osmId"].astype(str).values if "@osmId" in roads.columns else np.arange(len(lines)).astype(str)
    for ln, h, b, oid in zip(lines, hws, brs, ids):
        coords = list(ln.coords)
        start = 0
        for i in range(1, len(coords)):
            if i == len(coords) - 1 or counts[key(coords[i])] > 1:
                seg = LineString(coords[start:i + 1])
                start = i
                if seg.length == 0:
                    continue
                for pc in _pieces(seg, p.piece_m):
                    a, c = pc.coords[0], pc.coords[-1]
                    ua, uc = key(a), key(c)
                    if ua == uc:
                        continue
                    G.add_edge(ua, uc, length=float(pc.length), geom=pc, highway=h, bridge=bool(b), osm_id=oid)
    return G


def edge_hazard_table(G: nx.MultiGraph, hazard, p: CutoffParams) -> pd.DataFrame:
    """Per edge: metres inside hazard and whether a bridge-buffer touches it."""
    edges = list(G.edges(keys=True, data=True))
    geoms = np.array([d["geom"] for _, _, _, d in edges], dtype=object)
    n = len(edges)
    hit_len, near = np.zeros(n), np.zeros(n, bool)
    if n and not hazard.is_empty:
        shapely.prepare(hazard)
        cand = np.where(shapely.intersects(geoms, hazard))[0]
        if len(cand):
            hit_len[cand] = shapely.length(shapely.intersection(geoms[cand], hazard))
        isb = np.array([d["bridge"] for _, _, _, d in edges])
        if isb.any():
            bi = np.where(isb)[0]
            near[bi] = shapely.intersects(shapely.buffer(geoms[bi], p.bridge_buffer_m), hazard)
    return pd.DataFrame({"u": [e[0] for e in edges], "v": [e[1] for e in edges], "k": [e[2] for e in edges],
                         "bridge": [e[3]["bridge"] for e in edges], "hit_len": hit_len, "bridge_near": near})


def apply_blockage(G: nx.MultiGraph, table: pd.DataFrame, p: CutoffParams, scenario: str) -> nx.MultiGraph:
    """Copy of G without blocked pieces. scenario: 'optimistic' | 'conservative'."""
    if scenario not in ("optimistic", "conservative"):
        raise ValueError(scenario)
    if scenario == "optimistic":
        blocked = (~table["bridge"]) & (table["hit_len"] >= p.min_blocked_m)
    else:
        blocked = (((~table["bridge"]) & (table["hit_len"] >= p.min_blocked_m))
                   | (table["bridge"] & table["bridge_near"]))
    H = G.copy()
    H.remove_edges_from(list(table.loc[blocked, ["u", "v", "k"]].itertuples(index=False, name=None)))
    return H


# ----------------------------------------------------------------------------- settlements / destinations
def _centroids(g: gpd.GeoDataFrame):
    c = g.geometry.centroid
    return np.column_stack([c.x.values, c.y.values]) if len(g) else np.empty((0, 2))


def find_settlements(places: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame, p: CutoffParams) -> gpd.GeoDataFrame:
    rows = []
    if len(places) and "place" in places.columns:
        pl = places[places["place"].isin(SETTLEMENT_PLACES)]
        for _, r in pl.iterrows():
            rows.append({"name": r.get("name") if isinstance(r.get("name"), str) else f"unnamed {r['place']}",
                         "kind": r["place"], "source": "osm_place", "geometry": r.geometry.centroid})
    bxy = _centroids(buildings)
    if len(bxy) >= p.cluster_min_buildings:
        tree = cKDTree(bxy)
        pairs = tree.query_pairs(p.cluster_link_m, output_type="ndarray")
        m = len(bxy)
        ncomp, lab = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(m, m)),
                                          directed=False)
        existing = np.array([[g.x, g.y] for g in (r["geometry"] for r in rows)]) if rows else np.empty((0, 2))
        etree = cKDTree(existing) if len(existing) else None
        for c in range(ncomp):
            idx = np.where(lab == c)[0]
            if len(idx) < p.cluster_min_buildings:
                continue
            cen = bxy[idx].mean(axis=0)
            if etree is not None and etree.query(cen)[0] <= p.cluster_near_place_m:
                continue
            rows.append({"name": f"unnamed cluster ({len(idx)} buildings)", "kind": "cluster",
                         "source": "building_cluster", "geometry": Point(cen)})
    crs = places.crs or buildings.crs
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def find_destinations(places: gpd.GeoDataFrame, health: gpd.GeoDataFrame, hazard) -> gpd.GeoDataFrame:
    rows = []
    if len(places) and "place" in places.columns:
        for _, r in places[places["place"].isin(TOWN_PLACES)].iterrows():
            rows.append({"name": r.get("name"), "kind": f"place={r['place']}", "geometry": r.geometry.centroid})
    for _, r in health.iterrows():
        rows.append({"name": r.get("name"), "kind": r.get("amenity") or r.get("healthcare") or "health",
                     "geometry": r.geometry.centroid})
    d = gpd.GeoDataFrame(rows, geometry="geometry", crs=places.crs or health.crs)
    d["in_hazard"] = [bool(hazard.contains(g)) for g in d.geometry] if (len(d) and not hazard.is_empty) else False
    return d


# ----------------------------------------------------------------------------- reachability
def dist_to_nearest(G: nx.MultiGraph, dest_nodes: list) -> dict:
    """Road distance (m) from every node to its nearest destination node (multi-source Dijkstra)."""
    dest_nodes = [n for n in dest_nodes if n in G]
    if not dest_nodes:
        return {}
    return nx.multi_source_dijkstra_path_length(G, dest_nodes, weight="length")


def classify(pre_d, post_d, p: CutoffParams) -> str:
    """Pure function: distance to nearest destination before/after (m, None = unreachable)."""
    if pre_d is None:
        return "unreachable_pre"
    if post_d is None:
        return "cut_off"
    if post_d >= p.degraded_ratio * pre_d and (post_d - pre_d) / 1000 >= p.degraded_extra_km:
        return "degraded"
    return "ok"


def analyse(roads, places, health, buildings, flood, debris, p: CutoffParams | None = None):
    p = p or CutoffParams()
    hazard = unary_union([flood, debris])
    G = build_road_graph(roads, p)
    if G.number_of_nodes() == 0:
        raise RuntimeError("no motorable roads in the OSM snapshot; cannot run the cut-off analysis")
    table = edge_hazard_table(G, hazard, p)
    scen = {s: apply_blockage(G, table, p, s) for s in ("optimistic", "conservative")}
    nodes = list(G.nodes)
    ntree = cKDTree(np.array(nodes))

    sett = find_settlements(places, buildings, p)
    dests = find_destinations(places, health, hazard)
    sxy = np.array([[g.x, g.y] for g in sett.geometry]) if len(sett) else np.empty((0, 2))
    bxy = _centroids(buildings)
    btree = cKDTree(bxy) if len(bxy) else None

    def snap(xy):
        d, i = ntree.query(xy)
        return (nodes[i], float(d)) if d <= p.snap_m else (None, float(d))

    dest_nodes_pre = [snap([g.x, g.y])[0] for g in dests.geometry]
    dest_nodes_pre = [n for n in dest_nodes_pre if n is not None]
    ok_dest = [snap([g.x, g.y])[0] for g, h in zip(dests.geometry, dests["in_hazard"]) if not h] if len(dests) else []
    ok_dest = [n for n in ok_dest if n is not None]
    pre_dist = dist_to_nearest(G, dest_nodes_pre)
    post = {s: dist_to_nearest(H, ok_dest) for s, H in scen.items()}

    # nodes still connected to a surviving destination (for "distance to nearest intact road")
    intact = {s: np.array([n for n in post[s]]) for s in post}
    itree = {s: (cKDTree(a) if len(a) else None) for s, a in intact.items()}

    out = []
    for i, (xy, r) in enumerate(zip(sxy, sett.itertuples())):
        node, snap_d = snap(xy)
        n_b = int(len(btree.query_ball_point(xy, p.building_radius_m))) if btree is not None else 0
        row = {"name": r.name, "kind": r.kind, "source": r.source, "buildings_300m": n_b,
               "snap_distance_m": round(snap_d, 0),
               "in_hazard": bool(hazard.contains(Point(xy))) if not hazard.is_empty else False}
        if node is None:
            row.update(status_optimistic="no_road_access", status_conservative="no_road_access")
        else:
            pre = pre_dist.get(node)
            for s in ("optimistic", "conservative"):
                post_d = post[s].get(node)
                row[f"status_{s}"] = classify(pre, post_d, p)
                row[f"post_dist_km_{s}"] = None if post_d is None else round(post_d / 1000, 2)
                if post_d is None and itree[s] is not None:
                    row[f"to_intact_road_m_{s}"] = round(float(itree[s].query(xy)[0]), 0)
            row["pre_dist_km"] = None if pre is None else round(pre / 1000, 2)
        row["risk"] = overall_risk(row["status_optimistic"], row["status_conservative"])
        out.append(row)
    res = pd.DataFrame(out)
    if len(res):
        order = {"cut_off": 0, "cut_off_if_bridges_fail": 1, "degraded": 2, "ok": 3,
                 "unreachable_pre": 4, "no_road_access": 5}
        res["_o"] = res["risk"].map(order)
        res = res.sort_values(["_o", "buildings_300m"], ascending=[True, False]).drop(columns="_o")
    gdf = gpd.GeoDataFrame(res.reset_index(drop=True), geometry=list(sett.geometry[res.index]) if len(res) else [],
                           crs=sett.crs)
    qa = {
        "graph_nodes": G.number_of_nodes(), "graph_edges": G.number_of_edges(),
        "graph_components_pre": nx.number_connected_components(G),
        "edges_blocked_optimistic": G.number_of_edges() - scen["optimistic"].number_of_edges(),
        "edges_blocked_conservative": G.number_of_edges() - scen["conservative"].number_of_edges(),
        "destinations": int(len(dests)), "destinations_in_hazard": int(dests["in_hazard"].sum()) if len(dests) else 0,
        "settlements": int(len(gdf)),
        "settlements_without_road_access": int((gdf["risk"] == "no_road_access").sum()) if len(gdf) else 0,
        "settlements_unreachable_pre": int((gdf["risk"] == "unreachable_pre").sum()) if len(gdf) else 0,
    }
    return gdf, dests, qa


def overall_risk(opt: str, cons: str) -> str:
    if opt == "cut_off":
        return "cut_off"
    if cons == "cut_off":
        return "cut_off_if_bridges_fail"
    if "degraded" in (opt, cons):
        return "degraded"
    return opt if opt in ("ok", "unreachable_pre", "no_road_access") else cons


def summarise(gdf: gpd.GeoDataFrame, qa: dict) -> dict:
    counts = gdf["risk"].value_counts().to_dict() if len(gdf) else {}
    cut = gdf[gdf["risk"] == "cut_off"]
    maybe = gdf[gdf["risk"] == "cut_off_if_bridges_fail"]
    return {
        "counts_by_risk": {k: int(v) for k, v in counts.items()},
        "cut_off": [{"name": r["name"], "buildings_300m": int(r["buildings_300m"])} for _, r in cut.iterrows()],
        "cut_off_if_bridges_fail": [{"name": r["name"], "buildings_300m": int(r["buildings_300m"])}
                                    for _, r in maybe.iterrows()],
        "buildings_near_cut_off": int(cut["buildings_300m"].sum()) if len(cut) else 0,
        "buildings_near_cut_off_if_bridges_fail": int(maybe["buildings_300m"].sum()) if len(maybe) else 0,
        "qa": qa,
        "caveat": ("Road access only (walking paths excluded). OSM completeness varies, so settlements without a "
                   "mapped route are listed separately instead of being called cut off. Bridge survival is "
                   "unknown, hence two scenarios."),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--outputs", default="data/outputs")
    ap.add_argument("--osm", default="data/raw/osm/osm_pre_event.gpkg")
    ap.add_argument("--flood-source", default="auto", choices=["auto", "baseline", "ml", "union"])
    ap.add_argument("--include-paths", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_config(a.config)
    out, osm = Path(a.outputs), Path(a.osm)
    flood, debris, grid, src = load_hazard(out, a.flood_source)
    roads, places = load_layer(osm, "roads", grid.crs), load_layer(osm, "places", grid.crs)
    health, bld = load_layer(osm, "health", grid.crs), load_layer(osm, "buildings", grid.crs)
    gdf, dests, qa = analyse(roads, places, health, bld, flood, debris, CutoffParams(include_paths=a.include_paths))
    summ = summarise(gdf, qa) | {"flood_source": src}
    gp = out / "cutoff_settlements.gpkg"
    if gp.exists():
        gp.unlink()
    if len(gdf):
        gdf.to_file(gp, layer="settlements", driver="GPKG")
        gdf.drop(columns="geometry").to_csv(out / "cutoff_settlements.csv", index=False)
    if len(dests):
        dests.to_file(gp, layer="destinations", driver="GPKG")
    (out / "cutoff_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    log.info("cut off: %d | cut off if bridges fail: %d | %s", len(summ["cut_off"]),
             len(summ["cut_off_if_bridges_fail"]), qa)
    return 0


if __name__ == "__main__":
    sys.exit(main())
