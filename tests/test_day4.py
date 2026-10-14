
import numpy as np
import pytest
from shapely.geometry import LineString
from shapely.ops import unary_union

from src import cutoff, damage
from src.demo_data import build
from src.damage import load_hazard, load_layer, mask_to_polygons


@pytest.fixture(scope="module")
def town(tmp_path_factory):
    d = build(tmp_path_factory.mktemp("demo"))
    flood, debris, grid, src = load_hazard(d["outputs"], "baseline")
    return d, flood, debris, grid


def test_mask_to_polygons_roundtrip(town):
    d, flood, debris, grid = town
    assert flood.area == pytest.approx(80 * 4000 + 100 * 110, rel=1e-6)
    assert mask_to_polygons(np.zeros((5, 5), np.uint8), grid).is_empty


def test_building_damage_counts_and_severity(town):
    d, flood, debris, grid = town
    bld = load_layer(d["osm"], "buildings", grid.crs)
    hit = damage.damaged_buildings(bld, flood, debris)
    assert len(bld) == 12 * 7 + 6
    assert len(hit) == 6                                    # 5 in the corridor + 1 grazing its edge
    assert (hit["severity"] == "high").sum() == 5 and (hit["hazard"] == "flood").all()
    assert hit["centroid_hit"].sum() == 5
    assert hit.loc[hit["centroid_hit"] == False, "overlap_frac"].iloc[0] < 0.5  # noqa: E712


def test_roads_and_bridges(town):
    d, flood, debris, grid = town
    roads = load_layer(d["osm"], "roads", grid.crs)
    hz = unary_union([flood, debris])
    rh = damage.roads_in_hazard(roads, hz)
    # highway crossing the river corridor (80 m) + flooded V5 spur (110 m) + 0 elsewhere
    assert rh["length_hit_m"].sum() == pytest.approx(80 + 110, abs=1.0)
    br = damage.bridges_exposed(load_layer(d["osm"], "bridges", grid.crs), flood, debris)
    assert len(br) == 1 and br["hazard"].iloc[0] == "flood"


def test_damage_run_summary(town):
    d, flood, debris, grid = town
    *_, s = damage.run(d["osm"], flood, debris, grid, "baseline")
    assert s["buildings_hit"] == 6 and s["bridges_total_osm"] == 1 and s["bridges_exposed"] == 1
    assert s["road_km_in_hazard"] == pytest.approx(0.19, abs=0.002)
    assert "lower bounds" in s["caveat"]


@pytest.fixture(scope="module")
def cut(town):
    d, flood, debris, grid = town
    g = lambda n: load_layer(d["osm"], n, grid.crs)  # noqa: E731
    gdf, dests, qa = cutoff.analyse(g("roads"), g("places"), g("health"), g("buildings"), flood, debris)
    return gdf.set_index("name"), dests, qa


def test_cutoff_scenarios_match_known_answers(cut):
    gdf, dests, qa = cut
    risk = gdf["risk"].to_dict()
    assert risk["Village 5"] == "cut_off"                              # flooded ford: both scenarios
    assert risk["Village 2"] == risk["Village 4"] == "cut_off_if_bridges_fail"
    assert risk["Village 1"] == risk["Village 3"] == "ok"
    assert risk["Village 7"] == "no_road_access"                       # no road within snap distance
    assert risk["Village 8"] == "unreachable_pre"                      # never had a route: not blamed on the flood
    assert gdf.loc["Village 2", "status_optimistic"] == "ok" and gdf.loc["Village 2", "status_conservative"] == "cut_off"
    assert qa["edges_blocked_conservative"] > qa["edges_blocked_optimistic"] > 0
    assert qa["destinations"] == 2 and qa["destinations_in_hazard"] == 0


def test_cutoff_distances_and_ranking(cut):
    gdf, _, _ = cut
    assert gdf.loc["Village 1", "pre_dist_km"] == pytest.approx(gdf.loc["Village 1", "post_dist_km_optimistic"])
    assert gdf.loc["Village 5", "to_intact_road_m_optimistic"] > 0
    assert gdf["buildings_300m"].min() >= 0 and gdf.loc["Village 2", "buildings_300m"] >= 12
    assert gdf.iloc[0]["risk"] == "cut_off"                            # worst first


def test_destination_in_hazard_is_not_a_refuge(town):
    d, flood, debris, grid = town
    g = lambda n: load_layer(d["osm"], n, grid.crs)  # noqa: E731
    hospital_zone = g("health").geometry.iloc[0].buffer(100)
    gdf, dests, qa = cutoff.analyse(g("roads"), g("places"), g("health"), g("buildings"),
                                    unary_union([flood, hospital_zone]), debris)
    assert qa["destinations_in_hazard"] == 2          # hospital and the town both inside the hazard
    assert (gdf["status_optimistic"] == "cut_off").sum() >= 4


def test_classify_degraded_and_graph_construction():
    p = cutoff.CutoffParams()
    assert cutoff.classify(None, 5000, p) == "unreachable_pre"
    assert cutoff.classify(5000, None, p) == "cut_off"
    assert cutoff.classify(5000, 10000, p) == "degraded"
    assert cutoff.classify(5000, 5500, p) == "ok"
    assert cutoff.classify(1000, 3500, p) == "degraded"   # 3.5x and +2.5 km
    assert cutoff.classify(1000, 2500, p) == "ok"         # 2.5x but only +1.5 km: below the absolute floor


def test_graph_junctions_and_crossings():
    import geopandas as gpd
    # two ways sharing a vertex connect; two ways that merely cross do not; footpaths excluded
    ways = gpd.GeoDataFrame({"highway": ["primary", "primary", "primary", "primary", "footway"],
                             "geometry": [LineString([(0, 0), (100, 0), (200, 0)]), LineString([(100, 0), (100, 100)]),
                                          LineString([(300, -50), (300, 50)]), LineString([(250, 0), (350, 0)]),
                                          LineString([(0, 0), (0, 500)])]}, crs="EPSG:32645")
    G = cutoff.build_road_graph(ways, cutoff.CutoffParams())
    assert cutoff.nx.number_connected_components(G) == 3     # {way1+way2}, {way3}, {way4}: crossing without a shared node
    assert (0.0, 500.0) not in G
    assert all(d["length"] <= 100.0 + 1e-6 for *_, d in G.edges(data=True))


# ---------------------------------------------------------------- flow path
def test_priority_flood_fills_pit_and_trace_follows_valley():
    from src import flowpath
    z = np.add.outer(np.linspace(100, 0, 60), np.zeros(60))        # slopes south; row 0 high
    z += np.abs(np.arange(60) - 30)[None, :] * 2.0                  # V valley along column 30
    z[40, 30] -= 15                                                 # a pit on the thalweg
    filled = flowpath.priority_flood(z)
    assert filled[40, 30] >= z[40, 30] and not np.isnan(filled).any()
    path = flowpath.trace_downstream(filled, (2, 31))
    cols = [c for _, c in path]
    assert path[-1][0] == 59 and abs(np.mean(cols[-20:]) - 30) < 2   # leaves through the south border, in the valley


def test_flow_path_and_settlements_along_it(town):
    from shapely.geometry import Point
    import geopandas as gpd
    from src import flowpath
    d, flood, debris, grid = town
    h, w = grid.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dem = (1000 + ((h - yy) * 0.5) + np.abs(xx - 300) * 3.0).astype(np.float32)   # valley at column 300, draining south
    start = (grid.transform.c + 300 * grid.res, grid.transform.f - 5 * grid.res)
    line = flowpath.flow_path(dem, grid, start, coarse_m=30.0)
    assert line.length > 0.8 * (h - 10) * grid.res and abs(line.coords[-1][0] - (grid.transform.c + 3000)) < 150
    pts = gpd.GeoDataFrame({"name": ["near", "far"], "place": ["village", "village"],
                            "geometry": [Point(grid.transform.c + 3050, grid.transform.f - 3000),
                                         Point(grid.transform.c + 6000, grid.transform.f - 3000)]}, crs=grid.crs)
    along = flowpath.settlements_along(line, pts, 300)
    assert list(along["name"]) == ["near"] and along["km_downstream"].iloc[0] > 2.0
