from __future__ import annotations

import networkx as nx
import pytest

from src.config import GRAPH_PATH, LANDMARKS
from src.osm_graph import (
    DRIVE_HIGHWAY_TYPES,
    filter_drive_graph,
    graph_from_osm_xml,
    is_drive_highway,
    load_graph,
    nearest_node,
    nearest_node_with_distance,
    prepare_graph,
    to_routing_graph,
)
from src.routing import generate_candidates, graph_vmax_kph

CACHE_AVAILABLE = GRAPH_PATH.exists()


def synthetic_dirty_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    coordinates = {
        "drive_a": (67.1000, 24.9000),
        "drive_b": (67.1010, 24.9000),
        "footway_node": (67.1020, 24.9000),
        "pedestrian_node": (67.1030, 24.9000),
        "steps_node": (67.1040, 24.9000),
        "path_node": (67.1050, 24.9000),
        "cycleway_node": (67.1060, 24.9000),
        "track_node": (67.1070, 24.9000),
        "construction_node": (67.1080, 24.9000),
        "missing_highway_node": (67.1090, 24.9000),
    }
    for node, (x, y) in coordinates.items():
        graph.add_node(node, x=x, y=y)
    graph.add_edge(
        "drive_a",
        "drive_b",
        key=0,
        highway="residential",
        road_type="residential",
        length_m=100.0,
        speed_kph=36.0,
        base_time_s=10.0,
    )
    graph.add_edge(
        "drive_a",
        "drive_b",
        key=1,
        highway="footway",
        road_type="footway",
        length_m=100.0,
        speed_kph=36.0,
        base_time_s=1.0,
    )
    graph.add_edge(
        "drive_b",
        "drive_a",
        key=0,
        highway="service",
        road_type="service",
        length_m=100.0,
        speed_kph=36.0,
        base_time_s=10.0,
    )
    for node, highway in (
        ("footway_node", "footway"),
        ("pedestrian_node", "pedestrian"),
        ("steps_node", "steps"),
        ("path_node", "path"),
        ("cycleway_node", "cycleway"),
        ("track_node", "track"),
        ("construction_node", "construction"),
    ):
        graph.add_edge(
            "drive_b",
            node,
            key=0,
            highway=highway,
            road_type=highway,
            length_m=10.0,
            speed_kph=5.0,
            base_time_s=1.0,
        )
    graph.add_edge(
        "drive_b",
        "missing_highway_node",
        key=0,
        length_m=10.0,
        speed_kph=5.0,
        base_time_s=1.0,
    )
    return graph


def test_whitelist_accepts_drive_types() -> None:
    for highway in DRIVE_HIGHWAY_TYPES:
        assert is_drive_highway(highway)


def test_whitelist_rejects_non_drive_and_missing_types() -> None:
    for highway in (
        "footway",
        "pedestrian",
        "steps",
        "path",
        "cycleway",
        "track",
        "construction",
        None,
        "",
    ):
        assert not is_drive_highway(highway)


def test_whitelist_handles_graphml_lists() -> None:
    assert is_drive_highway("['service', 'residential']")
    assert is_drive_highway(["residential", "living_street"])
    assert not is_drive_highway(["service", "track"])
    assert not is_drive_highway(["steps", "footway"])


def test_filter_drive_graph_removes_non_drive_edges_and_nodes() -> None:
    filtered = filter_drive_graph(synthetic_dirty_graph())
    assert set(filtered.nodes) == {"drive_a", "drive_b"}
    assert filtered.number_of_edges() == 2
    assert all(
        is_drive_highway(data.get("highway"))
        for _, _, _, data in filtered.edges(keys=True, data=True)
    )


def test_prepare_graph_is_drive_only() -> None:
    prepared = prepare_graph(synthetic_dirty_graph())
    assert set(prepared.nodes) == {"drive_a", "drive_b"}
    assert all(
        is_drive_highway(data.get("highway"))
        for _, _, _, data in prepared.edges(keys=True, data=True)
    )
    assert all(
        "base_time_s" in data for _, _, _, data in prepared.edges(keys=True, data=True)
    )


def test_to_routing_graph_ignores_non_drive_parallel_edge() -> None:
    routing = to_routing_graph(synthetic_dirty_graph())
    assert set(routing.nodes) == {"drive_a", "drive_b"}
    assert routing.has_edge("drive_a", "drive_b")
    assert routing["drive_a"]["drive_b"]["road_type"] == "residential"
    assert routing["drive_a"]["drive_b"]["base_time_s"] == 10.0
    assert all(
        data["road_type"] in DRIVE_HIGHWAY_TYPES
        for _, _, data in routing.edges(data=True)
    )


def test_cached_graphml_load_filters_dirty_graph(tmp_path) -> None:
    ox = pytest.importorskip("osmnx")
    path = tmp_path / "dirty.graphml"
    dirty = nx.convert_node_labels_to_integers(synthetic_dirty_graph())
    ox.save_graphml(dirty, path)
    loaded = load_graph(path)
    assert loaded.number_of_nodes() == 2
    assert loaded.number_of_edges() == 2
    assert all(
        is_drive_highway(data.get("highway"))
        for _, _, _, data in loaded.edges(keys=True, data=True)
    )


def test_osm_xml_load_filters_drive_only(tmp_path) -> None:
    pytest.importorskip("osmnx")
    path = tmp_path / "dirty.osm"
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <node id="1" lat="24.9000" lon="67.1000"/>
  <node id="2" lat="24.9001" lon="67.1000"/>
  <node id="3" lat="24.9002" lon="67.1000"/>
  <node id="4" lat="24.9001" lon="67.1001"/>
  <way id="10">
    <nd ref="1"/><nd ref="2"/>
    <tag k="highway" v="residential"/>
    <tag k="oneway" v="no"/>
  </way>
  <way id="11">
    <nd ref="2"/><nd ref="3"/>
    <tag k="highway" v="footway"/>
  </way>
  <way id="12">
    <nd ref="2"/><nd ref="4"/>
  </way>
</osm>
""",
        encoding="utf-8",
    )
    graph = graph_from_osm_xml(path)
    assert set(graph.nodes) == {1, 2}
    assert graph.number_of_edges() == 2
    assert all(
        is_drive_highway(data.get("highway"))
        for _, _, _, data in graph.edges(keys=True, data=True)
    )


def test_nearest_node_with_distance_reports_distance() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=67.12596, y=24.90930)
    graph.add_node(2, x=67.12696, y=24.90930)
    node, distance = nearest_node_with_distance(graph, 24.90930, 67.12796)
    assert node == 2
    assert 50.0 < distance < 300.0
    assert nearest_node(graph, 24.90930, 67.12796) == node


@pytest.fixture(scope="module")
def cached_graph() -> nx.MultiDiGraph:
    if not CACHE_AVAILABLE:
        pytest.skip("Run scripts/fetch_graph.py to create the local graph cache.")
    pytest.importorskip("osmnx")
    return load_graph(GRAPH_PATH)


def test_cached_graph_is_drive_only(cached_graph: nx.MultiDiGraph) -> None:
    assert cached_graph.number_of_edges() > 0
    non_drive = [
        (start, end, key, data.get("highway"))
        for start, end, key, data in cached_graph.edges(keys=True, data=True)
        if not is_drive_highway(data.get("highway"))
    ]
    assert not non_drive, non_drive[:5]


def test_cached_routing_graph_excludes_non_drive(cached_graph: nx.MultiDiGraph) -> None:
    routing = to_routing_graph(cached_graph)
    assert routing.number_of_edges() > 0
    assert all(
        data["road_type"] in DRIVE_HIGHWAY_TYPES
        for _, _, data in routing.edges(data=True)
    )


def test_cached_safari_park_to_johar_is_drive_only_and_diverse(
    cached_graph: nx.MultiDiGraph,
) -> None:
    routing = to_routing_graph(cached_graph)
    start = nearest_node(cached_graph, *LANDMARKS["Safari Park"])
    goal = nearest_node(cached_graph, *LANDMARKS["Johar Chowk Mosque"])
    candidates = generate_candidates(
        routing,
        start,
        goal,
        vmax_kph=graph_vmax_kph(routing),
    )
    assert candidates
    for candidate in candidates:
        for start_node, end_node in candidate.edges:
            assert (
                routing[start_node][end_node]["road_type"] in DRIVE_HIGHWAY_TYPES
            )
    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            shared = len(set(first.edges) & set(second.edges))
            overlap = shared / max(1, min(len(first.edges), len(second.edges)))
            assert overlap <= 0.70
