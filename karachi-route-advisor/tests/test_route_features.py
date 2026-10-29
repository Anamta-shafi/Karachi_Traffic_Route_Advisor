from __future__ import annotations

from datetime import datetime

import networkx as nx
import pytest

from src.contracts import MapboxRoute, RouteCandidate
from src.route_features import (
    extract_route_context,
    road_composition,
    route_context_band,
    route_fingerprint,
)


def custom_route(**overrides) -> RouteCandidate:
    values = {
        "route_id": "osm-1",
        "source": "custom_astar",
        "provider": "OSM analytical",
        "geometry": ((24.9, 67.1), (24.91, 67.11)),
        "distance_m": 5000.0,
        "eta_min": 12.0,
        "risk": 0.3,
        "confidence": 0.55,
        "congestion": None,
        "incident_count": 0,
        "closure_count": 0,
        "traffic_available": False,
        "nodes": (1, 2),
        "edges": ((1, 2),),
        "unlit_ratio": 0.1,
    }
    values.update(overrides)
    return RouteCandidate(**values)


def mapbox_route(**overrides) -> MapboxRoute:
    values = {
        "route_id": "mapbox-1",
        "profile": "mapbox/driving-traffic",
        "duration_s": 900.0,
        "duration_typical_s": 750.0,
        "distance_m": 5000.0,
        "geometry": ((24.9, 67.1), (24.91, 67.11)),
        "congestion_values": ("moderate",),
        "congestion_numeric": (50,),
        "incident_count": 0,
        "closure_count": 0,
        "traffic_evidence": True,
        "fetched_at": "request-time",
    }
    values.update(overrides)
    return MapboxRoute(**values)


def test_route_fingerprint_is_stable_and_sensitive() -> None:
    first = route_fingerprint(custom_route())
    second = route_fingerprint(custom_route())
    assert first == second
    assert len(first) == 64
    assert first != route_fingerprint(custom_route(distance_m=5001.0))
    assert first != route_fingerprint(custom_route(geometry=((24.9, 67.1),)))


def test_road_composition_uses_edge_attributes() -> None:
    graph = nx.DiGraph()
    graph.add_edge(1, 2, road_type="primary", length_m=600.0)
    graph.add_edge(2, 3, road_type="residential", length_m=400.0)
    composition = road_composition(graph, (1, 2, 3))
    assert composition == (("primary", 0.6), ("residential", 0.4))


def test_context_band_thresholds() -> None:
    assert route_context_band(
        risk=0.70, unlit_ratio=0.0, delay_ratio=0.0, night=False
    ) == "high"
    assert route_context_band(
        risk=0.50, unlit_ratio=0.0, delay_ratio=0.0, night=False
    ) == "moderate"
    assert route_context_band(
        risk=0.10, unlit_ratio=0.0, delay_ratio=0.40, night=False
    ) == "moderate"
    assert route_context_band(
        risk=0.10, unlit_ratio=0.50, delay_ratio=0.0, night=True
    ) == "high"
    assert route_context_band(
        risk=None, unlit_ratio=None, delay_ratio=None, night=False
    ) == "unknown"


def test_custom_route_context_uses_graph_and_is_not_mapbox() -> None:
    graph = nx.DiGraph()
    graph.add_edge(
        1,
        2,
        road_type="primary",
        length_m=5000.0,
        speed_kph=50.0,
        surface="asphalt",
        lit=True,
    )
    context = extract_route_context(
        custom_route(),
        graph=graph,
        departure=datetime(2026, 10, 30, 18, 0),
    )
    assert context.available is True
    assert context.distance_km == 5.0
    assert context.ordinary_eta_min is not None
    assert context.free_flow_eta_min is not None
    assert context.road_composition == (("primary", 1.0),)
    assert "not a crash history" in context.context_basis
    assert context.fingerprint


def test_mapbox_route_context_is_separate_and_unknown() -> None:
    context = extract_route_context(
        mapbox_route(),
        graph=None,
        departure=datetime(2026, 10, 30, 18, 0),
    )
    assert context.available is True
    assert context.context_band == "unknown"
    assert context.ordinary_eta_min == 15.0
    assert context.free_flow_eta_min == 12.5
    assert context.delay_ratio == pytest.approx(0.2)
    assert "reported separately" in context.context_basis
