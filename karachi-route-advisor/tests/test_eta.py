from __future__ import annotations

from datetime import datetime

import networkx as nx

from src.config import time_factor
from src.eta import (
    edge_risk,
    parse_maxspeed,
    route_eta_min,
    route_impassable,
    route_unlit_ratio,
)


def build_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_edge(
        "a",
        "b",
        length_m=1000.0,
        speed_kph=30.0,
        base_time_s=120.0,
        road_type="residential",
        surface="asphalt",
        lit="no",
        toll=False,
        access="",
    )
    return graph


def test_parse_maxspeed() -> None:
    assert round(parse_maxspeed("50 mph"), 3) == 80.467
    assert parse_maxspeed("40") == 40.0
    assert parse_maxspeed(["30", "40"]) == 30.0
    assert parse_maxspeed(None) is None


def test_time_factors() -> None:
    assert time_factor(8) == 0.65
    assert time_factor(12) == 0.90
    assert time_factor(18) == 0.55
    assert time_factor(23) == 1.10


def test_route_eta() -> None:
    graph = build_graph()
    noon = datetime(2026, 10, 26, 12, 0)
    morning = datetime(2026, 10, 26, 8, 0)
    assert round(route_eta_min(graph, ("a", "b"), noon, "dry"), 3) == 2.222
    assert round(route_eta_min(graph, ("a", "b"), morning, "dry"), 3) == 3.077


def test_edge_risk_and_lighting() -> None:
    data = {
        "road_type": "residential",
        "surface": "unpaved",
        "lit": "no",
        "speed_kph": 30.0,
    }
    assert edge_risk(data, night=True) > edge_risk(data, night=False)


def test_route_flags() -> None:
    graph = build_graph()
    assert route_unlit_ratio(graph, ("a", "b")) == 1.0
    assert route_impassable(graph, ("a", "b")) is False
