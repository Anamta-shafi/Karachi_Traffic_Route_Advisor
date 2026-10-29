from __future__ import annotations

import networkx as nx
import pytest

from src.routing import dijkstra, generate_candidates, graph_vmax_kph, search


def build_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    for node, x, y in (
        ("a", 0.0, 0.0),
        ("b", 0.01, 0.0),
        ("c", 0.0, 0.01),
        ("d", 0.01, 0.01),
    ):
        graph.add_node(node, x=x, y=y)
    graph.add_edge("a", "b", base_time_s=10.0, length_m=100.0, speed_kph=36.0)
    graph.add_edge("b", "d", base_time_s=10.0, length_m=100.0, speed_kph=36.0)
    graph.add_edge("a", "c", base_time_s=12.0, length_m=120.0, speed_kph=36.0)
    graph.add_edge("c", "d", base_time_s=13.0, length_m=130.0, speed_kph=36.0)
    return graph


def test_astar_and_dijkstra_match() -> None:
    graph = build_graph()
    astar_result = search(graph, "a", "d")
    dijkstra_result = dijkstra(graph, "a", "d")
    assert astar_result is not None
    assert dijkstra_result is not None
    assert astar_result.nodes == ("a", "b", "d")
    assert dijkstra_result.nodes == ("a", "b", "d")
    assert astar_result.base_time_s == 20.0
    assert astar_result.base_time_s == pytest.approx(dijkstra_result.base_time_s)
    assert astar_result.expanded_nodes > 0
    assert dijkstra_result.expanded_nodes >= astar_result.expanded_nodes


def test_no_route() -> None:
    graph = build_graph()
    assert search(graph, "d", "a") is None


def test_disconnected_component_has_no_route() -> None:
    graph = build_graph()
    graph.add_node("island", x=0.02, y=0.02)
    assert search(graph, "a", "island") is None


def test_missing_start_or_goal_returns_none() -> None:
    graph = build_graph()
    assert search(graph, "missing", "d") is None
    assert search(graph, "a", "missing") is None


def test_start_equals_goal() -> None:
    graph = build_graph()
    result = search(graph, "a", "a")
    assert result is not None
    assert result.nodes == ("a",)
    assert result.edges == ()
    assert result.base_time_s == 0.0
    assert result.penalized_time_s == 0.0
    assert result.distance_m == 0.0


def test_alternatives_are_diverse() -> None:
    graph = build_graph()
    candidates = generate_candidates(graph, "a", "d")
    assert len(candidates) == 2
    assert candidates[0].nodes == ("a", "b", "d")
    assert candidates[1].nodes == ("a", "c", "d")


def build_pairwise_overlap_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    primary = ["a", "p1", "p2", "d"]
    first = ["a", "a1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "d"]
    second = ["a", "b1", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "d"]
    for node in set(primary + first + second):
        graph.add_node(node, x=0.0, y=0.0)
    for start, end in zip(primary, primary[1:]):
        graph.add_edge(start, end, base_time_s=3.3, length_m=33.0, speed_kph=36.0)
    for start, end in zip(first, first[1:]):
        cost = 1.0
        graph.add_edge(start, end, base_time_s=cost, length_m=10.0, speed_kph=36.0)
    for start, end in zip(second, second[1:]):
        cost = 2.5 if (start, end) == ("a", "b1") else 1.0
        graph.add_edge(start, end, base_time_s=cost, length_m=cost * 10.0, speed_kph=36.0)
    return graph


def test_alternatives_reject_pairwise_overlap() -> None:
    graph = build_pairwise_overlap_graph()
    candidates = generate_candidates(graph, "a", "d")
    assert len(candidates) == 2
    assert candidates[0].nodes == ("a", "p1", "p2", "d")
    assert candidates[1].nodes == (
        "a",
        "a1",
        "m1",
        "m2",
        "m3",
        "m4",
        "m5",
        "m6",
        "m7",
        "m8",
        "d",
    )
    first_edges = set(candidates[0].edges)
    second_edges = set(candidates[1].edges)
    assert len(first_edges & second_edges) == 0


def test_penalty_changes_choice() -> None:
    graph = build_graph()
    result = search(
        graph,
        "a",
        "d",
        penalty_edges=frozenset({("a", "b"), ("b", "d")}),
        penalty_factor=3.0,
    )
    assert result is not None
    assert result.nodes == ("a", "c", "d")


def test_graph_vmax() -> None:
    graph = build_graph()
    assert graph_vmax_kph(graph) == 36.0
