from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from typing import Any

import networkx as nx

from ..config import MAX_SPEED_KPH
from ..eta import haversine_m


@dataclass(frozen=True)
class PathResult:
    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    base_time_s: float
    penalized_time_s: float
    distance_m: float
    expanded_nodes: int = 0


def heuristic(graph: nx.DiGraph, node: int, goal: int, vmax_kph: float) -> float:
    start = graph.nodes[node]
    end = graph.nodes[goal]
    return haversine_m(
        float(start["y"]),
        float(start["x"]),
        float(end["y"]),
        float(end["x"]),
    ) / (max(vmax_kph, 1.0) * 1000.0 / 3600.0)


def search(
    graph: nx.DiGraph,
    start: int,
    goal: int,
    use_heuristic: bool = True,
    penalty_edges: frozenset[tuple[int, int]] = frozenset(),
    penalty_factor: float = 1.0,
    vmax_kph: float = MAX_SPEED_KPH,
) -> PathResult | None:
    if start not in graph or goal not in graph:
        return None
    if start == goal:
        return PathResult((start,), (), 0.0, 0.0, 0.0, expanded_nodes=1)

    counter = itertools.count()
    expanded_nodes = 0
    open_heap: list[tuple[float, float, int, int]] = []
    initial_h = heuristic(graph, start, goal, vmax_kph) if use_heuristic else 0.0
    heapq.heappush(open_heap, (initial_h, 0.0, next(counter), start))
    g_score: dict[int, float] = {start: 0.0}
    parent: dict[int, int] = {}
    closed: set[int] = set()

    while open_heap:
        _, current_g, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        expanded_nodes += 1
        if current == goal:
            return reconstruct(
                graph,
                parent,
                start,
                goal,
                current_g,
                expanded_nodes=expanded_nodes,
            )
        closed.add(current)

        for neighbor, data in graph[current].items():
            if neighbor in closed:
                continue
            weight = float(data.get("base_time_s", 0.0))
            if (current, neighbor) in penalty_edges:
                weight *= penalty_factor
            tentative = current_g + weight
            if tentative >= g_score.get(neighbor, float("inf")):
                continue
            g_score[neighbor] = tentative
            parent[neighbor] = current
            h = heuristic(graph, neighbor, goal, vmax_kph) if use_heuristic else 0.0
            heapq.heappush(open_heap, (tentative + h, tentative, next(counter), neighbor))

    return None


def reconstruct(
    graph: nx.DiGraph,
    parent: dict[int, int],
    start: int,
    goal: int,
    penalized_time_s: float,
    expanded_nodes: int = 0,
) -> PathResult:
    nodes = [goal]
    current = goal
    while current != start:
        current = parent[current]
        nodes.append(current)
    nodes.reverse()
    edges = tuple(zip(nodes, nodes[1:]))
    base_time_s = sum(float(graph.edges[u, v]["base_time_s"]) for u, v in edges)
    distance_m = sum(float(graph.edges[u, v]["length_m"]) for u, v in edges)
    return PathResult(
        nodes=tuple(nodes),
        edges=edges,
        base_time_s=base_time_s,
        penalized_time_s=penalized_time_s,
        distance_m=distance_m,
        expanded_nodes=expanded_nodes,
    )
