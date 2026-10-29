from __future__ import annotations

from typing import Any

import networkx as nx

from .core import PathResult, search


def dijkstra(
    graph: nx.DiGraph,
    start: int,
    goal: int,
    penalty_edges: frozenset[tuple[int, int]] = frozenset(),
    penalty_factor: float = 1.0,
) -> PathResult | None:
    return search(
        graph,
        start,
        goal,
        use_heuristic=False,
        penalty_edges=penalty_edges,
        penalty_factor=penalty_factor,
    )


def dijkstra_path(
    graph: nx.DiGraph,
    start: int,
    goal: int,
    **kwargs: Any,
) -> PathResult | None:
    return dijkstra(graph, start, goal, **kwargs)
