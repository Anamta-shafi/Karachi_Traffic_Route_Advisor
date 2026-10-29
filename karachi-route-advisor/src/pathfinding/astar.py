from __future__ import annotations

from typing import Any

import networkx as nx

from ..config import MAX_SPEED_KPH
from .core import PathResult, search


def astar(
    graph: nx.DiGraph,
    start: int,
    goal: int,
    penalty_edges: frozenset[tuple[int, int]] = frozenset(),
    penalty_factor: float = 1.0,
    vmax_kph: float = MAX_SPEED_KPH,
) -> PathResult | None:
    return search(
        graph,
        start,
        goal,
        use_heuristic=True,
        penalty_edges=penalty_edges,
        penalty_factor=penalty_factor,
        vmax_kph=vmax_kph,
    )


def astar_path(
    graph: nx.DiGraph,
    start: int,
    goal: int,
    **kwargs: Any,
) -> PathResult | None:
    return astar(graph, start, goal, **kwargs)
