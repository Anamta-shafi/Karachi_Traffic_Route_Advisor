from __future__ import annotations

import networkx as nx

from ..config import MAX_SPEED_KPH
from .astar import astar
from .core import PathResult


def overlap_ratio(
    first: tuple[tuple[int, int], ...],
    second: tuple[tuple[int, int], ...],
) -> float:
    if not first or not second:
        return 0.0
    shared = len(set(first) & set(second))
    return shared / max(1, min(len(first), len(second)))


def generate_candidates(
    graph: nx.DiGraph,
    start: int,
    goal: int,
    max_alternatives: int = 3,
    vmax_kph: float = MAX_SPEED_KPH,
) -> list[PathResult]:
    primary = astar(graph, start, goal, vmax_kph=vmax_kph)
    if primary is None:
        return []

    candidates = [primary]
    penalty_edges = set(primary.edges)
    for factor in (1.8, 3.0, 5.0):
        if len(candidates) >= 1 + max_alternatives:
            break
        candidate = astar(
            graph,
            start,
            goal,
            penalty_edges=frozenset(penalty_edges),
            penalty_factor=factor,
            vmax_kph=vmax_kph,
        )
        if candidate is None:
            continue
        if any(candidate.edges == existing.edges for existing in candidates):
            continue
        if primary.base_time_s > 0 and candidate.base_time_s / primary.base_time_s > 1.45:
            continue
        if any(
            overlap_ratio(candidate.edges, existing.edges) > 0.70
            for existing in candidates
        ):
            continue
        candidates.append(candidate)
        penalty_edges.update(candidate.edges)

    return candidates


def graph_vmax_kph(graph: nx.DiGraph) -> float:
    speeds = [float(data.get("speed_kph", 0.0)) for _, _, data in graph.edges(data=True)]
    return max(speeds, default=MAX_SPEED_KPH)


_overlap_ratio = overlap_ratio
