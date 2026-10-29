from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from .contracts import MapboxRoute, RouteCandidate, RouteContext
from .eta import route_base_time_min, route_eta_min, route_risk

def route_geometry(
    route: RouteCandidate | MapboxRoute,
) -> tuple[tuple[float, float], ...]:
    return tuple(route.geometry)


def route_fingerprint(route: RouteCandidate | MapboxRoute) -> str:
    payload = {
        "source": route.source if isinstance(route, RouteCandidate) else "mapbox",
        "distance_m": round(float(route.distance_m), 3),
        "geometry": [
            [round(float(lat), 6), round(float(lon), 6)]
            for lat, lon in route.geometry
        ],
    }
    if isinstance(route, RouteCandidate):
        payload["nodes"] = list(route.nodes)
        payload["edges"] = [list(edge) for edge in route.edges]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def road_composition(
    graph: Any,
    nodes: tuple[int, ...],
) -> tuple[tuple[str, float], ...]:
    lengths: dict[str, float] = {}
    total = 0.0
    for start, end in zip(nodes, nodes[1:]):
        data = graph.edges[start, end]
        road = data.get("road_type", data.get("highway", "unclassified"))
        if isinstance(road, (list, tuple)):
            road = road[0] if road else "unclassified"
        length = float(data.get("length_m", data.get("length", 0.0)))
        lengths[str(road)] = lengths.get(str(road), 0.0) + length
        total += length
    if total <= 0:
        return ()
    return tuple(
        sorted(
            ((road, length / total) for road, length in lengths.items()),
            key=lambda item: item[1],
            reverse=True,
        )
    )


def route_context_band(
    *,
    risk: float | None,
    unlit_ratio: float | None,
    delay_ratio: float | None,
    night: bool,
) -> str:
    if risk is None and unlit_ratio is None and delay_ratio is None:
        return "unknown"
    if (risk is not None and risk >= 0.70) or (
        night and unlit_ratio is not None and unlit_ratio >= 0.50
    ):
        return "high"
    if (risk is not None and risk >= 0.50) or (
        delay_ratio is not None and delay_ratio >= 0.40
    ):
        return "moderate"
    return "low"


def extract_route_context(
    route: RouteCandidate | MapboxRoute,
    *,
    graph: Any | None,
    departure: datetime,
    weather_condition: str = "dry",
) -> RouteContext:
    fingerprint = route_fingerprint(route)
    distance_km = route.distance_m / 1000.0
    night = departure.hour >= 19 or departure.hour < 6

    if isinstance(route, RouteCandidate) and graph is not None and route.nodes:
        ordinary_eta = route_eta_min(
            graph, route.nodes, departure, weather_condition
        )
        free_flow_eta = route_base_time_min(graph, route.nodes)
        risk = route_risk(graph, route.nodes, departure)
        composition = road_composition(graph, route.nodes)
        delay_ratio = (
            (ordinary_eta - free_flow_eta) / free_flow_eta
            if free_flow_eta > 0
            else None
        )
        band = route_context_band(
            risk=risk,
            unlit_ratio=route.unlit_ratio,
            delay_ratio=delay_ratio,
            night=night,
        )
        return RouteContext(
            distance_km=distance_km,
            ordinary_eta_min=ordinary_eta,
            free_flow_eta_min=free_flow_eta,
            delay_ratio=delay_ratio,
            risk=risk,
            unlit_ratio=route.unlit_ratio,
            road_composition=composition,
            context_band=band,
            context_basis=(
                "OSM route structure and analytical ETA. This is road context, "
                "not a crash history."
            ),
            fingerprint=fingerprint,
            available=True,
        )

    if isinstance(route, MapboxRoute):
        return RouteContext(
            distance_km=distance_km,
            ordinary_eta_min=route.duration_s / 60.0,
            free_flow_eta_min=(
                route.duration_typical_s / 60.0
                if route.duration_typical_s is not None
                else None
            ),
            delay_ratio=(
                route.duration_delta_s / route.duration_typical_s
                if route.duration_delta_s is not None
                and route.duration_typical_s
                else None
            ),
            risk=None,
            unlit_ratio=None,
            road_composition=(),
            context_band="unknown",
            context_basis=(
                "Mapbox route geometry has no OSM edge-level road context. "
                "Requested-departure Mapbox evidence is reported separately."
            ),
            fingerprint=fingerprint,
            available=True,
        )

    return RouteContext(
        distance_km=distance_km,
        ordinary_eta_min=None,
        free_flow_eta_min=None,
        delay_ratio=None,
        risk=route.risk if isinstance(route, RouteCandidate) else None,
        unlit_ratio=(
            route.unlit_ratio if isinstance(route, RouteCandidate) else None
        ),
        road_composition=(),
        context_band="unknown",
        context_basis="Route context is unavailable without the OSM graph.",
        fingerprint=fingerprint,
        available=False,
    )
