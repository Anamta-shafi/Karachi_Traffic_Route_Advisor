from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Mapping

from .config import (
    DEFAULT_MAX_SPEED_KPH,
    HIGHWAY_SPEEDS_KPH,
    WEATHER_FACTORS,
    time_factor,
)

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def parse_maxspeed(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            parsed = parse_maxspeed(item)
            if parsed is not None:
                return parsed
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if ";" in text:
        text = text.split(";", 1)[0].strip()
    if "mph" in text:
        number = _first_number(text)
        return number * 1.609344 if number is not None else None
    if "knot" in text:
        number = _first_number(text)
        return number * 1.852 if number is not None else None
    number = _first_number(text)
    return number


def _first_number(text: str) -> float | None:
    current = []
    for char in text:
        if char.isdigit() or char == ".":
            current.append(char)
        elif current:
            break
    if not current:
        return None
    try:
        return float("".join(current))
    except ValueError:
        return None


def road_type(data: Mapping[str, Any]) -> str:
    value = data.get("road_type", data.get("highway", "unclassified"))
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else "unclassified"
    return str(value)


def base_speed_kph(data: Mapping[str, Any]) -> float:
    explicit = data.get("speed_kph")
    if explicit is not None:
        try:
            return max(1.0, float(explicit))
        except (TypeError, ValueError):
            pass
    parsed = parse_maxspeed(data.get("maxspeed"))
    if parsed is not None and parsed > 0:
        return parsed
    return HIGHWAY_SPEEDS_KPH.get(road_type(data), DEFAULT_MAX_SPEED_KPH)


def effective_speed_kph(
    data: Mapping[str, Any],
    departure: datetime,
    weather_condition: str,
) -> float:
    weather = WEATHER_FACTORS.get(weather_condition, 1.0)
    return max(5.0, base_speed_kph(data) * time_factor(departure.hour) * weather)


def edge_eta_s(
    data: Mapping[str, Any],
    departure: datetime,
    weather_condition: str,
) -> float:
    length_m = float(data.get("length_m", data.get("length", 0.0)))
    speed_kph = effective_speed_kph(data, departure, weather_condition)
    return length_m / (speed_kph * 1000.0 / 3600.0)


def edge_risk(data: Mapping[str, Any], night: bool) -> float:
    road = road_type(data)
    risk = {
        "motorway": 0.30,
        "motorway_link": 0.35,
        "trunk": 0.30,
        "trunk_link": 0.35,
        "primary": 0.25,
        "primary_link": 0.30,
        "secondary": 0.30,
        "secondary_link": 0.35,
        "tertiary": 0.35,
        "tertiary_link": 0.40,
        "residential": 0.40,
        "unclassified": 0.45,
        "service": 0.50,
        "living_street": 0.40,
        "road": 0.45,
    }.get(road, 0.45)
    surface = str(data.get("surface", "unknown")).lower()
    if surface in {"unpaved", "gravel", "dirt", "ground", "sand", "mud"}:
        risk += 0.20
    elif surface in {"cobblestone", "sett", "paving_stones"}:
        risk += 0.10
    lit = str(data.get("lit", "unknown")).lower()
    if night and lit in {"no", "false", "0"}:
        risk += 0.20
    speed = base_speed_kph(data)
    if speed > 70.0:
        risk += 0.10
    return max(0.0, min(1.0, risk))


def route_eta_min(
    graph: Any,
    nodes: tuple[int, ...] | list[int],
    departure: datetime,
    weather_condition: str,
) -> float:
    total_s = 0.0
    for start, end in zip(nodes, nodes[1:]):
        data = graph.edges[start, end]
        total_s += edge_eta_s(data, departure, weather_condition)
    return total_s / 60.0


def route_base_time_min(
    graph: Any,
    nodes: tuple[int, ...] | list[int],
) -> float:
    total_s = 0.0
    for start, end in zip(nodes, nodes[1:]):
        data = graph.edges[start, end]
        total_s += float(data.get("base_time_s", 0.0))
    return total_s / 60.0


def route_distance_m(graph: Any, nodes: tuple[int, ...] | list[int]) -> float:
    total = 0.0
    for start, end in zip(nodes, nodes[1:]):
        data = graph.edges[start, end]
        total += float(data.get("length_m", data.get("length", 0.0)))
    return total


def route_risk(
    graph: Any,
    nodes: tuple[int, ...] | list[int],
    departure: datetime,
) -> float:
    if len(nodes) < 2:
        return 0.0
    night = departure.hour >= 19 or departure.hour < 6
    risks = []
    for start, end in zip(nodes, nodes[1:]):
        data = graph.edges[start, end]
        risks.append(edge_risk(data, night))
    return sum(risks) / len(risks)


def route_toll(graph: Any, nodes: tuple[int, ...] | list[int]) -> bool:
    for start, end in zip(nodes, nodes[1:]):
        data = graph.edges[start, end]
        if bool(data.get("toll", False)):
            return True
    return False


def route_impassable(graph: Any, nodes: tuple[int, ...] | list[int]) -> bool:
    for start, end in zip(nodes, nodes[1:]):
        data = graph.edges[start, end]
        access = str(data.get("access", "")).lower()
        if access in {"no", "private"}:
            return True
    return False


def route_unlit_ratio(graph: Any, nodes: tuple[int, ...] | list[int]) -> float:
    if len(nodes) < 2:
        return 0.0
    unlit = 0
    total = 0
    for start, end in zip(nodes, nodes[1:]):
        data = graph.edges[start, end]
        lit = str(data.get("lit", "unknown")).lower()
        if lit in {"no", "false", "0"}:
            unlit += 1
        total += 1
    return unlit / total if total else 0.0
