from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Iterable, Sequence

from .config import MAPBOX_PROFILE
from .contracts import (
    MapboxRoute,
    RouteAssessment,
    RouteCandidate,
    RouteTrafficPrediction,
    RouteVerdict,
    TrafficEvidenceMatch,
)
from .eta import edge_eta_s, haversine_m, route_base_time_min, route_eta_min
from .mapbox_client import traffic_evidence_status

MIN_CORRIDOR_OVERLAP = 0.65
MATCH_TOLERANCE_M = 35.0
MIN_SEGMENT_COVERAGE = 0.40
MIN_BASELINE_RATIO = 0.40
MAX_BASELINE_RATIO = 2.50
MIN_DELAY_RATIO = 0.50
MAX_DELAY_RATIO = 3.00
MAX_SEGMENT_DELAY_FACTOR = 1.00
TRAFFIC_DELAY_FACTOR = 0.60
TRAFFIC_SEVERITY = {"low": 0.0, "moderate": 0.5, "high": 1.0}
_CONGESTION_STRING_VALUE = {
    "low": 0.20,
    "moderate": 0.50,
    "heavy": 0.70,
    "severe": 0.90,
}


def _densify(
    points: Sequence[tuple[float, float]],
    spacing_m: float = 25.0,
) -> list[tuple[float, float]]:
    if len(points) < 2:
        return list(points)
    densified = [points[0]]
    for start, end in zip(points, points[1:]):
        distance = haversine_m(start[0], start[1], end[0], end[1])
        steps = max(1, math.ceil(distance / spacing_m))
        for step in range(1, steps + 1):
            fraction = step / steps
            densified.append(
                (
                    start[0] + (end[0] - start[0]) * fraction,
                    start[1] + (end[1] - start[1]) * fraction,
                )
            )
    return densified


def _sample_points(
    points: Sequence[tuple[float, float]],
    limit: int = 120,
) -> list[tuple[float, float]]:
    densified = _densify(points)
    if len(densified) <= limit:
        return densified
    step = max(1, math.ceil(len(densified) / limit))
    sampled = list(densified[::step])
    if sampled[-1] != densified[-1]:
        sampled.append(densified[-1])
    return sampled


def _point_to_segment_m(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    lat, lon = point
    scale = math.cos(math.radians(lat))
    px, py = lon * scale, lat
    ax, ay = start[1] * scale, start[0]
    bx, by = end[1] * scale, end[0]
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return haversine_m(lat, lon, start[0], start[1])
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    closest = (ay + t * dy, (ax + t * dx) / scale)
    return haversine_m(lat, lon, closest[0], closest[1])


def _point_to_polyline_m(
    point: tuple[float, float],
    polyline: Sequence[tuple[float, float]],
) -> float:
    if not polyline:
        return float("inf")
    if len(polyline) == 1:
        return haversine_m(point[0], point[1], polyline[0][0], polyline[0][1])
    return min(
        _point_to_segment_m(point, start, end)
        for start, end in zip(polyline, polyline[1:])
    )


def polyline_overlap(
    first: Sequence[tuple[float, float]],
    second: Sequence[tuple[float, float]],
    tolerance_m: float = MATCH_TOLERANCE_M,
) -> float:
    if not first or not second:
        return 0.0
    samples = _sample_points(first)
    matches = sum(
        1
        for point in samples
        if _point_to_polyline_m(point, second) <= tolerance_m
    )
    return matches / len(samples)


def symmetric_overlap(
    first: Sequence[tuple[float, float]],
    second: Sequence[tuple[float, float]],
    tolerance_m: float = MATCH_TOLERANCE_M,
) -> float:
    return min(
        polyline_overlap(first, second, tolerance_m),
        polyline_overlap(second, first, tolerance_m),
    )


def _edge_midpoint(graph: Any, start: int, end: int) -> tuple[float, float]:
    data = graph.edges[start, end]
    geometry = data.get("geometry")
    if geometry is not None and hasattr(geometry, "coords"):
        coords = list(geometry.coords)
        if coords:
            lon, lat = coords[len(coords) // 2]
            return float(lat), float(lon)
    start_data = graph.nodes[start]
    end_data = graph.nodes[end]
    return (
        (float(start_data["y"]) + float(end_data["y"])) / 2.0,
        (float(start_data["x"]) + float(end_data["x"])) / 2.0,
    )


def _edge_length_m(graph: Any, start: int, end: int) -> float:
    data = graph.edges[start, end]
    return float(data.get("length_m", data.get("length", 0.0)))


def _congestion_factor(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() == "unknown":
            return None
        numeric = _CONGESTION_STRING_VALUE.get(value.strip().lower())
    else:
        try:
            numeric = float(value) / 100.0
        except (TypeError, ValueError):
            return None
    if numeric is None:
        return None
    numeric = max(0.0, min(1.0, float(numeric)))
    return 1.0 + numeric * MAX_SEGMENT_DELAY_FACTOR


def _aligned_segment_values(
    route: MapboxRoute,
) -> tuple[tuple[Any, ...], str] | None:
    segment_count = len(route.geometry) - 1
    if segment_count < 1:
        return None
    if len(route.congestion_numeric) == segment_count:
        return route.congestion_numeric, "congestion_numeric"
    if len(route.congestion_values) == segment_count:
        return route.congestion_values, "congestion"
    return None


def _segment_adjusted_eta(
    route: RouteCandidate,
    provider: MapboxRoute,
    graph: Any,
    departure: datetime,
    weather_condition: str,
) -> tuple[float, float, int] | None:
    aligned = _aligned_segment_values(provider)
    if aligned is None or len(provider.geometry) < 2:
        return None
    values, _ = aligned
    segments = list(zip(provider.geometry, provider.geometry[1:]))
    total_length = 0.0
    matched_length = 0.0
    known_matches = 0
    adjusted_s = 0.0
    for start, end in zip(route.nodes, route.nodes[1:]):
        data = graph.edges[start, end]
        length_m = _edge_length_m(graph, start, end)
        total_length += length_m
        midpoint = _edge_midpoint(graph, start, end)
        best_index: int | None = None
        best_distance = float("inf")
        for index, (seg_start, seg_end) in enumerate(segments):
            distance = _point_to_segment_m(midpoint, seg_start, seg_end)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        factor = None
        if best_index is not None and best_distance <= MATCH_TOLERANCE_M:
            factor = _congestion_factor(values[best_index])
            if factor is not None:
                matched_length += length_m
                known_matches += 1
        analytical_edge_s = edge_eta_s(data, departure, weather_condition)
        if factor is not None:
            adjusted_s += analytical_edge_s * factor
        else:
            adjusted_s += analytical_edge_s
    if total_length <= 0 or known_matches == 0:
        return None
    coverage = matched_length / total_length
    return adjusted_s / 60.0, coverage, known_matches


def _usable_prediction(prediction: RouteTrafficPrediction | None) -> bool:
    return (
        prediction is not None
        and prediction.status in {"ok", "partial"}
        and prediction.traffic_band in TRAFFIC_SEVERITY
        and bool(prediction.traffic_probabilities)
    )


def _expected_severity(prediction: RouteTrafficPrediction | None) -> float | None:
    if not _usable_prediction(prediction):
        return None
    assert prediction is not None
    total = 0.0
    weight = 0.0
    for band, probability in prediction.traffic_probabilities:
        if band in TRAFFIC_SEVERITY:
            value = float(probability)
            total += TRAFFIC_SEVERITY[band] * value
            weight += value
    return total / weight if weight > 0 else None


def match_traffic_evidence(
    route: RouteCandidate,
    mapbox_routes: Iterable[MapboxRoute],
    graph: Any,
    departure: datetime,
    weather_condition: str = "dry",
) -> TrafficEvidenceMatch | None:
    if graph is None or not route.nodes:
        return None
    analytical_eta = route_eta_min(
        graph, route.nodes, departure, weather_condition
    )
    best: TrafficEvidenceMatch | None = None
    for provider in mapbox_routes:
        if provider.profile != MAPBOX_PROFILE or provider.duration_s <= 0:
            continue
        status = traffic_evidence_status((provider,))
        if status == "no_traffic_evidence":
            continue
        overlap = symmetric_overlap(route.geometry, provider.geometry)
        if overlap < MIN_CORRIDOR_OVERLAP:
            continue
        limitations: list[str] = [
            "The provider ETA is for the requested departure and is a provider "
            "estimate, not an observed measurement.",
            "Corridor matching requires at least 65% two-way geometry overlap; "
            "sharing only an origin and destination is not sufficient.",
        ]
        ratio = (
            provider.duration_s / provider.duration_typical_s
            if provider.duration_typical_s
            else None
        )
        typical_min = (
            provider.duration_typical_s / 60.0
            if provider.duration_typical_s is not None
            else None
        )
        baseline_ratio = (
            typical_min / analytical_eta
            if typical_min is not None and analytical_eta > 0
            else None
        )
        segment_result = _segment_adjusted_eta(
            route, provider, graph, departure, weather_condition
        )
        if segment_result is not None:
            segment_eta, coverage, known_matches = segment_result
            if coverage >= MIN_SEGMENT_COVERAGE:
                limitations.append(
                    f"Matched segment congestion covers {coverage:.0%} of the OSM "
                    f"route across {known_matches} segment(s); unmatched segments "
                    "retain the analytical estimate."
                )
                candidate = TrafficEvidenceMatch(
                    route_id=route.route_id,
                    provider_route_id=provider.route_id,
                    method="segment_congestion",
                    geometry_overlap=overlap,
                    segment_coverage=coverage,
                    provider_eta_min=provider.duration_s / 60.0,
                    provider_typical_min=typical_min,
                    provider_delay_ratio=ratio,
                    adjusted_eta_min=segment_eta,
                    evidence_status=status,
                    requested_departure=departure,
                    fetched_at=provider.fetched_at,
                    incident_count=provider.incident_count,
                    closure_count=provider.closure_count,
                    applied=True,
                    limitations=tuple(limitations),
                )
                if best is None or _match_priority(candidate) > _match_priority(best):
                    best = candidate
                continue
        if (
            ratio is not None
            and MIN_DELAY_RATIO <= ratio <= MAX_DELAY_RATIO
            and baseline_ratio is not None
            and MIN_BASELINE_RATIO <= baseline_ratio <= MAX_BASELINE_RATIO
        ):
            limitations.append(
                "No sufficiently covered segment-level congestion mapping was "
                "available, so the provider's departure/typical duration ratio was "
                "applied to the OSM analytical ETA. The provider's absolute ETA was "
                "not copied onto the OSM route."
            )
            candidate = TrafficEvidenceMatch(
                route_id=route.route_id,
                provider_route_id=provider.route_id,
                method="corridor_ratio",
                geometry_overlap=overlap,
                segment_coverage=None,
                provider_eta_min=provider.duration_s / 60.0,
                provider_typical_min=typical_min,
                provider_delay_ratio=ratio,
                adjusted_eta_min=analytical_eta * ratio,
                evidence_status=status,
                requested_departure=departure,
                fetched_at=provider.fetched_at,
                incident_count=provider.incident_count,
                closure_count=provider.closure_count,
                applied=True,
                limitations=tuple(limitations),
            )
            if best is None or _match_priority(candidate) > _match_priority(best):
                best = candidate
            continue
        limitations.append(
            "The matched provider route did not provide a defensible baseline or "
            "departure/typical delay ratio, so no traffic-adjusted ETA was applied."
        )
        candidate = TrafficEvidenceMatch(
            route_id=route.route_id,
            provider_route_id=provider.route_id,
            method="unavailable",
            geometry_overlap=overlap,
            segment_coverage=(
                segment_result[1] if segment_result is not None else None
            ),
            provider_eta_min=provider.duration_s / 60.0,
            provider_typical_min=typical_min,
            provider_delay_ratio=ratio,
            adjusted_eta_min=analytical_eta,
            evidence_status=status,
            requested_departure=departure,
            fetched_at=provider.fetched_at,
            incident_count=provider.incident_count,
            closure_count=provider.closure_count,
            applied=False,
            limitations=tuple(limitations),
        )
        if best is None or _match_priority(candidate) > _match_priority(best):
            best = candidate
    return best


def _match_priority(match: TrafficEvidenceMatch) -> tuple[int, int, float, float]:
    return (
        1 if match.applied else 0,
        1 if match.method == "segment_congestion" else 0,
        match.geometry_overlap,
        match.segment_coverage or 0.0,
    )


def build_route_assessment(
    *,
    route: RouteCandidate,
    verdict: RouteVerdict,
    prediction: RouteTrafficPrediction | None,
    traffic_evidence: TrafficEvidenceMatch | None,
    departure: datetime,
    graph: Any,
    weather_condition: str,
) -> RouteAssessment:
    from .ai.optimizer import build_incident_risk

    if graph is not None and route.nodes:
        analytical_eta = route_eta_min(
            graph, route.nodes, departure, weather_condition
        )
        free_flow_eta = route_base_time_min(graph, route.nodes)
    else:
        analytical_eta = route.eta_min
        free_flow_eta = route.free_flow_eta_min
    if free_flow_eta is not None and free_flow_eta <= 0:
        free_flow_eta = None
    severity = _expected_severity(prediction)
    ml_eta = (
        free_flow_eta * (1.0 + TRAFFIC_DELAY_FACTOR * severity)
        if severity is not None and free_flow_eta is not None
        else None
    )
    evidence = traffic_evidence if traffic_evidence and traffic_evidence.applied else None
    if evidence is not None:
        decision_eta = evidence.adjusted_eta_min
        eta_kind = "traffic_adjusted"
        eta_label = "Traffic-adjusted ETA (matched provider evidence)"
        basis = (
            "Matched Mapbox provider evidence; "
            + (
                "segment-level congestion applied to matched OSM segments."
                if evidence.method == "segment_congestion"
                else "departure/typical delay ratio applied to the OSM analytical ETA."
            )
        )
        evidence_source = "Mapbox Directions API driving-traffic"
        confidence = "Medium" if evidence.geometry_overlap >= 0.80 else "Low"
        confidence_note = (
            f"Provider evidence matched {evidence.geometry_overlap:.0%} of the OSM "
            "corridor. It is a provider estimate for the requested departure, not an "
            "observed Karachi measurement."
        )
    elif ml_eta is not None and _usable_prediction(prediction):
        decision_eta = ml_eta
        eta_kind = "analytical"
        eta_label = "Analytical OSM ETA with future-ML scenario adjustment"
        basis = "OSM free-flow ETA plus one ML future-traffic premium."
        evidence_source = "OSM route-traffic ML (synthetic-trained scenario model)"
        confidence = prediction.confidence if prediction is not None else "Unavailable"
        confidence_note = (
            prediction.confidence_note
            if prediction is not None
            else "No valid ML prediction."
        )
    else:
        decision_eta = analytical_eta
        eta_kind = "analytical"
        eta_label = "Analytical OSM ETA"
        basis = "OSM analytical ETA; no matched traffic evidence or valid ML prediction."
        evidence_source = "OSM analytical ETA"
        confidence = "Unavailable"
        confidence_note = (
            "No matched traffic evidence or valid future-ML prediction is available. "
            "The ETA is an analytical estimate, not a measured traffic-aware ETA."
        )
    future_band = (
        prediction.traffic_band if _usable_prediction(prediction) else None
    )
    future_probability = (
        prediction.traffic_probability if _usable_prediction(prediction) else None
    )
    incident = build_incident_risk(
        route,
        prediction,
        traffic_evidence=traffic_evidence,
    )
    limitations = [
        "No route-specific Karachi crash history is available; incident risk is "
        "shown as unavailable unless a matched provider route reports an incident "
        "or closure.",
        "The future-ML traffic labels are synthetic scenarios calibrated to external "
        "profiles, not observed Karachi traffic.",
        "OSM turn restrictions, lane-level details, and live incidents are not fully "
        "modelled.",
    ]
    if traffic_evidence is not None:
        limitations.extend(traffic_evidence.limitations)
    if prediction is not None:
        limitations.extend(prediction.limitations)
    details = [
        f"Analytical OSM ETA: {analytical_eta:.1f} min",
        f"OSM free-flow base: "
        f"{free_flow_eta:.1f} min" if free_flow_eta is not None else "OSM free-flow base: unavailable",
        f"Future-ML scenario ETA: "
        f"{ml_eta:.1f} min" if ml_eta is not None else "Future-ML scenario ETA: unavailable",
        f"Traffic-adjusted ETA: "
        f"{evidence.adjusted_eta_min:.1f} min" if evidence is not None else "Traffic-adjusted ETA: unavailable",
        f"Selected decision ETA: {decision_eta:.1f} min ({basis})",
        f"Evidence source: {evidence_source}",
        f"Confidence: {confidence}",
    ]
    if evidence is not None and ml_eta is not None:
        details.append(
            "The future-ML scenario is shown for context only and was not added to "
            "the provider-adjusted ETA."
        )
    if evidence is not None:
        details.extend(
            [
                f"Matched provider route: {evidence.provider_route_id}",
                f"Match method: {evidence.method}",
                f"Geometry overlap: {evidence.geometry_overlap:.0%}",
                f"Segment coverage: "
                f"{evidence.segment_coverage:.0%}"
                if evidence.segment_coverage is not None
                else "Segment coverage: not available",
                f"Provider ETA: {evidence.provider_eta_min:.1f} min",
                f"Provider typical duration: "
                f"{evidence.provider_typical_min:.1f} min"
                if evidence.provider_typical_min is not None
                else "Provider typical duration: unavailable",
            ]
        )
    return RouteAssessment(
        route=route,
        requested_departure=departure,
        analytical_eta_min=analytical_eta,
        free_flow_eta_min=free_flow_eta,
        ml_eta_min=ml_eta,
        traffic_adjusted_eta_min=(
            evidence.adjusted_eta_min if evidence is not None else None
        ),
        decision_eta_min=decision_eta,
        decision_eta_basis=basis,
        eta_kind=eta_kind,
        eta_label=eta_label,
        evidence_source=evidence_source,
        evidence_timestamp=(
            evidence.fetched_at
            if evidence is not None
            else (
                prediction.generated_at.isoformat(timespec="seconds")
                if prediction is not None
                else None
            )
        ),
        evidence_coverage=(
            evidence.segment_coverage
            if evidence is not None and evidence.segment_coverage is not None
            else (evidence.geometry_overlap if evidence is not None else None)
        ),
        traffic_evidence=traffic_evidence,
        traffic_prediction=prediction,
        future_traffic_band=future_band,
        future_traffic_probability=future_probability,
        confidence=confidence,
        confidence_note=confidence_note,
        incident_risk=incident,
        expert_verdict=verdict,
        limitations=tuple(dict.fromkeys(limitations)),
        details=tuple(details),
    )
