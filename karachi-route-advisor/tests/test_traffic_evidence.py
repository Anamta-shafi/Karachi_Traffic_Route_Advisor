from __future__ import annotations

from datetime import datetime

import networkx as nx
import pytest

from src.ai.optimizer import build_final_recommendation
from src.contracts import (
    MapboxRoute,
    RankedRoute,
    RouteCandidate,
    RouteVerdict,
)
from src.traffic_evidence import build_route_assessment, match_traffic_evidence


def make_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    coordinates = (
        (24.900, 67.120),
        (24.910, 67.120),
        (24.920, 67.120),
    )
    for index, (lat, lon) in enumerate(coordinates):
        graph.add_node(index, x=lon, y=lat)
    for index in range(len(coordinates) - 1):
        graph.add_edge(
            index,
            index + 1,
            length_m=1110.0,
            base_time_s=100.0,
            speed_kph=40.0,
            highway="residential",
            lit="yes",
        )
    return graph


def make_route(route_id: str = "osm-1") -> RouteCandidate:
    return RouteCandidate(
        route_id=route_id,
        source="custom_astar",
        provider="OSM analytical",
        geometry=((24.900, 67.120), (24.910, 67.120), (24.920, 67.120)),
        distance_m=2220.0,
        eta_min=5.12,
        risk=0.4,
        confidence=0.6,
        congestion=None,
        incident_count=0,
        closure_count=0,
        traffic_available=False,
        nodes=(0, 1, 2),
        edges=((0, 1), (1, 2)),
        free_flow_eta_min=3.33,
    )


def make_provider(
    *,
    route_id: str = "mapbox-driving-traffic-1",
    duration_s: float = 240.0,
    typical_s: float = 130.0,
    congestion_numeric: tuple[int | None, ...] = (),
    geometry: tuple[tuple[float, float], ...] = (
        (24.900, 67.120),
        (24.910, 67.120),
        (24.920, 67.120),
    ),
    traffic_evidence: bool = True,
) -> MapboxRoute:
    return MapboxRoute(
        route_id=route_id,
        profile="mapbox/driving-traffic",
        duration_s=duration_s,
        duration_typical_s=typical_s,
        distance_m=2220.0,
        geometry=geometry,
        congestion_values=(),
        congestion_numeric=congestion_numeric,
        incident_count=0,
        closure_count=0,
        traffic_evidence=traffic_evidence,
        requested_departure="2026-11-02T04:00:00Z",
    )


def make_verdict(route_id: str = "osm-1") -> RouteVerdict:
    return RouteVerdict(
        route_id=route_id,
        vetoed=False,
        veto_reasons=(),
        penalty=0.0,
        bonus=0.0,
        trace=(),
    )


def test_segment_congestion_is_preferred_when_alignment_is_defensible() -> None:
    match = match_traffic_evidence(
        route=make_route(),
        mapbox_routes=(
            make_provider(congestion_numeric=(50, 50)),
        ),
        graph=make_graph(),
        departure=datetime(2026, 11, 2, 9, 0),
    )

    assert match is not None
    assert match.applied is True
    assert match.method == "segment_congestion"
    assert match.geometry_overlap == pytest.approx(1.0)
    assert match.segment_coverage == pytest.approx(1.0)
    assert match.adjusted_eta_min == pytest.approx(7.68, abs=0.05)


def test_corridor_ratio_is_used_when_segment_alignment_is_unavailable() -> None:
    match = match_traffic_evidence(
        route=make_route(),
        mapbox_routes=(make_provider(),),
        graph=make_graph(),
        departure=datetime(2026, 11, 2, 9, 0),
    )

    assert match is not None
    assert match.applied is True
    assert match.method == "corridor_ratio"
    assert match.provider_delay_ratio == pytest.approx(240.0 / 130.0)
    assert match.adjusted_eta_min == pytest.approx(5.12 * 240.0 / 130.0, abs=0.05)


def test_shared_endpoints_without_corridor_overlap_are_not_matched() -> None:
    match = match_traffic_evidence(
        route=make_route(),
        mapbox_routes=(
            make_provider(
                geometry=((24.900, 67.120), (24.700, 67.400), (24.920, 67.120)),
            ),
        ),
        graph=make_graph(),
        departure=datetime(2026, 11, 2, 9, 0),
    )

    assert match is None


def test_provider_without_traffic_evidence_is_not_matched() -> None:
    match = match_traffic_evidence(
        route=make_route(),
        mapbox_routes=(
            make_provider(
                duration_s=130.0,
                typical_s=130.0,
                traffic_evidence=False,
            ),
        ),
        graph=make_graph(),
        departure=datetime(2026, 11, 2, 9, 0),
    )

    assert match is None


def test_assessment_uses_matched_eta_once_and_keeps_incident_risk_unavailable() -> None:
    route = make_route()
    match = match_traffic_evidence(
        route=route,
        mapbox_routes=(make_provider(),),
        graph=make_graph(),
        departure=datetime(2026, 11, 2, 9, 0),
    )
    assessment = build_route_assessment(
        route=route,
        verdict=make_verdict(),
        prediction=None,
        traffic_evidence=match,
        departure=datetime(2026, 11, 2, 9, 0),
        graph=make_graph(),
        weather_condition="dry",
    )

    assert assessment.eta_kind == "traffic_adjusted"
    assert assessment.decision_eta_min == pytest.approx(
        match.adjusted_eta_min if match is not None else 0.0
    )
    assert assessment.traffic_adjusted_eta_min is not None
    assert assessment.incident_risk.level == "UNAVAILABLE"
    assert any("provider estimate" in item for item in assessment.limitations)


def test_final_recommendation_consumes_assessment_times_without_mapbox_candidates() -> None:
    fast = make_route("osm-1")
    slow = make_route("osm-2")
    fast_assessment = build_route_assessment(
        route=fast,
        verdict=make_verdict("osm-1"),
        prediction=None,
        traffic_evidence=match_traffic_evidence(
            route=fast,
            mapbox_routes=(make_provider(),),
            graph=make_graph(),
            departure=datetime(2026, 11, 2, 9, 0),
        ),
        departure=datetime(2026, 11, 2, 9, 0),
        graph=make_graph(),
        weather_condition="dry",
    )
    slow_assessment = build_route_assessment(
        route=slow,
        verdict=make_verdict("osm-2"),
        prediction=None,
        traffic_evidence=None,
        departure=datetime(2026, 11, 2, 9, 0),
        graph=make_graph(),
        weather_condition="dry",
    )
    final = build_final_recommendation(
        routes=(slow, fast),
        verdicts=(make_verdict("osm-1"), make_verdict("osm-2")),
        academic_ranked=(
            RankedRoute("osm-1", 1, 90, 0.1, 0.1, 1.0, 0.0, "academic"),
            RankedRoute("osm-2", 2, 80, 0.2, 0.2, 1.0, 0.0, "academic"),
        ),
        predictions={},
        departure=datetime(2026, 11, 2, 9, 0),
        preference="Fastest",
        assessments={
            "osm-1": fast_assessment,
            "osm-2": slow_assessment,
        },
    )

    assert final.recommended_route_id == "osm-1"
    assert final.routes[0].assessment == fast_assessment
    assert final.routes[0].decision_time_min == pytest.approx(
        fast_assessment.decision_eta_min
    )
    assert all(decision.route.source == "custom_astar" for decision in final.routes)
    assert "mapbox" not in final.participating_sources
