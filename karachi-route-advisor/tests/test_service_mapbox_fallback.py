from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest

from src.config import GRAPH_PATH, MAPBOX_PROFILE
from src.contracts import MapboxRoute, WeatherSnapshot
from src.mapbox_client import MapboxError
from src.service import RouteService

pytestmark = pytest.mark.skipif(
    not GRAPH_PATH.exists(),
    reason="Run scripts/fetch_graph.py to create the local graph cache.",
)


def ordinary_route(profile: str) -> MapboxRoute:
    return MapboxRoute(
        route_id="mapbox-driving-1",
        profile=profile,
        duration_s=600.0,
        duration_typical_s=None,
        distance_m=5000.0,
        geometry=((24.90930, 67.12596), (24.93187, 67.15488)),
        congestion_values=(),
        congestion_numeric=(),
        incident_count=0,
        closure_count=0,
        traffic_evidence=False,
    )


def traffic_route(profile: str) -> MapboxRoute:
    return MapboxRoute(
        route_id="mapbox-driving-traffic-1",
        profile=profile,
        duration_s=700.0,
        duration_typical_s=650.0,
        distance_m=5200.0,
        geometry=((24.90930, 67.12596), (24.93187, 67.15488)),
        congestion_values=("severe", "heavy"),
        congestion_numeric=(90, 80),
        incident_count=0,
        closure_count=0,
        traffic_evidence=True,
    )


def test_service_uses_traffic_aware_mapbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAPBOX_ACCESS_TOKEN", "test-token")
    service = RouteService()

    def fake_fetch(origin, destination, profile, alternatives=True, depart_at=None):
        return (traffic_route(profile),)

    monkeypatch.setattr("src.service.fetch_routes", fake_fetch)
    response = service.route(
        origin_name="Johar Chowk Mosque",
        destination_name="Block 8 bus stop",
        departure=datetime(2026, 10, 26, 9, 0),
        preference="Fastest",
        weather=WeatherSnapshot(condition="dry", source="test"),
        include_mapbox=True,
    )
    assert response.mapbox_routes[0].profile == "mapbox/driving-traffic"
    assert response.mapbox_routes[0].traffic_evidence is True
    assert len(response.mapbox_verdicts) == 1
    assert any(
        entry.rule_id == "R11" for entry in response.mapbox_verdicts[0].trace
    )
    assert not any(
        entry.rule_id in {"R04", "R05", "R13", "R16"}
        for entry in response.mapbox_verdicts[0].trace
    )
    assert response.mapbox_verdicts[0].penalty == pytest.approx(0.0)
    assert response.mapbox_verdicts[0].bonus == pytest.approx(0.08)
    assert response.mapbox_verdicts[0].warnings == ()
    assert any("driving-traffic" in item for item in response.provenance)


def test_service_requests_mapbox_at_the_selected_departure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAPBOX_ACCESS_TOKEN", "test-token")
    service = RouteService()
    calls: list[tuple[str, str | None]] = []

    def fake_fetch(origin, destination, profile, alternatives=True, depart_at=None):
        calls.append((profile, depart_at))
        return (traffic_route(profile),)

    monkeypatch.setattr("src.service.fetch_routes", fake_fetch)
    service.route(
        origin_name="Johar Chowk Mosque",
        destination_name="Block 8 bus stop",
        departure=datetime(2026, 10, 26, 9, 0),
        preference="Fastest",
        weather=WeatherSnapshot(condition="dry", source="test"),
        include_mapbox=True,
    )

    assert calls == [("mapbox/driving-traffic", "2026-10-26T04:00:00Z")]


def test_stale_mapbox_evidence_is_excluded_from_refreshed_ai_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAPBOX_ACCESS_TOKEN", "test-token")
    service = RouteService()

    def fake_fetch(origin, destination, profile, alternatives=True, depart_at=None):
        return (traffic_route(profile),)

    monkeypatch.setattr("src.service.fetch_routes", fake_fetch)
    response = service.route(
        origin_name="Johar Chowk Mosque",
        destination_name="Block 8 bus stop",
        departure=datetime(2026, 10, 26, 9, 0),
        preference="Fastest",
        weather=WeatherSnapshot(condition="dry", source="test"),
        include_mapbox=True,
    )

    refreshed = service.recommend_routes(
        response,
        datetime(2026, 10, 26, 18, 0),
    )

    assert refreshed.mapbox_considered is False
    assert "mapbox" not in refreshed.participating_sources
    assert any(
        "fetched for the previous departure" in warning
        for warning in refreshed.warnings
    )

    assessment = service.assess_traffic(
        response,
        "mapbox-driving-traffic-1",
        datetime(2026, 10, 26, 18, 0),
    )
    assert assessment.route_source == "mapbox"
    assert assessment.traffic_level == "unknown"
    assert any(
        "previous departure" in item for item in assessment.current_evidence
    )


def test_service_reports_both_mapbox_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAPBOX_ACCESS_TOKEN", "test-token")
    service = RouteService()

    def fake_fetch(origin, destination, profile, alternatives=True, depart_at=None):
        raise MapboxError("unavailable", status=503, code="NoRoute")

    monkeypatch.setattr("src.service.fetch_routes", fake_fetch)
    response = service.route(
        origin_name="Johar Chowk Mosque",
        destination_name="Block 8 bus stop",
        departure=datetime(2026, 10, 26, 9, 0),
        preference="Fastest",
        weather=WeatherSnapshot(condition="dry", source="test"),
        include_mapbox=True,
    )
    assert response.mapbox_routes == ()
    assert any("Mapbox unavailable" in warning for warning in response.warnings)
    assert response.academic_routes


def test_service_falls_back_to_ordinary_mapbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAPBOX_ACCESS_TOKEN", "test-token")
    service = RouteService()

    def fake_fetch(origin, destination, profile, alternatives=True, depart_at=None):
        if profile == "mapbox/driving-traffic":
            raise MapboxError("traffic unavailable", status=500, code="NoRoute")
        return (ordinary_route(profile),)

    monkeypatch.setattr("src.service.fetch_routes", fake_fetch)
    response = service.route(
        origin_name="Johar Chowk Mosque",
        destination_name="Block 8 bus stop",
        departure=datetime(2026, 10, 26, 9, 0),
        preference="Fastest",
        weather=WeatherSnapshot(condition="dry", source="test"),
        include_mapbox=True,
    )
    assert len(response.mapbox_routes) == 1
    assert response.mapbox_routes[0].profile == "mapbox/driving"
    assert any("non-traffic-aware" in warning for warning in response.warnings)
    assert len(response.mapbox_verdicts) == 1
    assert any(
        "No live traffic measurement" in warning
        for warning in response.mapbox_verdicts[0].warnings
    )
    assert any("Mapbox Directions API driving" in item for item in response.provenance)

def test_matched_mapbox_evidence_adjusts_osm_route_without_competing() -> None:
    service = RouteService()
    response = service.route(
        origin_name="Johar Chowk Mosque",
        destination_name="Block 8 bus stop",
        departure=datetime(2026, 10, 26, 9, 0),
        preference="Fastest",
        weather=WeatherSnapshot(condition="dry", source="test"),
        include_mapbox=False,
    )
    target = response.academic_routes[0]
    segment_count = max(1, len(target.geometry) - 1)
    typical_s = target.eta_min * 60.0 * 0.9
    provider = MapboxRoute(
        route_id="mapbox-driving-traffic-1",
        profile=MAPBOX_PROFILE,
        duration_s=typical_s * 1.5,
        duration_typical_s=typical_s,
        distance_m=target.distance_m,
        geometry=target.geometry,
        congestion_values=("heavy",) * segment_count,
        congestion_numeric=(40,) * segment_count,
        incident_count=1,
        closure_count=0,
        traffic_evidence=True,
    )
    enriched = replace(
        response,
        academic_routes=(target,),
        verdicts=tuple(
            verdict for verdict in response.verdicts if verdict.route_id == target.route_id
        ),
        ranked=tuple(
            ranked for ranked in response.ranked if ranked.route_id == target.route_id
        ),
        mapbox_routes=(provider,),
        mapbox_verdicts=(),
        traffic_status="annotated",
    )

    final = service.recommend_routes(enriched, response.departure)

    assert final.recommended_route_id == target.route_id
    assert final.mapbox_considered is True
    assert final.participating_sources == ("custom_astar",)
    decision = final.routes[0]
    assert decision.route.source == "custom_astar"
    assert decision.assessment is not None
    assert decision.assessment.eta_kind == "traffic_adjusted"
    assert decision.assessment.traffic_evidence is not None
    assert decision.assessment.traffic_evidence.applied is True
    assert decision.assessment.traffic_evidence.method == "segment_congestion"
    assert decision.assessment.decision_eta_min == pytest.approx(
        target.eta_min * 1.4,
        abs=0.1,
    )
    assert decision.incident_risk.route_specific is True
    assert decision.incident_risk.level == "MODERATE"
    assert any(
        "considered only as traffic evidence" in warning for warning in final.warnings
    )
