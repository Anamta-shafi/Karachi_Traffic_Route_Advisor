from __future__ import annotations

from datetime import datetime

import pytest

from src.config import GRAPH_PATH
from src.contracts import MapboxRoute, WeatherSnapshot
from src.service import RouteService
from src.traffic_prediction import (
    EvidenceBasedTrafficPredictor,
    TrafficPredictionError,
)

pytestmark = pytest.mark.skipif(
    not GRAPH_PATH.exists(),
    reason="Run scripts/fetch_graph.py to create the local graph cache.",
)


@pytest.fixture(scope="module")
def service() -> RouteService:
    return RouteService()


def dry_weather() -> WeatherSnapshot:
    return WeatherSnapshot(condition="dry", source="test")


def mapbox_route(
    *,
    route_id: str,
    duration_s: float,
    duration_typical_s: float | None,
    congestion_values: tuple[str, ...] = (),
    congestion_numeric: tuple[int | None, ...] = (),
    traffic_evidence: bool = False,
    profile: str = "mapbox/driving-traffic",
) -> MapboxRoute:
    return MapboxRoute(
        route_id=route_id,
        profile=profile,
        duration_s=duration_s,
        duration_typical_s=duration_typical_s,
        distance_m=5000.0,
        geometry=((24.90930, 67.12596), (24.93187, 67.15488)),
        congestion_values=congestion_values,
        congestion_numeric=congestion_numeric,
        incident_count=0,
        closure_count=0,
        traffic_evidence=traffic_evidence,
        fetched_at="2026-10-26T09:00:00+05:00",
    )


def test_custom_route_assessment_is_route_specific(service: RouteService) -> None:
    response = service.route(
        origin_name="Johar Chowk Mosque",
        destination_name="Block 8 bus stop",
        departure=datetime(2026, 10, 26, 9, 0),
        preference="Fastest",
        weather=dry_weather(),
        include_mapbox=False,
    )
    route = response.academic_routes[0]
    assessment = service.assess_traffic(
        response,
        route.route_id,
        datetime(2026, 10, 26, 9, 0),
    )

    assert assessment.route_id == route.route_id
    assert assessment.route_source == route.source
    assert assessment.distance_km == pytest.approx(route.distance_m / 1000.0)
    assert assessment.ordinary_eta_min is not None
    assert assessment.ordinary_eta_min > 0
    assert assessment.observed_eta_min is None
    assert assessment.predicted_eta_min is None
    assert assessment.predicted_eta_label.startswith("Not available")
    assert assessment.traffic_level == "unknown"
    assert assessment.evidence_status == "no_route_specific_evidence"
    assert assessment.road_composition
    assert assessment.trained is False
    assert assessment.historical_data_available is False


def test_switching_alternatives_changes_assessment(service: RouteService) -> None:
    response = service.route(
        origin_name="Johar Chowk Mosque",
        destination_name="Block 8 bus stop",
        departure=datetime(2026, 10, 26, 9, 0),
        preference="Fastest",
        weather=dry_weather(),
        include_mapbox=False,
    )
    assert len(response.academic_routes) >= 2
    first = service.assess_traffic(
        response,
        response.academic_routes[0].route_id,
        datetime(2026, 10, 26, 9, 0),
    )
    second = service.assess_traffic(
        response,
        response.academic_routes[1].route_id,
        datetime(2026, 10, 26, 9, 0),
    )

    assert first.route_id != second.route_id
    assert (first.distance_km, first.ordinary_eta_min) != (
        second.distance_km,
        second.ordinary_eta_min,
    )


def test_departure_change_updates_custom_assessment(service: RouteService) -> None:
    response = service.route(
        origin_name="Johar Chowk Mosque",
        destination_name="Block 8 bus stop",
        departure=datetime(2026, 10, 26, 9, 0),
        preference="Fastest",
        weather=dry_weather(),
        include_mapbox=False,
    )
    route_id = response.academic_routes[0].route_id
    morning = service.assess_traffic(
        response,
        route_id,
        datetime(2026, 10, 26, 9, 0),
    )
    evening = service.assess_traffic(
        response,
        route_id,
        datetime(2026, 10, 26, 18, 0),
    )

    assert morning.requested_departure != evening.requested_departure
    assert morning.ordinary_eta_min != evening.ordinary_eta_min
    assert morning.predicted_eta_min is None
    assert evening.predicted_eta_min is None


def test_annotated_mapbox_route_has_traffic_level_and_delay() -> None:
    predictor = EvidenceBasedTrafficPredictor()
    route = mapbox_route(
        route_id="mapbox-driving-traffic-1",
        duration_s=700.0,
        duration_typical_s=650.0,
        congestion_values=("severe", "heavy"),
        congestion_numeric=(90, 80),
        traffic_evidence=True,
    )
    assessment = predictor.assess(
        route=None,
        mapbox_route=route,
        departure=datetime(2026, 10, 26, 18, 0),
        weather=dry_weather(),
    )

    assert assessment.observed_eta_min == pytest.approx(700.0 / 60.0)
    assert assessment.baseline_eta_min == pytest.approx(650.0 / 60.0)
    assert assessment.delay_vs_baseline_min == pytest.approx(50.0 / 60.0)
    assert assessment.traffic_level == "severe"
    assert assessment.evidence_status == "annotated"
    assert assessment.predicted_eta_min is None
    assert assessment.trained is False


def test_partial_delta_only_keeps_traffic_level_unknown() -> None:
    predictor = EvidenceBasedTrafficPredictor()
    route = mapbox_route(
        route_id="mapbox-driving-traffic-1",
        duration_s=601.0,
        duration_typical_s=600.0,
        congestion_values=("unknown",),
        congestion_numeric=(None,),
    )
    assessment = predictor.assess(
        route=None,
        mapbox_route=route,
        departure=datetime(2026, 10, 26, 9, 0),
        weather=dry_weather(),
    )

    assert assessment.evidence_status == "partial_eta_delta_only"
    assert assessment.traffic_level == "unknown"
    assert assessment.delay_vs_baseline_min == pytest.approx(1.0 / 60.0)
    assert assessment.predicted_eta_min is None
    assert any(
        "only the ETA-versus-typical delta" in item
        for item in assessment.current_evidence
    )


def test_no_traffic_evidence_is_not_treated_as_free_flow() -> None:
    predictor = EvidenceBasedTrafficPredictor()
    route = mapbox_route(
        route_id="mapbox-driving-traffic-1",
        duration_s=600.0,
        duration_typical_s=None,
        congestion_values=("unknown",),
        congestion_numeric=(None,),
    )
    assessment = predictor.assess(
        route=None,
        mapbox_route=route,
        departure=datetime(2026, 10, 26, 9, 0),
        weather=dry_weather(),
    )

    assert assessment.evidence_status == "no_traffic_evidence"
    assert assessment.traffic_level == "unknown"
    assert assessment.delay_vs_baseline_min is None
    assert assessment.predicted_eta_min is None
    assert any(
        "No non-unknown congestion" in item for item in assessment.current_evidence
    )


def test_missing_graph_degrades_custom_assessment(service: RouteService) -> None:
    response = service.route(
        origin_name="Johar Chowk Mosque",
        destination_name="Block 8 bus stop",
        departure=datetime(2026, 10, 26, 9, 0),
        preference="Fastest",
        weather=dry_weather(),
        include_mapbox=False,
    )
    predictor = EvidenceBasedTrafficPredictor()
    assessment = predictor.assess(
        route=response.academic_routes[0],
        mapbox_route=None,
        departure=datetime(2026, 10, 26, 9, 0),
        weather=dry_weather(),
        graph=None,
    )

    assert assessment.status == "degraded"
    assert assessment.baseline_eta_min is None
    assert assessment.predicted_eta_min is None
    assert any(
        "routing graph was not available" in item for item in assessment.limitations
    )


def test_predictor_failure_returns_degraded_assessment(
    service: RouteService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = service.route(
        origin_name="Johar Chowk Mosque",
        destination_name="Block 8 bus stop",
        departure=datetime(2026, 10, 26, 9, 0),
        preference="Fastest",
        weather=dry_weather(),
        include_mapbox=False,
    )

    class BrokenPredictor:
        def assess(self, **kwargs):
            raise TrafficPredictionError("forced assessment failure")

    monkeypatch.setattr(service, "traffic_predictor", BrokenPredictor())
    assessment = service.assess_traffic(
        response,
        response.academic_routes[0].route_id,
        datetime(2026, 10, 26, 9, 0),
    )

    assert assessment.status == "degraded"
    assert assessment.predicted_eta_min is None
    assert "forced assessment failure" in assessment.explanation


def test_unknown_selected_route_returns_degraded_assessment(
    service: RouteService,
) -> None:
    response = service.route(
        origin_name="Johar Chowk Mosque",
        destination_name="Block 8 bus stop",
        departure=datetime(2026, 10, 26, 9, 0),
        preference="Fastest",
        weather=dry_weather(),
        include_mapbox=False,
    )
    assessment = service.assess_traffic(
        response,
        "missing-route",
        datetime(2026, 10, 26, 9, 0),
    )

    assert assessment.status == "degraded"
    assert "was not found" in assessment.explanation


def test_mapbox_status_is_route_specific(
    service: RouteService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAPBOX_ACCESS_TOKEN", "test-token")
    annotated = mapbox_route(
        route_id="mapbox-driving-traffic-1",
        duration_s=700.0,
        duration_typical_s=650.0,
        congestion_values=("severe", "heavy"),
        congestion_numeric=(90, 80),
        traffic_evidence=True,
    )
    unknown = mapbox_route(
        route_id="mapbox-driving-traffic-2",
        duration_s=600.0,
        duration_typical_s=None,
        congestion_values=("unknown",),
        congestion_numeric=(None,),
    )

    def fake_fetch(origin, destination, profile, alternatives=True, depart_at=None):
        return (annotated, unknown)

    monkeypatch.setattr("src.service.fetch_routes", fake_fetch)
    response = service.route(
        origin_name="Johar Chowk Mosque",
        destination_name="Block 8 bus stop",
        departure=datetime(2026, 10, 26, 9, 0),
        preference="Fastest",
        weather=dry_weather(),
        include_mapbox=True,
    )
    annotated_assessment = service.assess_traffic(
        response,
        annotated.route_id,
        datetime(2026, 10, 26, 9, 0),
    )
    unknown_assessment = service.assess_traffic(
        response,
        unknown.route_id,
        datetime(2026, 10, 26, 9, 0),
    )

    assert response.traffic_status == "annotated"
    assert annotated_assessment.evidence_status == "annotated"
    assert annotated_assessment.traffic_level == "severe"
    assert unknown_assessment.evidence_status == "no_traffic_evidence"
    assert unknown_assessment.traffic_level == "unknown"
    assert unknown_assessment.predicted_eta_min is None
