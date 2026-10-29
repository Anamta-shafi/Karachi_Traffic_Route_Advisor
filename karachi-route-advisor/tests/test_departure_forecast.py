from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from src.contracts import MapboxRoute, RouteCandidate, RouteContext
from src.departure_forecast import (
    ACCIDENT_DISCLOSURE,
    DISCLOSURE,
    DepartureForecastService,
    _final_assessment,
    forecast_rows,
)
from src.traffic_prediction import degraded_assessment


def custom_route(**overrides) -> RouteCandidate:
    values = {
        "route_id": "osm-1",
        "source": "custom_astar",
        "provider": "OSM analytical",
        "geometry": ((24.9, 67.1), (24.91, 67.11)),
        "distance_m": 5000.0,
        "eta_min": 12.0,
        "risk": 0.3,
        "confidence": 0.55,
        "congestion": None,
        "incident_count": 0,
        "closure_count": 0,
        "traffic_available": False,
        "unlit_ratio": 0.1,
    }
    values.update(overrides)
    return RouteCandidate(**values)


def mapbox_route(**overrides) -> MapboxRoute:
    values = {
        "route_id": "mapbox-1",
        "profile": "mapbox/driving-traffic",
        "duration_s": 900.0,
        "duration_typical_s": 750.0,
        "distance_m": 5000.0,
        "geometry": ((24.9, 67.1), (24.91, 67.11)),
        "congestion_values": ("severe",),
        "congestion_numeric": (90,),
        "incident_count": 0,
        "closure_count": 0,
        "traffic_evidence": True,
        "fetched_at": "request-time",
    }
    values.update(overrides)
    return MapboxRoute(**values)


def mapbox_assessment(route_id: str, level: str = "severe"):
    assessment = degraded_assessment(
        route_id=route_id,
        route_source="mapbox",
        distance_km=5.0,
        departure=datetime(2026, 10, 30, 18, 0),
        reason="test fixture",
    )
    return replace(
        assessment,
        traffic_level=level,
        traffic_level_basis="Test Mapbox congestion annotations.",
    )


def test_model_availability_and_disclosure() -> None:
    service = DepartureForecastService()
    assert service.available is True
    forecast = service.forecast(
        route=custom_route(),
        departure=datetime(2026, 10, 30, 18, 0),
    )
    assert forecast.status == "ok"
    assert forecast.evidence_status == "experimental_external_calendar_only"
    assert forecast.disclosure == (DISCLOSURE, ACCIDENT_DISCLOSURE)
    assert DISCLOSURE in forecast.disclosure
    assert "not locally validated for Gulistan-e-Jauhar" in DISCLOSURE
    assert "not locally validated" in ACCIDENT_DISCLOSURE
    assert "corridor-matched" in ACCIDENT_DISCLOSURE
    assert forecast.model_version == "external-calendar-v1"


def test_predictions_are_bands_with_probabilities() -> None:
    service = DepartureForecastService()
    forecast = service.forecast(
        route=custom_route(),
        departure=datetime(2026, 10, 30, 18, 0),
    )
    assert forecast.predicted_traffic in {"low", "moderate", "high"}
    assert forecast.accident_risk in {"low", "moderate", "high"}
    assert forecast.final_assessment in {"low", "moderate", "high"}
    assert forecast.traffic_probability is not None
    assert forecast.accident_probability is not None
    assert 0.0 <= forecast.traffic_probability <= 1.0
    assert 0.0 <= forecast.accident_probability <= 1.0
    assert forecast.traffic.available is True
    assert forecast.accident.available is True
    assert forecast.traffic.value is not None
    assert forecast.accident.value is not None
    assert forecast.traffic.units == "vehicles per hour"
    assert forecast.accident.units == "severity-weighted collisions per hour"
    assert forecast.route_context.available is False
    assert forecast.route_context.context_band == "unknown"


def test_custom_route_does_not_apply_mapbox_evidence() -> None:
    service = DepartureForecastService()
    assessment = mapbox_assessment("osm-1", "severe")
    forecast = service.forecast(
        route=custom_route(),
        departure=datetime(2026, 10, 30, 18, 0),
        current_mapbox_assessment=assessment,
    )
    assert forecast.current_mapbox_applied is False
    assert forecast.current_mapbox_traffic == "severe"


def test_applied_mapbox_evidence_can_worsen_but_not_improve() -> None:
    service = DepartureForecastService()
    route = mapbox_route()
    severe = mapbox_assessment(route.route_id, "severe")
    forecast = service.forecast(
        route=route,
        departure=datetime(2026, 10, 30, 18, 0),
        current_mapbox_assessment=severe,
    )
    assert forecast.current_mapbox_applied is True
    assert forecast.final_assessment == "high"

    low = mapbox_assessment(route.route_id, "low")
    forecast_low = service.forecast(
        route=route,
        departure=datetime(2026, 10, 30, 3, 0),
        current_mapbox_assessment=low,
    )
    without_mapbox = service.forecast(
        route=route,
        departure=datetime(2026, 10, 30, 3, 0),
    )
    assert forecast_low.final_assessment == without_mapbox.final_assessment


def test_final_assessment_is_worst_component() -> None:
    service = DepartureForecastService()
    forecast = service.forecast(
        route=custom_route(),
        departure=datetime(2026, 10, 30, 18, 0),
    )
    order = {"low": 0, "moderate": 1, "high": 2}
    assert order[forecast.final_assessment] == max(
        order[forecast.predicted_traffic],
        order[forecast.accident_risk],
    )


def test_high_route_context_escalates_low_to_moderate() -> None:
    context = RouteContext(
        distance_km=5.0,
        ordinary_eta_min=15.0,
        free_flow_eta_min=10.0,
        delay_ratio=0.5,
        risk=0.3,
        unlit_ratio=0.1,
        road_composition=(),
        context_band="high",
        context_basis="test",
        fingerprint="abc",
        available=True,
    )
    final, applied, level = _final_assessment(
        traffic_band="low",
        accident_band="low",
        route_context=context,
        current_mapbox_assessment=None,
        route_source="custom_astar",
        route_id="osm-1",
    )
    assert final == "moderate"
    assert applied is False
    assert level == "not_applied"


def test_route_comparison_rows_cover_all_routes() -> None:
    service = DepartureForecastService()
    routes = (
        custom_route(route_id="osm-1"),
        custom_route(route_id="osm-2", distance_m=6000.0),
    )
    forecasts = service.compare(
        routes,
        datetime(2026, 10, 30, 18, 0),
    )
    rows = forecast_rows(forecasts)
    assert [row["Route"] for row in rows] == ["osm-1", "osm-2"]
    assert set(rows[0]) == {
        "Route",
        "Source",
        "Traffic band",
        "Traffic probability",
        "Accident prior",
        "Accident probability",
        "Route context",
        "Final assessment",
    }


def test_missing_model_returns_unavailable_forecast(tmp_path: Path) -> None:
    service = DepartureForecastService(tmp_path)
    assert service.available is False
    forecast = service.forecast(
        route=custom_route(),
        departure=datetime(2026, 10, 30, 18, 0),
    )
    assert forecast.status == "unavailable"
    assert forecast.final_assessment == "unknown"
    assert forecast.traffic.available is False
    assert forecast.accident.available is False
    assert forecast.traffic_probability is None
    assert forecast.accident_probability is None
    assert any("metadata.json" in item for item in forecast.limitations)


def test_prediction_does_not_use_mapbox_values_as_features() -> None:
    service = DepartureForecastService()
    baseline = service.forecast(
        route=mapbox_route(),
        departure=datetime(2026, 10, 30, 18, 0),
    )
    with_severe = service.forecast(
        route=mapbox_route(),
        departure=datetime(2026, 10, 30, 18, 0),
        current_mapbox_assessment=mapbox_assessment("mapbox-1", "severe"),
    )
    assert baseline.predicted_traffic == with_severe.predicted_traffic
    assert baseline.accident_risk == with_severe.accident_risk
    assert baseline.traffic_probability == pytest.approx(
        with_severe.traffic_probability
    )
    assert baseline.accident_probability == pytest.approx(
        with_severe.accident_probability
    )
