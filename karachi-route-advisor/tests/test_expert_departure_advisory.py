from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from src.contracts import RouteCandidate
from src.departure_forecast import DepartureForecastService
from src.expert_system import (
    build_departure_rules,
    build_rules,
    evaluate_departure_advisory,
)


def custom_route() -> RouteCandidate:
    return RouteCandidate(
        route_id="osm-1",
        source="custom_astar",
        provider="OSM analytical",
        geometry=((24.9, 67.1), (24.91, 67.11)),
        distance_m=5000.0,
        eta_min=12.0,
        risk=0.3,
        confidence=0.55,
        congestion=None,
        incident_count=0,
        closure_count=0,
        traffic_available=False,
        unlit_ratio=0.1,
    )


def base_forecast():
    return DepartureForecastService().forecast(
        route=custom_route(),
        departure=datetime(2026, 10, 30, 18, 0),
    )


def rule_ids(advisory) -> list[str]:
    return [entry.rule_id for entry in advisory.trace]


def test_departure_rules_are_separate_from_route_rules() -> None:
    route_ids = {rule.id for rule in build_rules()}
    departure_ids = {rule.id for rule in build_departure_rules()}
    assert {f"R{index:02d}" for index in range(1, 17)} <= route_ids
    assert departure_ids == {f"R{index:02d}" for index in range(17, 23)}
    assert route_ids.isdisjoint(departure_ids)


def test_high_forecast_fires_disclosures_and_warning() -> None:
    advisory = evaluate_departure_advisory(base_forecast())
    assert advisory.final_assessment == "high"
    assert "R17" in rule_ids(advisory)
    assert "R18" in rule_ids(advisory)
    assert "R19" in rule_ids(advisory)
    assert any(
        "ML model trained on external traffic data" in warning
        for warning in advisory.warnings
    )
    assert any(
        "external UK STATS19" in warning for warning in advisory.warnings
    )
    assert any("High predicted calendar-slot risk" in warning for warning in advisory.warnings)


def test_low_forecast_produces_positive_note() -> None:
    forecast = replace(
        base_forecast(),
        final_assessment="low",
        predicted_traffic="low",
        accident_risk="low",
    )
    advisory = evaluate_departure_advisory(forecast)
    assert "R21" in rule_ids(advisory)
    assert not any(
        "High predicted calendar-slot risk" in warning
        for warning in advisory.warnings
    )
    assert not any(
        "Moderate predicted calendar-slot risk" in warning
        for warning in advisory.warnings
    )
    assert any(
        "Low predicted calendar-slot risk" in positive
        for positive in advisory.positives
    )


def test_moderate_forecast_produces_warning() -> None:
    forecast = replace(
        base_forecast(),
        final_assessment="moderate",
        predicted_traffic="moderate",
        accident_risk="low",
    )
    advisory = evaluate_departure_advisory(forecast)
    assert "R20" in rule_ids(advisory)
    assert any(
        "Moderate predicted calendar-slot risk" in warning
        for warning in advisory.warnings
    )


def test_applied_mapbox_evidence_fires_separation_rule() -> None:
    forecast = replace(
        base_forecast(),
        current_mapbox_traffic="severe",
        current_mapbox_applied=True,
    )
    advisory = evaluate_departure_advisory(forecast)
    assert "R22" in rule_ids(advisory)
    assert any(
        "Requested-departure Mapbox evidence" in warning
        for warning in advisory.warnings
    )
