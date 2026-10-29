from __future__ import annotations

from datetime import datetime

import pytest

from src.config import GRAPH_PATH
from src.contracts import WeatherSnapshot
from src.service import RouteService

pytestmark = pytest.mark.skipif(
    not GRAPH_PATH.exists(),
    reason="Run scripts/fetch_graph.py to create the local graph cache.",
)

DEPARTURE = datetime(2026, 11, 2, 9, 0)


@pytest.fixture(scope="module")
def service() -> RouteService:
    return RouteService()


@pytest.fixture(scope="module")
def response(service: RouteService):
    return service.route(
        origin_name="Johar Chowk Mosque",
        destination_name="Block 8 bus stop",
        departure=DEPARTURE,
        preference="Fastest",
        weather=WeatherSnapshot(condition="dry", source="test"),
        include_mapbox=False,
    )


def test_route_contains_final_recommendation_with_per_route_predictions(
    response,
) -> None:
    final = response.final_recommendation

    assert final is not None
    assert final.departure == DEPARTURE
    assert final.recommended_route_id is not None
    assert final.routes
    assert len(final.routes) <= 3
    assert [item.recommended for item in final.routes].count(True) == 1
    assert final.routes[0].recommended is True
    assert final.routes[0].route_id == final.recommended_route_id

    academic_ids = {route.route_id for route in response.academic_routes}
    for decision in final.routes:
        assert decision.route_id in academic_ids
        assert decision.traffic_prediction is not None
        assert decision.traffic_prediction.route_id == decision.route_id
        assert decision.traffic_prediction.requested_departure == DEPARTURE
        assert decision.traffic_prediction.status in {"ok", "partial"}
        assert decision.traffic_prediction.confidence in {"Low", "Medium"}
        assert decision.incident_risk.level == "UNAVAILABLE"
        assert decision.incident_risk.route_specific is False
        assert decision.incident_risk.external_prior_level in {
            "LOW",
            "MODERATE",
            "HIGH",
            None,
        }
        assert decision.assessment is not None
        assert decision.assessment.route.route_id == decision.route_id
        assert decision.assessment.decision_eta_min > 0
        assert decision.assessment.incident_risk == decision.incident_risk

    assert final.fallback_used is False
    assert final.weights == (
        ("time", 0.75),
        ("distance", 0.15),
        ("risk", 0.10),
    )
    assert any(
        "not Karachi crash history" in item for item in final.provenance
    )


def test_refresh_uses_the_current_departure_without_mutating_saved_response(
    service: RouteService,
    response,
) -> None:
    refreshed_departure = datetime(2026, 11, 2, 18, 0)

    refreshed = service.recommend_routes(response, refreshed_departure)

    assert refreshed.departure == refreshed_departure
    assert all(
        decision.traffic_prediction is not None
        and decision.traffic_prediction.requested_departure == refreshed_departure
        for decision in refreshed.routes
    )
    assert response.final_recommendation is not None
    assert response.final_recommendation.departure == DEPARTURE


def test_prediction_failure_uses_analytical_eta_without_inventing_evidence(
    service: RouteService,
    response,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert response.final_recommendation is not None
    target_id = response.final_recommendation.recommended_route_id
    original = service.predict_route_traffic

    def flaky_predict(
        response_arg,
        route_id: str,
        departure: datetime,
        weather: WeatherSnapshot | None = None,
    ):
        if route_id == target_id:
            raise RuntimeError("forced model failure")
        return original(response_arg, route_id, departure, weather)

    monkeypatch.setattr(service, "predict_route_traffic", flaky_predict)

    final = service.recommend_routes(
        response,
        datetime(2026, 11, 2, 18, 30),
    )

    assert final.fallback_used is False
    assert final.weights == (
        ("time", 0.75),
        ("distance", 0.15),
        ("risk", 0.10),
    )
    failed = next(
        decision for decision in final.routes if decision.route_id == target_id
    )
    assert failed.traffic_prediction is None
    assert failed.traffic_component is None
    assert failed.assessment is not None
    assert failed.assessment.ml_eta_min is None
    assert failed.assessment.eta_kind == "analytical"
    assert any("prediction failed" in warning for warning in final.warnings)
    assert any(
        "without inventing evidence" in warning for warning in final.warnings
    )
