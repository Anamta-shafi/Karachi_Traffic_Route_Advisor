from __future__ import annotations

from dataclasses import replace
from datetime import datetime, time, timedelta
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.config import GRAPH_PATH, MAPBOX_PROFILE
from src.contracts import MapboxRoute, RouteResponse, WeatherSnapshot
from src.service import RouteService

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

pytestmark = pytest.mark.skipif(
    not GRAPH_PATH.exists(),
    reason="Run scripts/fetch_graph.py to create the local graph cache.",
)


def run_app_without_mapbox() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=240)
    at.run()
    next(
        checkbox
        for checkbox in at.checkbox
        if checkbox.label == "Use matched Mapbox traffic evidence"
    ).set_value(False)
    next(
        button for button in at.button if button.label == "Find Best Route"
    ).click()
    at.run()
    return at


def run_app_with_saved_response(response: RouteResponse) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=240)
    at.run()
    at.session_state["response"] = response
    at.run()
    return at


def recommended_prediction(
    at: AppTest,
) -> tuple[str, str | None, str | None]:
    headline = next(
        item.value
        for item in at.markdown
        if item.value.startswith("**Predicted traffic at departure:**")
    )
    probability = next(
        (
            item.value
            for item in at.metric
            if item.label == "Predicted congestion probability"
        ),
        None,
    )
    departure = next(
        (item.value for item in at.metric if item.label == "Selected departure"),
        None,
    )
    return headline, probability, departure


def future_departure(hour: int) -> datetime:
    return (datetime.now() + timedelta(days=2)).replace(
        hour=hour,
        minute=0,
        second=0,
        microsecond=0,
    )


@pytest.fixture(scope="module")
def service() -> RouteService:
    return RouteService()


def service_response(
    service: RouteService,
    departure: datetime,
) -> RouteResponse:
    return service.route(
        origin_name="Johar Chowk Mosque",
        destination_name="Block 8 bus stop",
        departure=departure,
        preference="Fastest",
        weather=WeatherSnapshot(condition="dry", source="test"),
        include_mapbox=False,
    )


def test_clean_route_recommendation_and_technical_details() -> None:
    at = run_app_without_mapbox()
    assert [str(exc) for exc in at.exception] == []

    assert any(item.value == "Recommended route" for item in at.subheader)
    assert any(item.value == "Other best routes" for item in at.subheader)
    assert any(item.value == "Map" for item in at.subheader)
    assert any(
        expander.label == "Technical Details & Data Limitations"
        for expander in at.expander
    )
    metric_labels = {metric.label for metric in at.metric}
    assert {
        "ETA",
        "Distance",
        "Provider traffic evidence",
        "Incident risk",
        "Predicted congestion probability",
        "Confidence",
        "Selected departure",
    } <= metric_labels
    assert any(
        "No matched provider route reported an incident" in item.value
        for item in at.warning
    )
    assert any(
        "ETAs use analytical or future-ML scenario estimates" in item.value
        for item in at.warning
    )
    assert not any(item.label == "Find Best Routes" for item in at.button)
    assert not any(
        item.value == "AI Recommended Route" for item in at.subheader
    )
    assert not any(
        item.label == "Route for traffic assessment" for item in at.selectbox
    )
    assert not any(
        item.value == "**Traffic evidence at your departure**"
        for item in at.markdown
    )
    assert not any(
        item.value == "**Incident / Accident Risk**" for item in at.markdown
    )


def test_technical_details_disclose_evidence_and_limitations() -> None:
    at = run_app_without_mapbox()
    technical = next(
        expander
        for expander in at.expander
        if expander.label == "Technical Details & Data Limitations"
    )
    assert technical is not None
    assert any(
        "OSM A* vs Dijkstra baseline" in item.value for item in at.markdown
    )
    assert any(
        "Mapbox routes are evidence only" in item.value for item in at.markdown
    )
    assert any(
        "not Karachi crash history" in item.value for item in at.caption
    )


def test_recommended_card_shows_successful_ml_prediction() -> None:
    at = run_app_without_mapbox()
    assert [str(exc) for exc in at.exception] == []

    headline, probability, departure = recommended_prediction(at)

    assert headline != "**Predicted traffic at departure:** Unavailable"
    assert headline.rsplit(" ", 1)[-1] in {"LOW", "MODERATE", "HIGH"}
    assert probability is not None
    assert probability.endswith("%")
    assert departure is not None
    assert any(
        metric.label == "Confidence" and metric.value in {"Low", "Medium"}
        for metric in at.metric
    )
    assert any(
        "Prediction provenance: Mixed real + synthetic data" in item.value
        for item in at.caption
    )
    assert any(
        "Class probabilities:" in item.value for item in at.caption
    )


def test_changing_departure_time_refreshes_recommended_prediction() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=240)
    at.run()
    next(
        checkbox
        for checkbox in at.checkbox
        if checkbox.label == "Use matched Mapbox traffic evidence"
    ).set_value(False)
    next(
        time_input
        for time_input in at.time_input
        if time_input.label == "Departure time"
    ).set_value(time(8, 0))
    next(
        button for button in at.button if button.label == "Find Best Route"
    ).click()
    at.run()
    morning = recommended_prediction(at)

    next(
        time_input
        for time_input in at.time_input
        if time_input.label == "Departure time"
    ).set_value(time(22, 0))
    next(
        button for button in at.button if button.label == "Find Best Route"
    ).click()
    at.run()
    night = recommended_prediction(at)

    assert morning[2] is not None and morning[2].endswith("08:00")
    assert night[2] is not None and night[2].endswith("22:00")
    assert morning[:2] != night[:2]


def test_switching_recommended_route_uses_that_routes_prediction(
    service: RouteService,
) -> None:
    response = service_response(service, future_departure(8))
    final = response.final_recommendation
    assert final is not None
    assert len(final.routes) >= 2
    target = final.routes[1]
    assert target.traffic_prediction is not None

    switched_final = replace(
        final,
        routes=(replace(target, rank=1, recommended=True),),
        recommended_route_id=target.route_id,
    )
    switched_response = replace(response, final_recommendation=switched_final)
    at = run_app_with_saved_response(switched_response)
    assert [str(exc) for exc in at.exception] == []

    headline, probability, departure = recommended_prediction(at)
    expected_prediction = target.traffic_prediction
    assert headline == (
        f"**Predicted traffic at departure:** "
        f"{expected_prediction.traffic_band.upper()}"
    )
    assert probability == f"{expected_prediction.traffic_probability:.0%}"
    assert departure == (
        f"{expected_prediction.requested_departure:%Y-%m-%d %H:%M}"
    )
    assert any(
        f"OSM route {target.route_id}" in item.value for item in at.caption
    )


def test_unavailable_prediction_is_labelled_and_never_fabricated(
    service: RouteService,
) -> None:
    response = service_response(service, future_departure(9))
    final = response.final_recommendation
    assert final is not None
    original = final.routes[0]
    assessment = original.assessment
    assert assessment is not None

    failed_assessment = replace(
        assessment,
        ml_eta_min=None,
        traffic_prediction=None,
        future_traffic_band=None,
        future_traffic_probability=None,
    )
    failed_decision = replace(
        original,
        traffic_prediction=None,
        assessment=failed_assessment,
    )
    failed_final = replace(
        final,
        routes=(failed_decision,),
        recommended_route_id=failed_decision.route_id,
    )
    failed_response = replace(response, final_recommendation=failed_final)
    at = run_app_with_saved_response(failed_response)
    assert [str(exc) for exc in at.exception] == []

    headline, probability, _ = recommended_prediction(at)

    assert headline == "**Predicted traffic at departure:** Unavailable"
    assert probability is None
    assert not any(
        metric.label == "Predicted congestion probability"
        for metric in at.metric
    )
    assert any(
        "No ML congestion band is shown" in item.value for item in at.caption
    )
    assert any(
        "no prediction was generated" in item.value for item in at.caption
    )


def test_provider_evidence_and_ml_prediction_are_separately_labelled(
    service: RouteService,
) -> None:
    departure = future_departure(9)
    response = service_response(service, departure)
    providers = tuple(
        MapboxRoute(
            route_id=f"mapbox-test-{index}",
            profile=MAPBOX_PROFILE,
            duration_s=route.eta_min * 60.0 * 0.9,
            duration_typical_s=route.eta_min * 60.0,
            distance_m=route.distance_m,
            geometry=route.geometry,
            congestion_values=(),
            congestion_numeric=(),
            incident_count=0,
            closure_count=0,
            traffic_evidence=True,
            fetched_at="test",
            requested_departure=departure.isoformat(),
        )
        for index, route in enumerate(response.academic_routes, start=1)
    )
    response_with_evidence = replace(
        response,
        mapbox_routes=providers,
        mapbox_verdicts=(),
    )
    final = service.recommend_routes(
        response_with_evidence,
        departure,
        WeatherSnapshot(condition="dry", source="test"),
    )
    recommended = final.routes[0]
    assert recommended.assessment is not None
    evidence = recommended.assessment.traffic_evidence
    assert evidence is not None and evidence.applied is True
    assert recommended.traffic_prediction is not None

    at = run_app_with_saved_response(
        replace(response_with_evidence, final_recommendation=final)
    )
    assert [str(exc) for exc in at.exception] == []

    provider_metric = next(
        metric
        for metric in at.metric
        if metric.label == "Provider traffic evidence"
    )
    assert provider_metric.value == "Matched provider evidence"
    assert provider_metric.label != "Predicted traffic at departure"

    headline, probability, _ = recommended_prediction(at)
    prediction = recommended.traffic_prediction
    assert headline == (
        f"**Predicted traffic at departure:** "
        f"{prediction.traffic_band.upper()}"
    )
    assert probability == f"{prediction.traffic_probability:.0%}"
    assert any(
        item.value == "**Future-traffic ML evidence**" for item in at.markdown
    )
    assert any(
        item.value == "**Matched traffic evidence**" for item in at.markdown
    )
