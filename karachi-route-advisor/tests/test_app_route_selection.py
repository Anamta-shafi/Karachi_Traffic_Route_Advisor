from __future__ import annotations

from dataclasses import replace
from datetime import datetime, time, timedelta
from pathlib import Path

import folium
import pytest
from streamlit.testing.v1 import AppTest

import app
from src.config import GRAPH_PATH, MAPBOX_PROFILE
from src.contracts import MapboxRoute, RouteResponse, WeatherSnapshot
from src.service import RouteService

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

pytestmark = pytest.mark.skipif(
    not GRAPH_PATH.exists(),
    reason="Run scripts/fetch_graph.py to create the local graph cache.",
)


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


@pytest.fixture(scope="module")
def response(service: RouteService) -> RouteResponse:
    return service.route(
        origin_name="Johar Chowk Mosque",
        destination_name="Block 7 Decent Houses",
        departure=future_departure(8),
        preference="Fastest",
        weather=WeatherSnapshot(condition="dry", source="test"),
        include_mapbox=False,
    )


def run_app_with_response(response: RouteResponse) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=240)
    at.run()
    at.session_state["response"] = response
    at.run()
    return at


def click_route(at: AppTest, route_id: str) -> None:
    next(
        button
        for button in at.button
        if button.key == f"select-route-{route_id}"
    ).click()
    at.run()


def active_route_markdown(at: AppTest) -> str:
    return next(
        item.value
        for item in at.markdown
        if item.value.startswith("**Active route:**")
    )


def test_selection_defaults_to_recommended_route(
    response: RouteResponse,
) -> None:
    final = response.final_recommendation
    assert final is not None
    assert len(final.routes) == 3

    at = run_app_with_response(response)

    assert [str(exc) for exc in at.exception] == []
    assert at.session_state["selected_route_id"] == final.routes[0].route_id
    assert active_route_markdown(at) == (
        f"**Active route:** OSM route {final.routes[0].route_id}"
    )
    assert any(
        item.value == "Selected route details" for item in at.subheader
    )
    assert {button.key for button in at.button} >= {
        f"select-route-{decision.route_id}" for decision in final.routes
    }
    labels = {button.label for button in at.button}
    assert f"Selected: {app._route_label(final.routes[0])}" in labels
    for decision in final.routes[1:]:
        assert f"Select {app._route_label(decision)}" in labels


def test_clicking_each_alternative_updates_selection_and_details(
    response: RouteResponse,
) -> None:
    final = response.final_recommendation
    assert final is not None
    at = run_app_with_response(response)

    for decision in final.routes[1:]:
        click_route(at, decision.route_id)
        assert [str(exc) for exc in at.exception] == []
        assert at.session_state["selected_route_id"] == decision.route_id
        assert active_route_markdown(at) == (
            f"**Active route:** OSM route {decision.route_id}"
        )
        assert any(
            metric.label == "Selected ETA"
            and metric.value
            == f"{decision.assessment.decision_eta_min:.0f} min"
            for metric in at.metric
        )
        assert any(
            button.label
            == f"Selected: {app._route_label(decision)}"
            for button in at.button
        )


def test_map_highlights_only_the_selected_route(
    response: RouteResponse,
) -> None:
    final = response.final_recommendation
    assert final is not None
    for decision in final.routes:
        map_object = app.build_map(response, final, decision.route_id)
        lines = [
            child
            for child in map_object._children.values()
            if isinstance(child, folium.PolyLine)
        ]
        assert len(lines) == len(final.routes)

        selected_lines = [
            line
            for line in lines
            if line.options["color"] == app.SELECTED_ROUTE_COLOR
        ]
        other_lines = [
            line
            for line in lines
            if line.options["color"] == app.UNSELECTED_ROUTE_COLOR
        ]
        assert len(selected_lines) == 1
        assert selected_lines[0].options["weight"] == 10
        assert selected_lines[0].options["opacity"] == 1.0
        assert selected_lines[0].options["dashArray"] is None
        assert len(other_lines) == len(final.routes) - 1
        assert all(line.options["dashArray"] == "6 6" for line in other_lines)

        html = map_object.get_root().render()
        assert (
            f"<b>Selected OSM route:</b> {decision.route_id}" in html
        )
        assert "Other OSM routes" in html


def test_selected_alternative_uses_its_own_evidence_and_prediction(
    service: RouteService,
    response: RouteResponse,
) -> None:
    departure = future_departure(9)
    base = service.route(
        origin_name=response.origin,
        destination_name=response.destination,
        departure=departure,
        preference=response.preference,
        weather=WeatherSnapshot(condition="dry", source="test"),
        include_mapbox=False,
    )
    providers = tuple(
        MapboxRoute(
            route_id=f"mapbox-selection-{index}",
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
        for index, route in enumerate(base.academic_routes, start=1)
    )
    with_evidence = replace(
        base,
        mapbox_routes=providers,
        mapbox_verdicts=(),
    )
    final = service.recommend_routes(
        with_evidence,
        departure,
        WeatherSnapshot(condition="dry", source="test"),
    )
    target = final.routes[1]
    assert target.assessment is not None
    evidence = target.assessment.traffic_evidence
    prediction = target.traffic_prediction
    assert evidence is not None and evidence.applied is True
    assert prediction is not None

    target_prediction = replace(
        prediction,
        advice_label="Selftest selected-route advice",
        advice_message="This advice is attached to the selected alternative.",
    )
    target_risk = replace(
        target.incident_risk,
        level="high",
        source="Selftest selected-route incident feed",
        explanation="Selftest incident explanation for the selected route.",
        evidence_type="ROUTE_SPECIFIC",
        route_specific=True,
    )
    updated_target = replace(
        target,
        traffic_prediction=target_prediction,
        incident_risk=target_risk,
    )
    updated_final = replace(
        final,
        routes=(
            final.routes[0],
            updated_target,
            *final.routes[2:],
        ),
    )
    custom_response = replace(
        with_evidence,
        final_recommendation=updated_final,
    )
    at = run_app_with_response(custom_response)
    click_route(at, target.route_id)

    assert [str(exc) for exc in at.exception] == []
    assert active_route_markdown(at) == (
        f"**Active route:** OSM route {target.route_id}"
    )
    assert any(
        f"Provider route: {evidence.provider_route_id}" in item.value
        for item in at.caption
    )
    assert any(
        "Traffic-adjusted ETA from matched provider evidence:" in item.value
        for item in at.markdown
    )
    assert any(
        item.value
        == "**Incident risk for this route:** High"
        for item in at.markdown
    )
    assert any(
        "Source: Selftest selected-route incident feed" in item.value
        for item in at.caption
    )
    assert any(
        "Selftest selected-route advice" in item.value
        and "attached to the selected alternative" in item.value
        for item in at.caption
    )


def test_new_trip_selection_resets_to_recommended(
    response: RouteResponse,
) -> None:
    final = response.final_recommendation
    assert final is not None
    at = run_app_with_response(response)
    click_route(at, final.routes[1].route_id)
    assert at.session_state["selected_route_id"] == final.routes[1].route_id

    next(
        time_input
        for time_input in at.time_input
        if time_input.label == "Departure time"
    ).set_value(time(22, 0))
    next(
        button for button in at.button if button.label == "Find Best Route"
    ).click()
    at.run()

    refreshed = at.session_state["response"]
    refreshed_final = refreshed.final_recommendation
    assert refreshed_final is not None
    assert at.session_state["selected_route_id"] == (
        refreshed_final.routes[0].route_id
    )
    assert active_route_markdown(at) == (
        f"**Active route:** OSM route {refreshed_final.routes[0].route_id}"
    )


def test_invalid_saved_selection_falls_back_to_recommended_route(
    response: RouteResponse,
) -> None:
    final = response.final_recommendation
    assert final is not None
    at = run_app_with_response(response)
    at.session_state["selected_route_id"] = "missing-route"
    at.run()

    assert at.session_state["selected_route_id"] == final.routes[0].route_id
    assert active_route_markdown(at) == (
        f"**Active route:** OSM route {final.routes[0].route_id}"
    )
