from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path

import folium
import pytest
from streamlit.testing.v1 import AppTest

import app
from src.config import GRAPH_PATH
from src.contracts import RouteResponse, WeatherSnapshot
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


def test_selection_defaults_to_recommended_route(
    response: RouteResponse,
) -> None:
    final = response.final_recommendation
    assert final is not None
    assert len(final.routes) == 3

    at = run_app_with_response(response)

    assert [str(exc) for exc in at.exception] == []
    assert at.session_state["selected_route_id"] == final.routes[0].route_id
    assert [subheader.value for subheader in at.subheader] == [
        "Map",
        "Selected route details",
        "Other best routes",
    ]
    assert {button.key for button in at.button} >= {
        f"select-route-{decision.route_id}" for decision in final.routes
    }
    labels = {button.label for button in at.button}
    assert f"Selected: {app._route_label(final.routes[0])}" in labels
    for decision in final.routes[1:]:
        assert f"Select {app._route_label(decision)}" in labels


def test_clicking_each_alternative_updates_selection_and_eta(
    response: RouteResponse,
) -> None:
    final = response.final_recommendation
    assert final is not None
    at = run_app_with_response(response)

    for decision in final.routes[1:]:
        click_route(at, decision.route_id)
        assert [str(exc) for exc in at.exception] == []
        assert at.session_state["selected_route_id"] == decision.route_id
        assert any(
            metric.label == "Selected ETA"
            and metric.value
            == f"{decision.assessment.decision_eta_min:.0f} min"
            for metric in at.metric
        )
        assert any(
            button.label == f"Selected: {app._route_label(decision)}"
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
        assert f"<b>Selected OSM route:</b> {decision.route_id}" in html
        assert "Other OSM routes" in html


def test_route_cards_explain_why_choose_and_why_not(
    response: RouteResponse,
) -> None:
    at = run_app_with_response(response)

    assert [str(exc) for exc in at.exception] == []
    assert any(
        item.value == "**Why choose this route**" for item in at.markdown
    )
    assert any(item.value == "**Why not**" for item in at.markdown)
    assert not any(
        expander.label == "Technical Details & Data Limitations"
        for expander in at.expander
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


def test_invalid_saved_selection_falls_back_to_recommended_route(
    response: RouteResponse,
) -> None:
    final = response.final_recommendation
    assert final is not None
    at = run_app_with_response(response)
    at.session_state["selected_route_id"] = "missing-route"
    at.run()

    assert at.session_state["selected_route_id"] == final.routes[0].route_id
