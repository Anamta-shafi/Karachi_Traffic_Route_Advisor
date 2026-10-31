from __future__ import annotations

from datetime import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.config import GRAPH_PATH

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


def test_dashboard_is_map_first_and_minimal() -> None:
    at = run_app_without_mapbox()
    assert [str(exc) for exc in at.exception] == []
    assert [subheader.value for subheader in at.subheader] == [
        "Map",
        "Selected route details",
        "Other best routes",
    ]
    assert not any(
        expander.label == "Technical Details & Data Limitations"
        for expander in at.expander
    )
    metric_labels = {metric.label for metric in at.metric}
    assert {"Selected ETA", "ETA", "Distance", "Traffic"} <= metric_labels
    assert "Predicted congestion probability" not in metric_labels
    assert any(
        item.value == "**Why choose this route**" for item in at.markdown
    )
    assert any(item.value == "**Why not**" for item in at.markdown)


def test_warning_when_live_traffic_is_unavailable() -> None:
    at = run_app_without_mapbox()
    assert any("Live traffic" in warning.value for warning in at.warning)


def test_departure_time_change_updates_response() -> None:
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
    first_departure = at.session_state["response"].departure

    next(
        time_input
        for time_input in at.time_input
        if time_input.label == "Departure time"
    ).set_value(time(22, 0))
    next(
        button for button in at.button if button.label == "Find Best Route"
    ).click()
    at.run()
    second_departure = at.session_state["response"].departure

    assert first_departure.hour == 8
    assert second_departure.hour == 22
