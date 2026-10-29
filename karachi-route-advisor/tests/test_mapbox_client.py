from __future__ import annotations

import builtins
from typing import Any

import pytest
import requests

from src.contracts import MapboxRoute
from src.mapbox_client import (
    MapboxError,
    fetch_routes,
    traffic_evidence_status,
)

ORIGIN = (24.90930, 67.12596)
DESTINATION = (24.93187, 67.15488)
TOKEN_ENV = "MAPBOX_ACCESS_TOKEN"


class FakeResponse:
    def __init__(self, status_code: int, payload: Any):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def make_route(
    *,
    duration: float = 600.0,
    duration_typical: float | None = None,
    distance: float = 5000.0,
    congestion: list[str] | None = None,
    congestion_numeric: list[int | None] | None = None,
    annotation_durations: list[float | None] | None = None,
    annotation_distances: list[float | None] | None = None,
    annotation_speeds: list[float | None] | None = None,
    incidents: list[dict[str, Any]] | None = None,
    closures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "duration": duration,
        "distance": distance,
        "duration_typical": duration_typical,
        "geometry": {
            "coordinates": [
                [ORIGIN[1], ORIGIN[0]],
                [DESTINATION[1], DESTINATION[0]],
            ]
        },
        "legs": [
            {
                "annotation": {
                    "congestion": congestion if congestion is not None else [],
                    "congestion_numeric": (
                        congestion_numeric if congestion_numeric is not None else []
                    ),
                    "duration": (
                        annotation_durations
                        if annotation_durations is not None
                        else []
                    ),
                    "distance": (
                        annotation_distances
                        if annotation_distances is not None
                        else []
                    ),
                    "speed": (
                        annotation_speeds
                        if annotation_speeds is not None
                        else []
                    ),
                },
                "incidents": incidents if incidents is not None else [],
                "closures": closures if closures is not None else [],
            }
        ],
    }


def make_payload(*routes: dict[str, Any]) -> dict[str, Any]:
    return {"code": "Ok", "routes": list(routes)}


def install_response(
    monkeypatch: pytest.MonkeyPatch,
    payload: Any,
    status: int = 200,
) -> None:
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    monkeypatch.setattr(
        "src.mapbox_client.requests.get",
        lambda *args, **kwargs: FakeResponse(status, payload),
    )


def test_missing_token_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    with pytest.raises(MapboxError) as error:
        fetch_routes(ORIGIN, DESTINATION)
    assert error.value.code == "missing_token"
    assert error.value.status is None


def test_successful_response_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = make_payload(
        make_route(
            duration=600.0,
            duration_typical=650.0,
            congestion=["low", "heavy"],
            congestion_numeric=[10, 80],
            incidents=[{"id": "incident-1"}],
        )
    )
    install_response(monkeypatch, payload)

    routes = fetch_routes(ORIGIN, DESTINATION)
    assert len(routes) == 1
    route = routes[0]
    assert route.duration_s == 600.0
    assert route.distance_m == 5000.0
    assert route.duration_delta_s == -50.0
    assert route.duration_delta_pct == pytest.approx(-50.0 / 650.0 * 100.0)
    assert route.traffic_evidence is True
    assert route.incident_count == 1
    assert route.geometry[0] == (24.90930, 67.12596)
    assert route.fetched_at


def test_traffic_annotation_arrays_and_requested_departure_are_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = make_payload(
        make_route(
            duration=600.0,
            duration_typical=650.0,
            congestion=["low"],
            congestion_numeric=[20],
            annotation_durations=[420.0],
            annotation_distances=[5000.0],
            annotation_speeds=[42.5],
        )
    )
    install_response(monkeypatch, payload)

    routes = fetch_routes(
        ORIGIN,
        DESTINATION,
        depart_at="2026-10-26T04:00:00Z",
    )
    route = routes[0]
    assert route.requested_departure == "2026-10-26T04:00:00Z"
    assert route.annotation_durations_s == (420.0,)
    assert route.annotation_distances_m == (5000.0,)
    assert route.annotation_speeds_kph == (42.5,)


@pytest.mark.parametrize("duration", [601.0, 599.0])
def test_delta_only_status_is_exact_literal(
    monkeypatch: pytest.MonkeyPatch,
    duration: float,
) -> None:
    payload = make_payload(
        make_route(
            duration=duration,
            duration_typical=600.0,
            congestion=["unknown"],
            congestion_numeric=[None],
        )
    )
    install_response(monkeypatch, payload)

    routes = fetch_routes(ORIGIN, DESTINATION)
    assert routes[0].traffic_evidence is False
    assert routes[0].duration_delta_s == pytest.approx(duration - 600.0)
    assert traffic_evidence_status(routes) == "partial_eta_delta_only"


def test_annotated_status_from_congestion(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = make_payload(
        make_route(
            congestion=["unknown", "low"],
            congestion_numeric=[None, 25],
        )
    )
    install_response(monkeypatch, payload)

    routes = fetch_routes(ORIGIN, DESTINATION)
    assert routes[0].traffic_evidence is True
    assert traffic_evidence_status(routes) == "annotated"


def test_unknown_congestion_and_small_delta_have_no_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = make_payload(
        make_route(
            duration=600.5,
            duration_typical=600.0,
            congestion=["unknown"],
            congestion_numeric=[None],
        )
    )
    install_response(monkeypatch, payload)

    routes = fetch_routes(ORIGIN, DESTINATION)
    route = routes[0]
    assert route.traffic_evidence is False
    assert route.congestion_values == ("unknown",)
    assert route.congestion_average is None
    assert traffic_evidence_status(routes) == "no_traffic_evidence"
    assert traffic_evidence_status(()) == "no_traffic_evidence"


def test_error_response_is_reported_and_token_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"code": "NoToken", "message": "Not authorized for test-token."}
    install_response(monkeypatch, payload, status=401)

    with pytest.raises(MapboxError) as error:
        fetch_routes(ORIGIN, DESTINATION)
    assert error.value.status == 401
    assert error.value.code == "NoToken"
    assert "test-token" not in str(error.value)
    assert "[redacted]" in str(error.value)


def test_malformed_json_becomes_mapbox_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_response(monkeypatch, ValueError("not json"))

    with pytest.raises(MapboxError) as error:
        fetch_routes(ORIGIN, DESTINATION)
    assert error.value.code == "invalid_json"


def test_non_object_json_becomes_mapbox_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_response(monkeypatch, ["not", "an", "object"])

    with pytest.raises(MapboxError) as error:
        fetch_routes(ORIGIN, DESTINATION)
    assert error.value.code == "invalid_json"


def test_network_exception_becomes_mapbox_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_ENV, "test-token")

    def fake_get(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr("src.mapbox_client.requests.get", fake_get)
    with pytest.raises(MapboxError) as error:
        fetch_routes(ORIGIN, DESTINATION)
    assert error.value.code == "network_error"
    assert error.value.status is None
    assert "ConnectionError" in str(error.value)


def test_incident_and_closure_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = make_payload(
        make_route(
            congestion=["unknown"],
            congestion_numeric=[None],
            incidents=[{"id": "incident-1"}, {"id": "incident-2"}],
            closures=[{"id": "closure-1"}],
        )
    )
    install_response(monkeypatch, payload)

    routes = fetch_routes(ORIGIN, DESTINATION)
    route = routes[0]
    assert route.incident_count == 2
    assert route.closure_count == 1
    assert route.traffic_evidence is True
    assert traffic_evidence_status(routes) == "annotated"


def test_congestion_numeric_average_and_string_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = make_payload(
        make_route(
            congestion=["low", "heavy"],
            congestion_numeric=[20, 60, None],
        ),
        make_route(congestion=["low", "heavy"]),
    )
    install_response(monkeypatch, payload)

    numeric_route, string_route = fetch_routes(ORIGIN, DESTINATION)
    assert numeric_route.congestion_numeric == (20, 60, None)
    assert numeric_route.congestion_average == pytest.approx(0.40)
    assert string_route.congestion_numeric == ()
    assert string_route.congestion_average == pytest.approx(0.45)


def test_client_returns_in_memory_objects_without_file_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = make_payload(
        make_route(congestion=["low"], congestion_numeric=[30])
    )
    install_response(monkeypatch, payload)

    def fail_open(*args, **kwargs):
        raise AssertionError("Mapbox client attempted file access")

    monkeypatch.setattr(builtins, "open", fail_open)
    routes = fetch_routes(ORIGIN, DESTINATION)
    assert isinstance(routes, tuple)
    assert routes
    assert all(isinstance(route, MapboxRoute) for route in routes)
