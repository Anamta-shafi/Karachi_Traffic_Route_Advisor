from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from src.config import GRAPH_PATH
from src.contracts import MapboxRoute, WeatherSnapshot
from src.mapbox_client import fetch_routes, traffic_evidence_status
from src.service import RouteService

ORIGIN = (24.90930, 67.12596)
DESTINATION = (24.93187, 67.15488)
TOKEN_ENV = "MAPBOX_ACCESS_TOKEN"


class FakeResponse:
    def __init__(self, payload: dict[str, Any]):
        self.status_code = 200
        self._payload = payload

    def json(self):
        return self._payload


def make_payload(
    *,
    duration: float,
    duration_typical: float,
    congestion: list[str],
    incidents: list[dict[str, Any]] | None = None,
    closures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "code": "Ok",
        "routes": [
            {
                "duration": duration,
                "distance": 5000.0,
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
                            "congestion": congestion,
                            "congestion_numeric": [None for _ in congestion],
                        },
                        "incidents": incidents if incidents is not None else [],
                        "closures": closures if closures is not None else [],
                    }
                ],
            }
        ],
    }


def test_parsed_delta_only_payload_reports_partial_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    monkeypatch.setattr(
        "src.mapbox_client.requests.get",
        lambda *args, **kwargs: FakeResponse(
            make_payload(
                duration=601.0,
                duration_typical=600.0,
                congestion=["unknown"],
            )
        ),
    )

    routes = fetch_routes(ORIGIN, DESTINATION)
    assert routes[0].traffic_evidence is False
    assert traffic_evidence_status(routes) == "partial_eta_delta_only"


def test_depart_at_is_forwarded_to_mapbox_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    captured: dict[str, Any] = {}

    def fake_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return FakeResponse(
            make_payload(
                duration=600.0,
                duration_typical=600.0,
                congestion=[],
            )
        )

    monkeypatch.setattr("src.mapbox_client.requests.get", fake_get)
    fetch_routes(ORIGIN, DESTINATION, depart_at="1793000000")

    assert captured["params"]["depart_at"] == "1793000000"
    assert captured["params"]["access_token"] == "test-token"
    assert captured["timeout"] > 0
    expected_coordinates = (
        f"{ORIGIN[1]},{ORIGIN[0]};{DESTINATION[1]},{DESTINATION[0]}"
    )
    assert captured["url"].endswith(expected_coordinates)


@pytest.mark.skipif(
    not GRAPH_PATH.exists(),
    reason="Run scripts/fetch_graph.py to create the local graph cache.",
)
def test_service_surfaces_delta_only_mapbox_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_ENV, "test-token")
    route = MapboxRoute(
        route_id="mapbox-driving-traffic-1",
        profile="mapbox/driving-traffic",
        duration_s=601.0,
        duration_typical_s=600.0,
        distance_m=5000.0,
        geometry=(ORIGIN, DESTINATION),
        congestion_values=("unknown",),
        congestion_numeric=(None,),
        incident_count=0,
        closure_count=0,
        traffic_evidence=False,
    )

    def fake_fetch(origin, destination, profile, alternatives=True, depart_at=None):
        return (route,)

    monkeypatch.setattr("src.service.fetch_routes", fake_fetch)
    service = RouteService()
    response = service.route(
        origin_name="Johar Chowk Mosque",
        destination_name="Block 8 bus stop",
        departure=datetime(2026, 10, 26, 9, 0),
        preference="Fastest",
        weather=WeatherSnapshot(condition="dry", source="test"),
        include_mapbox=True,
    )

    assert response.traffic_status == "partial_eta_delta_only"
    assert traffic_evidence_status(response.mapbox_routes) == "partial_eta_delta_only"
    assert any("ETA delta versus typical" in warning for warning in response.warnings)
