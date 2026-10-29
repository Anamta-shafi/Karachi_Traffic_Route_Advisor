from __future__ import annotations

from datetime import datetime

import pytest

from src.config import GRAPH_PATH, LANDMARKS, SNAP_WARNING_DISTANCE_M
from src.contracts import WeatherSnapshot
from src.osm_graph import nearest_node_with_distance
from src.service import RouteService

pytestmark = pytest.mark.skipif(
    not GRAPH_PATH.exists(),
    reason="Run scripts/fetch_graph.py to create the local graph cache.",
)


@pytest.fixture(scope="module")
def service() -> RouteService:
    return RouteService()


def dry_weather() -> WeatherSnapshot:
    return WeatherSnapshot(condition="dry", source="test")


def test_route_exposes_dijkstra_baseline(service: RouteService) -> None:
    response = service.route(
        origin_name="Johar Chowk Mosque",
        destination_name="Block 8 bus stop",
        departure=datetime(2026, 10, 26, 9, 0),
        preference="Fastest",
        weather=dry_weather(),
        include_mapbox=False,
    )
    assert response.academic_routes
    assert response.ranked
    assert response.baseline is not None
    assert response.baseline.same_cost is True
    assert response.baseline.astar_expanded_nodes > 0
    assert response.baseline.dijkstra_expanded_nodes > 0
    assert response.baseline.dijkstra_route.route_id == "dijkstra-baseline"
    assert response.baseline.dijkstra_route.source == "dijkstra"
    assert response.baseline.dijkstra_route.geometry
    assert response.mapbox_routes == ()
    assert response.mapbox_verdicts == ()
    assert response.traffic_status == "no_mapbox"
    assert all(route.geometry for route in response.academic_routes)
    assert all(
        route.route_id != "dijkstra-baseline" for route in response.academic_routes
    )
    assert len({route.edges for route in response.academic_routes}) == len(
        response.academic_routes
    )


def test_alternatives_are_diverse_against_primary(service: RouteService) -> None:
    response = service.route(
        origin_name="Johar Chowk Mosque",
        destination_name="Block 8 bus stop",
        departure=datetime(2026, 10, 26, 9, 0),
        preference="Fastest",
        weather=dry_weather(),
        include_mapbox=False,
    )
    primary = response.academic_routes[0]
    primary_edges = set(primary.edges)
    for route in response.academic_routes[1:]:
        shared = len(primary_edges & set(route.edges))
        overlap = shared / max(1, min(len(primary_edges), len(route.edges)))
        assert overlap <= 0.70
        assert route.eta_min <= primary.eta_min * 1.45


def test_landmarks_are_close_to_the_drive_network(service: RouteService) -> None:
    for name, point in LANDMARKS.items():
        _, distance = nearest_node_with_distance(service.graph, *point)
        assert distance <= SNAP_WARNING_DISTANCE_M, (name, distance)


def test_invalid_inputs_are_rejected(service: RouteService) -> None:
    with pytest.raises(ValueError, match="Unknown origin"):
        service.route(
            origin_name="Not a landmark",
            destination_name="Block 8 bus stop",
            departure=datetime(2026, 10, 26, 9, 0),
            preference="Fastest",
            weather=dry_weather(),
            include_mapbox=False,
        )
    with pytest.raises(ValueError, match="must be different"):
        service.route(
            origin_name="Shell Jauhar",
            destination_name="Shell Jauhar",
            departure=datetime(2026, 10, 26, 9, 0),
            preference="Fastest",
            weather=dry_weather(),
            include_mapbox=False,
        )
    with pytest.raises(ValueError, match="Unknown preference"):
        service.route(
            origin_name="Johar Chowk Mosque",
            destination_name="Block 8 bus stop",
            departure=datetime(2026, 10, 26, 9, 0),
            preference="Fastestest",
            weather=dry_weather(),
            include_mapbox=False,
        )


def test_no_route_response_is_clear(
    service: RouteService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.service.generate_candidates", lambda *args, **kwargs: [])
    response = service.route(
        origin_name="Johar Chowk Mosque",
        destination_name="Block 8 bus stop",
        departure=datetime(2026, 10, 26, 9, 0),
        preference="Fastest",
        weather=dry_weather(),
        include_mapbox=False,
    )
    assert response.academic_routes == ()
    assert response.ranked == ()
    assert response.baseline is None
    assert response.traffic_status == "no_mapbox"
    assert any("No feasible route" in warning for warning in response.warnings)


def test_missing_mapbox_token_uses_osm_fallback(
    service: RouteService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MAPBOX_ACCESS_TOKEN", raising=False)
    response = service.route(
        origin_name="Johar Chowk Mosque",
        destination_name="Block 8 bus stop",
        departure=datetime(2026, 10, 26, 9, 0),
        preference="Fastest",
        weather=dry_weather(),
        include_mapbox=True,
    )
    assert response.mapbox_routes == ()
    assert response.traffic_status == "no_mapbox"
    assert any("not set" in warning for warning in response.warnings)
