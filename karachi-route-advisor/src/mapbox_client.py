from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any, Literal

import requests

from .config import MAPBOX_PROFILE, MAPBOX_TIMEOUT_SECONDS, mapbox_token
from .contracts import MapboxRoute


TrafficEvidenceStatus = Literal[
    "annotated",
    "partial_eta_delta_only",
    "no_traffic_evidence",
]


class MapboxError(RuntimeError):
    def __init__(
        self,
        message: str,
        status: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


def _redact(text: str, token: str) -> str:
    return text.replace(token, "[redacted]") if token else text


def _parse_error(response: requests.Response, token: str) -> MapboxError:
    code = "http_error"
    message = ""
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if isinstance(payload, dict):
        code = _redact(str(payload.get("code", code)), token)
        message = str(payload.get("message", ""))
    safe_message = _redact(message, token)
    detail = f"Mapbox returned HTTP {response.status_code} ({code})."
    if safe_message:
        detail += f" {safe_message}"
    return MapboxError(detail, status=response.status_code, code=code)


def traffic_evidence_status(
    routes: Iterable[MapboxRoute],
) -> TrafficEvidenceStatus:
    route_list = tuple(routes)
    annotated = any(
        route.traffic_evidence
        or route.incident_count > 0
        or route.closure_count > 0
        or any(
            value not in (None, "unknown") for value in route.congestion_values
        )
        for route in route_list
    )
    if annotated:
        return "annotated"
    delta_only = any(
        route.duration_delta_s is not None
        and abs(route.duration_delta_s) >= 1.0
        for route in route_list
    )
    if delta_only:
        return "partial_eta_delta_only"
    return "no_traffic_evidence"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_route(
    route: Any,
    profile: str,
    index: int,
    requested_departure: str | None = None,
) -> MapboxRoute:
    legs = route.get("legs", [])
    leg = legs[0] if legs else {}
    annotation = leg.get("annotation", {})
    congestion_values = tuple(annotation.get("congestion", []))
    congestion_numeric = tuple(annotation.get("congestion_numeric", []))
    annotation_distances = tuple(
        _optional_float(value) for value in annotation.get("distance", [])
    )
    annotation_durations = tuple(
        _optional_float(value) for value in annotation.get("duration", [])
    )
    annotation_speeds = tuple(
        _optional_float(value) for value in annotation.get("speed", [])
    )
    incidents = leg.get("incidents", [])
    closures = leg.get("closures", [])
    geometry = route.get("geometry", {})
    coordinates_list = (
        geometry.get("coordinates", []) if isinstance(geometry, dict) else []
    )
    points = tuple(
        (float(point[1]), float(point[0]))
        for point in coordinates_list
        if isinstance(point, (list, tuple)) and len(point) >= 2
    )
    non_unknown = any(
        value not in (None, "unknown") for value in congestion_values
    )
    return MapboxRoute(
        route_id=f"mapbox-{profile.split('/')[-1]}-{index}",
        profile=profile,
        duration_s=float(route.get("duration", 0.0)),
        duration_typical_s=(
            float(route["duration_typical"])
            if route.get("duration_typical") is not None
            else None
        ),
        distance_m=float(route.get("distance", 0.0)),
        geometry=points,
        congestion_values=congestion_values,
        congestion_numeric=congestion_numeric,
        incident_count=len(incidents),
        closure_count=len(closures),
        traffic_evidence=bool(non_unknown or incidents or closures),
        fetched_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        requested_departure=requested_departure or "",
        annotation_distances_m=annotation_distances,
        annotation_durations_s=annotation_durations,
        annotation_speeds_kph=annotation_speeds,
    )


def fetch_routes(
    origin: tuple[float, float],
    destination: tuple[float, float],
    profile: str = MAPBOX_PROFILE,
    alternatives: bool = True,
    depart_at: str | None = None,
) -> tuple[MapboxRoute, ...]:
    token = mapbox_token()
    if token is None:
        raise MapboxError(
            "MAPBOX_ACCESS_TOKEN is not set.",
            code="missing_token",
        )

    coordinates = f"{origin[1]},{origin[0]};{destination[1]},{destination[0]}"
    url = f"https://api.mapbox.com/directions/v5/{profile}/{coordinates}"
    params: dict[str, Any] = {
        "alternatives": "true" if alternatives else "false",
        "geometries": "geojson",
        "overview": "full",
        "annotations": "duration,distance,speed,congestion,congestion_numeric,closure",
        "steps": "false",
        "access_token": token,
    }
    if depart_at:
        params["depart_at"] = depart_at

    try:
        response = requests.get(url, params=params, timeout=MAPBOX_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise MapboxError(
            f"Mapbox request failed ({type(exc).__name__}).",
            code="network_error",
        ) from None

    if response.status_code != 200:
        raise _parse_error(response, token)

    try:
        payload = response.json()
    except ValueError:
        raise MapboxError(
            "Mapbox returned invalid JSON.",
            code="invalid_json",
        ) from None

    if not isinstance(payload, dict):
        raise MapboxError("Mapbox returned invalid JSON.", code="invalid_json")

    if payload.get("code") != "Ok":
        code = _redact(str(payload.get("code", "unknown")), token)
        message = _redact(str(payload.get("message", "")), token)
        detail = f"Mapbox returned {code}."
        if message:
            detail += f" {message}"
        raise MapboxError(detail, status=response.status_code, code=code)

    routes = payload.get("routes", [])
    if not isinstance(routes, list):
        raise MapboxError(
            "Mapbox returned an invalid routes payload.",
            code="invalid_response",
        )
    if not routes:
        raise MapboxError("Mapbox returned no routes.", code="NoRoutes")

    try:
        return tuple(
            _parse_route(route, profile, index, depart_at)
            for index, route in enumerate(routes, start=1)
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        raise MapboxError(
            "Mapbox returned malformed route data.",
            code="invalid_response",
        ) from None
