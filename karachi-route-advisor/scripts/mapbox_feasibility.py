from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

API_ROOT = "https://api.mapbox.com/directions/v5"
ORIGIN = (24.90930, 67.12596)
DESTINATION = (24.93187, 67.15488)
TIMEOUT_SECONDS = 20
PROFILES = ("mapbox/driving-traffic", "mapbox/driving")


def _redact(text: str, token: str) -> str:
    return text.replace(token, "[redacted]") if token else text


def _build_url(
    profile: str,
    token: str,
    origin: tuple[float, float],
    destination: tuple[float, float],
) -> str:
    coordinates = f"{origin[1]},{origin[0]};{destination[1]},{destination[0]}"
    query = urllib.parse.urlencode(
        {
            "alternatives": "true",
            "geometries": "geojson",
            "overview": "full",
            "annotations": "duration,distance,speed,congestion,congestion_numeric,closure",
            "steps": "false",
            "access_token": token,
        }
    )
    return f"{API_ROOT}/{profile}/{coordinates}?{query}"


def _request(
    profile: str,
    token: str,
    origin: tuple[float, float],
    destination: tuple[float, float],
) -> dict[str, Any]:
    request = urllib.request.Request(
        _build_url(profile, token, origin, destination),
        headers={"User-Agent": "karachi-route-advisor-feasibility/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            status = response.status
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = exc.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return {
            "http_status": None,
            "code": "network_error",
            "message": type(exc).__name__,
            "routes": [],
        }

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return {
            "http_status": status,
            "code": "invalid_json",
            "message": "Response was not valid JSON.",
            "routes": [],
        }

    if status != 200 or payload.get("code") != "Ok":
        code = payload.get("code") or f"http_{status}"
        return {
            "http_status": status,
            "code": code,
            "message": _redact(str(payload.get("message", "")), token),
            "routes": [],
        }

    routes = []
    for route in payload.get("routes", []):
        legs = route.get("legs", [])
        leg = legs[0] if legs else {}
        annotation = leg.get("annotation", {})
        congestion = annotation.get("congestion", [])
        congestion_numeric = annotation.get("congestion_numeric", [])
        incidents = leg.get("incidents", [])
        closures = leg.get("closures", [])
        geometry = route.get("geometry", {})
        coordinates = geometry.get("coordinates", []) if isinstance(geometry, dict) else []
        non_unknown = sum(1 for value in congestion if value not in (None, "unknown"))
        duration = route.get("duration")
        duration_typical = route.get("duration_typical")
        duration_delta = (
            float(duration) - float(duration_typical)
            if duration is not None and duration_typical is not None
            else None
        )
        routes.append(
            {
                "duration_s": duration,
                "duration_typical_s": duration_typical,
                "duration_delta_s": duration_delta,
                "duration_delta_pct": (
                    duration_delta / float(duration_typical) * 100.0
                    if duration_delta is not None and float(duration_typical) != 0.0
                    else None
                ),
                "distance_m": route.get("distance"),
                "geometry_points": len(coordinates),
                "congestion_values": len(congestion),
                "non_unknown_congestion": non_unknown,
                "incidents": len(incidents),
                "closures": len(closures),
                "annotation_evidence": bool(non_unknown or incidents or closures),
            }
        )

    return {
        "http_status": status,
        "code": payload.get("code"),
        "message": "",
        "routes": routes,
    }


def _print_profile_result(profile: str, result: dict[str, Any]) -> None:
    print(f"profile={profile}")
    print(f"http_status={result['http_status']}")
    print(f"code={result['code']}")
    if result["message"]:
        print(f"message={result['message']}")
    print(f"routes={len(result['routes'])}")
    for index, route in enumerate(result["routes"]):
        print(
            "route"
            f"[{index}]: duration_s={route['duration_s']}"
            f" duration_typical_s={route['duration_typical_s']}"
            f" duration_delta_s={route['duration_delta_s']}"
            f" duration_delta_pct={route['duration_delta_pct']}"
            f" distance_m={route['distance_m']}"
            f" geometry_points={route['geometry_points']}"
            f" congestion_values={route['congestion_values']}"
            f" non_unknown_congestion={route['non_unknown_congestion']}"
            f" incidents={route['incidents']}"
            f" closures={route['closures']}"
            f" annotation_evidence={route['annotation_evidence']}"
        )
    print()


def main() -> int:
    token = os.getenv("MAPBOX_ACCESS_TOKEN", "").strip()
    if not token:
        print("MAPBOX_ACCESS_TOKEN is not set. Set it in the environment and rerun.")
        return 2

    print("Mapbox feasibility test")
    print("token_present=yes")
    print(f"origin=Johar Chowk Mosque {ORIGIN}")
    print(f"destination=Block 8 bus stop {DESTINATION}")
    print()

    results = {}
    for profile in PROFILES:
        results[profile] = _request(profile, token, ORIGIN, DESTINATION)
        _print_profile_result(profile, results[profile])

    traffic_result = results["mapbox/driving-traffic"]
    routes = traffic_result["routes"]
    annotation_evidence = any(route["annotation_evidence"] for route in routes)
    eta_delta_evidence = any(
        route["duration_delta_s"] is not None
        and abs(float(route["duration_delta_s"])) >= 1.0
        for route in routes
    )
    if annotation_evidence:
        traffic_aware_evidence = "yes"
    elif eta_delta_evidence:
        traffic_aware_evidence = "partial_eta_delta_only"
    else:
        traffic_aware_evidence = "no"
    traffic_ok = traffic_result["http_status"] == 200 and bool(routes)
    print(f"traffic_profile_ok={'yes' if traffic_ok else 'no'}")
    print(
        "traffic_annotation_evidence="
        f"{'yes' if annotation_evidence else 'no'}"
    )
    print(f"traffic_eta_delta_evidence={'yes' if eta_delta_evidence else 'no'}")
    print(f"traffic_aware_evidence={traffic_aware_evidence}")
    print(
        "note=Non-unknown congestion annotations, incidents, or closures are direct "
        "segment-level evidence. A duration-vs-typical delta is only evidence that the "
        "traffic profile changed the ETA; it does not provide segment-level congestion."
    )
    return 0 if traffic_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
