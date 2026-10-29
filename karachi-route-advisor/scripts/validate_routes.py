from __future__ import annotations

import sys
from datetime import datetime
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import LANDMARKS
from src.contracts import WeatherSnapshot
from src.service import RouteService


def main() -> int:
    service = RouteService()
    weather = WeatherSnapshot(condition="dry", source="validation")
    counts: dict[int, int] = {}
    failures: list[tuple[str, str, str]] = []
    etas: list[float] = []
    distances: list[float] = []
    for origin, destination in permutations(LANDMARKS, 2):
        try:
            response = service.route(
                origin_name=origin,
                destination_name=destination,
                departure=datetime(2026, 10, 26, 9, 0),
                preference="Fastest",
                weather=weather,
                include_mapbox=False,
            )
        except Exception as exc:
            failures.append((origin, destination, f"{type(exc).__name__}: {exc}"))
            continue
        route_count = len(response.academic_routes)
        counts[route_count] = counts.get(route_count, 0) + 1
        if not response.academic_routes:
            failures.append((origin, destination, "no routes"))
            continue
        etas.append(min(route.eta_min for route in response.academic_routes))
        distances.append(min(route.distance_m for route in response.academic_routes))

    print(f"pairs={sum(counts.values()) + len(failures)}")
    print(f"route_count_distribution={dict(sorted(counts.items()))}")
    print(f"failures={failures}")
    if etas:
        print(f"eta_range_min={min(etas):.1f}")
        print(f"eta_range_max={max(etas):.1f}")
        print(f"distance_range_km_min={min(distances) / 1000.0:.2f}")
        print(f"distance_range_km_max={max(distances) / 1000.0:.2f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
