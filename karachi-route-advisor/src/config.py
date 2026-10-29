from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
GRAPH_PATH = CACHE_DIR / "gulistan_jauhar_drive.graphml"

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class BBox:
    south: float
    west: float
    north: float
    east: float

    @property
    def osmnx(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    def contains(self, lat: float, lon: float) -> bool:
        return self.south <= lat <= self.north and self.west <= lon <= self.east


BBOX = BBox(south=24.898, west=67.108, north=24.944, east=67.168)
FALLBACK_BBOX = BBox(south=24.902, west=67.111, north=24.940, east=67.164)

LANDMARKS: dict[str, tuple[float, float]] = {
    "Johar Chowk Mosque": (24.90930, 67.12596),
    "Gulistan-e-Jauhar Police Station": (24.92329, 67.14750),
    "Jauhar Inn Banquet Block 1": (24.92724, 67.11290),
    "Block 8 bus stop": (24.93187, 67.15488),
    "Shell Jauhar": (24.90780, 67.13518),
    "Block 7 Decent Houses": (24.93527, 67.14852),
    "NED University": (24.93274, 67.11028),
    "Safari Park": (24.9195915, 67.1115831),
    "Karachi Race Club": (24.9360141, 67.1612179),
}

SNAP_WARNING_DISTANCE_M = 75.0

PREFERENCE_WEIGHTS: dict[str, dict[str, float]] = {
    "Fastest": {"time": 0.55, "distance": 0.15, "congestion": 0.20, "risk": 0.10},
    "Shortest": {"time": 0.15, "distance": 0.55, "congestion": 0.20, "risk": 0.10},
    "Least congested": {"time": 0.20, "distance": 0.10, "congestion": 0.60, "risk": 0.10},
    "Safest": {"time": 0.15, "distance": 0.10, "congestion": 0.15, "risk": 0.60},
}

HIGHWAY_SPEEDS_KPH: dict[str, float] = {
    "motorway": 80.0,
    "motorway_link": 50.0,
    "trunk": 70.0,
    "trunk_link": 45.0,
    "primary": 60.0,
    "primary_link": 40.0,
    "secondary": 50.0,
    "secondary_link": 35.0,
    "tertiary": 40.0,
    "tertiary_link": 30.0,
    "unclassified": 30.0,
    "residential": 30.0,
    "living_street": 20.0,
    "service": 20.0,
    "road": 30.0,
}

WEATHER_FACTORS: dict[str, float] = {
    "dry": 1.00,
    "light": 0.95,
    "moderate": 0.85,
    "heavy": 0.70,
}

DEFAULT_MAX_SPEED_KPH = 30.0
MAX_SPEED_KPH = 120.0
MAPBOX_PROFILE = "mapbox/driving-traffic"
MAPBOX_DRIVING_PROFILE = "mapbox/driving"
MAPBOX_TIMEOUT_SECONDS = 20


def mapbox_token() -> str | None:
    token = os.getenv("MAPBOX_ACCESS_TOKEN", "").strip()
    return token or None


def mapbox_token_present() -> bool:
    return mapbox_token() is not None


def time_factor(hour: int) -> float:
    if 7 <= hour < 10:
        return 0.65
    if 17 <= hour < 20:
        return 0.55
    if 10 <= hour < 17:
        return 0.90
    if 20 <= hour < 22:
        return 0.85
    return 1.10
