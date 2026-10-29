from __future__ import annotations

from typing import Any

import requests

from .contracts import WeatherSnapshot

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_SECONDS = 10


def _condition_from_code(code: int, precipitation: float) -> str:
    if code in {0, 1, 2, 3}:
        return "dry"
    if code in {45, 48, 51, 53, 55, 56, 57, 61}:
        return "light"
    if code in {63, 66, 80, 81}:
        return "moderate"
    if code in {65, 67, 82, 95, 96, 99}:
        return "heavy"
    if code in {71, 73, 75, 77, 85, 86}:
        return "moderate" if precipitation > 0 else "light"
    return "dry"


def fetch_weather(lat: float, lon: float) -> WeatherSnapshot:
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,precipitation,rain,weather_code",
        "timezone": "Asia/Karachi",
    }
    try:
        response = requests.get(WEATHER_URL, params=params, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        current = payload.get("current", {})
        code = int(current.get("weather_code", 0))
        precipitation = float(current.get("precipitation", 0.0) or 0.0)
        condition = _condition_from_code(code, precipitation)
        return WeatherSnapshot(
            condition=condition,
            source="Open-Meteo",
            temperature_c=(
                float(current["temperature_2m"])
                if current.get("temperature_2m") is not None
                else None
            ),
            precipitation_mm=precipitation,
            observed_at=str(current.get("time", "")) or None,
        )
    except Exception:
        return WeatherSnapshot(
            condition="dry",
            source="fallback",
            warning="Open-Meteo unavailable; dry assumption used.",
        )
