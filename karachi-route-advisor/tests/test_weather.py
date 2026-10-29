from __future__ import annotations

from typing import Any

import pytest
import requests

from src.weather import WEATHER_URL, fetch_weather


class FakeResponse:
    def __init__(self, payload: Any = None, json_error: Exception | None = None):
        self._payload = payload
        self._json_error = json_error

    def raise_for_status(self) -> None:
        return None

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload


def test_fetch_weather_parses_open_meteo_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "current": {
            "time": "2026-10-26T09:00",
            "temperature_2m": 27.5,
            "precipitation": 0.4,
            "rain": 0.4,
            "weather_code": 61,
        }
    }
    captured: dict[str, Any] = {}

    def fake_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return FakeResponse(payload)

    monkeypatch.setattr("src.weather.requests.get", fake_get)
    snapshot = fetch_weather(24.90930, 67.12596)

    assert snapshot.condition == "light"
    assert snapshot.source == "Open-Meteo"
    assert snapshot.temperature_c == 27.5
    assert snapshot.precipitation_mm == 0.4
    assert snapshot.observed_at == "2026-10-26T09:00"
    assert snapshot.warning is None
    assert captured["url"] == WEATHER_URL
    assert captured["params"]["latitude"] == 24.90930
    assert captured["params"]["longitude"] == 67.12596
    assert captured["params"]["timezone"] == "Asia/Karachi"
    assert captured["timeout"] > 0


def test_fetch_weather_falls_back_to_dry_on_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr("src.weather.requests.get", fake_get)
    snapshot = fetch_weather(24.90930, 67.12596)

    assert snapshot.condition == "dry"
    assert snapshot.source == "fallback"
    assert snapshot.temperature_c is None
    assert snapshot.precipitation_mm is None
    assert snapshot.observed_at is None
    assert "dry assumption" in snapshot.warning


def test_fetch_weather_falls_back_to_dry_on_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.weather.requests.get",
        lambda *args, **kwargs: FakeResponse(
            json_error=ValueError("invalid json")
        ),
    )
    snapshot = fetch_weather(24.90930, 67.12596)

    assert snapshot.condition == "dry"
    assert snapshot.source == "fallback"
    assert "dry assumption" in snapshot.warning
