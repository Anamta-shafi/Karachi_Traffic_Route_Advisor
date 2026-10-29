from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from .calendar_features import calendar_features
from .config import WEATHER_FACTORS
from .contracts import RouteCandidate
from .route_features import extract_route_context

FEATURE_NAMES: tuple[str, ...] = (
    "distance_km",
    "free_flow_eta_min",
    "analytical_eta_min",
    "analytical_delay_ratio",
    "segment_count",
    "mean_free_flow_speed_kph",
    "mean_segment_length_m",
    "primary_share",
    "secondary_share",
    "tertiary_share",
    "residential_share",
    "trunk_share",
    "motorway_share",
    "service_share",
    "other_share",
    "unlit_ratio",
    "is_toll",
    "risk",
    "context_band_rank",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_friday",
    "is_morning_peak",
    "is_evening_peak",
    "is_night",
    "weather_factor",
    "is_raining",
    "external_traffic_index",
    "external_accident_index",
)

ROAD_BUCKETS: tuple[str, ...] = (
    "primary",
    "secondary",
    "tertiary",
    "residential",
    "trunk",
    "motorway",
    "service",
    "other",
)

CONTEXT_BAND_RANK = {"unknown": -1.0, "low": 0.0, "moderate": 1.0, "high": 2.0}

FEATURE_LABELS: dict[str, str] = {
    "distance_km": "route distance",
    "free_flow_eta_min": "free-flow travel time",
    "analytical_eta_min": "analytical ETA",
    "analytical_delay_ratio": "analytical time-of-day delay",
    "segment_count": "number of road segments",
    "mean_free_flow_speed_kph": "mean free-flow speed",
    "mean_segment_length_m": "mean segment length",
    "primary_share": "primary-road share",
    "secondary_share": "secondary-road share",
    "tertiary_share": "tertiary-road share",
    "residential_share": "residential-road share",
    "trunk_share": "trunk-road share",
    "motorway_share": "motorway share",
    "service_share": "service-road share",
    "other_share": "other-road share",
    "unlit_ratio": "unlit-road share",
    "is_toll": "toll exposure",
    "risk": "OSM road risk",
    "context_band_rank": "OSM route-context band",
    "hour_sin": "departure hour (cyclic)",
    "hour_cos": "departure hour (cyclic)",
    "dow_sin": "day of week (cyclic)",
    "dow_cos": "day of week (cyclic)",
    "month_sin": "month (cyclic)",
    "month_cos": "month (cyclic)",
    "is_weekend": "weekend departure",
    "is_friday": "Friday departure",
    "is_morning_peak": "weekday morning peak",
    "is_evening_peak": "weekday evening peak",
    "is_night": "night-time departure",
    "weather_factor": "weather slowdown factor",
    "is_raining": "rain or wet weather",
    "external_traffic_index": "external time-of-week traffic index",
    "external_accident_index": "external time-of-week collision index",
}

SHARE_FEATURES = {
    "primary_share",
    "secondary_share",
    "tertiary_share",
    "residential_share",
    "trunk_share",
    "motorway_share",
    "service_share",
    "other_share",
    "unlit_ratio",
}

FLAG_FEATURES = {
    "is_toll",
    "is_weekend",
    "is_friday",
    "is_morning_peak",
    "is_evening_peak",
    "is_night",
    "is_raining",
}


def _normalise_composition(
    composition: tuple[tuple[str, float], ...],
) -> dict[str, float]:
    buckets = {name: 0.0 for name in ROAD_BUCKETS}
    for road, share in composition:
        base = str(road)
        if base.endswith("_link"):
            base = base[: -len("_link")]
        bucket = base if base in buckets else "other"
        buckets[bucket] += float(share)
    total = sum(buckets.values())
    if total > 0:
        buckets = {name: value / total for name, value in buckets.items()}
    return buckets


def route_feature_values(
    route: RouteCandidate,
    departure: datetime,
    graph: Any,
    weather_condition: str = "dry",
    *,
    external_traffic_index: float = 1.0,
    external_accident_index: float = 1.0,
) -> dict[str, float] | None:
    if graph is None or not route.nodes:
        return None
    context = extract_route_context(
        route,
        graph=graph,
        departure=departure,
        weather_condition=weather_condition,
    )
    if (
        not context.available
        or context.free_flow_eta_min is None
        or context.ordinary_eta_min is None
        or context.free_flow_eta_min <= 0
    ):
        return None

    segment_count = max(1, len(route.nodes) - 1)
    distance_km = route.distance_m / 1000.0
    composition = _normalise_composition(context.road_composition)
    calendar = calendar_features(departure)
    dow = int(departure.weekday())
    hour = int(departure.hour)
    weather_factor = float(WEATHER_FACTORS.get(weather_condition, 1.0))
    values: dict[str, float] = {
        "distance_km": float(distance_km),
        "free_flow_eta_min": float(context.free_flow_eta_min),
        "analytical_eta_min": float(context.ordinary_eta_min),
        "analytical_delay_ratio": float(context.delay_ratio or 0.0),
        "segment_count": float(segment_count),
        "mean_free_flow_speed_kph": float(
            distance_km / (context.free_flow_eta_min / 60.0)
        ),
        "mean_segment_length_m": float(route.distance_m / segment_count),
        **{
            f"{name}_share": float(composition[name])
            for name in ROAD_BUCKETS
        },
        "unlit_ratio": float(route.unlit_ratio),
        "is_toll": float(bool(route.toll)),
        "risk": float(context.risk if context.risk is not None else route.risk),
        "context_band_rank": float(
            CONTEXT_BAND_RANK.get(context.context_band, -1.0)
        ),
        **calendar,
        "is_friday": float(dow == 4),
        "is_morning_peak": float(dow < 5 and 7 <= hour < 10),
        "is_evening_peak": float(dow < 5 and 17 <= hour < 20),
        "is_night": float(hour >= 19 or hour < 6),
        "weather_factor": weather_factor,
        "is_raining": float(weather_condition != "dry"),
        "external_traffic_index": float(external_traffic_index),
        "external_accident_index": float(external_accident_index),
    }
    return {name: values[name] for name in FEATURE_NAMES}


def feature_frame(values: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [[float(values[name]) for name in FEATURE_NAMES]],
        columns=list(FEATURE_NAMES),
    )


def describe_feature(name: str, value: float) -> str:
    label = FEATURE_LABELS.get(name, name.replace("_", " "))
    if name in SHARE_FEATURES:
        return f"{label} {value:.0%}"
    if name in FLAG_FEATURES:
        return f"{label}: {'yes' if value >= 0.5 else 'no'}"
    if name == "segment_count":
        return f"{label} {int(round(value))}"
    if name.endswith("_min"):
        return f"{label} {value:.1f} min"
    if name == "distance_km":
        return f"{label} {value:.2f} km"
    if name == "mean_segment_length_m":
        return f"{label} {value:.0f} m"
    if name == "mean_free_flow_speed_kph":
        return f"{label} {value:.1f} km/h"
    if name == "context_band_rank":
        bands = {-1.0: "unknown", 0.0: "low", 1.0: "moderate", 2.0: "high"}
        return f"{label} {bands.get(round(value), 'unknown')}"
    if name in {"external_traffic_index", "external_accident_index"}:
        return f"{label} {value:.2f}"
    if name.endswith("_sin") or name.endswith("_cos"):
        return ""
    return f"{label} {value:.2f}"
