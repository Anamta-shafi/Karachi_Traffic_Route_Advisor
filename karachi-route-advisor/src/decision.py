from __future__ import annotations

from dataclasses import replace

from .config import PREFERENCE_WEIGHTS
from .contracts import RankedRoute, RouteCandidate, RouteVerdict


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high - low < 1e-12:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def rank_routes(
    routes: tuple[RouteCandidate, ...] | list[RouteCandidate],
    verdicts: tuple[RouteVerdict, ...] | list[RouteVerdict],
    preference: str,
) -> list[RankedRoute]:
    verdict_by_id = {verdict.route_id: verdict for verdict in verdicts}
    survivors = [
        route
        for route in routes
        if route.route_id in verdict_by_id and not verdict_by_id[route.route_id].vetoed
    ]
    if not survivors:
        return []

    weights = dict(PREFERENCE_WEIGHTS.get(preference, PREFERENCE_WEIGHTS["Fastest"]))
    include_congestion = all(route.congestion is not None for route in survivors)
    if not include_congestion:
        weights.pop("congestion", None)
    weight_total = sum(weights.values())
    if weight_total <= 0:
        weights = {"time": 1.0}
        weight_total = 1.0
    weights = {key: value / weight_total for key, value in weights.items()}

    time_values = _normalize([route.eta_min for route in survivors])
    distance_values = _normalize([route.distance_m for route in survivors])
    risk_values = _normalize([route.risk for route in survivors])
    congestion_values = (
        _normalize([float(route.congestion) for route in survivors])
        if include_congestion
        else [0.0 for _ in survivors]
    )

    ranked: list[RankedRoute] = []
    for index, route in enumerate(survivors):
        verdict = verdict_by_id[route.route_id]
        base_cost = (
            weights.get("time", 0.0) * time_values[index]
            + weights.get("distance", 0.0) * distance_values[index]
            + weights.get("congestion", 0.0) * congestion_values[index]
            + weights.get("risk", 0.0) * risk_values[index]
        )
        multiplier = max(0.75, min(1.25, 1.0 + verdict.penalty - verdict.bonus))
        confidence_penalty = 0.20 * (1.0 - route.confidence)
        final_cost = base_cost * multiplier + confidence_penalty
        score = max(0, min(100, round(100.0 * (1.0 - min(final_cost, 1.25) / 1.25))))
        congestion_text = (
            f"{float(route.congestion):.2f}" if route.congestion is not None else "unavailable"
        )
        explanation = (
            f"time={time_values[index]:.2f}, distance={distance_values[index]:.2f}, "
            f"risk={risk_values[index]:.2f}, congestion={congestion_text}; "
            f"penalty_multiplier={multiplier:.2f}; "
            f"confidence_penalty={confidence_penalty:.2f}"
        )
        if not include_congestion:
            explanation += "; congestion omitted and weights renormalized"
        ranked.append(
            RankedRoute(
                route_id=route.route_id,
                rank=0,
                score=score,
                final_cost=final_cost,
                base_cost=base_cost,
                multiplier=multiplier,
                confidence_penalty=confidence_penalty,
                explanation=explanation,
            )
        )

    route_by_id = {route.route_id: route for route in survivors}
    ranked.sort(
        key=lambda item: (
            item.final_cost,
            route_by_id[item.route_id].risk,
            route_by_id[item.route_id].congestion
            if route_by_id[item.route_id].congestion is not None
            else 0.5,
            route_by_id[item.route_id].eta_min,
            item.route_id,
        )
    )
    return [replace(item, rank=index + 1) for index, item in enumerate(ranked)]
