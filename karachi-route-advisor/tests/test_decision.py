from __future__ import annotations

import pytest

from src.contracts import RouteCandidate, RouteVerdict
from src.decision import rank_routes


def make_route(**overrides) -> RouteCandidate:
    values = {
        "route_id": "osm-1",
        "source": "custom_astar",
        "provider": "OSM analytical",
        "geometry": ((24.9, 67.1),),
        "distance_m": 1000.0,
        "eta_min": 10.0,
        "risk": 0.3,
        "confidence": 0.55,
        "congestion": None,
        "incident_count": 0,
        "closure_count": 0,
        "traffic_available": False,
    }
    values.update(overrides)
    return RouteCandidate(**values)


def make_verdict(
    route_id: str,
    vetoed: bool = False,
    penalty: float = 0.0,
    bonus: float = 0.0,
) -> RouteVerdict:
    return RouteVerdict(
        route_id=route_id,
        vetoed=vetoed,
        veto_reasons=("closed",) if vetoed else (),
        penalty=penalty,
        bonus=bonus,
        trace=(),
    )


def by_id(ranked):
    return {item.route_id: item for item in ranked}


def test_fastest_ranks_first() -> None:
    routes = [
        make_route(route_id="osm-1", eta_min=10.0, distance_m=2000.0),
        make_route(route_id="osm-2", eta_min=20.0, distance_m=1000.0),
    ]
    verdicts = [make_verdict("osm-1"), make_verdict("osm-2")]
    ranked = rank_routes(routes, verdicts, "Fastest")
    assert ranked[0].route_id == "osm-1"


def test_shortest_ranks_first() -> None:
    routes = [
        make_route(route_id="osm-1", eta_min=10.0, distance_m=2000.0),
        make_route(route_id="osm-2", eta_min=20.0, distance_m=1000.0),
    ]
    verdicts = [make_verdict("osm-1"), make_verdict("osm-2")]
    ranked = rank_routes(routes, verdicts, "Shortest")
    assert ranked[0].route_id == "osm-2"


def test_least_congested_ranks_first() -> None:
    routes = [
        make_route(route_id="osm-1", congestion=0.8),
        make_route(route_id="osm-2", congestion=0.2),
    ]
    verdicts = [make_verdict("osm-1"), make_verdict("osm-2")]
    ranked = rank_routes(routes, verdicts, "Least congested")
    assert ranked[0].route_id == "osm-2"


def test_safest_ranks_lowest_risk_first() -> None:
    routes = [
        make_route(
            route_id="osm-1",
            eta_min=10.0,
            distance_m=1000.0,
            risk=0.80,
            congestion=0.20,
        ),
        make_route(
            route_id="osm-2",
            eta_min=20.0,
            distance_m=2000.0,
            risk=0.10,
            congestion=0.20,
        ),
    ]
    verdicts = [make_verdict("osm-1"), make_verdict("osm-2")]
    ranked = rank_routes(routes, verdicts, "Safest")
    assert ranked[0].route_id == "osm-2"


def test_vetoed_route_is_excluded() -> None:
    routes = [
        make_route(route_id="osm-1"),
        make_route(route_id="osm-2", eta_min=11.0),
    ]
    verdicts = [make_verdict("osm-1", vetoed=True), make_verdict("osm-2")]
    ranked = rank_routes(routes, verdicts, "Fastest")
    assert [item.route_id for item in ranked] == ["osm-2"]


def test_all_vetoed_returns_empty() -> None:
    routes = [make_route(route_id="osm-1")]
    verdicts = [make_verdict("osm-1", vetoed=True)]
    assert rank_routes(routes, verdicts, "Fastest") == []


def test_confidence_penalty_lowers_confidence() -> None:
    routes = [
        make_route(route_id="low", confidence=0.50),
        make_route(route_id="high", confidence=1.00),
    ]
    verdicts = [make_verdict("low"), make_verdict("high")]
    ranked = rank_routes(routes, verdicts, "Fastest")
    items = by_id(ranked)

    assert items["low"].confidence_penalty == pytest.approx(0.10)
    assert items["high"].confidence_penalty == pytest.approx(0.0)
    assert items["low"].final_cost == pytest.approx(0.10)
    assert items["high"].final_cost == pytest.approx(0.0)
    assert items["low"].score < items["high"].score
    assert ranked[0].route_id == "high"


def test_penalty_and_bonus_multipliers_apply_to_base_cost() -> None:
    routes = [
        make_route(
            route_id="a",
            eta_min=10.0,
            distance_m=1000.0,
            risk=0.0,
            confidence=1.0,
        ),
        make_route(
            route_id="b",
            eta_min=15.0,
            distance_m=1500.0,
            risk=0.5,
            confidence=1.0,
        ),
        make_route(
            route_id="c",
            eta_min=20.0,
            distance_m=2000.0,
            risk=1.0,
            confidence=1.0,
        ),
    ]
    verdicts = [
        make_verdict("a"),
        make_verdict("b", penalty=0.10),
        make_verdict("c", bonus=0.25),
    ]
    ranked = rank_routes(routes, verdicts, "Fastest")
    items = by_id(ranked)

    assert items["a"].base_cost == pytest.approx(0.0)
    assert items["b"].base_cost == pytest.approx(0.5)
    assert items["c"].base_cost == pytest.approx(1.0)
    assert items["a"].multiplier == pytest.approx(1.00)
    assert items["b"].multiplier == pytest.approx(1.10)
    assert items["c"].multiplier == pytest.approx(0.75)
    assert items["b"].final_cost == pytest.approx(0.55)
    assert items["c"].final_cost == pytest.approx(0.75)
    assert items["b"].confidence_penalty == pytest.approx(0.0)
    expected_score = max(
        0,
        min(
            100,
            round(100.0 * (1.0 - min(items["b"].final_cost, 1.25) / 1.25)),
        ),
    )
    assert items["b"].score == expected_score


def test_multiplier_clamps_at_bounds() -> None:
    routes = [
        make_route(route_id="a", eta_min=10.0, distance_m=1000.0, risk=0.0),
        make_route(route_id="b", eta_min=20.0, distance_m=2000.0, risk=1.0),
    ]
    verdicts = [
        make_verdict("a", penalty=0.50),
        make_verdict("b", bonus=0.50),
    ]
    ranked = rank_routes(routes, verdicts, "Fastest")
    items = by_id(ranked)

    assert items["a"].multiplier == pytest.approx(1.25)
    assert items["b"].multiplier == pytest.approx(0.75)


def test_congestion_is_renormalized_when_unavailable() -> None:
    routes = [
        make_route(
            route_id="short",
            eta_min=10.0,
            distance_m=1000.0,
            risk=0.3,
            congestion=None,
        ),
        make_route(
            route_id="long",
            eta_min=20.0,
            distance_m=2000.0,
            risk=0.3,
            congestion=None,
        ),
    ]
    verdicts = [make_verdict("short"), make_verdict("long")]
    ranked = rank_routes(routes, verdicts, "Fastest")
    items = by_id(ranked)

    assert items["short"].base_cost == pytest.approx(0.0)
    assert items["long"].base_cost == pytest.approx(0.875)
    assert "congestion omitted and weights renormalized" in items["long"].explanation
    assert "congestion=unavailable" in items["long"].explanation


def test_full_congestion_weight_when_all_routes_have_data() -> None:
    routes = [
        make_route(
            route_id="short",
            eta_min=10.0,
            distance_m=1000.0,
            risk=0.3,
            congestion=0.2,
        ),
        make_route(
            route_id="long",
            eta_min=20.0,
            distance_m=2000.0,
            risk=0.3,
            congestion=0.8,
        ),
    ]
    verdicts = [make_verdict("short"), make_verdict("long")]
    ranked = rank_routes(routes, verdicts, "Fastest")
    items = by_id(ranked)

    assert items["short"].base_cost == pytest.approx(0.0)
    assert items["long"].base_cost == pytest.approx(0.90)
    assert "renormalized" not in items["long"].explanation
    assert "congestion=0.80" in items["long"].explanation


def test_es_multiplier_can_change_ranking() -> None:
    routes = [
        make_route(
            route_id="baseline",
            eta_min=10.0,
            distance_m=1000.0,
            risk=0.0,
            confidence=1.0,
        ),
        make_route(
            route_id="penalized",
            eta_min=15.0,
            distance_m=1800.0,
            risk=0.5,
            confidence=1.0,
        ),
        make_route(
            route_id="bonused",
            eta_min=17.5,
            distance_m=2000.0,
            risk=1.0,
            confidence=1.0,
        ),
    ]
    no_effects = [
        make_verdict("baseline"),
        make_verdict("penalized"),
        make_verdict("bonused"),
    ]
    with_effects = [
        make_verdict("baseline"),
        make_verdict("penalized", penalty=0.50),
        make_verdict("bonused", bonus=0.50),
    ]

    without = by_id(rank_routes(routes, no_effects, "Fastest"))
    with_es = by_id(rank_routes(routes, with_effects, "Fastest"))

    assert without["penalized"].rank < without["bonused"].rank
    assert with_es["bonused"].rank < with_es["penalized"].rank
    assert with_es["penalized"].multiplier == pytest.approx(1.25)
    assert with_es["bonused"].multiplier == pytest.approx(0.75)
