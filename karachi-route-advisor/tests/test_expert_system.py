from __future__ import annotations

import pytest

from src.contracts import RouteCandidate
from src.expert_system import Rule, build_rules, evaluate_route


def make_route(**overrides) -> RouteCandidate:
    values = {
        "route_id": "osm-1",
        "source": "custom_astar",
        "provider": "OSM analytical",
        "geometry": ((24.9, 67.1), (24.91, 67.11)),
        "distance_m": 1000.0,
        "eta_min": 10.0,
        "risk": 0.3,
        "confidence": 0.55,
        "congestion": None,
        "incident_count": 0,
        "closure_count": 0,
        "traffic_available": False,
        "toll": False,
        "impassable": False,
        "unlit_ratio": 0.0,
    }
    values.update(overrides)
    return RouteCandidate(**values)


def min_values(**overrides):
    values = {
        "eta": 10.0,
        "distance": 1000.0,
        "risk": 0.3,
        "congestion": None,
    }
    values.update(overrides)
    return values


def evaluate(
    route: RouteCandidate,
    preference: str = "Fastest",
    minima: dict[str, float | None] | None = None,
    *,
    night: bool = False,
    weather: str = "dry",
    avoid_tolls: bool = False,
):
    return evaluate_route(
        route=route,
        preference=preference,
        min_values=min_values() if minima is None else minima,
        night=night,
        weather_condition=weather,
        avoid_tolls=avoid_tolls,
    )


def trace_ids(verdict) -> list[str]:
    return [entry.rule_id for entry in verdict.trace]


def test_rule_catalog_preserves_r01_to_r15() -> None:
    rules = build_rules()
    ids = [rule.id for rule in rules]
    assert len(ids) == len(set(ids))
    assert {f"R{index:02d}" for index in range(1, 16)} <= set(ids)
    derived = next(rule for rule in rules if rule.id == "R16")
    assert "derived fact" in derived.effect.lower()
    assert "R04" in derived.why and "R05" in derived.why


def test_closure_veto() -> None:
    verdict = evaluate(make_route(closure_count=1))
    assert verdict.vetoed is True
    assert "closure" in verdict.veto_reasons[0].lower()
    entry = next(entry for entry in verdict.trace if entry.rule_id == "R01")
    assert "closure_count=1" in entry.evidence


def test_impassable_segment_veto() -> None:
    verdict = evaluate(make_route(impassable=True))
    assert verdict.vetoed is True
    assert any("impassable" in reason.lower() for reason in verdict.veto_reasons)
    entry = next(entry for entry in verdict.trace if entry.rule_id == "R02")
    assert "impassable=true" in entry.evidence


def test_strict_toll_veto() -> None:
    verdict = evaluate(make_route(toll=True), avoid_tolls=True)
    assert verdict.vetoed is True
    assert any("toll" in reason.lower() for reason in verdict.veto_reasons)
    entry = next(entry for entry in verdict.trace if entry.rule_id == "R03")
    assert "avoid_tolls=true" in entry.evidence
    assert "toll=true" in entry.evidence

    allowed = evaluate(make_route(toll=True), avoid_tolls=False)
    assert allowed.vetoed is False
    assert "R03" not in trace_ids(allowed)


def test_heavy_congestion_penalty() -> None:
    verdict = evaluate(
        make_route(congestion=0.65, traffic_available=True),
        minima=min_values(congestion=0.65),
    )
    ids = trace_ids(verdict)
    assert verdict.vetoed is False
    assert verdict.penalty == pytest.approx(0.10)
    assert "R16" in ids
    assert "R04" in ids
    assert "R05" not in ids
    entry = next(entry for entry in verdict.trace if entry.rule_id == "R04")
    assert "congestion=0.65" in entry.evidence
    assert "congestion_band=heavy" in entry.evidence


def test_severe_congestion_penalty() -> None:
    verdict = evaluate(
        make_route(congestion=0.85, traffic_available=True),
        minima=min_values(congestion=0.85),
    )
    ids = trace_ids(verdict)
    assert verdict.penalty == pytest.approx(0.20)
    assert "R16" in ids
    assert "R05" in ids
    assert "R04" not in ids
    entry = next(entry for entry in verdict.trace if entry.rule_id == "R05")
    assert "congestion=0.85" in entry.evidence
    assert "congestion_band=severe" in entry.evidence


def test_incident_penalty() -> None:
    verdict = evaluate(make_route(incident_count=1))
    assert verdict.penalty == pytest.approx(0.08)
    assert "R06" in trace_ids(verdict)
    entry = next(entry for entry in verdict.trace if entry.rule_id == "R06")
    assert "incident_count=1" in entry.evidence


def test_high_road_risk_penalty() -> None:
    verdict = evaluate(
        make_route(risk=0.75),
        minima=min_values(risk=0.75),
    )
    assert verdict.penalty == pytest.approx(0.10)
    assert "R07" in trace_ids(verdict)
    entry = next(entry for entry in verdict.trace if entry.rule_id == "R07")
    assert "risk=0.75" in entry.evidence


def test_rain_and_risk_penalty() -> None:
    verdict = evaluate(
        make_route(risk=0.60),
        minima=min_values(risk=0.60),
        weather="moderate",
    )
    ids = trace_ids(verdict)
    assert verdict.penalty == pytest.approx(0.10)
    assert "R08" in ids
    assert "R07" not in ids
    entry = next(entry for entry in verdict.trace if entry.rule_id == "R08")
    assert "weather=moderate" in entry.evidence
    assert "risk=0.60" in entry.evidence


def test_night_and_unlit_penalty() -> None:
    verdict = evaluate(
        make_route(unlit_ratio=0.50),
        night=True,
    )
    assert verdict.penalty == pytest.approx(0.08)
    assert "R09" in trace_ids(verdict)
    entry = next(entry for entry in verdict.trace if entry.rule_id == "R09")
    assert "night=true" in entry.evidence
    assert "unlit_ratio=0.50" in entry.evidence


def test_slow_route_penalty() -> None:
    verdict = evaluate(
        make_route(eta_min=13.0),
        minima=min_values(eta=10.0),
    )
    assert verdict.penalty == pytest.approx(0.05)
    assert verdict.bonus == 0.0
    assert "R10" in trace_ids(verdict)
    entry = next(entry for entry in verdict.trace if entry.rule_id == "R10")
    assert "eta=13.00" in entry.evidence
    assert "best_eta=10.00" in entry.evidence


@pytest.mark.parametrize(
    ("preference", "route_overrides", "minimum_overrides", "rule_id", "evidence"),
    [
        (
            "Fastest",
            {},
            {},
            "R11",
            "eta=10.00",
        ),
        (
            "Shortest",
            {},
            {},
            "R12",
            "distance=1000.0",
        ),
        (
            "Least congested",
            {"congestion": 0.20, "traffic_available": True},
            {"congestion": 0.20},
            "R13",
            "congestion=0.20",
        ),
        (
            "Safest",
            {},
            {},
            "R14",
            "risk=0.30",
        ),
    ],
)
def test_preference_bonuses(
    preference: str,
    route_overrides: dict,
    minimum_overrides: dict,
    rule_id: str,
    evidence: str,
) -> None:
    verdict = evaluate(
        make_route(**route_overrides),
        preference=preference,
        minima=min_values(**minimum_overrides),
    )
    assert verdict.bonus == pytest.approx(0.08)
    assert rule_id in trace_ids(verdict)
    entry = next(entry for entry in verdict.trace if entry.rule_id == rule_id)
    assert evidence in entry.evidence
    assert f"preference={preference}" in entry.evidence


def test_traffic_warning_is_warning_only() -> None:
    missing = evaluate(make_route(traffic_available=False))
    present = evaluate(make_route(traffic_available=True))

    assert "R15" in trace_ids(missing)
    assert "R15" not in trace_ids(present)
    assert any("traffic" in warning.lower() for warning in missing.warnings)
    assert present.warnings == ()
    assert missing.penalty == present.penalty == 0.0
    assert missing.bonus == present.bonus == pytest.approx(0.08)
    entry = next(entry for entry in missing.trace if entry.rule_id == "R15")
    assert "traffic_available=false" in entry.evidence
    assert "source=custom_astar" in entry.evidence


def test_derived_congestion_fact_is_consumed_by_later_rules() -> None:
    severe = evaluate(
        make_route(congestion=0.85, traffic_available=True),
        minima=min_values(congestion=0.85),
    )
    severe_ids = trace_ids(severe)
    assert severe_ids.count("R16") == 1
    assert severe_ids.index("R16") < severe_ids.index("R05")
    derivation = next(entry for entry in severe.trace if entry.rule_id == "R16")
    consumer = next(entry for entry in severe.trace if entry.rule_id == "R05")
    assert "congestion=0.85" in derivation.evidence
    assert "congestion_band=severe" in derivation.evidence
    assert "congestion=0.85" in consumer.evidence
    assert "congestion_band=severe" in consumer.evidence

    heavy = evaluate(
        make_route(congestion=0.65, traffic_available=True),
        minima=min_values(congestion=0.65),
    )
    heavy_ids = trace_ids(heavy)
    assert heavy_ids.index("R16") < heavy_ids.index("R04")
    heavy_consumer = next(entry for entry in heavy.trace if entry.rule_id == "R04")
    assert "congestion_band=heavy" in heavy_consumer.evidence


def test_conflict_resolution_orders_by_stratum_priority_specificity_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specifications = [
        ("T03", "data", 99, 1),
        ("T01", "soft", 2, 1),
        ("T00", "soft", 2, 2),
        ("T02", "soft", 2, 2),
        ("T04", "soft", 1, 9),
    ]
    rules = [
        Rule(
            id=rule_id,
            name=f"Rule {rule_id}",
            stratum=stratum,
            priority=priority,
            specificity=specificity,
            why="Always applicable.",
            effect="Trace only",
            condition=lambda wm: True,
            action=lambda wm: None,
        )
        for rule_id, stratum, priority, specificity in specifications
    ]
    monkeypatch.setattr("src.expert_system.build_rules", lambda: rules)

    verdict = evaluate(make_route())
    assert trace_ids(verdict) == ["T00", "T02", "T01", "T04", "T03"]


def test_forward_chaining_can_require_a_second_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer = Rule(
        id="T01",
        name="Derived fact consumer",
        stratum="soft",
        priority=100,
        specificity=1,
        why="Derived fact is present.",
        effect="Bonus 0.01",
        condition=lambda wm: wm.has("derived_ready", wm.route_id, True),
        action=lambda wm: wm.bonus(0.01, "Consumed derived fact."),
    )
    producer = Rule(
        id="T02",
        name="Derived fact producer",
        stratum="soft",
        priority=1,
        specificity=1,
        why="ETA fact is present.",
        effect="Assert derived_ready",
        condition=lambda wm: wm.has("eta", wm.route_id, 10.0),
        action=lambda wm: wm.assert_fact("derived_ready", wm.route_id, True),
    )
    monkeypatch.setattr(
        "src.expert_system.build_rules", lambda: [consumer, producer]
    )

    verdict = evaluate(make_route())
    assert trace_ids(verdict) == ["T02", "T01"]
    assert verdict.bonus == pytest.approx(0.01)


def test_deterministic_trace_order() -> None:
    route = make_route(
        eta_min=13.0,
        risk=0.75,
        congestion=0.65,
        incident_count=1,
        unlit_ratio=0.50,
    )
    minima = min_values(eta=10.0, risk=0.75, congestion=0.65)
    first = evaluate(route, minima=minima, night=True, weather="heavy")
    second = evaluate(route, minima=minima, night=True, weather="heavy")

    expected = ["R16", "R04", "R06", "R08", "R07", "R09", "R10", "R15"]
    assert trace_ids(first) == expected
    assert trace_ids(second) == expected
    assert len(expected) == len(set(expected))
    assert [entry.evidence for entry in first.trace] == [
        entry.evidence for entry in second.trace
    ]
