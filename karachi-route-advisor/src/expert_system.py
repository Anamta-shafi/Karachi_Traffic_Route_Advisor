from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .contracts import (
    DepartureAdvisory,
    DepartureForecast,
    RouteCandidate,
    RouteVerdict,
    RuleTrace,
)

STRATUM_ORDER = {"veto": 0, "soft": 1, "preference": 2, "data": 3}
MAX_RULE_PASSES = 200
MAX_TRACE_ENTRIES = 200


@dataclass(frozen=True)
class Fact:
    predicate: str
    args: tuple[Any, ...] = ()


@dataclass
class WorkingMemory:
    route_id: str
    facts: set[Fact] = field(default_factory=set)
    fired: set[str] = field(default_factory=set)
    veto_reasons: list[str] = field(default_factory=list)
    penalties: float = 0.0
    bonuses: float = 0.0
    warnings: list[str] = field(default_factory=list)
    positives: list[str] = field(default_factory=list)
    trace: list[RuleTrace] = field(default_factory=list)

    def assert_fact(self, predicate: str, *args: Any) -> None:
        self.facts.add(Fact(predicate, tuple(args)))

    def has(self, predicate: str, *args: Any) -> bool:
        return Fact(predicate, tuple(args)) in self.facts

    def veto(self, reason: str) -> None:
        if reason not in self.veto_reasons:
            self.veto_reasons.append(reason)

    def penalty(self, amount: float, reason: str) -> None:
        self.penalties += amount

    def bonus(self, amount: float, reason: str) -> None:
        self.bonuses += amount

    def warning(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def positive(self, message: str) -> None:
        if message not in self.positives:
            self.positives.append(message)


@dataclass(frozen=True)
class Rule:
    id: str
    name: str
    stratum: str
    priority: int
    specificity: int
    why: str
    effect: str
    condition: Callable[[WorkingMemory], bool]
    action: Callable[[WorkingMemory], None]
    evidence: Callable[[WorkingMemory], str] | None = None

    def explain(self, wm: WorkingMemory) -> str:
        return self.evidence(wm) if self.evidence is not None else self.why


def _route_value(wm: WorkingMemory, predicate: str, default: Any = None) -> Any:
    for fact in wm.facts:
        if fact.predicate == predicate and fact.args and fact.args[0] == wm.route_id:
            return fact.args[1] if len(fact.args) > 1 else True
    return default


def _global_value(wm: WorkingMemory, predicate: str, default: Any = None) -> Any:
    for fact in wm.facts:
        if fact.predicate == predicate:
            return fact.args[0] if fact.args else True
    return default


def _close(first: float | None, second: float | None) -> bool:
    if first is None or second is None:
        return False
    return abs(float(first) - float(second)) < 1e-9


def _num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "unavailable"
    return f"{float(value):.{digits}f}"


def _boolean(value: Any) -> str:
    return "true" if bool(value) else "false"


def _count(value: Any) -> str:
    return str(int(value or 0))


def _congestion_band(wm: WorkingMemory) -> str | None:
    congestion = _route_value(wm, "congestion")
    if congestion is None:
        return None
    value = float(congestion)
    if value >= 0.80:
        return "severe"
    if value >= 0.60:
        return "heavy"
    if value >= 0.40:
        return "moderate"
    return "low"


def build_rules() -> list[Rule]:
    return [
        Rule(
            id="R01",
            name="Closed route veto",
            stratum="veto",
            priority=100,
            specificity=3,
            why="A live closure was reported on the route.",
            effect="Reject route",
            condition=lambda wm: (_route_value(wm, "closure_count", 0) or 0) > 0,
            action=lambda wm: wm.veto("Route has a live closure."),
            evidence=lambda wm: (
                f"closure_count={_count(_route_value(wm, 'closure_count', 0))} > 0"
            ),
        ),
        Rule(
            id="R02",
            name="Impassable segment veto",
            stratum="veto",
            priority=95,
            specificity=3,
            why="The route uses an OSM segment marked as inaccessible.",
            effect="Reject route",
            condition=lambda wm: bool(_route_value(wm, "impassable", False)),
            action=lambda wm: wm.veto("Route uses an impassable segment."),
            evidence=lambda wm: (
                f"impassable={_boolean(_route_value(wm, 'impassable', False))}"
            ),
        ),
        Rule(
            id="R03",
            name="Strict toll avoidance veto",
            stratum="veto",
            priority=90,
            specificity=3,
            why="The user selected strict toll avoidance and the route uses a toll segment.",
            effect="Reject route",
            condition=lambda wm: bool(_global_value(wm, "avoid_tolls", False))
            and bool(_route_value(wm, "toll", False)),
            action=lambda wm: wm.veto("Strict avoid-tolls preference violated."),
            evidence=lambda wm: (
                f"avoid_tolls={_boolean(_global_value(wm, 'avoid_tolls', False))}; "
                f"toll={_boolean(_route_value(wm, 'toll', False))}"
            ),
        ),
        Rule(
            id="R04",
            name="Heavy congestion penalty",
            stratum="soft",
            priority=80,
            specificity=2,
            why="Average congestion is between 60% and 80%.",
            effect="Penalty 0.10",
            condition=lambda wm: wm.has(
                "congestion_band", wm.route_id, "heavy"
            ),
            action=lambda wm: wm.penalty(0.10, "Heavy congestion."),
            evidence=lambda wm: (
                f"congestion={_num(_route_value(wm, 'congestion'))} in [0.60, 0.80); "
                f"derived congestion_band="
                f"{_route_value(wm, 'congestion_band', 'unavailable')}"
            ),
        ),
        Rule(
            id="R05",
            name="Severe congestion penalty",
            stratum="soft",
            priority=85,
            specificity=2,
            why="Average congestion is at least 80%.",
            effect="Penalty 0.20",
            condition=lambda wm: wm.has(
                "congestion_band", wm.route_id, "severe"
            ),
            action=lambda wm: wm.penalty(0.20, "Severe congestion."),
            evidence=lambda wm: (
                f"congestion={_num(_route_value(wm, 'congestion'))} >= 0.80; "
                f"derived congestion_band="
                f"{_route_value(wm, 'congestion_band', 'unavailable')}"
            ),
        ),
        Rule(
            id="R06",
            name="Incident penalty",
            stratum="soft",
            priority=70,
            specificity=2,
            why="At least one incident was reported on the route.",
            effect="Penalty 0.08",
            condition=lambda wm: (_route_value(wm, "incident_count", 0) or 0) >= 1,
            action=lambda wm: wm.penalty(0.08, "Incident on route."),
            evidence=lambda wm: (
                f"incident_count={_count(_route_value(wm, 'incident_count', 0))} >= 1"
            ),
        ),
        Rule(
            id="R07",
            name="High road-risk penalty",
            stratum="soft",
            priority=60,
            specificity=1,
            why="The OSM-derived road risk score is at least 0.70.",
            effect="Penalty 0.10",
            condition=lambda wm: (_route_value(wm, "risk", 0.0) or 0.0) >= 0.70,
            action=lambda wm: wm.penalty(0.10, "High road risk."),
            evidence=lambda wm: (
                f"risk={_num(_route_value(wm, 'risk', 0.0))} >= 0.70"
            ),
        ),
        Rule(
            id="R08",
            name="Rain and risk penalty",
            stratum="soft",
            priority=65,
            specificity=2,
            why="Moderate or heavy rain combines with a high-risk road score.",
            effect="Penalty 0.10",
            condition=lambda wm: _global_value(wm, "weather", "dry")
            in {"moderate", "heavy"}
            and (_route_value(wm, "risk", 0.0) or 0.0) >= 0.60,
            action=lambda wm: wm.penalty(0.10, "Rain plus high-risk road."),
            evidence=lambda wm: (
                f"weather={_global_value(wm, 'weather', 'unavailable')} in "
                "{moderate, heavy}; "
                f"risk={_num(_route_value(wm, 'risk', 0.0))} >= 0.60"
            ),
        ),
        Rule(
            id="R09",
            name="Night and unlit penalty",
            stratum="soft",
            priority=55,
            specificity=2,
            why="The trip is at night and at least half of the route is unlit.",
            effect="Penalty 0.08",
            condition=lambda wm: bool(_global_value(wm, "night", False))
            and (_route_value(wm, "unlit_ratio", 0.0) or 0.0) >= 0.50,
            action=lambda wm: wm.penalty(0.08, "Night plus unlit road."),
            evidence=lambda wm: (
                f"night={_boolean(_global_value(wm, 'night', False))}; "
                f"unlit_ratio={_num(_route_value(wm, 'unlit_ratio', 0.0))} >= 0.50"
            ),
        ),
        Rule(
            id="R10",
            name="Slow route penalty",
            stratum="soft",
            priority=50,
            specificity=1,
            why="The route ETA is at least 25% above the best candidate.",
            effect="Penalty 0.05",
            condition=lambda wm: (_global_value(wm, "min_eta", 0.0) or 0.0) > 0
            and (_route_value(wm, "eta", 0.0) or 0.0)
            > float(_global_value(wm, "min_eta", 0.0)) * 1.25,
            action=lambda wm: wm.penalty(0.05, "ETA significantly above best."),
            evidence=lambda wm: (
                f"eta={_num(_route_value(wm, 'eta'))} > 1.25 * "
                f"best_eta={_num(_global_value(wm, 'min_eta'))}"
            ),
        ),
        Rule(
            id="R11",
            name="Fastest preference bonus",
            stratum="preference",
            priority=40,
            specificity=1,
            why="The route matches the user's fastest preference.",
            effect="Bonus 0.08",
            condition=lambda wm: _global_value(wm, "preference") == "Fastest"
            and _close(_route_value(wm, "eta"), _global_value(wm, "min_eta")),
            action=lambda wm: wm.bonus(0.08, "Matches fastest preference."),
            evidence=lambda wm: (
                f"preference={_global_value(wm, 'preference', 'unavailable')}; "
                f"eta={_num(_route_value(wm, 'eta'))} equals "
                f"best_eta={_num(_global_value(wm, 'min_eta'))}"
            ),
        ),
        Rule(
            id="R12",
            name="Shortest preference bonus",
            stratum="preference",
            priority=39,
            specificity=1,
            why="The route matches the user's shortest preference.",
            effect="Bonus 0.08",
            condition=lambda wm: _global_value(wm, "preference") == "Shortest"
            and _close(_route_value(wm, "distance"), _global_value(wm, "min_distance")),
            action=lambda wm: wm.bonus(0.08, "Matches shortest preference."),
            evidence=lambda wm: (
                f"preference={_global_value(wm, 'preference', 'unavailable')}; "
                f"distance={_num(_route_value(wm, 'distance'), 1)} equals "
                f"best_distance={_num(_global_value(wm, 'min_distance'), 1)}"
            ),
        ),
        Rule(
            id="R13",
            name="Least-congested preference bonus",
            stratum="preference",
            priority=38,
            specificity=1,
            why="The route matches the user's least-congested preference.",
            effect="Bonus 0.08",
            condition=lambda wm: _global_value(wm, "preference") == "Least congested"
            and _close(_route_value(wm, "congestion"), _global_value(wm, "min_congestion")),
            action=lambda wm: wm.bonus(0.08, "Matches least-congested preference."),
            evidence=lambda wm: (
                f"preference={_global_value(wm, 'preference', 'unavailable')}; "
                f"congestion={_num(_route_value(wm, 'congestion'))} equals "
                f"best_congestion={_num(_global_value(wm, 'min_congestion'))}"
            ),
        ),
        Rule(
            id="R14",
            name="Safest preference bonus",
            stratum="preference",
            priority=37,
            specificity=1,
            why="The route matches the user's safest preference.",
            effect="Bonus 0.08",
            condition=lambda wm: _global_value(wm, "preference") == "Safest"
            and _close(_route_value(wm, "risk"), _global_value(wm, "min_risk")),
            action=lambda wm: wm.bonus(0.08, "Matches safest preference."),
            evidence=lambda wm: (
                f"preference={_global_value(wm, 'preference', 'unavailable')}; "
                f"risk={_num(_route_value(wm, 'risk'))} equals "
                f"best_risk={_num(_global_value(wm, 'min_risk'))}"
            ),
        ),
        Rule(
            id="R15",
            name="Traffic provenance warning",
            stratum="data",
            priority=30,
            specificity=1,
            why="No live traffic measurement is available for this route.",
            effect="Warning only",
            condition=lambda wm: not bool(_route_value(wm, "traffic_available", False)),
            action=lambda wm: wm.warning(
                "No live traffic measurement is available for this route."
            ),
            evidence=lambda wm: (
                f"traffic_available="
                f"{_boolean(_route_value(wm, 'traffic_available', False))}; "
                f"source={_route_value(wm, 'source', 'unavailable')}"
            ),
        ),
        Rule(
            id="R16",
            name="Congestion band derivation",
            stratum="soft",
            priority=86,
            specificity=1,
            why=(
                "Congestion is available; classify it into a derived congestion_band "
                "fact consumed by R04 and R05."
            ),
            effect="Assert derived fact congestion_band (no direct score change)",
            condition=lambda wm: _congestion_band(wm) is not None,
            action=lambda wm: wm.assert_fact(
                "congestion_band", wm.route_id, _congestion_band(wm)
            ),
            evidence=lambda wm: (
                f"congestion={_num(_route_value(wm, 'congestion'))} -> "
                f"congestion_band="
                f"{_route_value(wm, 'congestion_band', 'unavailable')}"
            ),
        ),
    ]


def build_departure_rules() -> list[Rule]:
    disclosure = (
        "ML model trained on external traffic data — not locally validated for "
        "Gulistan-e-Jauhar."
    )
    accident_disclosure = (
        "Accident-risk model trained on external UK STATS19 collision data — not "
        "locally validated for Gulistan-e-Jauhar and not corridor-matched to Karachi."
    )
    return [
        Rule(
            id="R17",
            name="External traffic model disclosure",
            stratum="data",
            priority=29,
            specificity=1,
            why="The departure forecast uses an external calendar-only model.",
            effect="Warning only",
            condition=lambda wm: bool(
                _route_value(wm, "external_model", False)
            ),
            action=lambda wm: wm.warning(disclosure),
            evidence=lambda wm: (
                f"model={_route_value(wm, 'model_kind', 'unavailable')}; "
                f"route_context={_route_value(wm, 'route_context_band', 'unavailable')}"
            ),
        ),
        Rule(
            id="R18",
            name="External accident prior disclosure",
            stratum="data",
            priority=28,
            specificity=1,
            why="The accident component is an external UK collision-time prior.",
            effect="Warning only",
            condition=lambda wm: bool(
                _route_value(wm, "external_accident_model", False)
            ),
            action=lambda wm: wm.warning(accident_disclosure),
            evidence=lambda wm: (
                f"accident_band="
                f"{_route_value(wm, 'accident_band', 'unavailable')}; "
                f"not_corridor_matched=true"
            ),
        ),
        Rule(
            id="R19",
            name="High departure assessment warning",
            stratum="soft",
            priority=75,
            specificity=2,
            why="The combined departure-time assessment is high.",
            effect="Warning only",
            condition=lambda wm: wm.has(
                "final_assessment", wm.route_id, "high"
            ),
            action=lambda wm: wm.warning(
                "High predicted calendar-slot risk. Consider shifting departure "
                "outside this window."
            ),
            evidence=lambda wm: (
                f"traffic={_route_value(wm, 'traffic_band', 'unavailable')}; "
                f"accident={_route_value(wm, 'accident_band', 'unavailable')}; "
                "policy=worst_component_with_route_context"
            ),
        ),
        Rule(
            id="R20",
            name="Moderate departure assessment warning",
            stratum="soft",
            priority=74,
            specificity=2,
            why="The combined departure-time assessment is moderate.",
            effect="Warning only",
            condition=lambda wm: wm.has(
                "final_assessment", wm.route_id, "moderate"
            ),
            action=lambda wm: wm.warning(
                "Moderate predicted calendar-slot risk. Allow extra time and "
                "compare routes."
            ),
            evidence=lambda wm: (
                f"traffic={_route_value(wm, 'traffic_band', 'unavailable')}; "
                f"accident={_route_value(wm, 'accident_band', 'unavailable')}; "
                f"route_context="
                f"{_route_value(wm, 'route_context_band', 'unavailable')}"
            ),
        ),
        Rule(
            id="R21",
            name="Low departure assessment note",
            stratum="preference",
            priority=36,
            specificity=1,
            why="The combined departure-time assessment is low.",
            effect="Positive note only",
            condition=lambda wm: wm.has(
                "final_assessment", wm.route_id, "low"
            ),
            action=lambda wm: wm.positive(
                "Low predicted calendar-slot risk. Current conditions may differ; "
                "check Mapbox and route context."
            ),
            evidence=lambda wm: (
                f"traffic={_route_value(wm, 'traffic_band', 'unavailable')}; "
                f"accident={_route_value(wm, 'accident_band', 'unavailable')}"
            ),
        ),
        Rule(
            id="R22",
            name="Requested-departure Mapbox evidence separation",
            stratum="data",
            priority=27,
            specificity=1,
            why=(
                "Mapbox evidence is requested for the selected departure and is "
                "not a training label or the OSM future-traffic ML prediction."
            ),
            effect="Warning only",
            condition=lambda wm: wm.has(
                "mapbox_available", wm.route_id, True
            ),
            action=lambda wm: wm.warning(
                "Requested-departure Mapbox evidence is applied only when it "
                "belongs to the selected Mapbox route."
                if bool(_route_value(wm, "mapbox_applied", False))
                else "Requested-departure Mapbox evidence was not applied to this "
                "route; it is shown separately."
            ),
            evidence=lambda wm: (
                f"mapbox_applied="
                f"{_boolean(_route_value(wm, 'mapbox_applied', False))}; "
                f"mapbox_evidence={_route_value(wm, 'mapbox_level', 'unavailable')}"
            ),
        ),
    ]


def evaluate_departure_advisory(forecast: DepartureForecast) -> DepartureAdvisory:
    wm = WorkingMemory(forecast.route_id)
    wm.assert_fact("final_assessment", forecast.route_id, forecast.final_assessment)
    wm.assert_fact("traffic_band", forecast.route_id, forecast.predicted_traffic)
    wm.assert_fact("accident_band", forecast.route_id, forecast.accident_risk)
    wm.assert_fact("route_context_band", forecast.route_id, forecast.route_context.context_band)
    wm.assert_fact("model_kind", forecast.route_id, forecast.traffic.model_kind)
    wm.assert_fact("external_model", forecast.route_id, forecast.traffic.available)
    wm.assert_fact(
        "external_accident_model", forecast.route_id, forecast.accident.available
    )
    wm.assert_fact(
        "mapbox_available",
        forecast.route_id,
        forecast.current_mapbox_traffic not in {"not_applied", "unknown"},
    )
    wm.assert_fact("mapbox_applied", forecast.route_id, forecast.current_mapbox_applied)
    wm.assert_fact("mapbox_level", forecast.route_id, forecast.current_mapbox_traffic)

    rules = sorted(
        build_departure_rules(),
        key=lambda rule: (
            STRATUM_ORDER[rule.stratum],
            -rule.priority,
            -rule.specificity,
            rule.id,
        ),
    )
    for _ in range(MAX_RULE_PASSES):
        fired_any = False
        for rule in rules:
            if rule.id in wm.fired:
                continue
            if not rule.condition(wm):
                continue
            rule.action(wm)
            wm.fired.add(rule.id)
            wm.trace.append(
                RuleTrace(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    stratum=rule.stratum,
                    priority=rule.priority,
                    route_id=forecast.route_id,
                    evidence=rule.explain(wm),
                    effect=rule.effect,
                )
            )
            fired_any = True
            if len(wm.trace) >= MAX_TRACE_ENTRIES:
                break
        if not fired_any or len(wm.trace) >= MAX_TRACE_ENTRIES:
            break

    return DepartureAdvisory(
        route_id=forecast.route_id,
        final_assessment=forecast.final_assessment,
        warnings=tuple(wm.warnings),
        positives=tuple(wm.positives),
        trace=tuple(wm.trace),
        evidence_status=forecast.evidence_status,
        model_version=forecast.model_version,
    )


def evaluate_route(
    route: RouteCandidate,
    preference: str,
    min_values: dict[str, float | None],
    night: bool,
    weather_condition: str,
    avoid_tolls: bool,
    excluded_rule_ids: frozenset[str] = frozenset(),
) -> RouteVerdict:
    wm = WorkingMemory(route.route_id)
    wm.assert_fact("eta", route.route_id, route.eta_min)
    wm.assert_fact("distance", route.route_id, route.distance_m)
    wm.assert_fact("risk", route.route_id, route.risk)
    wm.assert_fact("congestion", route.route_id, route.congestion)
    wm.assert_fact("incident_count", route.route_id, route.incident_count)
    wm.assert_fact("closure_count", route.route_id, route.closure_count)
    wm.assert_fact("traffic_available", route.route_id, route.traffic_available)
    wm.assert_fact("toll", route.route_id, route.toll)
    wm.assert_fact("impassable", route.route_id, route.impassable)
    wm.assert_fact("unlit_ratio", route.route_id, route.unlit_ratio)
    wm.assert_fact("source", route.route_id, route.source)
    wm.assert_fact("preference", preference)
    wm.assert_fact("avoid_tolls", avoid_tolls)
    wm.assert_fact("night", night)
    wm.assert_fact("weather", weather_condition)
    wm.assert_fact("min_eta", min_values.get("eta"))
    wm.assert_fact("min_distance", min_values.get("distance"))
    wm.assert_fact("min_risk", min_values.get("risk"))
    wm.assert_fact("min_congestion", min_values.get("congestion"))

    rules = sorted(
        (
            rule
            for rule in build_rules()
            if rule.id not in excluded_rule_ids
        ),
        key=lambda rule: (
            STRATUM_ORDER[rule.stratum],
            -rule.priority,
            -rule.specificity,
            rule.id,
        ),
    )
    for _ in range(MAX_RULE_PASSES):
        fired_any = False
        for rule in rules:
            if rule.id in wm.fired:
                continue
            if not rule.condition(wm):
                continue
            rule.action(wm)
            wm.fired.add(rule.id)
            wm.trace.append(
                RuleTrace(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    stratum=rule.stratum,
                    priority=rule.priority,
                    route_id=route.route_id,
                    evidence=rule.explain(wm),
                    effect=rule.effect,
                )
            )
            fired_any = True
            if len(wm.trace) >= MAX_TRACE_ENTRIES:
                break
        if not fired_any or len(wm.trace) >= MAX_TRACE_ENTRIES:
            break

    return RouteVerdict(
        route_id=route.route_id,
        vetoed=bool(wm.veto_reasons),
        veto_reasons=tuple(wm.veto_reasons),
        penalty=wm.penalties,
        bonus=wm.bonuses,
        trace=tuple(wm.trace),
        warnings=tuple(wm.warnings),
    )
