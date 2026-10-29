from __future__ import annotations

from datetime import datetime

from src.contracts import (
    RankedRoute,
    RouteCandidate,
    RouteTrafficPrediction,
    RouteVerdict,
)
from src.final_optimizer import (
    build_final_recommendation,
    build_incident_risk,
)

DEPARTURE = datetime(2026, 10, 26, 9, 0)


def make_route(
    route_id: str,
    *,
    source: str = "custom_astar",
    eta_min: float = 10.0,
    risk: float = 0.2,
    edges: tuple[tuple[int, int], ...] | None = None,
    incident_count: int = 0,
    closure_count: int = 0,
    traffic_available: bool = True,
) -> RouteCandidate:
    selected_edges = edges or ((1, 2), (2, 3), (3, 4))
    return RouteCandidate(
        route_id=route_id,
        source=source,
        provider="OSM",
        geometry=((24.9 + len(route_id) / 1000, 67.1), (24.95, 67.15)),
        distance_m=1000.0 + len(route_id) * 10,
        eta_min=eta_min,
        risk=risk,
        confidence=0.9,
        congestion=0.3,
        incident_count=incident_count,
        closure_count=closure_count,
        traffic_available=traffic_available,
        nodes=tuple(sorted({node for edge in selected_edges for node in edge})),
        edges=selected_edges,
    )


def make_mapbox_route(
    route_id: str,
    *,
    eta_min: float = 7.0,
    evidence_kind: str = "mapbox_traffic_eta",
    risk_available: bool = False,
    toll_available: bool = False,
    distance_m: float = 1050.0,
    geometry: tuple[tuple[float, float], ...] = (
        (24.92, 67.13),
        (24.94, 67.16),
    ),
) -> RouteCandidate:
    return RouteCandidate(
        route_id=route_id,
        source="mapbox",
        provider="Mapbox",
        geometry=geometry,
        distance_m=distance_m,
        eta_min=eta_min,
        risk=0.0,
        confidence=0.70,
        congestion=0.0,
        incident_count=0,
        closure_count=0,
        traffic_available=evidence_kind == "mapbox_traffic_eta",
        risk_available=risk_available,
        toll_available=toll_available,
        free_flow_eta_min=eta_min,
        eta_evidence_kind=evidence_kind,
        eta_evidence_label="Mapbox ETA for the requested departure",
    )


def make_verdict(route_id: str, *, vetoed: bool = False) -> RouteVerdict:
    return RouteVerdict(
        route_id=route_id,
        vetoed=vetoed,
        veto_reasons=("road closed",) if vetoed else (),
        penalty=0.0,
        bonus=0.0,
        trace=(),
    )


def make_ranked(route_id: str, rank: int, final_cost: float) -> RankedRoute:
    return RankedRoute(
        route_id=route_id,
        rank=rank,
        score=max(0, round(100 * (1 - final_cost))),
        final_cost=final_cost,
        base_cost=final_cost,
        multiplier=1.0,
        confidence_penalty=0.0,
        explanation="test ranking",
    )


def make_prediction(
    route_id: str,
    *,
    band: str = "low",
    probabilities: tuple[tuple[str, float], ...] | None = None,
    accident_risk: str = "low",
    accident_probability: float | None = 0.10,
    accident_evidence: str = "REAL",
    confidence: str = "Medium",
    status: str = "ok",
) -> RouteTrafficPrediction:
    selected = probabilities or (
        ("low", 0.7),
        ("moderate", 0.2),
        ("high", 0.1),
    )
    return RouteTrafficPrediction(
        route_id=route_id,
        route_source="custom_astar",
        requested_departure=DEPARTURE,
        status=status,
        traffic_band=band,
        traffic_probability=dict(selected).get(band),
        traffic_probabilities=selected,
        confidence=confidence,
        confidence_note="test",
        accident_risk=accident_risk,
        accident_probability=accident_probability,
        accident_evidence=accident_evidence,
        accident_source="UK STATS19 calendar model",
        advice_code="good_time",
        advice_label="Good time to leave",
        advice_message="test",
        explanation="test",
        top_drivers=(),
        data_provenance="mixed",
        data_provenance_label="Mixed real + synthetic data",
        model_name="test",
        model_version="test",
        training_dataset_size=10,
        real_records=5,
        synthetic_records=5,
        evaluation_accuracy=None,
        evaluation_precision=None,
        evaluation_recall=None,
        evaluation_f1=None,
        confusion_matrix=(),
        features=(),
        limitations=(),
        provenance=(),
        generated_at=DEPARTURE,
    )


def build(
    routes: list[RouteCandidate],
    ranked: list[RankedRoute],
    predictions: dict[str, RouteTrafficPrediction | None],
    *,
    verdicts: list[RouteVerdict] | None = None,
    errors: dict[str, str] | None = None,
    mapbox_candidates: list[RouteCandidate] | None = None,
    mapbox_verdicts: list[RouteVerdict] | None = None,
    avoid_tolls: bool = False,
):
    return build_final_recommendation(
        routes=routes,
        verdicts=verdicts or [make_verdict(route.route_id) for route in routes],
        academic_ranked=ranked,
        predictions=predictions,
        departure=DEPARTURE,
        preference="Fastest",
        prediction_errors=errors,
        mapbox_candidates=mapbox_candidates or [],
        mapbox_verdicts=mapbox_verdicts or [],
        avoid_tolls=avoid_tolls,
    )


def test_exactly_one_recommended_route_and_at_most_three_displayed() -> None:
    routes = [
        make_route("r1", edges=((1, 2), (2, 3))),
        make_route("r2", edges=((1, 2), (2, 4))),
        make_route("r3", edges=((1, 2), (2, 5))),
        make_route("r4", edges=((1, 2), (2, 6))),
    ]
    ranked = [
        make_ranked("r1", 1, 1.0),
        make_ranked("r2", 2, 1.1),
        make_ranked("r3", 3, 1.2),
        make_ranked("r4", 4, 1.3),
    ]
    predictions = {
        route.route_id: make_prediction(route.route_id) for route in routes
    }

    final = build(routes, ranked, predictions)

    assert len(final.routes) == 3
    assert final.recommended_route_id == "r1"
    assert [decision.recommended for decision in final.routes].count(True) == 1
    assert final.routes[0].recommended is True
    assert all(decision.rank == index for index, decision in enumerate(final.routes, 1))
    assert final.fallback_used is False
    assert final.weights == (
        ("time", 0.75),
        ("distance", 0.15),
        ("risk", 0.10),
    )


def test_hard_veto_cannot_be_recommended() -> None:
    routes = [
        make_route("vetoed", edges=((1, 2), (2, 3))),
        make_route("valid", edges=((1, 2), (2, 4))),
    ]
    ranked = [make_ranked("vetoed", 1, 1.0), make_ranked("valid", 2, 1.2)]
    predictions = {
        "vetoed": make_prediction("vetoed"),
        "valid": make_prediction("valid"),
    }
    verdicts = [make_verdict("vetoed", vetoed=True), make_verdict("valid")]

    final = build(routes, ranked, predictions, verdicts=verdicts)

    assert [decision.route_id for decision in final.routes] == ["valid"]
    assert final.recommended_route_id == "valid"
    assert any("hard veto" in warning for warning in final.warnings)


def test_future_traffic_can_change_the_final_ranking() -> None:
    routes = [
        make_route("academic-first", edges=((1, 2), (2, 3))),
        make_route("low-traffic", edges=((1, 2), (2, 4))),
        make_route("slow", edges=((1, 2), (2, 5))),
    ]
    ranked = [
        make_ranked("academic-first", 1, 1.0),
        make_ranked("low-traffic", 2, 1.2),
        make_ranked("slow", 3, 1.5),
    ]
    predictions = {
        "academic-first": make_prediction(
            "academic-first",
            band="high",
            probabilities=(("low", 0.0), ("moderate", 0.0), ("high", 1.0)),
            accident_risk="high",
            accident_probability=0.4,
        ),
        "low-traffic": make_prediction(
            "low-traffic",
            band="low",
            probabilities=(("low", 1.0), ("moderate", 0.0), ("high", 0.0)),
        ),
        "slow": make_prediction(
            "slow",
            band="moderate",
            probabilities=(("low", 0.0), ("moderate", 1.0), ("high", 0.0)),
        ),
    }

    final = build(routes, ranked, predictions)

    assert final.recommended_route_id == "low-traffic"
    assert final.routes[0].academic_rank == 2
    assert final.routes[0].traffic_component == 0.0
    assert final.routes[0].traffic_severity == 0.0
    assert final.fallback_used is False


def test_missing_prediction_falls_back_to_academic_and_is_not_zero_risk() -> None:
    routes = [
        make_route("r1", edges=((1, 2), (2, 3))),
        make_route("r2", edges=((1, 2), (2, 4))),
    ]
    ranked = [make_ranked("r1", 1, 1.0), make_ranked("r2", 2, 1.2)]
    predictions = {
        "r1": make_prediction("r1"),
        "r2": None,
    }

    final = build(
        routes,
        ranked,
        predictions,
        errors={"r2": "RuntimeError: model unavailable"},
    )

    assert final.fallback_used is True
    assert final.weights == (("academic_rank", 1.0),)
    assert final.recommended_route_id == "r1"
    assert all(decision.traffic_component is None for decision in final.routes)
    missing = next(decision for decision in final.routes if decision.route_id == "r2")
    assert missing.traffic_prediction is None
    assert missing.traffic_severity is None
    assert any("not treated as zero" in warning for warning in final.warnings)
    assert any("model unavailable" in warning for warning in final.warnings)


def test_incident_risk_is_unavailable_for_custom_routes_with_separate_prior() -> None:
    route = make_route("r1")
    prediction = make_prediction(
        "r1",
        accident_risk="high",
        accident_probability=0.35,
        accident_evidence="REAL",
    )

    incident = build_incident_risk(route, prediction)

    assert incident.level == "UNAVAILABLE"
    assert incident.probability is None
    assert incident.route_specific is False
    assert incident.evidence_type == "EXTERNAL_PRIOR"
    assert incident.external_prior_level == "HIGH"
    assert incident.external_prior_probability == 0.35
    assert "not Karachi crash history" in incident.explanation
    assert "not a crash probability for this route" in incident.explanation


def test_route_level_annotations_are_labelled_route_specific() -> None:
    route = make_route(
        "mapbox-1",
        source="mapbox",
        closure_count=1,
        incident_count=2,
    )

    incident = build_incident_risk(route, None)

    assert incident.level == "HIGH"
    assert incident.route_specific is True
    assert incident.evidence_type == "ROUTE_SPECIFIC"
    assert "Reported incidents: 2" in incident.explanation


def test_all_vetoed_routes_produce_no_recommendation() -> None:
    routes = [make_route("r1"), make_route("r2")]
    ranked = [make_ranked("r1", 1, 1.0), make_ranked("r2", 2, 1.1)]
    verdicts = [
        make_verdict("r1", vetoed=True),
        make_verdict("r2", vetoed=True),
    ]

    final = build(
        routes,
        ranked,
        {"r1": make_prediction("r1"), "r2": make_prediction("r2")},
        verdicts=verdicts,
    )

    assert final.routes == ()
    assert final.recommended_route_id is None
    assert final.advice_code == "insufficient_evidence"
    assert final.fallback_used is True


def test_duplicate_fingerprint_is_removed() -> None:
    edges = ((1, 2), (2, 3))
    first = make_route("r1", edges=edges)
    duplicate = RouteCandidate(
        route_id="r1-copy",
        source=first.source,
        provider=first.provider,
        geometry=first.geometry,
        distance_m=first.distance_m,
        eta_min=first.eta_min,
        risk=first.risk,
        confidence=first.confidence,
        congestion=first.congestion,
        incident_count=first.incident_count,
        closure_count=first.closure_count,
        traffic_available=first.traffic_available,
        nodes=first.nodes,
        edges=first.edges,
    )
    second = make_route("r2", edges=((1, 2), (2, 4)))
    ranked = [
        make_ranked("r1", 1, 1.0),
        make_ranked("r1-copy", 2, 1.05),
        make_ranked("r2", 3, 1.2),
    ]
    predictions = {
        route.route_id: make_prediction(route.route_id)
        for route in (first, duplicate, second)
    }

    final = build([first, duplicate, second], ranked, predictions)

    assert [decision.route_id for decision in final.routes] == ["r1", "r2"]
    assert any("duplicate fingerprint" in warning for warning in final.warnings)


def test_near_duplicate_overlap_is_removed() -> None:
    shared = ((1, 2), (2, 3), (3, 4), (4, 5))
    base = make_route("r1", edges=shared)
    near = make_route("r2", edges=shared + ((5, 6),))
    distinct = make_route("r3", edges=((10, 11), (11, 12)))
    ranked = [
        make_ranked("r1", 1, 1.0),
        make_ranked("r2", 2, 1.1),
        make_ranked("r3", 3, 1.2),
    ]
    predictions = {
        route.route_id: make_prediction(route.route_id)
        for route in (base, near, distinct)
    }

    final = build([base, near, distinct], ranked, predictions)

    assert "r2" not in {decision.route_id for decision in final.routes}
    assert any("near-duplicate" in warning for warning in final.warnings)


def test_ties_are_resolved_deterministically() -> None:
    routes = [
        make_route("b-route", edges=((1, 2), (2, 3))),
        make_route("a-route", edges=((1, 2), (2, 4))),
    ]
    ranked = [
        make_ranked("b-route", 1, 1.0),
        make_ranked("a-route", 2, 1.0),
    ]
    predictions = {
        "b-route": make_prediction("b-route"),
        "a-route": make_prediction("a-route"),
    }

    first = build(routes, ranked, predictions)
    second = build(routes, ranked, predictions)

    assert first.recommended_route_id == "a-route"
    assert second.recommended_route_id == "a-route"
    assert [item.route_id for item in first.routes] == [
        item.route_id for item in second.routes
    ]


def test_advice_uses_prediction_band_and_accident_prior() -> None:
    routes = [make_route("r1")]
    ranked = [make_ranked("r1", 1, 1.0)]
    prediction = make_prediction(
        "r1",
        band="high",
        probabilities=(("low", 0.0), ("moderate", 0.0), ("high", 1.0)),
        accident_risk="high",
        accident_probability=0.4,
    )

    final = build(routes, ranked, {"r1": prediction})

    assert final.advice_code == "not_recommended"
    assert final.advice_label == "Not recommended"


def test_moderate_traffic_advises_extra_time() -> None:
    routes = [make_route("r1")]
    ranked = [make_ranked("r1", 1, 1.0)]
    prediction = make_prediction(
        "r1",
        band="moderate",
        probabilities=(("low", 0.1), ("moderate", 0.8), ("high", 0.1)),
        accident_risk="low",
    )

    final = build(routes, ranked, {"r1": prediction})

    assert final.advice_code == "allow_extra_time"
    assert "extra time" in final.advice_message.lower()


def test_requested_departure_mapbox_eta_can_win_final_ai_decision() -> None:
    osm = make_route("osm-1")
    mapbox = make_mapbox_route("mapbox-1", eta_min=7.0)
    prediction = make_prediction(
        "osm-1",
        band="high",
        probabilities=(("low", 0.0), ("moderate", 0.0), ("high", 1.0)),
    )

    final = build(
        [osm],
        [make_ranked("osm-1", 1, 1.0)],
        {"osm-1": prediction},
        mapbox_candidates=[mapbox],
        mapbox_verdicts=[make_verdict("mapbox-1")],
    )

    recommended = final.routes[0]
    assert final.recommended_route_id == "mapbox-1"
    assert recommended.route.source == "mapbox"
    assert recommended.academic_rank is None
    assert recommended.traffic_component is None
    assert recommended.evidence_kind == "mapbox_traffic_eta"
    assert final.mapbox_considered is True
    assert set(final.participating_sources) == {"custom_astar", "mapbox"}
    assert final.advice_code == "provider_eta_only"
    assert any(
        "unknown Mapbox road risk was scored conservatively" in warning
        for warning in final.warnings
    )


def test_mapbox_driving_fallback_cannot_win_or_compete() -> None:
    osm = make_route("osm-1")
    ordinary = make_mapbox_route(
        "mapbox-ordinary",
        eta_min=1.0,
        evidence_kind="mapbox_driving_eta",
    )

    final = build(
        [osm],
        [make_ranked("osm-1", 1, 1.0)],
        {"osm-1": make_prediction("osm-1")},
        mapbox_candidates=[ordinary],
        mapbox_verdicts=[make_verdict("mapbox-ordinary")],
    )

    assert final.recommended_route_id == "osm-1"
    assert final.participating_sources == ("custom_astar",)
    assert any("no usable requested-departure" in warning for warning in final.warnings)


def test_strict_toll_avoidance_excludes_mapbox_with_unknown_toll_status() -> None:
    osm = make_route("osm-1")
    mapbox = make_mapbox_route("mapbox-1", eta_min=1.0, toll_available=False)

    final = build(
        [osm],
        [make_ranked("osm-1", 1, 1.0)],
        {"osm-1": make_prediction("osm-1")},
        mapbox_candidates=[mapbox],
        mapbox_verdicts=[make_verdict("mapbox-1")],
        avoid_tolls=True,
    )

    assert final.recommended_route_id == "osm-1"
    assert any(
        "Mapbox toll status is unknown" in warning for warning in final.warnings
    )


def test_cross_source_geometry_duplicate_is_removed() -> None:
    osm = make_route("osm-1")
    duplicate = make_mapbox_route(
        "mapbox-copy",
        eta_min=1.0,
        geometry=osm.geometry,
    )

    final = build(
        [osm],
        [make_ranked("osm-1", 1, 1.0)],
        {"osm-1": make_prediction("osm-1")},
        mapbox_candidates=[duplicate],
        mapbox_verdicts=[make_verdict("mapbox-copy")],
    )

    assert final.recommended_route_id == "osm-1"
    assert any(
        "near-duplicate" in warning for warning in final.warnings
    )


def test_mapbox_is_not_mixed_into_partial_ml_fallback() -> None:
    osm = make_route("osm-1")
    mapbox = make_mapbox_route("mapbox-1", eta_min=1.0)

    final = build(
        [osm],
        [make_ranked("osm-1", 1, 1.0)],
        {"osm-1": None},
        errors={"osm-1": "model unavailable"},
        mapbox_candidates=[mapbox],
        mapbox_verdicts=[make_verdict("mapbox-1")],
    )

    assert final.recommended_route_id == "osm-1"
    assert final.fallback_used is True
    assert final.participating_sources == ("custom_astar",)
    assert any(
        "not mixed into a partial ML decision" in warning
        for warning in final.warnings
    )
