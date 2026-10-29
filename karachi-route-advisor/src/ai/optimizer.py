from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from ..config import PREFERENCE_WEIGHTS
from ..contracts import (
    FinalRecommendation,
    FinalRouteDecision,
    IncidentRiskAssessment,
    RankedRoute,
    RouteAssessment,
    RouteCandidate,
    RouteTrafficPrediction,
    RouteVerdict,
    TrafficEvidenceMatch,
)
from ..eta import haversine_m
from ..route_features import route_fingerprint

POLICY_VERSION = "ai-final-v2"
TRAFFIC_DELAY_FACTOR = 0.60
MAX_DISPLAYED_ROUTES = 3
MAX_EDGE_OVERLAP = 0.70
MAX_GEOMETRY_OVERLAP = 0.70
GEOMETRY_TOLERANCE_M = 35.0
UNKNOWN_RISK_NORM = 0.50
UNKNOWN_RISK_NORM_SAFEST = 1.00

TRAFFIC_SEVERITY = {"low": 0.0, "moderate": 0.5, "high": 1.0}
TRAFFIC_BANDS = frozenset(TRAFFIC_SEVERITY)
INCIDENT_LEVELS = frozenset({"low", "moderate", "high"})

BAND_LABELS = {
    "low": "LOW",
    "moderate": "MODERATE",
    "high": "HIGH",
    "severe": "SEVERE",
}

ADVICE_LABELS = {
    "good_time": "Good time to leave",
    "allow_extra_time": "Allow extra time",
    "consider_earlier": "Consider leaving earlier",
    "not_recommended": "Not recommended",
    "provider_eta_only": "Traffic-aware provider ETA only",
    "traffic_evidence": "Traffic-adjusted ETA applied",
    "elevated_incident_risk": "Route has elevated incident risk",
    "insufficient_evidence": "Insufficient evidence for a reliable recommendation",
}

POLICY_DESCRIPTION = (
    "Hard constraints first. OSM A*/Dijkstra candidates compete in one decision "
    "layer. Matched Mapbox traffic evidence adjusts the OSM analytical ETA once "
    "when the corridor overlap and provider baseline are defensible; otherwise "
    "the future-ML scenario premium or the analytical OSM ETA is used once. "
    "Provider ETAs are not copied onto OSM routes and are not mixed with the ML "
    "premium. Academic costs remain unchanged and are not added again. Missing "
    "evidence is disclosed and never treated as zero risk."
)

_PROVENANCE = (
    "A* and Dijkstra retain their original OSM cost function; expert-system rules "
    "R01-R16 are unchanged.",
    "OSM future traffic: route-specific ML probabilities converted to one "
    "free-flow travel-time premium; synthetic labels are disclosed.",
    "Mapbox traffic: requested-departure provider evidence matched to an OSM "
    "corridor and used once to adjust that OSM ETA; the provider ETA is not "
    "copied onto the OSM route or presented as the future-traffic ML forecast.",
    "Incident/closures: route-specific OSM incident history is unavailable; "
    "Mapbox route-level annotations are labelled provider evidence.",
    "Accident prior: external UK STATS19 calendar model; not Karachi crash "
    "history and not a crash probability for a selected route.",
    "Missing road-risk or incident evidence is unknown, not zero.",
)


def build_final_recommendation(
    *,
    routes: Sequence[RouteCandidate],
    verdicts: Sequence[RouteVerdict],
    academic_ranked: Sequence[RankedRoute] | None = None,
    ranked_routes: Sequence[RankedRoute] | None = None,
    predictions: Mapping[str, RouteTrafficPrediction | None] | Sequence[RouteTrafficPrediction],
    departure: datetime,
    preference: str,
    prediction_errors: Mapping[str, str] | None = None,
    generated_at: datetime | None = None,
    mapbox_candidates: Sequence[RouteCandidate] = (),
    mapbox_verdicts: Sequence[RouteVerdict] = (),
    avoid_tolls: bool = False,
    assessments: Mapping[str, RouteAssessment] | Sequence[RouteAssessment] | None = None,
) -> FinalRecommendation:
    ranked_source = academic_ranked if academic_ranked is not None else ranked_routes
    if ranked_source is None:
        ranked_source = ()
    assessment_by_id = (
        dict(assessments)
        if isinstance(assessments, Mapping)
        else {assessment.route.route_id: assessment for assessment in assessments}
        if assessments is not None
        else {}
    )
    if assessments is not None:
        return _build_assessment_recommendation(
            routes=routes,
            verdicts=verdicts,
            ranked_source=ranked_source,
            assessments=assessment_by_id,
            departure=departure,
            preference=preference,
            generated_at=generated_at,
            mapbox_considered=bool(mapbox_candidates),
            avoid_tolls=avoid_tolls,
        )
    prediction_by_id = (
        dict(predictions)
        if isinstance(predictions, Mapping)
        else {prediction.route_id: prediction for prediction in predictions}
    )
    errors = dict(prediction_errors or {})
    warnings: list[str] = []
    verdict_by_id = {verdict.route_id: verdict for verdict in verdicts}
    ranked_by_id = {ranked.route_id: ranked for ranked in ranked_source}
    mapbox_verdict_by_id = {
        verdict.route_id: verdict for verdict in mapbox_verdicts
    }

    osm: list[tuple[RouteCandidate, RouteVerdict, RankedRoute]] = []
    for route in routes:
        verdict = verdict_by_id.get(route.route_id)
        ranked = ranked_by_id.get(route.route_id)
        if verdict is None:
            warnings.append(
                f"Route {route.route_id} was excluded because it has no expert "
                "verdict."
            )
            continue
        if verdict.vetoed:
            reasons = "; ".join(verdict.veto_reasons) or "hard safety veto"
            warnings.append(
                f"Route {route.route_id} was excluded by a hard veto: {reasons}."
            )
            continue
        if ranked is None:
            warnings.append(
                f"Route {route.route_id} was excluded because it has no academic "
                "ranking row."
            )
            continue
        osm.append((route, verdict, ranked))

    missing_prediction_routes = [
        route.route_id
        for route, _, _ in osm
        if not _usable_prediction(prediction_by_id.get(route.route_id))
    ]
    fallback_used = bool(missing_prediction_routes)
    fallback_basis = ""

    eligible_osm = list(osm)
    eligible_mapbox: list[tuple[RouteCandidate, RouteVerdict]] = []

    if fallback_used:
        fallback_basis = (
            "Existing preference-aware academic rank because at least one "
            "eligible OSM route lacked usable future-traffic evidence."
        )
        warnings.append(
            "Future-traffic evidence was unavailable for at least one eligible "
            "route. The final ranking fell back to the existing academic ranking "
            "for all routes; missing evidence was not treated as zero risk."
        )
        for route_id, reason in sorted(errors.items()):
            warnings.append(f"Traffic prediction for {route_id} failed: {reason}")
        if mapbox_candidates:
            warnings.append(
                "Mapbox candidates were not mixed into a partial ML decision, so "
                "every competing candidate retains comparably complete evidence."
            )
    else:
        for route in mapbox_candidates:
            verdict = mapbox_verdict_by_id.get(route.route_id)
            if verdict is None:
                warnings.append(
                    f"Route {route.route_id} was excluded because it has no final "
                    "Mapbox expert verdict."
                )
                continue
            if verdict.vetoed:
                reasons = "; ".join(verdict.veto_reasons) or "hard safety veto"
                warnings.append(
                    f"Route {route.route_id} was excluded by a hard Mapbox veto: "
                    f"{reasons}."
                )
                continue
            if not _usable_mapbox_eta(route):
                warnings.append(
                    f"Route {route.route_id} was excluded because no usable "
                    "requested-departure traffic-aware Mapbox ETA was available."
                )
                continue
            if avoid_tolls and not route.toll_available:
                warnings.append(
                    f"Route {route.route_id} was excluded under strict toll "
                    "avoidance because Mapbox toll status is unknown, not assumed "
                    "toll-free."
                )
                continue
            eligible_mapbox.append((route, verdict))

    eligible, duplicate_warnings = _distinct_candidates(
        eligible_osm,
        eligible_mapbox,
    )
    warnings.extend(duplicate_warnings)
    if not eligible:
        warnings.append(
            "No non-vetoed, distinct route with comparable evidence is available, "
            "so no route is recommended."
        )
        return _empty_recommendation(
            departure,
            warnings,
            bool(mapbox_candidates),
            generated_at,
        )

    academic_norms = _normalize(
        [float(ranked.final_cost) for _, _, ranked in osm]
    )
    academic_norm_by_id = {
        route.route_id: academic_norms[index]
        for index, (route, _, _) in enumerate(osm)
    }
    academic_rank_by_id = {
        route.route_id: ranked.rank for route, _, ranked in osm
    }
    academic_score_by_id = {
        route.route_id: ranked.score for route, _, ranked in osm
    }

    severity_by_id: dict[str, float | None] = {}
    for record in eligible:
        route = record["route"]
        severity_by_id[route.route_id] = (
            None
            if route.source == "mapbox"
            else _expected_severity(prediction_by_id.get(route.route_id))
        )

    decision_times = [
        _decision_time(
            record["route"],
            prediction_by_id.get(record["route"].route_id),
            severity_by_id[record["route"].route_id],
        )
        for record in eligible
    ]
    time_norms = _normalize([value for value, _ in decision_times])
    distance_norms = _normalize(
        [record["route"].distance_m / 1000.0 for record in eligible]
    )

    risk_values = [
        record["route"].risk
        for record in eligible
        if record["route"].risk_available
    ]
    risk_norms = _normalize(risk_values)
    unknown_risk_norm = (
        UNKNOWN_RISK_NORM_SAFEST if preference == "Safest" else UNKNOWN_RISK_NORM
    )
    risk_norm_by_id: dict[str, float] = {}
    available_index = 0
    for record in eligible:
        route = record["route"]
        if route.risk_available:
            risk_norm_by_id[route.route_id] = risk_norms[available_index]
            available_index += 1
        else:
            risk_norm_by_id[route.route_id] = unknown_risk_norm

    component_weights = _component_weights(preference)
    decisions: list[FinalRouteDecision] = []

    for index, record in enumerate(eligible):
        route = record["route"]
        verdict = record["verdict"]
        prediction = prediction_by_id.get(route.route_id)
        severity = severity_by_id[route.route_id]
        decision_time, time_basis = decision_times[index]
        academic_norm = academic_norm_by_id.get(route.route_id)
        academic_rank = academic_rank_by_id.get(route.route_id)
        academic_score = academic_score_by_id.get(route.route_id)

        if fallback_used:
            final_cost = academic_norm if academic_norm is not None else 1.0
        else:
            final_cost = (
                component_weights["time"] * time_norms[index]
                + component_weights["distance"] * distance_norms[index]
                + component_weights["risk"] * risk_norm_by_id[route.route_id]
            )
            final_cost *= _verdict_multiplier(verdict)
            if prediction is not None:
                final_cost += _confidence_penalty(prediction)

        final_score = max(
            0,
            min(100, round(100.0 * (1.0 - min(final_cost, 1.25) / 1.25))),
        )
        evidence_kind = _evidence_kind(route, fallback_used)
        evidence_label = _evidence_label(route, fallback_used)
        traffic_component = (
            None
            if fallback_used or route.source == "mapbox"
            else severity
        )
        decisions.append(
            FinalRouteDecision(
                route_id=route.route_id,
                rank=0,
                recommended=False,
                route=route,
                traffic_prediction=prediction,
                incident_risk=build_incident_risk(route, prediction),
                academic_rank=academic_rank,
                academic_score=academic_score,
                final_score=final_score,
                final_cost=final_cost,
                academic_component=academic_norm,
                traffic_component=traffic_component,
                traffic_severity=severity,
                ranking_reason=_ranking_reason(
                    route=route,
                    verdict=verdict,
                    prediction=prediction,
                    severity=severity,
                    decision_time=decision_time,
                    component_weights=component_weights,
                    academic_rank=academic_rank,
                    fallback_used=fallback_used,
                ),
                details=_decision_details(
                    route=route,
                    prediction=prediction,
                    verdict=verdict,
                    academic_rank=academic_rank,
                    severity=severity,
                    decision_time=decision_time,
                    time_basis=time_basis,
                    final_cost=final_cost,
                    fallback_used=fallback_used,
                ),
                evidence_kind=evidence_kind,
                evidence_label=evidence_label,
                decision_time_min=decision_time,
                decision_time_basis=time_basis,
                risk_available=route.risk_available,
                traffic_evidence_kind=evidence_kind,
                traffic_evidence_label=evidence_label,
            )
        )

    decisions.sort(
        key=lambda decision: (
            decision.final_cost,
            _SOURCE_PRIORITY.get(decision.route.source, 9),
            decision.decision_time_min,
            decision.route_id,
        )
    )
    ranked_decisions = []
    for rank, decision in enumerate(decisions[:MAX_DISPLAYED_ROUTES], start=1):
        ranked_decisions.append(
            FinalRouteDecision(
                route_id=decision.route_id,
                rank=rank,
                recommended=rank == 1,
                route=decision.route,
                traffic_prediction=decision.traffic_prediction,
                incident_risk=decision.incident_risk,
                academic_rank=decision.academic_rank,
                academic_score=decision.academic_score,
                final_score=decision.final_score,
                final_cost=decision.final_cost,
                academic_component=decision.academic_component,
                traffic_component=decision.traffic_component,
                traffic_severity=decision.traffic_severity,
                ranking_reason=decision.ranking_reason,
                details=decision.details,
                evidence_kind=decision.evidence_kind,
                evidence_label=decision.evidence_label,
                decision_time_min=decision.decision_time_min,
                decision_time_basis=decision.decision_time_basis,
                risk_available=decision.risk_available,
                traffic_evidence_kind=decision.traffic_evidence_kind,
                traffic_evidence_label=decision.traffic_evidence_label,
            )
        )

    recommended = ranked_decisions[0]
    if recommended.route.source == "mapbox":
        advice_code, advice_label, advice_message = _mapbox_advice(recommended)
    else:
        advice_code, advice_label, advice_message = _advice(
            recommended.route,
            recommended.traffic_prediction,
            recommended.incident_risk,
        )

    if not any(
        decision.incident_risk.route_specific for decision in ranked_decisions
    ):
        warnings.append(
            "Route-specific incident history is unavailable for custom OSM "
            "routes. The separate external accident prior is shown and was not "
            "described as Karachi crash probability."
        )
    if any(
        decision.route.source == "mapbox" and not decision.route.risk_available
        for decision in ranked_decisions
    ):
        warnings.append(
            "Mapbox does not expose OSM road-risk features; unknown Mapbox road "
            "risk was scored conservatively, not as zero."
        )
    if preference not in PREFERENCE_WEIGHTS:
        warnings.append(
            f"Unknown preference {preference!r}; the final policy used Fastest "
            "component weights."
        )

    return FinalRecommendation(
        departure=departure,
        routes=tuple(ranked_decisions),
        recommended_route_id=recommended.route_id,
        advice_code=advice_code,
        advice_label=advice_label,
        advice_message=advice_message,
        policy_version=POLICY_VERSION,
        policy_description=POLICY_DESCRIPTION,
        weights=(
            (("academic_rank", 1.0),)
            if fallback_used
            else tuple(component_weights.items())
        ),
        fallback_used=fallback_used,
        fallback_basis=fallback_basis,
        warnings=tuple(dict.fromkeys(warnings)),
        provenance=_PROVENANCE,
        generated_at=generated_at or _now_utc(),
        participating_sources=tuple(
            dict.fromkeys(decision.route.source for decision in ranked_decisions)
        ),
        mapbox_considered=bool(mapbox_candidates),
    )


def _build_assessment_recommendation(
    *,
    routes: Sequence[RouteCandidate],
    verdicts: Sequence[RouteVerdict],
    ranked_source: Sequence[RankedRoute],
    assessments: Mapping[str, RouteAssessment],
    departure: datetime,
    preference: str,
    generated_at: datetime | None,
    mapbox_considered: bool,
    avoid_tolls: bool,
) -> FinalRecommendation:
    warnings: list[str] = []
    verdict_by_id = {verdict.route_id: verdict for verdict in verdicts}
    ranked_by_id = {ranked.route_id: ranked for ranked in ranked_source}
    eligible: list[dict] = []
    for route in routes:
        verdict = verdict_by_id.get(route.route_id)
        ranked = ranked_by_id.get(route.route_id)
        if verdict is None:
            warnings.append(
                f"Route {route.route_id} was excluded because it has no expert "
                "verdict."
            )
            continue
        if verdict.vetoed:
            reasons = "; ".join(verdict.veto_reasons) or "hard safety veto"
            warnings.append(
                f"Route {route.route_id} was excluded by a hard veto: {reasons}."
            )
            continue
        if ranked is None:
            warnings.append(
                f"Route {route.route_id} was excluded because it has no academic "
                "ranking row."
            )
            continue
        assessment = assessments.get(route.route_id)
        if assessment is None:
            assessment = _fallback_assessment(route, verdict, departure)
            warnings.append(
                f"Route {route.route_id} had no precomputed assessment; the "
                "analytical OSM ETA was used without inventing traffic evidence."
            )
        eligible.append(
            {
                "route": route,
                "verdict": verdict,
                "academic": ranked,
                "assessment": assessment,
            }
        )
    if avoid_tolls:
        toll_unknown = [
            record for record in eligible if not record["route"].toll_available
        ]
        if toll_unknown:
            warnings.append(
                "Strict toll avoidance was requested, but OSM toll status is "
                "unknown for at least one route; unknown toll status is not "
                "treated as toll-free."
            )
    eligible, duplicate_warnings = _distinct_assessment_candidates(eligible)
    warnings.extend(duplicate_warnings)
    if not eligible:
        warnings.append(
            "No non-vetoed, distinct OSM route with an assessment is available, "
            "so no route is recommended."
        )
        return _empty_recommendation(
            departure,
            warnings,
            mapbox_considered,
            generated_at,
        )

    time_norms = _normalize(
        [float(record["assessment"].decision_eta_min) for record in eligible]
    )
    distance_norms = _normalize(
        [record["route"].distance_m / 1000.0 for record in eligible]
    )
    risk_values = [
        record["route"].risk
        for record in eligible
        if record["route"].risk_available
    ]
    risk_norms = _normalize(risk_values)
    unknown_risk_norm = (
        UNKNOWN_RISK_NORM_SAFEST if preference == "Safest" else UNKNOWN_RISK_NORM
    )
    risk_norm_by_id: dict[str, float] = {}
    available_index = 0
    for record in eligible:
        route = record["route"]
        if route.risk_available:
            risk_norm_by_id[route.route_id] = risk_norms[available_index]
            available_index += 1
        else:
            risk_norm_by_id[route.route_id] = unknown_risk_norm

    component_weights = _component_weights(preference)
    decisions: list[FinalRouteDecision] = []
    for index, record in enumerate(eligible):
        route = record["route"]
        verdict = record["verdict"]
        academic = record["academic"]
        assessment = record["assessment"]
        prediction = assessment.traffic_prediction
        final_cost = (
            component_weights["time"] * time_norms[index]
            + component_weights["distance"] * distance_norms[index]
            + component_weights["risk"] * risk_norm_by_id[route.route_id]
        )
        final_cost *= _verdict_multiplier(verdict)
        if (
            assessment.eta_kind == "analytical"
            and assessment.ml_eta_min is not None
            and prediction is not None
        ):
            final_cost += _confidence_penalty(prediction)
        final_score = max(
            0,
            min(100, round(100.0 * (1.0 - min(final_cost, 1.25) / 1.25))),
        )
        decisions.append(
            FinalRouteDecision(
                route_id=route.route_id,
                rank=0,
                recommended=False,
                route=route,
                traffic_prediction=prediction,
                incident_risk=assessment.incident_risk,
                academic_rank=academic.rank,
                academic_score=academic.score,
                final_score=final_score,
                final_cost=final_cost,
                academic_component=None,
                traffic_component=(
                    None
                    if assessment.eta_kind == "traffic_adjusted"
                    else _expected_severity(prediction)
                ),
                traffic_severity=_expected_severity(prediction),
                ranking_reason=_assessment_ranking_reason(
                    assessment,
                    academic.rank,
                ),
                details=tuple(
                    dict.fromkeys(
                        (
                            f"AI final policy: {POLICY_VERSION}",
                            *assessment.details,
                            f"AI final cost: {final_cost:.3f}",
                            f"Expert evidence: penalty={verdict.penalty:.2f} "
                            f"bonus={verdict.bonus:.2f}",
                            f"Academic comparison rank: {academic.rank} "
                            "(not scored twice)",
                        )
                    )
                ),
                evidence_kind=assessment.eta_kind,
                evidence_label=assessment.eta_label,
                decision_time_min=assessment.decision_eta_min,
                decision_time_basis=assessment.decision_eta_basis,
                risk_available=route.risk_available,
                traffic_evidence_kind=assessment.eta_kind,
                traffic_evidence_label=assessment.eta_label,
                assessment=assessment,
            )
        )

    decisions.sort(
        key=lambda decision: (
            decision.final_cost,
            _SOURCE_PRIORITY.get(decision.route.source, 9),
            decision.decision_time_min,
            decision.route_id,
        )
    )
    ranked_decisions = [
        _with_rank(decision, rank)
        for rank, decision in enumerate(
            decisions[:MAX_DISPLAYED_ROUTES],
            start=1,
        )
    ]
    recommended = ranked_decisions[0]
    advice_code, advice_label, advice_message = _assessment_advice(recommended)
    if not any(
        decision.incident_risk.route_specific for decision in ranked_decisions
    ):
        warnings.append(
            "No matched provider route reported an incident or closure, and no "
            "route-specific Karachi incident history is available. Incident risk "
            "is shown as unavailable, not zero."
        )
    if not any(
        decision.assessment is not None
        and decision.assessment.traffic_evidence is not None
        and decision.assessment.traffic_evidence.applied
        for decision in ranked_decisions
    ):
        warnings.append(
            "No OSM route had defensible matched Mapbox traffic evidence for this "
            "departure. ETAs use analytical or future-ML scenario estimates, not a "
            "traffic-aware provider ETA."
        )
    if mapbox_considered:
        warnings.append(
            "Mapbox routes were considered only as traffic evidence for OSM "
            "candidates; they were not separate competing route options."
        )
    if preference not in PREFERENCE_WEIGHTS:
        warnings.append(
            f"Unknown preference {preference!r}; the final policy used Fastest "
            "component weights."
        )
    return FinalRecommendation(
        departure=departure,
        routes=tuple(ranked_decisions),
        recommended_route_id=recommended.route_id,
        advice_code=advice_code,
        advice_label=advice_label,
        advice_message=advice_message,
        policy_version=POLICY_VERSION,
        policy_description=POLICY_DESCRIPTION,
        weights=tuple(component_weights.items()),
        fallback_used=False,
        fallback_basis="",
        warnings=tuple(dict.fromkeys(warnings)),
        provenance=_PROVENANCE,
        generated_at=generated_at or _now_utc(),
        participating_sources=tuple(
            dict.fromkeys(decision.route.source for decision in ranked_decisions)
        ),
        mapbox_considered=mapbox_considered,
    )


def _with_rank(decision: FinalRouteDecision, rank: int) -> FinalRouteDecision:
    return FinalRouteDecision(
        route_id=decision.route_id,
        rank=rank,
        recommended=rank == 1,
        route=decision.route,
        traffic_prediction=decision.traffic_prediction,
        incident_risk=decision.incident_risk,
        academic_rank=decision.academic_rank,
        academic_score=decision.academic_score,
        final_score=decision.final_score,
        final_cost=decision.final_cost,
        academic_component=decision.academic_component,
        traffic_component=decision.traffic_component,
        traffic_severity=decision.traffic_severity,
        ranking_reason=decision.ranking_reason,
        details=decision.details,
        evidence_kind=decision.evidence_kind,
        evidence_label=decision.evidence_label,
        decision_time_min=decision.decision_time_min,
        decision_time_basis=decision.decision_time_basis,
        risk_available=decision.risk_available,
        traffic_evidence_kind=decision.traffic_evidence_kind,
        traffic_evidence_label=decision.traffic_evidence_label,
        assessment=decision.assessment,
    )


def _fallback_assessment(
    route: RouteCandidate,
    verdict: RouteVerdict,
    departure: datetime,
) -> RouteAssessment:
    incident = build_incident_risk(route, None)
    return RouteAssessment(
        route=route,
        requested_departure=departure,
        analytical_eta_min=route.eta_min,
        free_flow_eta_min=route.free_flow_eta_min,
        ml_eta_min=None,
        traffic_adjusted_eta_min=None,
        decision_eta_min=route.eta_min,
        decision_eta_basis=(
            "OSM analytical ETA; no precomputed traffic assessment was supplied."
        ),
        eta_kind="analytical",
        eta_label="Analytical OSM ETA",
        evidence_source="OSM analytical ETA",
        evidence_timestamp=None,
        evidence_coverage=None,
        traffic_evidence=None,
        traffic_prediction=None,
        future_traffic_band=None,
        future_traffic_probability=None,
        confidence="Unavailable",
        confidence_note=(
            "No precomputed assessment was available, so no traffic-aware claim "
            "is made."
        ),
        incident_risk=incident,
        expert_verdict=verdict,
        limitations=(
            "No matched traffic evidence was supplied for this route.",
        ),
        details=(f"Analytical OSM ETA: {route.eta_min:.1f} min",),
    )


def _distinct_assessment_candidates(
    records: Sequence[dict],
) -> tuple[list[dict], list[str]]:
    ordered = sorted(records, key=lambda item: item["academic"].rank)
    accepted: list[dict] = []
    fingerprints: set[str] = set()
    warnings: list[str] = []
    for record in ordered:
        route = record["route"]
        fingerprint = route_fingerprint(route)
        if fingerprint in fingerprints:
            warnings.append(
                f"Route {route.route_id} was removed as a duplicate fingerprint."
            )
            continue
        duplicate = False
        for existing in accepted:
            existing_route = existing["route"]
            if route.edges and existing_route.edges:
                if _edge_overlap(route.edges, existing_route.edges) > MAX_EDGE_OVERLAP:
                    duplicate = True
                    break
            elif route.geometry and existing_route.geometry:
                if (
                    _geometry_overlap(route.geometry, existing_route.geometry)
                    > MAX_GEOMETRY_OVERLAP
                ):
                    duplicate = True
                    break
        if duplicate:
            warnings.append(
                f"Route {route.route_id} was removed as a near-duplicate "
                f"(>{MAX_EDGE_OVERLAP:.0%} edge or geometry overlap)."
            )
            continue
        fingerprints.add(fingerprint)
        accepted.append(record)
    return accepted, warnings


def _assessment_ranking_reason(
    assessment: RouteAssessment,
    academic_rank: int,
) -> str:
    if (
        assessment.traffic_evidence is not None
        and assessment.traffic_evidence.applied
    ):
        return (
            "Ranked by matched provider traffic evidence: the OSM analytical ETA "
            f"was adjusted to {assessment.decision_eta_min:.1f} min using "
            f"{assessment.traffic_evidence.method}, plus distance, road risk, and "
            "expert constraints. The provider's absolute ETA was not copied. "
            f"Academic rank {academic_rank} is shown for comparison but not added "
            "again."
        )
    if assessment.ml_eta_min is not None:
        return (
            "Ranked by the OSM future-ML scenario ETA of "
            f"{assessment.decision_eta_min:.1f} min, distance, road risk, and "
            f"expert constraints. Academic rank {academic_rank} is shown for "
            "comparison but not added again."
        )
    return (
        "Ranked by the analytical OSM ETA of "
        f"{assessment.decision_eta_min:.1f} min, distance, road risk, and expert "
        f"constraints. No matched traffic evidence was available. Academic rank "
        f"{academic_rank} is shown for comparison but not added again."
    )


def _assessment_advice(
    decision: FinalRouteDecision,
) -> tuple[str, str, str]:
    assessment = decision.assessment
    incident = decision.incident_risk
    if incident.route_specific and incident.level in {"HIGH", "MODERATE"}:
        return (
            "elevated_incident_risk",
            ADVICE_LABELS["elevated_incident_risk"],
            "The recommended route has matched provider incident or closure "
            "evidence. Check provider details before departure.",
        )
    if assessment is None:
        return (
            "insufficient_evidence",
            ADVICE_LABELS["insufficient_evidence"],
            "No assessment is available for the recommended route.",
        )
    evidence = assessment.traffic_evidence
    if evidence is not None and evidence.applied:
        ratio = evidence.provider_delay_ratio
        if ratio is not None and ratio >= 1.15:
            delay_pct = (ratio - 1.0) * 100.0
            return (
                "allow_extra_time",
                ADVICE_LABELS["allow_extra_time"],
                f"Matched provider evidence indicates about {delay_pct:.0f}% more "
                "time than the provider's typical duration for this departure. "
                "Allow extra time; the provider ETA is an estimate, not an "
                "observed Karachi measurement.",
            )
        return (
            "traffic_evidence",
            ADVICE_LABELS["traffic_evidence"],
            "Matched provider traffic evidence was applied once to the OSM "
            "analytical ETA. The provider's absolute ETA was not copied.",
        )
    band = assessment.future_traffic_band
    if band == "high":
        return (
            "consider_earlier",
            ADVICE_LABELS["consider_earlier"],
            "The future-ML scenario indicates high traffic for this departure. "
            "Consider leaving earlier or later; the labels are synthetic scenarios, "
            "not observed Karachi traffic.",
        )
    if band == "moderate":
        return (
            "allow_extra_time",
            ADVICE_LABELS["allow_extra_time"],
            "The future-ML scenario indicates moderate traffic for this departure. "
            "Allow extra time; the labels are synthetic scenarios.",
        )
    if band == "low":
        return (
            "good_time",
            ADVICE_LABELS["good_time"],
            "The future-ML scenario indicates low traffic for this departure, with "
            "the usual caveats for synthetic traffic labels.",
        )
    return (
        "insufficient_evidence",
        ADVICE_LABELS["insufficient_evidence"],
        "No matched traffic evidence or usable future-ML prediction is available. "
        "The ETA is analytical, so no reliable departure-time advice can be given.",
    )


_SOURCE_PRIORITY = {"custom_astar": 0, "mapbox": 1}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _distinct_candidates(
    osm: Sequence[tuple[RouteCandidate, RouteVerdict, RankedRoute]],
    mapbox: Sequence[tuple[RouteCandidate, RouteVerdict]],
) -> tuple[list[dict], list[str]]:
    ordered = [
        {"route": route, "verdict": verdict, "academic": ranked}
        for route, verdict, ranked in sorted(osm, key=lambda item: item[2].rank)
    ]
    ordered.extend(
        {"route": route, "verdict": verdict, "academic": None}
        for route, verdict in sorted(
            mapbox,
            key=lambda item: (item[0].eta_min, item[0].route_id),
        )
    )

    accepted: list[dict] = []
    fingerprints: set[str] = set()
    warnings: list[str] = []
    for record in ordered:
        route = record["route"]
        fingerprint = route_fingerprint(route)
        if fingerprint in fingerprints:
            warnings.append(
                f"Route {route.route_id} was removed as a duplicate fingerprint."
            )
            continue
        duplicate = False
        for existing in accepted:
            existing_route = existing["route"]
            if route.edges and existing_route.edges:
                if _edge_overlap(route.edges, existing_route.edges) > MAX_EDGE_OVERLAP:
                    duplicate = True
                    break
            elif route.geometry and existing_route.geometry:
                if (
                    _geometry_overlap(route.geometry, existing_route.geometry)
                    > MAX_GEOMETRY_OVERLAP
                ):
                    duplicate = True
                    break
        if duplicate:
            warnings.append(
                f"Route {route.route_id} was removed as a near-duplicate "
                f"(>{MAX_EDGE_OVERLAP:.0%} edge or geometry overlap)."
            )
            continue
        fingerprints.add(fingerprint)
        accepted.append(record)
    return accepted, warnings


def _edge_overlap(
    first: tuple[tuple[int, int], ...],
    second: tuple[tuple[int, int], ...],
) -> float:
    if not first or not second:
        return 0.0
    shared = len(set(first) & set(second))
    return shared / max(1, min(len(first), len(second)))


def _geometry_overlap(
    first: tuple[tuple[float, float], ...],
    second: tuple[tuple[float, float], ...],
) -> float:
    if not first or not second:
        return 0.0
    samples = _sample_points(first)
    targets = _sample_points(second)
    matches = 0
    for lat, lon in samples:
        if any(
            haversine_m(lat, lon, target_lat, target_lon) <= GEOMETRY_TOLERANCE_M
            for target_lat, target_lon in targets
        ):
            matches += 1
    return matches / len(samples)


def _sample_points(
    points: Sequence[tuple[float, float]],
    limit: int = 90,
) -> list[tuple[float, float]]:
    if len(points) <= limit:
        return list(points)
    step = max(1, math.ceil(len(points) / limit))
    sampled = list(points[::step])
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def _usable_prediction(prediction: RouteTrafficPrediction | None) -> bool:
    return (
        prediction is not None
        and prediction.status in {"ok", "partial"}
        and prediction.traffic_band in TRAFFIC_BANDS
        and bool(prediction.traffic_probabilities)
    )


def _usable_mapbox_eta(route: RouteCandidate) -> bool:
    return (
        route.source == "mapbox"
        and route.eta_min > 0
        and route.eta_evidence_kind == "mapbox_traffic_eta"
    )


def _expected_severity(
    prediction: RouteTrafficPrediction | None,
) -> float | None:
    if not _usable_prediction(prediction):
        return None
    assert prediction is not None
    total = 0.0
    weight = 0.0
    for band, probability in prediction.traffic_probabilities:
        if band in TRAFFIC_SEVERITY:
            value = float(probability)
            total += TRAFFIC_SEVERITY[band] * value
            weight += value
    if weight <= 0:
        return None
    return total / weight


def _decision_time(
    route: RouteCandidate,
    prediction: RouteTrafficPrediction | None,
    severity: float | None,
) -> tuple[float, str]:
    if route.source == "mapbox":
        return route.eta_min, "Mapbox requested-departure traffic-aware ETA"
    if severity is None or prediction is None:
        return route.eta_min, "OSM analytical ETA without future ML adjustment"
    free_flow = route.free_flow_eta_min or route.eta_min
    adjusted = free_flow * (1.0 + TRAFFIC_DELAY_FACTOR * severity)
    return adjusted, "OSM free-flow ETA plus one ML future-traffic premium"


def _component_weights(preference: str) -> dict[str, float]:
    raw = dict(PREFERENCE_WEIGHTS.get(preference, PREFERENCE_WEIGHTS["Fastest"]))
    raw["time"] = raw.get("time", 0.0) + raw.pop("congestion", 0.0)
    total = raw["time"] + raw["distance"] + raw["risk"]
    return {
        "time": raw["time"] / total,
        "distance": raw["distance"] / total,
        "risk": raw["risk"] / total,
    }


def _normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high - low < 1e-12:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _verdict_multiplier(verdict: RouteVerdict) -> float:
    multiplier = 1.0 + 0.15 * verdict.penalty - 0.10 * verdict.bonus
    return max(0.75, multiplier)


def _confidence_penalty(prediction: RouteTrafficPrediction) -> float:
    return {
        "low": 0.20,
        "medium": 0.10,
        "high": 0.0,
    }.get(prediction.confidence.strip().lower(), 0.10)


def _evidence_kind(route: RouteCandidate, fallback_used: bool) -> str:
    if fallback_used:
        return "academic_fallback"
    if route.source == "mapbox":
        return "mapbox_traffic_eta"
    return "future_ml"


def _evidence_label(route: RouteCandidate, fallback_used: bool) -> str:
    if fallback_used:
        return "Academic fallback without complete future-ML evidence"
    if route.source == "mapbox":
        return "Mapbox traffic-aware ETA for the requested departure"
    return "OSM future-traffic ML prediction"


def _ranking_reason(
    route: RouteCandidate,
    verdict: RouteVerdict,
    prediction: RouteTrafficPrediction | None,
    severity: float | None,
    decision_time: float,
    component_weights: dict[str, float],
    academic_rank: int | None,
    fallback_used: bool,
) -> str:
    if fallback_used:
        return (
            "Ranked by the academic fallback; future-traffic evidence was "
            "incomplete and was not fabricated."
        )
    if route.source == "mapbox":
        return (
            "Ranked by the AI decision layer using a requested-departure "
            f"traffic-aware Mapbox ETA of {decision_time:.1f} min, distance, "
            "conservative unknown road risk, and hard expert constraints. The "
            "Mapbox ETA is provider evidence, not the OSM future-traffic ML "
            "forecast."
        )
    if prediction is None or severity is None:
        return "Future-traffic evidence was insufficient for a complete AI ranking."
    band = (
        "HIGH"
        if prediction.traffic_band == "high"
        else "MODERATE"
        if prediction.traffic_band == "moderate"
        else "LOW"
    )
    probability = (
        f"{prediction.traffic_probability:.0%}"
        if prediction.traffic_probability is not None
        else "unknown probability"
    )
    return (
        f"Ranked by the AI decision layer: OSM future ML traffic {band} "
        f"({probability}), one free-flow-plus-traffic time of {decision_time:.1f} "
        "min, distance, road risk, and expert constraints. Academic rank "
        f"{academic_rank} is shown for comparison but not added again."
    )


def _decision_details(
    route: RouteCandidate,
    prediction: RouteTrafficPrediction | None,
    verdict: RouteVerdict,
    academic_rank: int | None,
    severity: float | None,
    decision_time: float,
    time_basis: str,
    final_cost: float,
    fallback_used: bool,
) -> tuple[str, ...]:
    details = [
        f"AI final policy: {POLICY_VERSION}",
        f"Route evidence source: {route.source}",
        f"Decision travel time: {decision_time:.1f} min ({time_basis})",
        f"AI final cost: {final_cost:.3f}",
        f"Expert evidence: penalty={verdict.penalty:.2f} bonus={verdict.bonus:.2f}",
        f"Road-risk evidence available: {route.risk_available}",
    ]
    if fallback_used:
        details.append("Academic fallback active; future traffic was not fabricated.")
    if route.source == "mapbox":
        details.append(
            "Mapbox traffic-aware ETA is used once for decision time; congestion "
            "annotations are not scored a second time."
        )
        if not route.risk_available:
            details.append(
                "Mapbox road risk is unknown and handled conservatively, not as zero."
            )
    else:
        details.append(
            "OSM future ML severity is applied once as a free-flow time premium; "
            "academic ETA/cost is not added again."
        )
        if prediction is not None:
            details.append(
                f"Future ML: low={prediction.traffic_probability if prediction.traffic_band == 'low' else 0.0:.2f} "
                f"band={prediction.traffic_band} status={prediction.status}"
            )
        if severity is not None:
            details.append(
                f"Future traffic delay premium: {severity:.3f} from "
                "probability-weighted severity"
            )
    if academic_rank is not None:
        details.append(
            f"Academic comparison rank: {academic_rank} (not scored twice)"
        )
    return tuple(details)


def build_incident_risk(
    route: RouteCandidate,
    prediction: RouteTrafficPrediction | None,
    traffic_evidence: TrafficEvidenceMatch | None = None,
) -> IncidentRiskAssessment:
    if traffic_evidence is not None and (
        traffic_evidence.incident_count > 0 or traffic_evidence.closure_count > 0
    ):
        level = "HIGH" if traffic_evidence.closure_count > 0 else "MODERATE"
        return IncidentRiskAssessment(
            level=level,
            probability=None,
            evidence_type="ROUTE_SPECIFIC",
            route_specific=True,
            source="Matched Mapbox route incident/closure annotations",
            explanation=(
                "Matched provider evidence reports "
                f"{traffic_evidence.incident_count} incident(s) and "
                f"{traffic_evidence.closure_count} closure(s) on the matched "
                "provider route. No calibrated Karachi crash probability is "
                "available."
            ),
            limitations=(
                "Incidents and closures are only those exposed along the matched "
                "provider route, not a complete city incident feed.",
                "The match is geometric and may not cover the entire OSM route.",
            ),
        )
    route_specific_level: str | None = None
    if route.source not in {"custom_astar", "dijkstra"} and route.traffic_available:
        if route.closure_count > 0:
            route_specific_level = "high"
        elif route.incident_count > 0:
            route_specific_level = "moderate"
        else:
            route_specific_level = "low"

    if route_specific_level is not None:
        return IncidentRiskAssessment(
            level=route_specific_level.upper(),
            probability=None,
            evidence_type="ROUTE_SPECIFIC",
            route_specific=True,
            source="Route-level incident/closure annotations",
            explanation=(
                "Route-level annotations are available for this route. "
                f"Reported incidents: {route.incident_count}; closures: "
                f"{route.closure_count}. No calibrated crash probability is "
                "available."
            ),
            limitations=(
                "Incidents and closures are only those exposed along the returned "
                "route, not a complete city incident feed.",
            ),
        )

    if (
        prediction is not None
        and prediction.accident_risk in INCIDENT_LEVELS
        and prediction.accident_probability is not None
    ):
        if prediction.accident_evidence == "REAL":
            evidence_type = "EXTERNAL_PRIOR"
            source = prediction.accident_source
            prior_label = "real external UK STATS19 calendar prior"
        elif prediction.accident_evidence == "SYNTHETIC":
            evidence_type = "SYNTHETIC_PRIOR"
            source = prediction.accident_source
            prior_label = "synthetic accident prior"
        else:
            evidence_type = "UNAVAILABLE"
            source = prediction.accident_source or "Unavailable"
            prior_label = "unavailable prior"
        return IncidentRiskAssessment(
            level="UNAVAILABLE",
            probability=None,
            evidence_type=evidence_type,
            route_specific=False,
            source=source,
            explanation=(
                "No route-specific Karachi incident/accident feed is available "
                "for this custom OSM route, so incident risk is UNAVAILABLE "
                "rather than zero. A separate calendar accident prior is "
                f"{prediction.accident_risk.upper()} "
                f"({prediction.accident_probability:.0%} classifier confidence) "
                f"from a {prior_label}. It is not Karachi crash history and not "
                "a crash probability for this route."
            ),
            external_prior_level=prediction.accident_risk.upper(),
            external_prior_probability=prediction.accident_probability,
            limitations=(
                "The external accident prior depends only on calendar time, not "
                "route geometry, and is not validated for Karachi.",
                "Route-specific incident evidence is not available for custom "
                "OSM routes.",
            ),
        )

    reason = (
        "No route-specific incident feed and no usable accident prior are "
        "available for this request. Missing evidence is not treated as zero risk."
    )
    if prediction is not None:
        reason = prediction.accident_source or reason
    return IncidentRiskAssessment(
        level="UNAVAILABLE",
        probability=None,
        evidence_type="UNAVAILABLE",
        route_specific=False,
        source="Unavailable",
        explanation=reason,
        limitations=(
            "No route-specific Karachi incident history is available; no "
            "probability is invented.",
        ),
    )


def _advice(
    route: RouteCandidate,
    prediction: RouteTrafficPrediction | None,
    incident: IncidentRiskAssessment,
) -> tuple[str, str, str]:
    route_id = route.route_id
    if not _usable_prediction(prediction) or prediction is None:
        return (
            "insufficient_evidence",
            ADVICE_LABELS["insufficient_evidence"],
            f"No usable future-traffic prediction is available for {route_id} at "
            "this departure. The route is still the best valid fallback, but no "
            "reliable departure-time advice can be given.",
        )

    if incident.route_specific and incident.level == "HIGH":
        return (
            "elevated_incident_risk",
            ADVICE_LABELS["elevated_incident_risk"],
            f"{route_id} has route-level incident/closures reported. Consider a "
            "different route or departure time.",
        )

    traffic = prediction.traffic_band
    accident = prediction.accident_risk
    probability = (
        f"{prediction.traffic_probability:.0%}"
        if prediction.traffic_probability is not None
        else "unknown probability"
    )

    if traffic == "high" and accident == "high":
        return (
            "not_recommended",
            ADVICE_LABELS["not_recommended"],
            f"{route_id} has high predicted traffic ({probability}) and a high "
            "external accident prior. Consider a different departure window. "
            "The accident prior is not a route-specific Karachi crash probability.",
        )
    if traffic == "high" or accident == "high":
        return (
            "consider_earlier",
            ADVICE_LABELS["consider_earlier"],
            f"{route_id} has "
            f"{'high predicted traffic' if traffic == 'high' else 'a high external accident prior'} "
            f"for this departure ({probability} traffic confidence). "
            "Consider leaving earlier or later.",
        )
    if traffic == "moderate":
        return (
            "allow_extra_time",
            ADVICE_LABELS["allow_extra_time"],
            f"{route_id} has moderate predicted traffic ({probability}). "
            "Allow extra time; the traffic labels are synthetic.",
        )
    if accident not in INCIDENT_LEVELS:
        return (
            "allow_extra_time",
            ADVICE_LABELS["allow_extra_time"],
            f"{route_id} has low predicted traffic ({probability}), but the "
            "accident prior is unavailable. Allow extra time and treat the missing "
            "evidence as unknown, not safe.",
        )
    if prediction.confidence == "Low":
        return (
            "allow_extra_time",
            ADVICE_LABELS["allow_extra_time"],
            f"{route_id} has low predicted traffic, but the classifier confidence "
            "is Low. Allow extra time for uncertainty.",
        )
    if route.risk >= 0.60:
        return (
            "allow_extra_time",
            ADVICE_LABELS["allow_extra_time"],
            f"{route_id} has low predicted future traffic, but the OSM road-risk "
            f"score is elevated ({route.risk:.2f}). Allow extra time.",
        )
    return (
        "good_time",
        ADVICE_LABELS["good_time"],
        f"{route_id} has low predicted traffic ({probability}) and no high "
        "external accident prior at this departure. This is a good time to leave, "
        "with the usual caveats for synthetic traffic labels.",
    )


def _mapbox_advice(
    decision: FinalRouteDecision,
) -> tuple[str, str, str]:
    incident = decision.incident_risk
    if incident.route_specific and incident.level in {"HIGH", "MODERATE"}:
        return (
            "elevated_incident_risk",
            ADVICE_LABELS["elevated_incident_risk"],
            "Mapbox returned route-level incident or closure evidence for the "
            "recommended route. Check provider details before departure.",
        )
    return (
        "provider_eta_only",
        ADVICE_LABELS["provider_eta_only"],
        "Mapbox provides a traffic-aware ETA for the requested departure, but not "
        "the OSM future-traffic ML congestion band. The recommendation uses "
        "provider ETA, distance, conservative unknown risk, and expert rules.",
    )


def _empty_recommendation(
    departure: datetime,
    warnings: list[str],
    mapbox_considered: bool,
    generated_at: datetime | None,
) -> FinalRecommendation:
    return FinalRecommendation(
        departure=departure,
        routes=(),
        recommended_route_id=None,
        advice_code="insufficient_evidence",
        advice_label=ADVICE_LABELS["insufficient_evidence"],
        advice_message=(
            "No non-vetoed, distinct route is available, so no route is "
            "recommended. Missing evidence and safety vetoes were not bypassed."
        ),
        policy_version=POLICY_VERSION,
        policy_description=POLICY_DESCRIPTION,
        weights=(),
        fallback_used=True,
        fallback_basis="No eligible route after hard constraints and deduplication.",
        warnings=tuple(dict.fromkeys(warnings)),
        provenance=_PROVENANCE,
        generated_at=generated_at or _now_utc(),
        participating_sources=(),
        mapbox_considered=mapbox_considered,
    )
