from __future__ import annotations

from datetime import date, datetime, time, timedelta
from html import escape as html_escape

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.config import LANDMARKS, mapbox_token_present
from src.contracts import (
    FinalRecommendation,
    FinalRouteDecision,
    RouteAssessment,
    RouteResponse,
    RouteTrafficPrediction,
    WeatherSnapshot,
)
from src.local_traffic import local_traffic_summary
from src.service import RouteService
from src.weather import fetch_weather

st.set_page_config(page_title="Karachi Route Advisor", layout="wide")

SELECTED_ROUTE_COLOR = "#0b6e4f"
UNSELECTED_ROUTE_COLOR = "#7f8c8d"
PREDICTION_BANDS = {"low": "LOW", "moderate": "MODERATE", "high": "HIGH"}
USABLE_PREDICTION_STATUSES = frozenset({"ok", "partial"})
SELECTED_ROUTE_KEY = "selected_route_id"
SELECTION_CONTEXT_KEY = "selected_route_context"


@st.cache_resource(show_spinner=False)
def get_service() -> RouteService:
    return RouteService()


@st.cache_data(ttl=600, show_spinner=False)
def get_weather(lat: float, lon: float) -> WeatherSnapshot:
    return fetch_weather(lat, lon)


def _route_label(decision: FinalRouteDecision) -> str:
    if decision.recommended:
        return "Recommended route"
    return f"Alternative {decision.rank - 1}"


def _assessment(decision: FinalRouteDecision) -> RouteAssessment | None:
    return decision.assessment


def _eta_text(assessment: RouteAssessment | None, fallback: float) -> str:
    value = assessment.decision_eta_min if assessment is not None else fallback
    return f"{value:.0f} min"


def _prediction(decision: FinalRouteDecision) -> RouteTrafficPrediction | None:
    prediction = decision.traffic_prediction
    if prediction is not None and prediction.route_id == decision.route_id:
        return prediction
    assessment = _assessment(decision)
    if assessment is not None:
        candidate = assessment.traffic_prediction
        if candidate is not None and candidate.route_id == decision.route_id:
            return candidate
    return None


def _usable_prediction(
    prediction: RouteTrafficPrediction | None,
) -> bool:
    return (
        prediction is not None
        and prediction.status in USABLE_PREDICTION_STATUSES
        and prediction.traffic_band in PREDICTION_BANDS
        and bool(prediction.traffic_probabilities)
    )


def _prediction_departure(
    decision: FinalRouteDecision,
) -> datetime | None:
    prediction = _prediction(decision)
    if prediction is not None:
        return prediction.requested_departure
    assessment = _assessment(decision)
    if assessment is not None:
        return assessment.requested_departure
    return None


def _prediction_band_text(decision: FinalRouteDecision) -> str:
    prediction = _prediction(decision)
    if _usable_prediction(prediction):
        assert prediction is not None
        return PREDICTION_BANDS[prediction.traffic_band]
    return "Unavailable"


def _prediction_probability_text(decision: FinalRouteDecision) -> str:
    prediction = _prediction(decision)
    if _usable_prediction(prediction):
        assert prediction is not None
        if prediction.traffic_probability is not None:
            return f"{prediction.traffic_probability:.0%}"
    return "Unavailable"


def _prediction_confidence_text(decision: FinalRouteDecision) -> str:
    prediction = _prediction(decision)
    if _usable_prediction(prediction):
        assert prediction is not None
        return prediction.confidence
    return "Unavailable"


def _traffic_adjusted_eta_text(decision: FinalRouteDecision) -> str:
    assessment = _assessment(decision)
    if assessment is None:
        return "Unavailable"
    evidence = assessment.traffic_evidence
    if (
        evidence is not None
        and evidence.applied
        and assessment.traffic_adjusted_eta_min is not None
    ):
        return f"{assessment.traffic_adjusted_eta_min:.0f} min"
    return "Not applied"


def _incident_source_text(decision: FinalRouteDecision) -> str:
    risk = decision.incident_risk
    return risk.source or "Unavailable"


def _recommended_decision(
    final: FinalRecommendation,
) -> FinalRouteDecision:
    for decision in final.routes:
        if decision.recommended:
            return decision
    return final.routes[0]


def _selected_decision(
    final: FinalRecommendation,
    selected_route_id: str | None,
) -> FinalRouteDecision:
    for decision in final.routes:
        if decision.route_id == selected_route_id:
            return decision
    return _recommended_decision(final)


def _selection_context(response: RouteResponse) -> tuple[str, ...]:
    return (
        response.origin,
        response.destination,
        response.departure.isoformat(),
        response.preference,
        str(response.avoid_tolls),
        response.traffic_status,
    )


def _select_route(route_id: str) -> None:
    if st.session_state.get(SELECTED_ROUTE_KEY) != route_id:
        st.session_state[SELECTED_ROUTE_KEY] = route_id
        st.rerun()


def _provider_evidence_text(decision: FinalRouteDecision) -> str:
    assessment = _assessment(decision)
    if assessment is None:
        return "Unavailable"
    evidence = assessment.traffic_evidence
    if evidence is not None and evidence.applied:
        return "Matched provider evidence"
    if evidence is not None:
        return "No defensible match"
    return "Not available"


def _provider_evidence_help(decision: FinalRouteDecision) -> str:
    assessment = _assessment(decision)
    if assessment is None:
        return "No provider traffic evidence was matched to this OSM route."
    evidence = assessment.traffic_evidence
    if evidence is not None and evidence.applied:
        return (
            f"Matched Mapbox route {evidence.provider_route_id} for the requested "
            "departure. This is provider evidence, separate from the future-ML "
            "prediction below."
        )
    return (
        "Matched Mapbox provider evidence is used only when the corridor overlap "
        "and baseline are defensible. It is separate from the future-ML prediction."
    )


def _incident_text(decision: FinalRouteDecision) -> str:
    risk = decision.incident_risk
    if risk.route_specific:
        return risk.level.title()
    return "Unavailable"


def _traffic_status_text(response: RouteResponse) -> str:
    mapping = {
        "annotated": "Matched provider annotations available",
        "partial_eta_delta_only": "Provider ETA-versus-typical evidence only",
        "no_traffic_evidence": "No non-unknown provider traffic evidence",
        "no_mapbox": "Mapbox evidence not used",
    }
    return mapping.get(response.traffic_status, response.traffic_status)


def _route_line_style(selected: bool) -> tuple[str, int, float, str | None]:
    if selected:
        return SELECTED_ROUTE_COLOR, 10, 1.0, None
    return UNSELECTED_ROUTE_COLOR, 4, 0.55, "6 6"


def build_map(
    response: RouteResponse,
    final: FinalRecommendation,
    selected_route_id: str | None = None,
) -> folium.Map:
    origin = LANDMARKS[response.origin]
    destination = LANDMARKS[response.destination]
    center = [
        (origin[0] + destination[0]) / 2.0,
        (origin[1] + destination[1]) / 2.0,
    ]
    map_object = folium.Map(
        location=center,
        zoom_start=13,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    folium.Marker(
        location=origin,
        tooltip=f"Origin: {response.origin}",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(map_object)
    folium.Marker(
        location=destination,
        tooltip=f"Destination: {response.destination}",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(map_object)
    selected = _selected_decision(final, selected_route_id)
    ordered = [
        decision
        for decision in final.routes
        if decision.route_id != selected.route_id
    ]
    ordered.append(selected)
    for decision in ordered:
        route = decision.route
        is_selected = decision.route_id == selected.route_id
        color, weight, opacity, dash_array = _route_line_style(is_selected)
        assessment = _assessment(decision)
        eta = (
            assessment.decision_eta_min
            if assessment is not None
            else decision.decision_time_min
        )
        folium.PolyLine(
            locations=route.geometry,
            color=color,
            weight=weight,
            opacity=opacity,
            dash_array=dash_array,
            tooltip=(
                f"{'Selected - ' if is_selected else ''}"
                f"{_route_label(decision)} | {eta:.1f} min | "
                f"{route.distance_m / 1000.0:.2f} km"
            ),
        ).add_to(map_object)
    points = [origin, destination]
    for decision in final.routes:
        points.extend(decision.route.geometry)
    if points:
        latitudes = [point[0] for point in points]
        longitudes = [point[1] for point in points]
        map_object.fit_bounds(
            [
                [min(latitudes), min(longitudes)],
                [max(latitudes), max(longitudes)],
            ],
            padding=(30, 30),
        )
    legend_html = (
        '<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; '
        "background: rgba(255, 255, 255, 0.95); border: 1px solid #ccc; "
        "border-radius: 6px; padding: 8px 12px; font-size: 12px; "
        'line-height: 1.5;">'
        f"<b>Selected OSM route:</b> {html_escape(selected.route_id)} "
        "(bold green)<br>"
        "<span style=\"color: #7f8c8d;\">Other OSM routes</span> "
        "(thin dashed grey)"
        "</div>"
    )
    map_object.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=True).add_to(map_object)
    return map_object


def render_map(
    response: RouteResponse,
    final: FinalRecommendation,
    selected_route_id: str | None = None,
) -> None:
    selected = _selected_decision(final, selected_route_id)
    st_folium(
        build_map(response, final, selected.route_id),
        height=500,
        use_container_width=True,
        returned_objects=[],
        key=(
            f"map-{response.origin}-{response.destination}-"
            f"{response.departure.isoformat()}-{response.preference}-"
            f"{selected.route_id}"
        ),
    )


def render_prediction(decision: FinalRouteDecision) -> None:
    prediction = _prediction(decision)
    usable = _usable_prediction(prediction)
    band = (
        PREDICTION_BANDS[prediction.traffic_band]
        if usable and prediction is not None
        else "Unavailable"
    )
    st.markdown("#### Future-traffic ML prediction")
    st.markdown(f"**Predicted traffic at departure:** {band}")
    if not usable:
        reason = (
            prediction.confidence_note
            if prediction is not None
            else "No route-specific ML prediction was returned for this route."
        )
        departure = _prediction_departure(decision)
        departure_text = (
            f" for {departure:%Y-%m-%d %H:%M}" if departure is not None else ""
        )
        st.caption(
            f"No ML congestion band is shown for {decision.route_id}"
            f"{departure_text}. {reason}"
        )
        if prediction is not None:
            st.caption(
                f"Prediction provenance: {prediction.data_provenance_label}. "
                f"Model: {prediction.model_name} ({prediction.model_version})."
            )
        else:
            st.caption(
                "Prediction provenance: unavailable; no prediction was generated."
            )
        return

    assert prediction is not None
    probability = (
        f"{prediction.traffic_probability:.0%}"
        if prediction.traffic_probability is not None
        else "Unavailable"
    )
    columns = st.columns(3)
    columns[0].metric("Predicted congestion probability", probability)
    columns[1].metric("Confidence", prediction.confidence)
    columns[2].metric(
        "Selected departure",
        f"{prediction.requested_departure:%Y-%m-%d %H:%M}",
    )
    st.caption(
        f"Prediction provenance: {prediction.data_provenance_label}. "
        f"Model: {prediction.model_name} ({prediction.model_version})."
    )
    if prediction.traffic_probabilities:
        st.caption(
            "Class probabilities: "
            + ", ".join(
                f"{PREDICTION_BANDS.get(band_name, band_name.upper())}="
                f"{probability_value:.0%}"
                for band_name, probability_value in prediction.traffic_probabilities
            )
        )
    st.caption(prediction.confidence_note)


def render_route_card(
    decision: FinalRouteDecision,
    final: FinalRecommendation,
    selected_route_id: str,
) -> None:
    assessment = _assessment(decision)
    prediction = _prediction(decision)
    selected = decision.route_id == selected_route_id
    with st.container(border=True):
        title_columns = st.columns([3.2, 1.0])
        title_columns[0].markdown(f"### {_route_label(decision)}")
        title_columns[1].caption(
            "Currently selected" if selected else "Available route"
        )
        st.caption(
            f"OSM route {decision.route_id} | "
            f"{decision.route.distance_m / 1000.0:.2f} km | "
            f"{final.departure:%Y-%m-%d %H:%M}"
        )
        columns = st.columns(4)
        columns[0].metric(
            "ETA",
            _eta_text(assessment, decision.decision_time_min),
            help=(
                assessment.decision_eta_basis
                if assessment is not None
                else decision.decision_time_basis
            ),
        )
        columns[1].metric(
            "Distance",
            f"{decision.route.distance_m / 1000.0:.1f} km",
        )
        columns[2].metric(
            "Predicted traffic",
            _prediction_band_text(decision),
            help=(
                prediction.confidence_note
                if prediction is not None
                else "No route-specific ML prediction is available."
            ),
        )
        columns[3].metric(
            "Congestion probability",
            _prediction_probability_text(decision),
        )
        detail_columns = st.columns(4)
        detail_columns[0].metric(
            "Confidence",
            _prediction_confidence_text(decision),
        )
        detail_columns[1].metric(
            "Provider traffic evidence",
            _provider_evidence_text(decision),
            help=_provider_evidence_help(decision),
        )
        detail_columns[2].metric(
            "Traffic-adjusted ETA",
            _traffic_adjusted_eta_text(decision),
        )
        detail_columns[3].metric(
            "Incident risk",
            _incident_text(decision),
        )
        provenance = (
            prediction.data_provenance_label
            if prediction is not None
            else "Unavailable"
        )
        st.caption(
            f"Incident source: {_incident_source_text(decision)} | "
            f"Prediction provenance: {provenance}"
        )
        st.caption(decision.ranking_reason)
        if decision.recommended:
            if final.advice_code in {"not_recommended", "consider_earlier"}:
                st.warning(f"**{final.advice_label}**")
            elif final.advice_code == "good_time":
                st.success(f"**{final.advice_label}**")
            else:
                st.info(f"**{final.advice_label}**")
            st.write(final.advice_message)
        button_label = (
            f"Selected: {_route_label(decision)}"
            if selected
            else f"Select {_route_label(decision)}"
        )
        if st.button(
            button_label,
            key=f"select-route-{decision.route_id}",
            type="primary" if selected else "secondary",
            use_container_width=True,
        ):
            _select_route(decision.route_id)


def render_route_options(
    final: FinalRecommendation,
    selected_route_id: str,
) -> None:
    recommended = _recommended_decision(final)
    alternatives = [
        decision
        for decision in final.routes
        if decision.route_id != recommended.route_id
    ]
    st.subheader("Recommended route")
    render_route_card(recommended, final, selected_route_id)
    st.subheader("Other best routes")
    if not alternatives:
        st.caption(
            "No distinct valid alternative was found. Routes are not invented to "
            "fill this list."
        )
        return
    for decision in alternatives:
        render_route_card(decision, final, selected_route_id)


def render_selected_route(
    decision: FinalRouteDecision,
    final: FinalRecommendation,
) -> None:
    assessment = _assessment(decision)
    prediction = _prediction(decision)
    st.subheader("Selected route details")
    with st.container(border=True):
        st.markdown(f"### {_route_label(decision)}")
        st.markdown(f"**Active route:** OSM route {decision.route_id}")
        st.caption(
            f"{decision.route.distance_m / 1000.0:.2f} km | "
            f"Selected departure {final.departure:%Y-%m-%d %H:%M}"
        )
        columns = st.columns(4)
        columns[0].metric(
            "Selected ETA",
            _eta_text(assessment, decision.decision_time_min),
            help=(
                assessment.decision_eta_basis
                if assessment is not None
                else decision.decision_time_basis
            ),
        )
        columns[1].metric(
            "Distance",
            f"{decision.route.distance_m / 1000.0:.1f} km",
        )
        columns[2].metric(
            "Provider traffic evidence",
            _provider_evidence_text(decision),
            help=_provider_evidence_help(decision),
        )
        columns[3].metric("Incident risk", _incident_text(decision))
        if assessment is not None:
            st.caption(f"{assessment.eta_label}. {assessment.confidence_note}")

        render_prediction(decision)

        st.markdown("#### ETA breakdown")
        eta_columns = st.columns(4)
        eta_columns[0].metric(
            "Analytical ETA",
            (
                f"{assessment.analytical_eta_min:.0f} min"
                if assessment is not None
                else f"{decision.route.eta_min:.0f} min"
            ),
        )
        eta_columns[1].metric(
            "ML scenario ETA",
            (
                f"{assessment.ml_eta_min:.1f} min"
                if assessment is not None
                and assessment.ml_eta_min is not None
                else "Unavailable"
            ),
        )
        eta_columns[2].metric(
            "Traffic-adjusted ETA",
            _traffic_adjusted_eta_text(decision),
        )
        eta_columns[3].metric(
            "Decision ETA",
            _eta_text(assessment, decision.decision_time_min),
        )
        eta_basis = (
            assessment.decision_eta_basis
            if assessment is not None
            else decision.decision_time_basis
        )
        st.caption(f"Decision ETA basis: {eta_basis}")

        st.markdown("#### Provider traffic evidence")
        evidence = assessment.traffic_evidence if assessment is not None else None
        if evidence is not None:
            st.markdown(
                "**Provider traffic evidence for this route:** "
                f"{_provider_evidence_text(decision)}"
            )
            st.caption(
                f"Source: Mapbox Directions API | "
                f"Provider route: {evidence.provider_route_id} | "
                f"Method: {evidence.method} | "
                f"Geometry overlap: {evidence.geometry_overlap:.0%} | "
                f"Requested departure: "
                f"{evidence.requested_departure:%Y-%m-%d %H:%M}"
            )
            if evidence.applied:
                st.write(
                    "Traffic-adjusted ETA from matched provider evidence: "
                    f"{evidence.adjusted_eta_min:.1f} min"
                )
            else:
                st.caption(
                    "Matched provider evidence was not applied, so no "
                    "provider-adjusted ETA is shown."
                )
            for item in evidence.limitations:
                st.caption(item)
        else:
            st.markdown(
                "**Provider traffic evidence for this route:** Not available"
            )
            st.caption(
                "No Mapbox provider route met the geometric matching and baseline "
                "requirements for this route, so no provider-adjusted ETA is shown."
            )

        st.markdown("#### Incident risk")
        risk = decision.incident_risk
        st.markdown(f"**Incident risk for this route:** {_incident_text(decision)}")
        st.caption(
            f"Source: {_incident_source_text(decision)} | "
            f"Evidence type: {risk.evidence_type}"
        )
        st.write(risk.explanation)
        if risk.external_prior_level is not None:
            prior_probability = (
                f"{risk.external_prior_probability:.0%}"
                if risk.external_prior_probability is not None
                else "probability unavailable"
            )
            st.caption(
                f"External accident prior: {risk.external_prior_level} "
                f"({prior_probability}); not route-specific Karachi crash history."
            )
        for item in risk.limitations:
            st.caption(item)

        st.markdown("#### Decision explanation")
        st.write(decision.ranking_reason)
        if prediction is not None and _usable_prediction(prediction):
            st.caption(
                f"Route-specific ML advice: {prediction.advice_label}. "
                f"{prediction.advice_message}"
            )
        if assessment is not None:
            st.caption(
                "Expert verdict: "
                f"penalty={assessment.expert_verdict.penalty:.2f}, "
                f"bonus={assessment.expert_verdict.bonus:.2f}"
            )


def render_technical_details(
    response: RouteResponse,
    final: FinalRecommendation,
    selected_route_id: str | None = None,
) -> None:
    with st.expander("Technical Details & Data Limitations", expanded=False):
        st.markdown("**Route assessment summary**")
        st.dataframe(
            [
                {
                    "Rank": decision.rank,
                    "Route": decision.route_id,
                    "Selected": decision.route_id == selected_route_id,
                    "Recommended": decision.recommended,
                    "Decision ETA (min)": round(
                        decision.assessment.decision_eta_min
                        if decision.assessment is not None
                        else decision.decision_time_min,
                        2,
                    ),
                    "ETA basis": (
                        decision.assessment.decision_eta_basis
                        if decision.assessment is not None
                        else decision.decision_time_basis
                    ),
                    "Provider evidence": _provider_evidence_text(decision),
                    "Incident risk": decision.incident_risk.level,
                    "Academic rank": decision.academic_rank,
                    "Final cost": round(decision.final_cost, 3),
                }
                for decision in final.routes
            ],
            width="stretch",
            hide_index=True,
        )
        for decision in final.routes:
            assessment = _assessment(decision)
            selected_suffix = (
                " (selected)"
                if decision.route_id == selected_route_id
                else ""
            )
            with st.expander(
                f"{_route_label(decision)} technical details{selected_suffix}",
                expanded=False,
            ):
                if assessment is not None:
                    for item in assessment.details:
                        st.write(f"- {item}")
                    st.markdown("**Limitations**")
                    for item in assessment.limitations:
                        st.write(f"- {item}")
                st.write(decision.ranking_reason)
                st.markdown("**Incident risk evidence**")
                st.write(decision.incident_risk.explanation)
                for item in decision.incident_risk.limitations:
                    st.caption(item)
                st.markdown("**Expert-system trace**")
                trace_rows = [
                    {
                        "Rule": entry.rule_id,
                        "Name": entry.rule_name,
                        "Evidence": entry.evidence,
                        "Effect": entry.effect,
                    }
                    for entry in decision.assessment.expert_verdict.trace
                ] if decision.assessment is not None else []
                if trace_rows:
                    st.dataframe(trace_rows, width="stretch", hide_index=True)
                else:
                    st.caption("No expert-system trace is available.")
                prediction = _prediction(decision)
                if prediction is not None:
                    st.markdown("**Future-traffic ML evidence**")
                    st.write(
                        f"Model: {prediction.model_name} "
                        f"({prediction.model_version}) | "
                        f"Data: {prediction.data_provenance_label}"
                    )
                    st.write(
                        f"Band: {prediction.traffic_band.title()} | "
                        f"Confidence: {prediction.confidence} | "
                        f"Selected departure: "
                        f"{prediction.requested_departure:%Y-%m-%d %H:%M} | "
                        f"Real records: {prediction.real_records:,} | "
                        f"Synthetic records: {prediction.synthetic_records:,}"
                    )
                    if prediction.traffic_probabilities:
                        st.write(
                            "Probabilities: "
                            + ", ".join(
                                f"{band}={probability:.0%}"
                                for band, probability in prediction.traffic_probabilities
                            )
                        )
                    if prediction.top_drivers:
                        st.write("Top drivers: " + "; ".join(prediction.top_drivers))
                    for item in prediction.limitations:
                        st.caption(item)
        st.markdown("**Matched traffic evidence**")
        evidence_rows = []
        for decision in final.routes:
            assessment = _assessment(decision)
            evidence = assessment.traffic_evidence if assessment is not None else None
            if evidence is None:
                continue
            evidence_rows.append(
                {
                    "OSM route": evidence.route_id,
                    "Provider route": evidence.provider_route_id,
                    "Applied": evidence.applied,
                    "Method": evidence.method,
                    "Geometry overlap": f"{evidence.geometry_overlap:.0%}",
                    "Segment coverage": (
                        f"{evidence.segment_coverage:.0%}"
                        if evidence.segment_coverage is not None
                        else "unavailable"
                    ),
                    "Provider ETA (min)": round(evidence.provider_eta_min, 2),
                    "Provider typical (min)": (
                        round(evidence.provider_typical_min, 2)
                        if evidence.provider_typical_min is not None
                        else None
                    ),
                    "Delay ratio": (
                        round(evidence.provider_delay_ratio, 3)
                        if evidence.provider_delay_ratio is not None
                        else None
                    ),
                    "Adjusted OSM ETA (min)": round(evidence.adjusted_eta_min, 2),
                }
            )
        if evidence_rows:
            st.dataframe(evidence_rows, width="stretch", hide_index=True)
        else:
            st.caption(
                "No Mapbox provider route met the geometric matching and baseline "
                "requirements for this request."
            )
        st.markdown("**Provider and data status**")
        st.write(f"Mapbox traffic status: {_traffic_status_text(response)}")
        st.write(f"Mapbox routes returned: {len(response.mapbox_routes)}")
        st.write(
            "Mapbox token configured: "
            f"{'yes' if mapbox_token_present() else 'no'}"
        )
        st.write("Mapbox routes are evidence only; they are not route candidates.")
        if response.baseline is not None:
            baseline = response.baseline
            st.markdown("**OSM A* vs Dijkstra baseline**")
            baseline_columns = st.columns(4)
            baseline_columns[0].metric(
                "A* expanded nodes",
                baseline.astar_expanded_nodes,
            )
            baseline_columns[1].metric(
                "Dijkstra expanded nodes",
                baseline.dijkstra_expanded_nodes,
            )
            baseline_columns[2].metric(
                "Same path",
                "yes" if baseline.same_path else "no",
            )
            baseline_columns[3].metric(
                "Same cost",
                "yes" if baseline.same_cost else "no",
            )
            st.caption(
                "Dijkstra is an independent reference comparison, not a traffic "
                "source and not a second recommendation."
            )
        else:
            st.caption("Dijkstra baseline is unavailable for this request.")
        st.markdown("**Local observations**")
        local = local_traffic_summary()
        st.write(local.message)
        st.caption(
            f"Local file: {local.path} | Rows: {local.rows} | "
            f"Ready for training: {local.ready_for_training}"
        )
        st.markdown("**Warnings and provenance**")
        for warning in (*response.warnings, *final.warnings):
            st.warning(warning)
        for item in (*response.provenance, *final.provenance):
            st.write(f"- {item}")
        st.markdown("**How the recommendation is produced**")
        st.write(final.policy_description)
        st.write(
            "Policy version: "
            f"{final.policy_version} | Weights: "
            + ", ".join(
                f"{name}={weight:.2f}" for name, weight in final.weights
            )
        )
        st.caption(
            "OSM routing is A*/Dijkstra on the local OSM graph. Mapbox evidence "
            "adjusts an OSM ETA only when the corridor match is defensible. The "
            "future-traffic ML labels are synthetic scenarios. External UK STATS19 "
            "accident priors are not Karachi crash history and are not route-specific."
        )


def main() -> None:
    st.title("Karachi Route Advisor")
    st.caption(
        "Gulistan-e-Jauhar pilot | OSM A*/Dijkstra routing | matched Mapbox "
        "traffic evidence | future-traffic ML | expert-system assessment"
    )
    with st.form("trip_form"):
        origin_column, destination_column = st.columns(2)
        origin_name = origin_column.selectbox("From", list(LANDMARKS), index=0)
        destination_name = destination_column.selectbox(
            "To",
            list(LANDMARKS),
            index=3,
        )
        date_column, time_column, preference_column = st.columns(3)
        departure_date = date_column.date_input(
            "Departure date",
            value=date.today() + timedelta(days=1),
        )
        departure_time = time_column.time_input("Departure time", value=time(9, 0))
        preference = preference_column.selectbox(
            "Preference",
            ["Fastest", "Shortest", "Least congested", "Safest"],
        )
        toll_column, mapbox_column = st.columns(2)
        avoid_tolls = toll_column.checkbox("Strictly avoid tolls", value=False)
        include_mapbox = mapbox_column.checkbox(
            "Use matched Mapbox traffic evidence",
            value=True,
        )
        submitted = st.form_submit_button(
            "Find Best Route",
            type="primary",
            use_container_width=True,
        )
    requested_departure = datetime.combine(departure_date, departure_time)
    if submitted:
        if origin_name == destination_name:
            st.error("Origin and destination must be different.")
            return
        try:
            with st.spinner("Evaluating OSM routes and matching traffic evidence..."):
                service = get_service()
                try:
                    weather = get_weather(*LANDMARKS[origin_name])
                except Exception as exc:
                    weather = WeatherSnapshot(
                        condition="dry",
                        source="fallback",
                        warning=f"{type(exc).__name__}: {exc}",
                    )
                response = service.route(
                    origin_name=origin_name,
                    destination_name=destination_name,
                    departure=requested_departure,
                    preference=preference,
                    weather=weather,
                    avoid_tolls=avoid_tolls,
                    include_mapbox=include_mapbox,
                )
                st.session_state["response"] = response
                computed_final = response.final_recommendation
                if computed_final is not None and computed_final.routes:
                    st.session_state[SELECTION_CONTEXT_KEY] = _selection_context(
                        response
                    )
                    st.session_state[SELECTED_ROUTE_KEY] = (
                        _recommended_decision(computed_final).route_id
                    )
        except Exception as exc:
            st.error(f"Route request failed: {type(exc).__name__}: {exc}")
            return
    response = st.session_state.get("response")
    if response is None:
        st.info("Choose a trip, set a departure, and click Find Best Route.")
        st.write(
            "The recommendation uses OSM A*/Dijkstra candidates only. Mapbox is "
            "used as matched traffic evidence when available, not as a competing "
            "route. Missing evidence is shown as unavailable rather than zero."
        )
        return
    if response.departure != requested_departure:
        st.info(
            "Showing the saved result for the previous departure. Click Find Best "
            "Route to refresh for the current sidebar departure."
        )
    final = response.final_recommendation
    if final is None or not final.routes:
        st.error(
            "No non-vetoed, distinct route is available for this request. "
            "Safety constraints and missing evidence were not bypassed."
        )
        for warning in response.warnings:
            st.warning(warning)
        return
    recommended = _recommended_decision(final)
    context = _selection_context(response)
    if st.session_state.get(SELECTION_CONTEXT_KEY) != context:
        st.session_state[SELECTION_CONTEXT_KEY] = context
        st.session_state[SELECTED_ROUTE_KEY] = recommended.route_id
    route_ids = {decision.route_id for decision in final.routes}
    selected_route_id = st.session_state.get(SELECTED_ROUTE_KEY)
    if not isinstance(selected_route_id, str) or selected_route_id not in route_ids:
        selected_route_id = recommended.route_id
        st.session_state[SELECTED_ROUTE_KEY] = selected_route_id
    selected_decision = _selected_decision(final, selected_route_id)
    render_route_options(final, selected_route_id)
    render_selected_route(selected_decision, final)
    st.subheader("Map")
    render_map(response, final, selected_route_id)
    render_technical_details(response, final, selected_route_id)


if __name__ == "__main__":
    main()
