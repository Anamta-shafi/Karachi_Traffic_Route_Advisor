from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from html import escape as html_escape

import folium
import streamlit as st
from streamlit_folium import st_folium

from src.config import LANDMARKS
from src.contracts import (
    FinalRecommendation,
    FinalRouteDecision,
    RouteAssessment,
    RouteResponse,
    RouteTrafficPrediction,
    WeatherSnapshot,
)
from src.service import RouteService
from src.weather import fetch_weather

st.set_page_config(page_title="Karachi Route Advisor", layout="wide")

SELECTED_ROUTE_COLOR = "#0b6e4f"
UNSELECTED_ROUTE_COLOR = "#7f8c8d"
PREDICTION_BANDS = {"low": "LOW", "moderate": "MODERATE", "high": "HIGH"}
USABLE_PREDICTION_STATUSES = frozenset({"ok", "partial"})
SELECTED_ROUTE_KEY = "selected_route_id"
SELECTION_CONTEXT_KEY = "selected_route_context"
MAX_ALTERNATIVES = 2
STALE_EVIDENCE_MINUTES = 60

BADGE_STYLES = {
    "RECOMMENDED": "#0b6e4f",
    "LIVE": "#1f6feb",
    "FORECAST": "#6c4bb6",
    "SYNTHETIC": "#b7791f",
    "ESTIMATED": "#4a5568",
    "HISTORICAL": "#2b6cb0",
}
BAND_RANK = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
INCIDENT_RANK = {"Low": 0, "Moderate": 1, "High": 2}


@st.cache_resource(show_spinner=False)
def get_service() -> RouteService:
    return RouteService()


@st.cache_data(ttl=600, show_spinner=False)
def get_weather(lat: float, lon: float) -> WeatherSnapshot:
    return fetch_weather(lat, lon)


def _apply_readable_styles() -> None:
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"] { font-size: 18px; }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {
            font-size: 1.05rem;
            line-height: 1.55;
        }
        [data-testid="stMetricValue"] { font-size: 1.75rem; }
        [data-testid="stMetricLabel"] { font-size: 1rem; }
        [data-testid="stSidebar"] { font-size: 1rem; }
        h1 { font-size: 2.2rem; }
        h2 { font-size: 1.6rem; }
        h3 { font-size: 1.3rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _route_label(decision: FinalRouteDecision) -> str:
    if decision.recommended:
        return "Recommended route"
    return f"Alternative {decision.rank - 1}"


def _assessment(decision: FinalRouteDecision) -> RouteAssessment | None:
    return decision.assessment


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


def _usable_prediction(prediction: RouteTrafficPrediction | None) -> bool:
    return (
        prediction is not None
        and prediction.status in USABLE_PREDICTION_STATUSES
        and prediction.traffic_band in PREDICTION_BANDS
        and bool(prediction.traffic_probabilities)
    )


def _prediction_band_text(decision: FinalRouteDecision) -> str:
    prediction = _prediction(decision)
    if _usable_prediction(prediction):
        assert prediction is not None
        return PREDICTION_BANDS[prediction.traffic_band]
    return "Unavailable"


def _incident_text(decision: FinalRouteDecision) -> str:
    risk = decision.incident_risk
    return risk.level.title() if risk.route_specific else "Unavailable"


def _recommended_decision(final: FinalRecommendation) -> FinalRouteDecision:
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


def _route_line_style(selected: bool) -> tuple[str, int, float, str | None]:
    if selected:
        return SELECTED_ROUTE_COLOR, 10, 1.0, None
    return UNSELECTED_ROUTE_COLOR, 4, 0.55, "6 6"


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _evidence_is_stale(fetched_at: str | None) -> bool:
    parsed = _parse_timestamp(fetched_at)
    if parsed is None:
        return False
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return (now - parsed) > timedelta(minutes=STALE_EVIDENCE_MINUTES)


def _badge_html(kind: str, text: str) -> str:
    color = BADGE_STYLES.get(kind, "#4a5568")
    return (
        f'<span style="display:inline-block;margin:0 8px 6px 0;padding:4px 12px;'
        f"border-radius:999px;background:{color};color:#ffffff;font-size:0.85rem;"
        f'font-weight:600;">{html_escape(kind)} &middot; {html_escape(text)}</span>'
    )


def _decision_badges(decision: FinalRouteDecision) -> list[tuple[str, str]]:
    badges: list[tuple[str, str]] = []
    if decision.recommended:
        badges.append(("RECOMMENDED", "rank 1"))
    assessment = _assessment(decision)
    evidence = assessment.traffic_evidence if assessment is not None else None
    prediction = _prediction(decision)
    if evidence is not None and evidence.applied:
        badges.append(("LIVE", "Mapbox matched snapshot"))
        if _evidence_is_stale(evidence.fetched_at):
            badges.append(("HISTORICAL", "stale snapshot"))
    elif _usable_prediction(prediction):
        assert prediction is not None
        if prediction.data_provenance == "SYNTHETIC":
            badges.append(("SYNTHETIC", "ML scenario"))
        else:
            badges.append(("FORECAST", "ML scenario"))
        badges.append(("ESTIMATED", "OSM base"))
    else:
        badges.append(("ESTIMATED", "OSM analytical ETA"))
    return badges


def _traffic_banner(
    response: RouteResponse,
    final: FinalRecommendation,
) -> str | None:
    applied = [
        assessment.traffic_evidence
        for decision in final.routes
        if (assessment := _assessment(decision)) is not None
        and assessment.traffic_evidence is not None
        and assessment.traffic_evidence.applied
    ]
    usable_forecast = any(
        _usable_prediction(_prediction(decision)) for decision in final.routes
    )
    if applied:
        stale = [
            evidence
            for evidence in applied
            if _evidence_is_stale(evidence.fetched_at)
        ]
        if stale:
            return (
                "The matched live traffic snapshot is stale "
                f"(fetched {stale[0].fetched_at}). Treat it as historical "
                "evidence and refresh the route before relying on it."
            )
        if response.traffic_status == "partial_eta_delta_only":
            return (
                "Live traffic is partial: only a provider ETA-versus-typical "
                "delta is available. Segment-level congestion is unavailable."
            )
        return None
    if response.traffic_status in {"no_mapbox", "no_traffic_evidence"}:
        if usable_forecast:
            return (
                "Live traffic is unavailable. ETAs use the future-ML scenario "
                "forecast (synthetic labels) and OSM estimates, not live traffic."
            )
        return (
            "Live traffic and a usable forecast are unavailable. ETAs are "
            "analytical OSM estimates only; no traffic certainty is implied."
        )
    if response.traffic_status == "partial_eta_delta_only":
        return (
            "Live traffic is partial (provider ETA delta only). Segment "
            "congestion is unavailable; treat the ETA as provider-profile evidence."
        )
    return None


def _route_metrics(decision: FinalRouteDecision) -> dict[str, float | str | None]:
    assessment = _assessment(decision)
    return {
        "eta": (
            assessment.decision_eta_min
            if assessment is not None
            else decision.decision_time_min
        ),
        "km": decision.route.distance_m / 1000.0,
        "band": _prediction_band_text(decision),
        "incident": _incident_text(decision),
        "incident_rank": INCIDENT_RANK.get(_incident_text(decision)),
        "score": float(decision.final_score),
    }


def _why_choose(
    decision: FinalRouteDecision,
    recommended: FinalRouteDecision,
) -> str:
    metrics = _route_metrics(decision)
    rec = _route_metrics(recommended)
    if decision.route_id == recommended.route_id:
        return (
            f"Ranks first with a decision score of {metrics['score']:.0f}/100: "
            f"{metrics['eta']:.1f} min, {metrics['km']:.2f} km, "
            f"forecast traffic {metrics['band']}."
        )
    advantages: list[str] = []
    if metrics["eta"] < rec["eta"] - 0.5:
        advantages.append(f"{rec['eta'] - metrics['eta']:.1f} min faster")
    if metrics["km"] < rec["km"] - 0.05:
        advantages.append(f"{rec['km'] - metrics['km']:.1f} km shorter")
    alt_band = BAND_RANK.get(str(metrics["band"]))
    rec_band = BAND_RANK.get(str(rec["band"]))
    if alt_band is not None and rec_band is not None and alt_band < rec_band:
        advantages.append(
            f"lower forecast traffic ({metrics['band']} vs {rec['band']})"
        )
    if (
        metrics["incident_rank"] is not None
        and rec["incident_rank"] is not None
        and metrics["incident_rank"] < rec["incident_rank"]
    ):
        advantages.append(
            f"lower incident risk ({metrics['incident']} vs {rec['incident']})"
        )
    if advantages:
        return "It is " + ", ".join(advantages) + "."
    return (
        "No ETA, distance, forecast-traffic, or incident-risk advantage over "
        "the recommended route."
    )


def _why_not(
    decision: FinalRouteDecision,
    recommended: FinalRouteDecision,
    alternatives: list[FinalRouteDecision],
) -> str:
    metrics = _route_metrics(decision)
    if decision.route_id == recommended.route_id:
        better: list[str] = []
        for alternative in alternatives:
            alt = _route_metrics(alternative)
            if alt["eta"] < metrics["eta"] - 0.5:
                better.append(
                    f"{_route_label(alternative)} is "
                    f"{metrics['eta'] - alt['eta']:.1f} min faster"
                )
            if alt["km"] < metrics["km"] - 0.05:
                better.append(
                    f"{_route_label(alternative)} is "
                    f"{metrics['km'] - alt['km']:.1f} km shorter"
                )
            alt_band = BAND_RANK.get(str(alt["band"]))
            rec_band = BAND_RANK.get(str(metrics["band"]))
            if alt_band is not None and rec_band is not None and alt_band < rec_band:
                better.append(
                    f"{_route_label(alternative)} has lower forecast traffic"
                )
            if (
                alt["incident_rank"] is not None
                and metrics["incident_rank"] is not None
                and alt["incident_rank"] < metrics["incident_rank"]
            ):
                better.append(
                    f"{_route_label(alternative)} has lower incident risk"
                )
        if better:
            return "Trade-off: " + "; ".join(better) + "."
        return (
            "No alternative beats it on ETA, distance, forecast traffic, or "
            "incident risk."
        )
    rec = _route_metrics(recommended)
    disadvantages: list[str] = []
    if metrics["eta"] > rec["eta"] + 0.5:
        disadvantages.append(f"{metrics['eta'] - rec['eta']:.1f} min slower")
    if metrics["km"] > rec["km"] + 0.05:
        disadvantages.append(f"{metrics['km'] - rec['km']:.1f} km longer")
    alt_band = BAND_RANK.get(str(metrics["band"]))
    rec_band = BAND_RANK.get(str(rec["band"]))
    if alt_band is not None and rec_band is not None and alt_band > rec_band:
        disadvantages.append(
            f"higher forecast traffic ({metrics['band']} vs {rec['band']})"
        )
    if (
        metrics["incident_rank"] is not None
        and rec["incident_rank"] is not None
        and metrics["incident_rank"] > rec["incident_rank"]
    ):
        disadvantages.append(
            f"higher incident risk ({metrics['incident']} vs {rec['incident']})"
        )
    if metrics["score"] < rec["score"]:
        disadvantages.append(
            f"lower decision score ({metrics['score']:.0f} vs {rec['score']:.0f})"
        )
    if disadvantages:
        return "It is " + ", ".join(disadvantages) + "."
    return "It ranks lower on the overall decision score despite similar metrics."


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
    recommended = _recommended_decision(final)
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
    if recommended.route.geometry:
        midpoint = recommended.route.geometry[len(recommended.route.geometry) // 2]
        folium.CircleMarker(
            location=midpoint,
            radius=8,
            color=SELECTED_ROUTE_COLOR,
            fill=True,
            fill_color=SELECTED_ROUTE_COLOR,
            fill_opacity=0.9,
            tooltip="Recommended route",
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
        "border-radius: 6px; padding: 8px 12px; font-size: 13px; "
        'line-height: 1.5;">'
        f"<b>Selected OSM route:</b> {html_escape(selected.route_id)} "
        "(bold green)<br>"
        "<span style=\"color: #7f8c8d;\">Other OSM routes</span> "
        "(thin dashed grey)<br>"
        f"<span style=\"color: {SELECTED_ROUTE_COLOR};\">&#9679;</span> "
        f"Recommended route marker: {html_escape(recommended.route_id)}"
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
        height=640,
        use_container_width=True,
        returned_objects=[],
        key=(
            f"map-{response.origin}-{response.destination}-"
            f"{response.departure.isoformat()}-{response.preference}-"
            f"{selected.route_id}"
        ),
    )


def render_route_card(
    decision: FinalRouteDecision,
    final: FinalRecommendation,
    selected_route_id: str,
    recommended: FinalRouteDecision,
    alternatives: list[FinalRouteDecision],
) -> None:
    assessment = _assessment(decision)
    selected = decision.route_id == selected_route_id
    with st.container(border=True):
        title_columns = st.columns([3.0, 1.0])
        title_columns[0].markdown(f"### {_route_label(decision)}")
        if selected:
            title_columns[1].markdown("**Selected**")
        st.markdown(
            "".join(
                _badge_html(kind, text)
                for kind, text in _decision_badges(decision)
            ),
            unsafe_allow_html=True,
        )
        metric_columns = st.columns(3)
        metric_columns[0].metric(
            "Selected ETA" if selected else "ETA",
            (
                f"{assessment.decision_eta_min:.0f} min"
                if assessment is not None
                else f"{decision.decision_time_min:.0f} min"
            ),
        )
        metric_columns[1].metric(
            "Distance",
            f"{decision.route.distance_m / 1000.0:.1f} km",
        )
        metric_columns[2].metric(
            "Traffic",
            _prediction_band_text(decision),
        )
        st.markdown("**Why choose this route**")
        st.markdown(_why_choose(decision, recommended))
        st.markdown("**Why not**")
        st.markdown(_why_not(decision, recommended, alternatives))
        if selected:
            st.button(
                f"Selected: {_route_label(decision)}",
                key=f"select-route-{decision.route_id}",
                disabled=True,
                use_container_width=True,
            )
        else:
            if st.button(
                f"Select {_route_label(decision)}",
                key=f"select-route-{decision.route_id}",
                use_container_width=True,
            ):
                _select_route(decision.route_id)


def _swap_locations() -> None:
    origin = st.session_state.get("origin_select")
    destination = st.session_state.get("destination_select")
    if origin is not None and destination is not None:
        st.session_state["origin_select"] = destination
        st.session_state["destination_select"] = origin


def _ensure_trip_state() -> None:
    landmarks = list(LANDMARKS)
    if "origin_select" not in st.session_state:
        st.session_state["origin_select"] = landmarks[0]
    if "destination_select" not in st.session_state:
        st.session_state["destination_select"] = landmarks[3]


def render_sidebar() -> tuple[
    str,
    str,
    datetime,
    str,
    bool,
    bool,
    bool,
]:
    with st.sidebar:
        st.header("Plan your trip")
        origin_name = st.selectbox(
            "From",
            list(LANDMARKS),
            key="origin_select",
        )
        destination_name = st.selectbox(
            "To",
            list(LANDMARKS),
            key="destination_select",
        )
        st.button(
            "Swap origin and destination",
            on_click=_swap_locations,
            use_container_width=True,
        )
        with st.form("trip_form"):
            departure_date = st.date_input(
                "Departure date",
                value=date.today() + timedelta(days=1),
            )
            departure_time = st.time_input(
                "Departure time",
                value=time(9, 0),
            )
            preference = st.selectbox(
                "Preference",
                ["Fastest", "Shortest", "Least congested", "Safest"],
            )
            avoid_tolls = st.checkbox("Strictly avoid tolls", value=False)
            include_mapbox = st.checkbox(
                "Use matched Mapbox traffic evidence",
                value=True,
            )
            submitted = st.form_submit_button(
                "Find Best Route",
                type="primary",
                use_container_width=True,
            )
    requested_departure = datetime.combine(departure_date, departure_time)
    return (
        origin_name,
        destination_name,
        requested_departure,
        preference,
        avoid_tolls,
        include_mapbox,
        submitted,
    )


def main() -> None:
    _apply_readable_styles()
    st.title("Karachi Route Advisor")
    _ensure_trip_state()
    (
        origin_name,
        destination_name,
        requested_departure,
        preference,
        avoid_tolls,
        include_mapbox,
        submitted,
    ) = render_sidebar()
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
        return
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
    selected = _selected_decision(final, selected_route_id)
    alternatives = [
        decision
        for decision in final.routes
        if decision.route_id != selected.route_id
    ][:MAX_ALTERNATIVES]

    st.markdown(f"### {response.origin} &rarr; {response.destination}", unsafe_allow_html=True)
    st.caption(
        f"{final.departure:%a %d %b %Y, %H:%M} | "
        f"Preference: {response.preference} | "
        f"Weather: {response.weather.condition}"
    )
    banner = _traffic_banner(response, final)
    if banner is not None:
        st.warning(banner)

    st.subheader("Map")
    render_map(response, final, selected_route_id)

    st.subheader("Selected route details")
    render_route_card(selected, final, selected_route_id, recommended, alternatives)
    st.subheader("Other best routes")
    if not alternatives:
        st.caption("No distinct alternative route is available.")
    for decision in alternatives:
        render_route_card(
            decision,
            final,
            selected_route_id,
            recommended,
            alternatives,
        )


if __name__ == "__main__":
    main()
