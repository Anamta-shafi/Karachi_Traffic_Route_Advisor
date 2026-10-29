from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from .contracts import (
    MapboxRoute,
    RouteCandidate,
    TrafficAssessment,
    WeatherSnapshot,
)
from .eta import route_base_time_min, route_eta_min
from .mapbox_client import traffic_evidence_status


class TrafficPredictionError(RuntimeError):
    pass


def degraded_assessment(
    *,
    route_id: str,
    route_source: str,
    distance_km: float,
    departure: datetime,
    reason: str,
) -> TrafficAssessment:
    return TrafficAssessment(
        route_id=route_id,
        route_source=route_source,
        distance_km=distance_km,
        requested_departure=departure,
        ordinary_eta_min=None,
        ordinary_eta_label="Unavailable in degraded mode",
        observed_eta_min=None,
        observed_eta_label="Unavailable in degraded mode",
        predicted_eta_min=None,
        predicted_eta_label="Unavailable in degraded mode",
        baseline_label="Unavailable in degraded mode",
        baseline_eta_min=None,
        delay_vs_baseline_min=None,
        delay_basis="Unavailable in degraded mode.",
        traffic_level="unknown",
        traffic_level_basis="Unavailable in degraded mode.",
        evidence_status="no_route_specific_evidence",
        current_evidence=(reason,),
        confidence_label="Not applicable: assessment failed.",
        explanation=(
            f"Traffic assessment failed for the selected route: {reason}. No "
            "traffic prediction or traffic level is shown."
        ),
        limitations=(
            "The route-specific assessment could not be completed.",
            "No trained machine-learning traffic model is used.",
        ),
        model_kind="evidence_fallback_no_trained_model",
        trained=False,
        historical_data_available=False,
        status="degraded",
    )


class TrafficPredictor(Protocol):
    def assess(
        self,
        *,
        route: RouteCandidate | None,
        mapbox_route: MapboxRoute | None,
        departure: datetime,
        weather: WeatherSnapshot,
        graph: Any | None = None,
        evidence_status: str | None = None,
    ) -> TrafficAssessment:
        ...


class EvidenceBasedTrafficPredictor:
    model_kind = "evidence_fallback_no_trained_model"
    trained = False
    historical_data_available = False

    def assess(
        self,
        *,
        route: RouteCandidate | None,
        mapbox_route: MapboxRoute | None,
        departure: datetime,
        weather: WeatherSnapshot,
        graph: Any | None = None,
        evidence_status: str | None = None,
    ) -> TrafficAssessment:
        if mapbox_route is not None:
            return self._assess_mapbox(
                mapbox_route=mapbox_route,
                departure=departure,
                evidence_status=evidence_status,
            )
        if route is not None:
            return self._assess_custom(
                route=route,
                departure=departure,
                weather=weather,
                graph=graph,
            )
        raise TrafficPredictionError("No route was supplied for traffic assessment.")

    def _assess_custom(
        self,
        *,
        route: RouteCandidate,
        departure: datetime,
        weather: WeatherSnapshot,
        graph: Any | None,
    ) -> TrafficAssessment:
        degraded = graph is None or not route.nodes
        if degraded:
            ordinary_eta = route.eta_min
            baseline_eta = None
            road_composition: tuple[tuple[str, float], ...] = ()
        else:
            ordinary_eta = route_eta_min(
                graph,
                route.nodes,
                departure,
                weather.condition,
            )
            baseline_eta = route_base_time_min(graph, route.nodes)
            road_composition = self._road_composition(graph, route.nodes)
        delay = (
            ordinary_eta - baseline_eta
            if baseline_eta is not None
            else None
        )
        status_note = (
            "The OSM routing graph was unavailable, so the request-time analytical "
            "ETA is reused."
            if degraded
            else "The OSM routing graph was used to recompute the analytical ETA."
        )
        limitations = [
            "No timestamped historical travel-time observations are available for "
            "this route or its segments.",
            "No route-specific live traffic observations are available for this "
            "custom A* route.",
            "The OSM analytical ETA has not been validated against observed traffic.",
            "Weather is a current observation, not a forecast for the selected "
            "departure.",
            "No trained machine-learning traffic model is used.",
        ]
        if degraded:
            limitations.append(
                "The assessment is degraded because the routing graph was not "
                "available."
            )
        return TrafficAssessment(
            route_id=route.route_id,
            route_source=route.source,
            distance_km=route.distance_m / 1000.0,
            requested_departure=departure,
            ordinary_eta_min=ordinary_eta,
            ordinary_eta_label="OSM analytical estimate at selected departure",
            observed_eta_min=None,
            observed_eta_label="Not available for this route",
            predicted_eta_min=None,
            predicted_eta_label=(
                "Not available: no historical traffic model for this route."
            ),
            baseline_label="OSM free-flow base time (posted/default speeds)",
            baseline_eta_min=baseline_eta,
            delay_vs_baseline_min=delay,
            delay_basis=(
                "Analytical time-of-day/weather adjustment versus free-flow base "
                "time; not observed traffic."
            ),
            traffic_level="unknown",
            traffic_level_basis="No route-specific live traffic evidence.",
            evidence_status="no_route_specific_evidence",
            current_evidence=(
                "No route-specific current traffic observations are available for "
                "this custom A* route.",
                "Mapbox routes are separate geometries; their annotations are not "
                "applied to this route.",
                "OSM edge attributes contain no timestamped traffic observations.",
            ),
            confidence_label=(
                "Not applicable: no trained traffic model; analytical estimate only."
            ),
            explanation=(
                f"Custom A* route {route.route_id} is estimated at "
                f"{ordinary_eta:.1f} min for "
                f"{departure:%A %H:%M} using OSM distance, road composition, "
                "departure-time, and weather factors. No observed traffic or "
                "trained prediction is available for this exact route. The delay "
                "versus free-flow is an analytical adjustment, not measured "
                f"congestion. {status_note}"
            ),
            limitations=tuple(limitations),
            road_composition=road_composition,
            model_kind=self.model_kind,
            trained=self.trained,
            historical_data_available=self.historical_data_available,
            features=(
                "route distance",
                "road composition",
                "departure hour",
                "day of week",
                "weather condition",
                "OSM base speed",
            ),
            status="degraded" if degraded else "ok",
        )

    def _assess_mapbox(
        self,
        *,
        mapbox_route: MapboxRoute,
        departure: datetime,
        evidence_status: str | None,
    ) -> TrafficAssessment:
        if mapbox_route.duration_s <= 0:
            raise TrafficPredictionError(
                f"Mapbox route {mapbox_route.route_id} has no usable duration."
            )
        status = evidence_status or traffic_evidence_status((mapbox_route,))
        observed_eta = mapbox_route.duration_s / 60.0
        baseline_eta = (
            mapbox_route.duration_typical_s / 60.0
            if mapbox_route.duration_typical_s is not None
            else None
        )
        delta_s = mapbox_route.duration_delta_s
        delta_min = delta_s / 60.0 if delta_s is not None else None
        delta_pct = mapbox_route.duration_delta_pct
        congestion = mapbox_route.congestion_average
        traffic_level = (
            self._traffic_level(congestion)
            if congestion is not None and status == "annotated"
            else "unknown"
        )
        if traffic_level == "unknown":
            level_basis = "Unavailable: no non-unknown congestion annotations."
        else:
            level_basis = "Derived from non-unknown Mapbox congestion annotations."

        evidence = [
            f"Selected Mapbox route {mapbox_route.route_id} "
            f"({mapbox_route.profile}) observed at "
            f"{mapbox_route.fetched_at or 'request time'}.",
            f"Observed Mapbox ETA: {observed_eta:.1f} min.",
        ]
        if baseline_eta is not None:
            delay_text = (
                f"; observed delay {delta_min:+.1f} min"
                if delta_min is not None
                else ""
            )
            evidence.append(
                f"Mapbox typical duration: {baseline_eta:.1f} min{delay_text}."
            )
        if delta_pct is not None:
            evidence.append(f"ETA versus typical: {delta_pct:+.1f}%.")
        if congestion is not None:
            evidence.append(
                f"Average non-unknown congestion annotation: {congestion:.0%}."
            )
        evidence.append(
            f"Incidents: {mapbox_route.incident_count}; "
            f"closures: {mapbox_route.closure_count}."
        )
        if status == "partial_eta_delta_only":
            evidence.append(
                "No non-unknown congestion, incident, or closure annotations; only "
                "the ETA-versus-typical delta is available."
            )
        elif status == "no_traffic_evidence":
            evidence.append(
                "No non-unknown congestion, incident, or closure annotations and no "
                "ETA-versus-typical delta."
            )
        if mapbox_route.profile != "mapbox/driving-traffic":
            evidence.append(
                "The response uses the ordinary Mapbox driving profile and is "
                "non-traffic-aware."
            )

        if status == "annotated":
            explanation = (
                f"The selected Mapbox route has request-time traffic evidence. "
                f"The observed ETA is {observed_eta:.1f} min and the traffic level "
                f"is {traffic_level}. This is current/observed evidence for the "
                "exact selected route, not a forecast for the selected departure. "
                "No trained model is used."
            )
        elif status == "partial_eta_delta_only":
            explanation = (
                "The selected Mapbox route returned only an ETA-versus-typical "
                "delta. That is evidence that the traffic profile changed the ETA, "
                "but it does not provide a segment-level congestion level. The "
                "traffic level is therefore unknown, and no departure-time "
                "prediction is made."
            )
        else:
            explanation = (
                "The selected Mapbox route returned no usable traffic evidence. "
                "The ETA is displayed as a requested-departure Mapbox estimate, but "
                "no traffic level or departure-time prediction is claimed."
            )

        return TrafficAssessment(
            route_id=mapbox_route.route_id,
            route_source="mapbox",
            distance_km=mapbox_route.distance_m / 1000.0,
            requested_departure=departure,
            ordinary_eta_min=None,
            ordinary_eta_label="Not applicable: selected route is a Mapbox route.",
            observed_eta_min=observed_eta,
            observed_eta_label="Mapbox ETA for the requested departure",
            predicted_eta_min=None,
            predicted_eta_label=(
                "Not available: no historical traffic model or departure-time "
                "forecast for this route."
            ),
            baseline_label="Mapbox typical duration",
            baseline_eta_min=baseline_eta,
            delay_vs_baseline_min=delta_min,
            delay_basis=(
                "Mapbox requested-departure ETA minus Mapbox typical duration "
                "(provider evidence)."
            ),
            traffic_level=traffic_level,
            traffic_level_basis=level_basis,
            evidence_status=status,
            current_evidence=tuple(evidence),
            confidence_label=(
                "Not applicable: observed request-time evidence; no predictive "
                "model confidence."
            ),
            explanation=explanation,
            limitations=(
                "Mapbox evidence is request-time, not historical.",
                "Mapbox uses its default traffic departure; it is not aligned to "
                "the selected departure.",
                "Mapbox annotations may be incomplete and are not a citywide "
                "incident feed.",
                "No timestamped historical traffic observations are available, so "
                "no trained model or departure-time prediction is used.",
            ),
            model_kind=self.model_kind,
            trained=self.trained,
            historical_data_available=self.historical_data_available,
            features=(
                "Mapbox duration",
                "Mapbox typical duration",
                "congestion annotations",
                "incidents",
                "closures",
                "route distance",
            ),
            status="ok",
        )

    def _road_composition(
        self,
        graph: Any,
        nodes: tuple[int, ...],
    ) -> tuple[tuple[str, float], ...]:
        lengths: dict[str, float] = {}
        total = 0.0
        for start, end in zip(nodes, nodes[1:]):
            data = graph.edges[start, end]
            road = data.get("road_type", data.get("highway", "unclassified"))
            if isinstance(road, (list, tuple)):
                road = road[0] if road else "unclassified"
            length = float(data.get("length_m", data.get("length", 0.0)))
            lengths[str(road)] = lengths.get(str(road), 0.0) + length
            total += length
        if total <= 0:
            return ()
        return tuple(
            sorted(
                (
                    (road, length / total)
                    for road, length in lengths.items()
                ),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    def _traffic_level(self, congestion: float) -> str:
        if congestion < 0.35:
            return "low"
        if congestion < 0.60:
            return "moderate"
        if congestion < 0.85:
            return "heavy"
        return "severe"
