from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

from .config import (
    BBOX,
    GRAPH_PATH,
    LANDMARKS,
    MAPBOX_DRIVING_PROFILE,
    MAPBOX_PROFILE,
    PREFERENCE_WEIGHTS,
    SNAP_WARNING_DISTANCE_M,
    mapbox_token_present,
)
from .contracts import (
    BaselineComparison,
    DepartureForecast,
    FinalRecommendation,
    MapboxRoute,
    RouteAssessment,
    RouteCandidate,
    RouteResponse,
    RouteTrafficPrediction,
    RouteVerdict,
    TrafficAssessment,
    WeatherSnapshot,
)
from .decision import rank_routes
from .departure_forecast import DepartureForecastService
from .eta import (
    route_base_time_min,
    route_eta_min,
    route_impassable,
    route_risk,
    route_toll,
    route_unlit_ratio,
)
from .expert_system import evaluate_route
from .final_optimizer import build_final_recommendation
from .mapbox_client import MapboxError, fetch_routes, traffic_evidence_status
from .osm_graph import (
    graph_stats,
    load_graph,
    nearest_node_with_distance,
    path_geometry,
    to_routing_graph,
)
from .route_traffic_ml import RouteTrafficPredictionService
from .routing import dijkstra, generate_candidates, graph_vmax_kph
from .traffic_evidence import build_route_assessment, match_traffic_evidence
from .traffic_prediction import (
    EvidenceBasedTrafficPredictor,
    TrafficPredictor,
    degraded_assessment,
)

_KARACHI_TZ = timezone(timedelta(hours=5))


def _mapbox_depart_at(departure: datetime) -> str:
    if departure.tzinfo is None:
        local = departure.replace(tzinfo=_KARACHI_TZ)
    else:
        local = departure
    return local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RouteService:
    def __init__(
        self,
        graph_path: Any = GRAPH_PATH,
        traffic_predictor: TrafficPredictor | None = None,
        departure_forecast_service: DepartureForecastService | None = None,
        route_traffic_service: RouteTrafficPredictionService | None = None,
    ) -> None:
        self.graph = load_graph(graph_path)
        self.routing_graph = to_routing_graph(self.graph)
        self.vmax_kph = graph_vmax_kph(self.routing_graph)
        self.stats = graph_stats(self.graph)
        self.traffic_predictor = traffic_predictor or EvidenceBasedTrafficPredictor()
        self.departure_forecast_service = (
            departure_forecast_service or DepartureForecastService()
        )
        self.route_traffic_service = (
            route_traffic_service or RouteTrafficPredictionService()
        )

    def route(
        self,
        origin_name: str,
        destination_name: str,
        departure: datetime,
        preference: str,
        weather: WeatherSnapshot,
        avoid_tolls: bool = False,
        include_mapbox: bool = True,
    ) -> RouteResponse:
        if origin_name not in LANDMARKS:
            raise ValueError(f"Unknown origin: {origin_name}")
        if destination_name not in LANDMARKS:
            raise ValueError(f"Unknown destination: {destination_name}")
        if origin_name == destination_name:
            raise ValueError("Origin and destination must be different.")
        if preference not in PREFERENCE_WEIGHTS:
            raise ValueError(f"Unknown preference: {preference}")

        origin = LANDMARKS[origin_name]
        destination = LANDMARKS[destination_name]
        for name, point in ((origin_name, origin), (destination_name, destination)):
            if not BBOX.contains(*point):
                raise ValueError(f"{name} is outside the approved bbox.")

        start, origin_snap_m = nearest_node_with_distance(self.graph, *origin)
        goal, destination_snap_m = nearest_node_with_distance(
            self.graph, *destination
        )
        warnings: list[str] = []
        for label, name, distance in (
            ("Origin", origin_name, origin_snap_m),
            ("Destination", destination_name, destination_snap_m),
        ):
            if distance > SNAP_WARNING_DISTANCE_M:
                warnings.append(
                    f"{label} {name} is {distance:.0f} m from the nearest drive "
                    "road; routing starts at the snapped access point."
                )
        paths = generate_candidates(
            self.routing_graph,
            start,
            goal,
            vmax_kph=self.vmax_kph,
        )

        if not paths:
            warnings.append("No feasible route found in the selected OSM graph.")
            return RouteResponse(
                origin=origin_name,
                destination=destination_name,
                departure=departure,
                preference=preference,
                weather=weather,
                academic_routes=(),
                verdicts=(),
                ranked=(),
                mapbox_routes=(),
                warnings=tuple(warnings),
                provenance=self._provenance((), weather, "no_mapbox"),
                mapbox_verdicts=(),
                baseline=None,
                traffic_status="no_mapbox",
                avoid_tolls=avoid_tolls,
            )

        candidates = tuple(
            self._candidate(
                index=index,
                path=path,
                departure=departure,
                weather=weather,
            )
            for index, path in enumerate(paths, start=1)
        )
        baseline_path = dijkstra(self.routing_graph, start, goal)
        baseline = None
        if baseline_path is not None:
            baseline_route = self._candidate(
                index=0,
                path=baseline_path,
                departure=departure,
                weather=weather,
                route_id="dijkstra-baseline",
                source="dijkstra",
                provider="OSM Dijkstra baseline",
                warnings=("Dijkstra baseline; same OSM analytical ETA model.",),
            )
            baseline = BaselineComparison(
                dijkstra_route=baseline_route,
                astar_expanded_nodes=paths[0].expanded_nodes,
                dijkstra_expanded_nodes=baseline_path.expanded_nodes,
                same_path=paths[0].nodes == baseline_path.nodes,
                same_cost=abs(paths[0].base_time_s - baseline_path.base_time_s) < 1e-6,
            )
        if len(candidates) == 1:
            warnings.append(
                "No distinct alternative route was found for this origin-destination pair."
            )

        min_values = {
            "eta": min(route.eta_min for route in candidates),
            "distance": min(route.distance_m for route in candidates),
            "risk": min(route.risk for route in candidates),
            "congestion": None,
        }
        night = departure.hour >= 19 or departure.hour < 6
        verdicts: tuple[RouteVerdict, ...] = tuple(
            evaluate_route(
                route=route,
                preference=preference,
                min_values=min_values,
                night=night,
                weather_condition=weather.condition,
                avoid_tolls=avoid_tolls,
            )
            for route in candidates
        )
        ranked = tuple(rank_routes(candidates, verdicts, preference))

        mapbox_routes: tuple[MapboxRoute, ...] = ()
        mapbox_verdicts: tuple[RouteVerdict, ...] = ()
        traffic_status = "no_mapbox"
        requested_mapbox_departure = _mapbox_depart_at(departure)
        if include_mapbox:
            if mapbox_token_present():
                try:
                    mapbox_routes = fetch_routes(
                        origin=origin,
                        destination=destination,
                        profile=MAPBOX_PROFILE,
                        alternatives=True,
                        depart_at=requested_mapbox_departure,
                    )
                    traffic_status = traffic_evidence_status(mapbox_routes)
                    if traffic_status == "partial_eta_delta_only":
                        delta_notes = []
                        for route in mapbox_routes:
                            delta = route.duration_delta_s
                            if delta is not None and abs(delta) >= 1.0:
                                delta_notes.append(
                                    f"{route.route_id}: {delta:+.0f}s vs typical"
                                )
                        warnings.append(
                            "Mapbox driving-traffic returned an ETA delta versus "
                            "typical ("
                            + "; ".join(delta_notes)
                            + "), but no non-unknown congestion, incident, or "
                            "closure annotations. Treat the ETA as traffic-profile "
                            "evidence only; segment-level congestion is unavailable."
                        )
                    elif traffic_status == "no_traffic_evidence":
                        warnings.append(
                            "Mapbox returned driving-traffic results, but no "
                            "non-unknown congestion, incident, or closure "
                            "annotations and no ETA-versus-typical delta. Treat the "
                            "ETA as a traffic-profile estimate, not as proof of live "
                            "traffic."
                        )
                except MapboxError as traffic_error:
                    try:
                        mapbox_routes = fetch_routes(
                            origin=origin,
                            destination=destination,
                            profile=MAPBOX_DRIVING_PROFILE,
                            alternatives=True,
                            depart_at=requested_mapbox_departure,
                        )
                        traffic_status = traffic_evidence_status(mapbox_routes)
                        warnings.append(
                            "Mapbox driving-traffic was unavailable "
                            f"({traffic_error.code or 'error'}); an ordinary Mapbox "
                            "driving route is shown and is explicitly non-traffic-aware."
                        )
                    except MapboxError as driving_error:
                        traffic_status = "no_mapbox"
                        warnings.append(
                            "Mapbox unavailable "
                            f"(traffic: {traffic_error.code or 'error'}; "
                            f"driving: {driving_error.code or 'error'}); "
                            "using the OSM analytical fallback."
                        )
            else:
                warnings.append(
                    "MAPBOX_ACCESS_TOKEN is not set; using the OSM analytical fallback."
                )

        if mapbox_routes:
            mapbox_candidates = tuple(
                self._mapbox_candidate(route) for route in mapbox_routes
            )
            mapbox_verdicts = self._mapbox_verdicts(
                mapbox_candidates,
                preference,
                night,
                weather,
                avoid_tolls,
            )

        response = RouteResponse(
            origin=origin_name,
            destination=destination_name,
            departure=departure,
            preference=preference,
            weather=weather,
            academic_routes=candidates,
            verdicts=verdicts,
            ranked=ranked,
            mapbox_routes=mapbox_routes,
            warnings=tuple(warnings),
            provenance=self._provenance(mapbox_routes, weather, traffic_status),
            mapbox_verdicts=mapbox_verdicts,
            baseline=baseline,
            traffic_status=traffic_status,
            avoid_tolls=avoid_tolls,
        )
        return replace(
            response,
            final_recommendation=self.recommend_routes(
                response,
                departure,
                weather,
            ),
        )

    def recommend_routes(
        self,
        response: RouteResponse,
        departure: datetime,
        weather: WeatherSnapshot | None = None,
    ) -> FinalRecommendation:
        """Run the additive multi-factor final route policy.

        The existing A*/Dijkstra outputs, academic ranking, and R01-R16
        verdicts are consumed unchanged. Each eligible custom candidate gets
        its own prediction for the requested departure.
        """

        selected_weather = weather or response.weather
        verdict_by_id = {verdict.route_id: verdict for verdict in response.verdicts}
        predictions = {}
        prediction_errors: dict[str, str] = {}
        for route in response.academic_routes:
            verdict = verdict_by_id.get(route.route_id)
            if verdict is not None and verdict.vetoed:
                continue
            try:
                predictions[route.route_id] = self.predict_route_traffic(
                    response,
                    route.route_id,
                    departure,
                    selected_weather,
                )
            except Exception as exc:
                prediction_errors[route.route_id] = (
                    f"{type(exc).__name__}: {exc}"
                )
                predictions[route.route_id] = None
        mapbox_evidence_is_current = departure == response.departure
        evidence_routes = (
            response.mapbox_routes if mapbox_evidence_is_current else ()
        )
        mapbox_candidates = (
            tuple(
                self._mapbox_candidate(route) for route in response.mapbox_routes
            )
            if mapbox_evidence_is_current
            else ()
        )
        mapbox_verdicts = (
            response.mapbox_verdicts if mapbox_evidence_is_current else ()
        )
        if mapbox_candidates and not mapbox_verdicts:
            mapbox_verdicts = self._mapbox_verdicts(
                mapbox_candidates,
                response.preference,
                departure.hour >= 19 or departure.hour < 6,
                selected_weather,
                response.avoid_tolls,
            )
        assessments: dict[str, RouteAssessment] = {}
        for route in response.academic_routes:
            verdict = verdict_by_id.get(route.route_id)
            if verdict is None or verdict.vetoed:
                continue
            evidence = match_traffic_evidence(
                route=route,
                mapbox_routes=evidence_routes,
                graph=self.routing_graph,
                departure=departure,
                weather_condition=selected_weather.condition,
            )
            assessments[route.route_id] = build_route_assessment(
                route=route,
                verdict=verdict,
                prediction=predictions.get(route.route_id),
                traffic_evidence=evidence,
                departure=departure,
                graph=self.routing_graph,
                weather_condition=selected_weather.condition,
            )
        final = build_final_recommendation(
            routes=response.academic_routes,
            verdicts=response.verdicts,
            academic_ranked=response.ranked,
            predictions=predictions,
            departure=departure,
            preference=response.preference,
            prediction_errors=prediction_errors,
            mapbox_candidates=mapbox_candidates,
            mapbox_verdicts=mapbox_verdicts,
            avoid_tolls=response.avoid_tolls,
            assessments=assessments,
        )
        extra_warnings: list[str] = []
        if response.mapbox_routes and not mapbox_evidence_is_current:
            extra_warnings.append(
                "Mapbox evidence was excluded from the refreshed AI decision "
                "because it was fetched for the previous departure. Fetch Mapbox "
                "evidence again for the new departure."
            )
        if prediction_errors:
            extra_warnings.append(
                "Future-traffic prediction failed for at least one route; that "
                "route uses the analytical OSM ETA without inventing evidence."
            )
        if extra_warnings:
            final = replace(
                final,
                warnings=final.warnings + tuple(extra_warnings),
            )
        return final

    def assess_traffic(
        self,
        response: RouteResponse,
        route_id: str,
        departure: datetime,
        weather: WeatherSnapshot | None = None,
    ) -> TrafficAssessment:
        selected_weather = weather or response.weather
        academic = next(
            (
                route
                for route in response.academic_routes
                if route.route_id == route_id
            ),
            None,
        )
        mapbox = next(
            (
                route
                for route in response.mapbox_routes
                if route.route_id == route_id
            ),
            None,
        )
        if academic is None and mapbox is None:
            return degraded_assessment(
                route_id=route_id,
                route_source="unknown",
                distance_km=0.0,
                departure=departure,
                reason=f"Selected route {route_id} was not found.",
            )

        if mapbox is not None and departure != response.departure:
            return degraded_assessment(
                route_id=route_id,
                route_source="mapbox",
                distance_km=mapbox.distance_m / 1000.0,
                departure=departure,
                reason=(
                    "Mapbox evidence was fetched for the previous departure. "
                    "Click Find Best Routes to refresh Mapbox observations before "
                    "assessing this route."
                ),
            )

        try:
            if academic is not None:
                return self.traffic_predictor.assess(
                    route=academic,
                    mapbox_route=None,
                    departure=departure,
                    weather=selected_weather,
                    graph=self.routing_graph,
                )
            return self.traffic_predictor.assess(
                route=None,
                mapbox_route=mapbox,
                departure=departure,
                weather=selected_weather,
                graph=None,
                evidence_status=traffic_evidence_status((mapbox,)),
            )
        except Exception as exc:
            if academic is not None:
                source = academic.source
                distance_km = academic.distance_m / 1000.0
            elif mapbox is not None:
                source = "mapbox"
                distance_km = mapbox.distance_m / 1000.0
            else:
                source = "unknown"
                distance_km = 0.0
            return degraded_assessment(
                route_id=route_id,
                route_source=source,
                distance_km=distance_km,
                departure=departure,
                reason=f"{type(exc).__name__}: {exc}",
            )

    def forecast_departure(
        self,
        response: RouteResponse,
        route_id: str,
        departure: datetime,
        weather: WeatherSnapshot | None = None,
    ) -> DepartureForecast:
        selected_weather = weather or response.weather
        academic = next(
            (
                route
                for route in response.academic_routes
                if route.route_id == route_id
            ),
            None,
        )
        mapbox = next(
            (
                route
                for route in response.mapbox_routes
                if route.route_id == route_id
            ),
            None,
        )
        if academic is None and mapbox is None:
            raise ValueError(f"Selected route {route_id} was not found.")
        current_assessment = None
        if mapbox is not None:
            current_assessment = self.assess_traffic(
                response,
                route_id,
                departure,
                selected_weather,
            )
        return self.departure_forecast_service.forecast(
            route=academic if academic is not None else mapbox,
            departure=departure,
            graph=self.routing_graph if academic is not None else None,
            weather_condition=selected_weather.condition,
            current_mapbox_assessment=current_assessment,
        )

    def predict_route_traffic(
        self,
        response: RouteResponse,
        route_id: str,
        departure: datetime,
        weather: WeatherSnapshot | None = None,
    ) -> RouteTrafficPrediction:
        selected_weather = weather or response.weather
        academic = next(
            (
                route
                for route in response.academic_routes
                if route.route_id == route_id
            ),
            None,
        )
        mapbox = next(
            (
                route
                for route in response.mapbox_routes
                if route.route_id == route_id
            ),
            None,
        )
        if academic is None and mapbox is None:
            raise ValueError(f"Selected route {route_id} was not found.")
        if academic is None:
            assert mapbox is not None
            return self.route_traffic_service.predict(
                route=mapbox,
                departure=departure,
                graph=None,
                weather_condition=selected_weather.condition,
            )
        return self.route_traffic_service.predict(
            route=academic,
            departure=departure,
            graph=self.routing_graph,
            weather_condition=selected_weather.condition,
        )

    def compare_departures(
        self,
        response: RouteResponse,
        departure: datetime,
        weather: WeatherSnapshot | None = None,
        route_ids: tuple[str, ...] | None = None,
    ) -> tuple[DepartureForecast, ...]:
        selected = set(route_ids) if route_ids else None
        forecasts: list[DepartureForecast] = []
        for route in response.academic_routes:
            if selected is not None and route.route_id not in selected:
                continue
            forecasts.append(
                self.forecast_departure(
                    response, route.route_id, departure, weather
                )
            )
        for route in response.mapbox_routes:
            if selected is not None and route.route_id not in selected:
                continue
            forecasts.append(
                self.forecast_departure(
                    response, route.route_id, departure, weather
                )
            )
        return tuple(forecasts)

    def _candidate(
        self,
        index: int,
        path: Any,
        departure: datetime,
        weather: WeatherSnapshot,
        *,
        route_id: str | None = None,
        source: str = "custom_astar",
        provider: str = "OSM analytical",
        warnings: tuple[str, ...] | None = None,
    ) -> RouteCandidate:
        eta_min = route_eta_min(
            self.routing_graph,
            path.nodes,
            departure,
            weather.condition,
        )
        risk = route_risk(self.routing_graph, path.nodes, departure)
        geometry = path_geometry(self.routing_graph, path.nodes)
        confidence = 0.55
        if weather.source == "Open-Meteo":
            confidence += 0.05
        return RouteCandidate(
            route_id=route_id or f"osm-{index}",
            source=source,
            provider=provider,
            geometry=geometry,
            distance_m=path.distance_m,
            eta_min=eta_min,
            risk=risk,
            confidence=min(confidence, 0.60),
            congestion=None,
            incident_count=0,
            closure_count=0,
            traffic_available=False,
            nodes=path.nodes,
            edges=path.edges,
            warnings=warnings
            or ("OSM analytical ETA; no live traffic weights.",),
            toll=route_toll(self.routing_graph, path.nodes),
            impassable=route_impassable(self.routing_graph, path.nodes),
            unlit_ratio=route_unlit_ratio(self.routing_graph, path.nodes),
            free_flow_eta_min=route_base_time_min(
                self.routing_graph,
                path.nodes,
            ),
        )

    def _mapbox_verdicts(
        self,
        candidates: tuple[RouteCandidate, ...],
        preference: str,
        night: bool,
        weather: WeatherSnapshot,
        avoid_tolls: bool,
    ) -> tuple[RouteVerdict, ...]:
        congestion_values = [
            route.congestion
            for route in candidates
            if route.congestion is not None
        ]
        min_values = {
            "eta": min(route.eta_min for route in candidates),
            "distance": min(route.distance_m for route in candidates),
            "risk": None,
            "congestion": min(congestion_values) if congestion_values else None,
        }
        return tuple(
            evaluate_route(
                route=route,
                preference=preference,
                min_values=min_values,
                night=night,
                weather_condition=weather.condition,
                avoid_tolls=avoid_tolls,
                excluded_rule_ids=frozenset({"R04", "R05", "R13", "R16"}),
            )
            for route in candidates
        )

    def _mapbox_candidate(self, route: MapboxRoute) -> RouteCandidate:
        traffic_profile = route.profile == MAPBOX_PROFILE
        annotations_available = (
            route.traffic_evidence or route.duration_delta_s is not None
        )
        if traffic_profile:
            evidence_kind = "mapbox_traffic_eta"
            evidence_label = (
                "Mapbox traffic-aware ETA for the requested departure"
            )
        else:
            evidence_kind = "mapbox_driving_eta"
            evidence_label = "Mapbox ordinary driving ETA; not traffic-aware"
        return RouteCandidate(
            route_id=route.route_id,
            source="mapbox",
            provider="Mapbox",
            geometry=route.geometry,
            distance_m=route.distance_m,
            eta_min=route.duration_s / 60.0,
            risk=0.0,
            confidence=0.70 if annotations_available else 0.60,
            congestion=route.congestion_average,
            incident_count=route.incident_count,
            closure_count=route.closure_count,
            traffic_available=traffic_profile,
            warnings=(
                "Mapbox ETA is provider evidence for the requested departure; "
                "it is not the OSM future-traffic ML forecast.",
            ),
            risk_available=False,
            toll_available=False,
            free_flow_eta_min=(
                route.duration_typical_s / 60.0
                if route.duration_typical_s is not None
                else None
            ),
            eta_evidence_kind=evidence_kind,
            eta_evidence_label=evidence_label,
        )

    def _provenance(
        self,
        mapbox_routes: tuple[MapboxRoute, ...],
        weather: WeatherSnapshot,
        traffic_status: str,
    ) -> tuple[str, ...]:
        items = [
            "OSM road network: © OpenStreetMap contributors (ODbL).",
            "Custom A* and Dijkstra baseline: same OSM-derived weights; no live "
            "traffic weights. Dijkstra is a reference comparison, not a second "
            "traffic source.",
        ]
        if mapbox_routes:
            if any(route.profile == MAPBOX_PROFILE for route in mapbox_routes):
                if traffic_status == "annotated":
                    items.append(
                        "Mapbox Directions API driving-traffic: provider ETA requested "
                        "at the selected departure, with segment-level annotations; it "
                        "competes once in the final AI layer and is not used as OSM ML "
                        "evidence. Results are transient and are not cached or exported."
                    )
                elif traffic_status == "partial_eta_delta_only":
                    items.append(
                        "Mapbox Directions API driving-traffic: selected-departure ETA "
                        "returned with an ETA-versus-typical delta, but segment-level "
                        "annotations were unavailable; it competes as provider ETA "
                        "evidence and is not the OSM ML forecast. Results are transient."
                    )
                else:
                    items.append(
                        "Mapbox Directions API driving-traffic: selected-departure ETA "
                        "returned, but neither annotations nor an ETA-versus-typical "
                        "delta was available; it competes as provider ETA evidence and "
                        "is not the OSM ML forecast. Results are transient."
                    )
            else:
                items.append(
                    "Mapbox Directions API driving: ordinary non-traffic-aware route; "
                    "it is comparison-only and cannot win the final AI recommendation. "
                    "Results are transient and are not cached or exported."
                )
            items.append("Mapbox attribution: © Mapbox © OpenStreetMap.")
        elif mapbox_token_present():
            items.append("Mapbox token configured; no Mapbox route returned for this request.")
        else:
            items.append("Mapbox token not configured; OSM analytical fallback in use.")
        if weather.source == "Open-Meteo":
            items.append("Weather: Open-Meteo (CC BY 4.0).")
        else:
            items.append("Weather: fallback dry assumption; no live weather data.")
        return tuple(items)
