from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

from .calendar_features import FEATURE_NAMES, calendar_feature_frame
from .contracts import (
    DepartureForecast,
    ForecastComponent,
    MapboxRoute,
    RouteCandidate,
    RouteContext,
    TrafficAssessment,
)
from .route_features import extract_route_context

BAND_RANK = {"low": 0, "moderate": 1, "high": 2}
MAPBOX_BAND = {
    "low": "low",
    "moderate": "moderate",
    "heavy": "high",
    "severe": "high",
}
DISCLOSURE = (
    "ML model trained on external traffic data — not locally validated for "
    "Gulistan-e-Jauhar."
)
ACCIDENT_DISCLOSURE = (
    "Accident-risk model trained on external UK STATS19 collision data — not "
    "locally validated for Gulistan-e-Jauhar and not corridor-matched to Karachi."
)


class DepartureForecastError(RuntimeError):
    pass


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _band_for_probability(probabilities: np.ndarray, index: int) -> str:
    names = ("low", "moderate", "high")
    return names[int(index)] if 0 <= int(index) < len(names) else "unknown"


def _worse(first: str, second: str) -> str:
    if first == "unknown":
        return second
    if second == "unknown":
        return first
    return first if BAND_RANK[first] >= BAND_RANK[second] else second


def _component_unavailable(
    *,
    kind: str,
    model_version: str,
    dataset_name: str = "Unavailable",
    dataset_license: str = "Unavailable",
    coverage: str = "Unavailable",
    reason: str,
) -> ForecastComponent:
    return ForecastComponent(
        available=False,
        band="unknown",
        probability=None,
        value=None,
        units="unavailable",
        model_kind=kind,
        model_version=model_version,
        dataset_name=dataset_name,
        dataset_license=dataset_license,
        coverage=coverage,
        status="unavailable",
        reasons=(reason,),
        limitations=(reason,),
    )


def _traffic_component(
    artifact: dict[str, Any],
    metadata: dict[str, Any],
    departure: datetime,
) -> ForecastComponent:
    feature_frame = calendar_feature_frame(departure)
    classifier = artifact["classifier"]
    regressor = artifact["regressor"]
    probabilities = np.asarray(classifier.predict_proba(feature_frame), dtype=float)[0]
    predicted_class = int(np.argmax(probabilities))
    value = float(np.asarray(regressor.predict(feature_frame), dtype=float)[0])
    traffic_metadata = metadata["traffic"]
    dataset = traffic_metadata["dataset"]
    return ForecastComponent(
        available=True,
        band=_band_for_probability(probabilities, predicted_class),
        probability=float(probabilities[predicted_class]),
        value=value,
        units="vehicles per hour",
        model_kind="external_calendar_traffic",
        model_version=artifact["model_version"],
        dataset_name=dataset["name"],
        dataset_license=dataset["license"],
        coverage=dataset["geographic_coverage"],
        status="ok",
        reasons=(
            "Calendar-only external model. It predicts an I-94 traffic-volume band, "
            "not a Karachi route ETA or speed.",
        ),
        limitations=tuple(traffic_metadata["limitations"]),
    )


def _accident_component(
    artifact: dict[str, Any],
    metadata: dict[str, Any],
    departure: datetime,
) -> ForecastComponent:
    feature_frame = calendar_feature_frame(departure)
    classifier = artifact["classifier"]
    regressor = artifact["regressor"]
    probabilities = np.asarray(classifier.predict_proba(feature_frame), dtype=float)[0]
    predicted_class = int(np.argmax(probabilities))
    value = float(np.asarray(regressor.predict(feature_frame), dtype=float)[0])
    accident_metadata = metadata["accident"]
    dataset = accident_metadata["dataset"]
    return ForecastComponent(
        available=True,
        band=_band_for_probability(probabilities, predicted_class),
        probability=float(probabilities[predicted_class]),
        value=max(0.0, value),
        units="severity-weighted collisions per hour",
        model_kind="external_calendar_accident",
        model_version=artifact["model_version"],
        dataset_name=dataset["name"],
        dataset_license=dataset["license"],
        coverage=dataset["geographic_coverage"],
        status="ok",
        reasons=(
            "Calendar-only Great Britain collision-time prior. It is not a Karachi "
            "crash history, not corridor-matched, and not a crash probability for "
            "the selected route.",
        ),
        limitations=tuple(accident_metadata["limitations"]),
    )


def _final_assessment(
    *,
    traffic_band: str,
    accident_band: str,
    route_context: RouteContext,
    current_mapbox_assessment: TrafficAssessment | None,
    route_source: str,
    route_id: str,
) -> tuple[str, bool, str]:
    final = _worse(traffic_band, accident_band)
    if route_context.context_band == "high" and final == "low":
        final = "moderate"

    applied = (
        current_mapbox_assessment is not None
        and current_mapbox_assessment.route_source == "mapbox"
        and route_source == "mapbox"
        and current_mapbox_assessment.route_id == route_id
    )
    if applied:
        mapbox_band = MAPBOX_BAND.get(
            current_mapbox_assessment.traffic_level, "unknown"
        )
        final = _worse(final, mapbox_band)
    return final, applied, (
        current_mapbox_assessment.traffic_level
        if current_mapbox_assessment is not None
        else "not_applied"
    )


def _recommendation(final: str) -> str:
    if final == "high":
        return (
            "High predicted calendar-slot risk. Consider shifting departure outside "
            "this window; treat this as an external-data signal, not a "
            "Karachi-validated forecast."
        )
    if final == "moderate":
        return (
            "Moderate predicted calendar-slot risk. Compare routes and allow extra "
            "time; check requested-departure Mapbox evidence before departing."
        )
    if final == "low":
        return (
            "Low predicted calendar-slot risk. Still check requested-departure "
            "Mapbox evidence and route context before departing."
        )
    return (
        "Forecast unavailable. Rely on requested-departure Mapbox evidence and "
        "route context; no ML departure-time claim is made."
    )


class DepartureForecastService:
    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = model_dir or (
            Path(__file__).resolve().parents[1] / "models" / "departure_forecast"
        )
        self._metadata = _read_json(self.model_dir / "metadata.json")
        self._traffic_artifact: dict[str, Any] | None = None
        self._accident_artifact: dict[str, Any] | None = None
        self._load_error: str | None = None
        self._load()

    def _load(self) -> None:
        if self._metadata is None:
            self._load_error = "metadata.json is missing or invalid."
            return
        try:
            self._traffic_artifact = joblib.load(
                self.model_dir / "traffic_model.joblib"
            )
            self._accident_artifact = joblib.load(
                self.model_dir / "accident_model.joblib"
            )
        except (OSError, ValueError, EOFError) as exc:
            self._load_error = f"Model artifacts could not be loaded: {exc}"
            self._traffic_artifact = None
            self._accident_artifact = None
            return
        for artifact in (self._traffic_artifact, self._accident_artifact):
            if artifact is None:
                self._load_error = "A required model artifact is missing."
                return
            if tuple(artifact.get("feature_names", ())) != FEATURE_NAMES:
                self._load_error = "Model feature schema does not match the code."
                return

    @property
    def available(self) -> bool:
        return (
            self._metadata is not None
            and self._traffic_artifact is not None
            and self._accident_artifact is not None
            and self._load_error is None
        )

    def _metadata_for(self, key: str) -> dict[str, Any]:
        if self._metadata is None:
            return {}
        value = self._metadata.get(key)
        return value if isinstance(value, dict) else {}

    def _unavailable_forecast(
        self,
        *,
        route: RouteCandidate | MapboxRoute,
        departure: datetime,
        graph: Any | None,
        weather_condition: str,
        current_mapbox_assessment: TrafficAssessment | None,
        reason: str,
    ) -> DepartureForecast:
        context = extract_route_context(
            route,
            graph=graph,
            departure=departure,
            weather_condition=weather_condition,
        )
        model_version = (
            self._metadata.get("model_version", "unavailable")
            if self._metadata
            else "unavailable"
        )
        traffic = _component_unavailable(
            kind="external_calendar_traffic",
            model_version=model_version,
            reason=reason,
        )
        accident = _component_unavailable(
            kind="external_calendar_accident",
            model_version=model_version,
            reason=reason,
        )
        route_source = (
            route.source if isinstance(route, RouteCandidate) else "mapbox"
        )
        final, applied, mapbox_level = _final_assessment(
            traffic_band=traffic.band,
            accident_band=accident.band,
            route_context=context,
            current_mapbox_assessment=current_mapbox_assessment,
            route_source=route_source,
            route_id=route.route_id,
        )
        return DepartureForecast(
            route_id=route.route_id,
            route_source=route_source,
            requested_departure=departure,
            status="unavailable",
            final_assessment=final,
            predicted_traffic=traffic.band,
            traffic_probability=None,
            accident_risk=accident.band,
            accident_probability=None,
            current_mapbox_traffic=mapbox_level,
            current_mapbox_basis=(
                current_mapbox_assessment.traffic_level_basis
                if current_mapbox_assessment is not None
                else "No requested-departure Mapbox assessment was applied."
            ),
            current_mapbox_applied=applied,
            recommendation=_recommendation(final),
            route_context=context,
            traffic=traffic,
            accident=accident,
            evidence_status="model_unavailable",
            model_version=model_version,
            disclosure=(DISCLOSURE, ACCIDENT_DISCLOSURE),
            explanation=(
                "The experimental departure-time model is unavailable, so no ML "
                "traffic or accident prediction is shown. Route context and current "
                "evidence are reported separately."
            ),
            limitations=(reason, "No ML prediction is claimed."),
            provenance=(),
            generated_at=_now_utc(),
        )

    def forecast(
        self,
        *,
        route: RouteCandidate | MapboxRoute,
        departure: datetime,
        graph: Any | None = None,
        weather_condition: str = "dry",
        current_mapbox_assessment: TrafficAssessment | None = None,
    ) -> DepartureForecast:
        if not self.available:
            return self._unavailable_forecast(
                route=route,
                departure=departure,
                graph=graph,
                weather_condition=weather_condition,
                current_mapbox_assessment=current_mapbox_assessment,
                reason=self._load_error or "Experimental model is unavailable.",
            )
        assert self._traffic_artifact is not None
        assert self._accident_artifact is not None
        assert self._metadata is not None
        try:
            traffic = _traffic_component(
                self._traffic_artifact, self._metadata, departure
            )
        except (KeyError, ValueError, TypeError) as exc:
            traffic = _component_unavailable(
                kind="external_calendar_traffic",
                model_version=self._metadata.get("model_version", "unavailable"),
                reason=f"Traffic component failed: {exc}",
            )
        try:
            accident = _accident_component(
                self._accident_artifact, self._metadata, departure
            )
        except (KeyError, ValueError, TypeError) as exc:
            accident = _component_unavailable(
                kind="external_calendar_accident",
                model_version=self._metadata.get("model_version", "unavailable"),
                reason=f"Accident component failed: {exc}",
            )
        context = extract_route_context(
            route,
            graph=graph,
            departure=departure,
            weather_condition=weather_condition,
        )
        route_source = (
            route.source if isinstance(route, RouteCandidate) else "mapbox"
        )
        final, applied, mapbox_level = _final_assessment(
            traffic_band=traffic.band,
            accident_band=accident.band,
            route_context=context,
            current_mapbox_assessment=current_mapbox_assessment,
            route_source=route_source,
            route_id=route.route_id,
        )
        probability_text = (
            f"traffic {traffic.probability:.0%}, accident {accident.probability:.0%}"
            if traffic.probability is not None and accident.probability is not None
            else "component probabilities unavailable"
        )
        disclosure = (DISCLOSURE, ACCIDENT_DISCLOSURE)
        traffic_meta = self._metadata_for("traffic")
        accident_meta = self._metadata_for("accident")
        traffic_dataset = traffic_meta.get("dataset", {})
        accident_dataset = accident_meta.get("dataset", {})
        provenance = (
            f"Traffic: {traffic.dataset_name} ({traffic.dataset_license}); "
            f"{traffic.coverage}; {traffic_dataset.get('source_url', 'source unavailable')}.",
            f"Accident: {accident.dataset_name} ({accident.dataset_license}); "
            f"{accident.coverage}; {accident_dataset.get('source_url', 'source unavailable')}.",
        )
        limitations = (
            "This is an experimental external-data model, not a Karachi validation.",
            "Calendar-only features cannot distinguish Karachi routes.",
            "Predicted bands and probabilities are model outputs, not calibrated "
            "confidence intervals.",
            "The traffic target is vehicle volume, not speed or travel time.",
            "The accident target is a UK collision-time prior, not a Karachi crash "
            "probability.",
            "Requested-departure Mapbox evidence is not a training label and is "
            "not the OSM future-traffic ML prediction.",
        )
        explanation = (
            f"For departure {departure:%A %d %B %Y %H:%M}, the external calendar-only "
            f"traffic model predicts a {traffic.band} volume band, and the external "
            f"UK collision-time model predicts a {accident.band} accident prior. The "
            f"combined policy gives a {final} assessment. Requested-departure "
            f"Mapbox evidence is {'applied' if applied else 'not applied'} to this "
            f"assessment. Route "
            f"context is {context.context_band} and is reported separately. "
            f"Probabilities: {probability_text}."
        )
        return DepartureForecast(
            route_id=route.route_id,
            route_source=route_source,
            requested_departure=departure,
            status="ok"
            if traffic.available and accident.available
            else "partial",
            final_assessment=final,
            predicted_traffic=traffic.band,
            traffic_probability=traffic.probability,
            accident_risk=accident.band,
            accident_probability=accident.probability,
            current_mapbox_traffic=mapbox_level,
            current_mapbox_basis=(
                current_mapbox_assessment.traffic_level_basis
                if current_mapbox_assessment is not None
                else "No requested-departure Mapbox assessment was applied."
            ),
            current_mapbox_applied=applied,
            recommendation=_recommendation(final),
            route_context=context,
            traffic=traffic,
            accident=accident,
            evidence_status="experimental_external_calendar_only",
            model_version=self._metadata.get("model_version", "unavailable"),
            disclosure=disclosure,
            explanation=explanation,
            limitations=limitations,
            provenance=provenance,
            generated_at=_now_utc(),
        )

    def compare(
        self,
        routes: Iterable[RouteCandidate | MapboxRoute],
        departure: datetime,
        graph: Any | None = None,
        weather_condition: str = "dry",
        current_mapbox_assessments: dict[str, TrafficAssessment] | None = None,
    ) -> tuple[DepartureForecast, ...]:
        assessments = current_mapbox_assessments or {}
        return tuple(
            self.forecast(
                route=route,
                departure=departure,
                graph=graph,
                weather_condition=weather_condition,
                current_mapbox_assessment=assessments.get(route.route_id),
            )
            for route in routes
        )


def forecast_rows(
    forecasts: Iterable[DepartureForecast],
) -> list[dict[str, str]]:
    rows = []
    for forecast in forecasts:
        rows.append(
            {
                "Route": forecast.route_id,
                "Source": forecast.route_source,
                "Traffic band": forecast.predicted_traffic,
                "Traffic probability": (
                    f"{forecast.traffic_probability:.0%}"
                    if forecast.traffic_probability is not None
                    else "unavailable"
                ),
                "Accident prior": forecast.accident_risk,
                "Accident probability": (
                    f"{forecast.accident_probability:.0%}"
                    if forecast.accident_probability is not None
                    else "unavailable"
                ),
                "Route context": forecast.route_context.context_band,
                "Final assessment": forecast.final_assessment,
            }
        )
    return rows
