from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .calendar_features import FEATURE_NAMES as CALENDAR_FEATURE_NAMES
from .calendar_features import calendar_feature_frame
from .config import PROJECT_ROOT
from .contracts import MapboxRoute, RouteCandidate, RouteTrafficPrediction
from .route_traffic_features import (
    FEATURE_NAMES,
    describe_feature,
    feature_frame,
    route_feature_values,
)

MODEL_DIR = PROJECT_ROOT / "models" / "route_traffic"
ACCIDENT_MODEL_DIR = PROJECT_ROOT / "models" / "departure_forecast"
BAND_NAMES = ("low", "moderate", "high")
ADVICE_LABELS = {
    "good_time": "Good time to leave",
    "reasonable_time": "Reasonable time to leave",
    "consider_earlier": "Consider leaving earlier",
    "not_recommended": "Not recommended",
    "unavailable": "Prediction unavailable",
}


class RouteTrafficPredictionError(RuntimeError):
    pass


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone().replace(tzinfo=None)


def advice_for(
    traffic_band: str, accident_risk: str
) -> tuple[str, str, str]:
    if traffic_band not in BAND_NAMES:
        return (
            "unavailable",
            ADVICE_LABELS["unavailable"],
            "No valid traffic prediction is available for this departure.",
        )
    if accident_risk not in BAND_NAMES:
        if traffic_band == "high":
            return (
                "consider_earlier",
                ADVICE_LABELS["consider_earlier"],
                "High traffic is predicted around this departure time. The accident "
                "prior is unavailable, so leave extra time.",
            )
        if traffic_band == "moderate":
            return (
                "reasonable_time",
                ADVICE_LABELS["reasonable_time"],
                "Moderate traffic is predicted around this departure time. The "
                "accident prior is unavailable, so allow extra time.",
            )
        return (
            "reasonable_time",
            ADVICE_LABELS["reasonable_time"],
            "Low traffic is predicted, but the accident prior is unavailable. This "
            "is a reasonable time to leave with some caution.",
        )
    if traffic_band == "high" and accident_risk == "high":
        return (
            "not_recommended",
            ADVICE_LABELS["not_recommended"],
            "High traffic and a high accident prior are predicted around this "
            "departure time. This is not an ideal time to leave.",
        )
    if traffic_band == "high":
        return (
            "consider_earlier",
            ADVICE_LABELS["consider_earlier"],
            "High traffic is predicted around this departure time. Leaving earlier "
            "or later may reduce delay.",
        )
    if traffic_band == "moderate" and accident_risk == "high":
        return (
            "consider_earlier",
            ADVICE_LABELS["consider_earlier"],
            "Moderate traffic is predicted with a high accident prior. Allow extra "
            "time or consider a different departure window.",
        )
    if traffic_band == "moderate":
        return (
            "reasonable_time",
            ADVICE_LABELS["reasonable_time"],
            "Moderate traffic is predicted around this departure time. This is a "
            "reasonable time to leave, with some delay likely.",
        )
    if accident_risk == "high":
        return (
            "reasonable_time",
            ADVICE_LABELS["reasonable_time"],
            "Low traffic is predicted, but the accident prior is high. This is a "
            "reasonable time to leave, with extra caution advised.",
        )
    return (
        "good_time",
        ADVICE_LABELS["good_time"],
        "Low traffic and low accident risk are predicted around this departure "
        "time. This is a good time to leave.",
    )


def confidence_for(probabilities: list[float]) -> tuple[str, str]:
    ordered = sorted(probabilities, reverse=True)
    top = ordered[0] if ordered else 0.0
    margin = top - ordered[1] if len(ordered) > 1 else top
    if top >= 0.70 and margin >= 0.25:
        level = "High"
    elif top >= 0.45 and margin >= 0.10:
        level = "Medium"
    else:
        level = "Low"
    if level == "High":
        level = "Medium"
    note = (
        "Classifier certainty on simulated route-time scenarios. Confidence is "
        "capped at Medium because the traffic model is synthetic-trained and has no "
        "Karachi real-world validation."
    )
    return level, note


def _class_probabilities(
    classifier: Any, frame: pd.DataFrame
) -> dict[str, float]:
    raw = np.asarray(classifier.predict_proba(frame), dtype=float)[0]
    classes = [int(value) for value in classifier.classes_]
    probabilities = {name: 0.0 for name in BAND_NAMES}
    for index, class_value in enumerate(classes):
        if 0 <= class_value < len(BAND_NAMES):
            probabilities[BAND_NAMES[class_value]] = float(raw[index])
    total = sum(probabilities.values())
    if total > 0:
        probabilities = {
            name: value / total for name, value in probabilities.items()
        }
    return probabilities


class RouteTrafficPredictionService:
    def __init__(
        self,
        model_dir: Path | None = None,
        accident_model_dir: Path | None = None,
    ) -> None:
        self.model_dir = Path(model_dir or MODEL_DIR)
        self.accident_model_dir = Path(accident_model_dir or ACCIDENT_MODEL_DIR)
        self._artifact: dict[str, Any] | None = None
        self._metadata: dict[str, Any] | None = None
        self._evaluation: dict[str, Any] | None = None
        self._load_error: str | None = None
        self._accident_artifact: dict[str, Any] | None = None
        self._accident_metadata: dict[str, Any] | None = None
        self._accident_evaluation: dict[str, Any] | None = None
        self._accident_error: str | None = None
        self._load()

    def _load(self) -> None:
        model_path = self.model_dir / "traffic_model.joblib"
        metadata_path = self.model_dir / "metadata.json"
        evaluation_path = self.model_dir / "evaluation.json"
        missing = [
            str(path)
            for path in (model_path, metadata_path, evaluation_path)
            if not path.exists()
        ]
        if missing:
            self._load_error = "Missing route traffic artifact(s): " + ", ".join(
                missing
            )
        else:
            try:
                artifact = joblib.load(model_path)
                metadata = _read_json(metadata_path)
                evaluation = _read_json(evaluation_path)
                if not isinstance(artifact, dict):
                    raise TypeError("model artifact must be a dictionary")
                if tuple(artifact.get("feature_names", ())) != FEATURE_NAMES:
                    raise ValueError("model feature schema does not match the code")
                if metadata is None or evaluation is None:
                    raise ValueError("metadata or evaluation JSON is invalid")
                self._artifact = artifact
                self._metadata = metadata
                self._evaluation = evaluation
            except (OSError, ValueError, TypeError, EOFError) as exc:
                self._load_error = (
                    f"Route traffic model artifacts could not be loaded: {exc}"
                )
                self._artifact = None
                self._metadata = None
                self._evaluation = None

        accident_path = self.accident_model_dir / "accident_model.joblib"
        accident_metadata_path = self.accident_model_dir / "metadata.json"
        accident_evaluation_path = self.accident_model_dir / "evaluation.json"
        if not all(
            path.exists()
            for path in (
                accident_path,
                accident_metadata_path,
                accident_evaluation_path,
            )
        ):
            self._accident_error = "Accident prior artifact is unavailable."
            return
        try:
            artifact = joblib.load(accident_path)
            metadata = _read_json(accident_metadata_path)
            evaluation = _read_json(accident_evaluation_path)
            if not isinstance(artifact, dict):
                raise TypeError("accident artifact must be a dictionary")
            if tuple(artifact.get("feature_names", ())) != CALENDAR_FEATURE_NAMES:
                raise ValueError("accident feature schema does not match the code")
            if metadata is None or evaluation is None:
                raise ValueError("accident metadata or evaluation JSON is invalid")
            self._accident_artifact = artifact
            self._accident_metadata = metadata
            self._accident_evaluation = evaluation
        except (OSError, ValueError, TypeError, EOFError) as exc:
            self._accident_error = f"Accident prior could not be loaded: {exc}"
            self._accident_artifact = None
            self._accident_metadata = None
            self._accident_evaluation = None

    @property
    def available(self) -> bool:
        return self._artifact is not None and self._metadata is not None

    @property
    def accident_available(self) -> bool:
        return self._accident_artifact is not None

    def _external_index(self, values: Any, departure: datetime) -> float:
        if not isinstance(values, (list, tuple)) or len(values) != 168:
            return 1.0
        index = int(departure.weekday()) * 24 + int(departure.hour)
        return float(values[index])

    def _accident_prediction(
        self, departure: datetime
    ) -> tuple[str, float | None, str, str]:
        if self._accident_artifact is None:
            return (
                "unknown",
                None,
                "UNAVAILABLE",
                self._accident_error or "Accident prior is unavailable.",
            )
        artifact = self._accident_artifact
        classifier = artifact.get("classifier")
        if classifier is None:
            return "unknown", None, "UNAVAILABLE", "Accident classifier missing."
        probabilities = _class_probabilities(
            classifier, calendar_feature_frame(departure)
        )
        band = max(probabilities, key=probabilities.get)
        confidence = probabilities[band]
        return (
            band,
            float(confidence),
            "REAL",
            (
                "External UK STATS19 collision-time prior. This is real external "
                "data, but it is not Karachi crash history and not a crash "
                "probability for this route."
            ),
        )

    def _unavailable(
        self,
        *,
        route: RouteCandidate | MapboxRoute | None,
        departure: datetime,
        reason: str,
        status: str = "unavailable",
    ) -> RouteTrafficPrediction:
        route_id = route.route_id if route is not None else "unknown"
        route_source = (
            route.source if isinstance(route, RouteCandidate) else "mapbox"
        )
        accident_band, accident_probability, accident_evidence, accident_source = (
            self._accident_prediction(departure)
        )
        advice_code, advice_label, advice_message = advice_for(
            "unknown", accident_band
        )
        metadata = self._metadata or {}
        evaluation = self._evaluation or {}
        synthetic_rows = int(metadata.get("synthetic_rows", 0))
        return RouteTrafficPrediction(
            route_id=route_id,
            route_source=route_source,
            requested_departure=departure,
            status=status,
            traffic_band="unknown",
            traffic_probability=None,
            traffic_probabilities=(),
            confidence="Unavailable",
            confidence_note=reason,
            accident_risk=accident_band,
            accident_probability=accident_probability,
            accident_evidence=accident_evidence,
            accident_source=accident_source,
            advice_code=advice_code,
            advice_label=advice_label,
            advice_message=advice_message,
            explanation=(
                "No route-specific traffic prediction is shown. " + reason
            ),
            top_drivers=(),
            data_provenance=(
                "MIXED" if accident_evidence == "REAL" else "SYNTHETIC"
            ),
            data_provenance_label=(
                "Mixed real + synthetic data"
                if accident_evidence == "REAL"
                else "Synthetic simulation"
            ),
            model_name=str(metadata.get("model_name", "unavailable")),
            model_version=str(metadata.get("model_version", "unavailable")),
            training_dataset_size=int(metadata.get("training_rows", 0)),
            real_records=int(metadata.get("real_records", 0)),
            synthetic_records=synthetic_rows,
            evaluation_accuracy=evaluation.get("test", {}).get("accuracy"),
            evaluation_precision=evaluation.get("test", {}).get(
                "macro_precision"
            ),
            evaluation_recall=evaluation.get("test", {}).get("macro_recall"),
            evaluation_f1=evaluation.get("test", {}).get("macro_f1"),
            confusion_matrix=tuple(
                tuple(int(value) for value in row)
                for row in evaluation.get("confusion_matrix", [])
            ),
            features=FEATURE_NAMES,
            limitations=tuple(metadata.get("limitations", ())),
            provenance=tuple(metadata.get("provenance", ())),
            generated_at=_now_utc(),
        )

    def _top_drivers(
        self,
        model: Any,
        row: dict[str, float],
        base_probabilities: np.ndarray,
        predicted_index: int,
    ) -> tuple[str, ...]:
        medians = (self._artifact or {}).get("training_medians", {})
        if not isinstance(medians, dict):
            return ()
        base = float(base_probabilities[predicted_index])
        contributions: list[tuple[float, str, float]] = []
        for name in FEATURE_NAMES:
            median = medians.get(name)
            if median is None:
                continue
            if abs(float(row[name]) - float(median)) < 1e-9:
                continue
            perturbed = dict(row)
            perturbed[name] = float(median)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*sklearn.utils.parallel.delayed.*",
                )
                probabilities = np.asarray(
                    model.predict_proba(feature_frame(perturbed)), dtype=float
                )[0]
            delta = base - float(probabilities[predicted_index])
            if delta > 1e-6:
                contributions.append((delta, name, float(row[name])))
        contributions.sort(reverse=True)
        drivers: list[str] = []
        for _, name, value in contributions[:3]:
            description = describe_feature(name, value)
            if description:
                drivers.append(description)
        return tuple(drivers)

    def predict(
        self,
        *,
        route: RouteCandidate | MapboxRoute,
        departure: datetime,
        graph: Any | None = None,
        weather_condition: str = "dry",
        now: datetime | None = None,
    ) -> RouteTrafficPrediction:
        requested = _naive(departure)
        current = _naive(now or datetime.now())
        if requested <= current:
            return self._unavailable(
                route=route,
                departure=requested,
                reason=(
                    "The selected departure must be in the future; "
                    f"{requested:%Y-%m-%d %H:%M} is not."
                ),
                status="invalid_request",
            )
        if not self.available or self._metadata is None or self._artifact is None:
            return self._unavailable(
                route=route,
                departure=requested,
                reason=self._load_error or "Route traffic model is unavailable.",
            )
        if not isinstance(route, RouteCandidate):
            return self._unavailable(
                route=route,
                departure=requested,
                reason=(
                    "The route-specific model requires an OSM academic route with "
                    "edge-level features. Mapbox-only routes are not supported."
                ),
            )

        artifact = self._artifact
        metadata = self._metadata
        traffic_index = self._external_index(
            artifact.get("external_traffic_index"), requested
        )
        accident_index = self._external_index(
            artifact.get("external_accident_index"), requested
        )
        values = route_feature_values(
            route,
            requested,
            graph,
            weather_condition,
            external_traffic_index=traffic_index,
            external_accident_index=accident_index,
        )
        if values is None:
            return self._unavailable(
                route=route,
                departure=requested,
                reason=(
                    "The selected route has no usable OSM edge features, so a "
                    "route-specific prediction cannot be produced."
                ),
            )

        model = artifact["model"]
        frame = feature_frame(values)
        try:
            raw_probabilities = np.asarray(
                model.predict_proba(frame), dtype=float
            )[0]
        except Exception as exc:
            return self._unavailable(
                route=route,
                departure=requested,
                reason=(
                    "The route traffic model failed during prediction: "
                    f"{type(exc).__name__}: {exc}"
                ),
                status="degraded",
            )
        classes = [int(value) for value in model.classes_]
        probabilities = {name: 0.0 for name in BAND_NAMES}
        for index, class_value in enumerate(classes):
            if 0 <= class_value < len(BAND_NAMES):
                probabilities[BAND_NAMES[class_value]] = float(
                    raw_probabilities[index]
                )
        total = sum(probabilities.values())
        if total <= 0:
            return self._unavailable(
                route=route,
                departure=requested,
                reason="The route traffic model returned no class probabilities.",
                status="degraded",
            )
        probabilities = {
            name: value / total for name, value in probabilities.items()
        }
        ordered = sorted(
            probabilities.items(), key=lambda item: item[1], reverse=True
        )
        traffic_band = ordered[0][0]
        traffic_probability = ordered[0][1]
        confidence, confidence_note = confidence_for(
            [value for _, value in ordered]
        )
        predicted_index = classes.index(
            BAND_NAMES.index(traffic_band)
        ) if BAND_NAMES.index(traffic_band) in classes else int(
            np.argmax(raw_probabilities)
        )
        top_drivers = self._top_drivers(
            model, values, raw_probabilities, predicted_index
        )
        accident_band, accident_probability, accident_evidence, accident_source = (
            self._accident_prediction(requested)
        )
        advice_code, advice_label, advice_message = advice_for(
            traffic_band, accident_band
        )

        drivers_text = (
            "The largest approximate model contributions were: "
            + "; ".join(top_drivers)
            + ". "
            if top_drivers
            else ""
        )
        explanation = (
            f"For {requested:%A %d %B %Y at %H:%M}, the route-specific model "
            f"predicts {traffic_band.upper()} traffic with "
            f"{traffic_probability:.0%} probability. {drivers_text}"
            f"The accident prior is {accident_band.upper()}"
            + (
                f" ({accident_probability:.0%}) from real external UK STATS19 data. "
                if accident_probability is not None
                else ", but accident evidence is unavailable. "
            )
            + "The traffic label is learned from a synthetic Gulistan-e-Jauhar "
            "route simulation, not observed Karachi traffic, so this is a "
            "scenario-based advisory rather than a validated real-world forecast."
        )
        provenance = [
            "Traffic target labels: synthetic route-time simulation "
            "(BPR-style congestion simulation calibrated to real external "
            "time-of-week profiles).",
            "Route features: real OpenStreetMap Gulistan-e-Jauhar drive graph.",
            "External traffic profile: UCI 492 Metro Interstate Traffic Volume "
            "(real external Minnesota sensor; CC BY 4.0).",
            "Accident prior: real external UK STATS19 2023-2025 collision times "
            "(OGL v3.0).",
            "No observed Karachi route-level traffic records were used.",
        ]
        limitations = tuple(metadata.get("limitations", ()))
        return RouteTrafficPrediction(
            route_id=route.route_id,
            route_source=route.source,
            requested_departure=requested,
            status="ok" if accident_evidence == "REAL" else "partial",
            traffic_band=traffic_band,
            traffic_probability=float(traffic_probability),
            traffic_probabilities=tuple(ordered),
            confidence=confidence,
            confidence_note=confidence_note,
            accident_risk=accident_band,
            accident_probability=accident_probability,
            accident_evidence=accident_evidence,
            accident_source=accident_source,
            advice_code=advice_code,
            advice_label=advice_label,
            advice_message=advice_message,
            explanation=explanation,
            top_drivers=top_drivers,
            data_provenance=(
                "MIXED" if accident_evidence == "REAL" else "SYNTHETIC"
            ),
            data_provenance_label=(
                "Mixed real + synthetic data"
                if accident_evidence == "REAL"
                else "Synthetic simulation"
            ),
            model_name=str(artifact.get("model_name", "unknown")),
            model_version=str(artifact.get("model_version", "unknown")),
            training_dataset_size=int(metadata.get("training_rows", 0)),
            real_records=int(metadata.get("real_records", 0)),
            synthetic_records=int(metadata.get("synthetic_rows", 0)),
            evaluation_accuracy=(
                self._evaluation or {}
            ).get("test", {}).get("accuracy"),
            evaluation_precision=(
                self._evaluation or {}
            ).get("test", {}).get("macro_precision"),
            evaluation_recall=(
                self._evaluation or {}
            ).get("test", {}).get("macro_recall"),
            evaluation_f1=(
                self._evaluation or {}
            ).get("test", {}).get("macro_f1"),
            confusion_matrix=tuple(
                tuple(int(value) for value in row)
                for row in (self._evaluation or {}).get(
                    "confusion_matrix", []
                )
            ),
            features=FEATURE_NAMES,
            limitations=limitations,
            provenance=tuple(provenance),
            generated_at=_now_utc(),
        )
