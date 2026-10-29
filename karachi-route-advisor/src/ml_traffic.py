from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from .config import PROJECT_ROOT
from .contracts import MLTrafficAssessment
from .local_traffic import LocalTrafficSummary

MODEL_DIR = PROJECT_ROOT / "models" / "traffic_flow"
MODEL_PATH = MODEL_DIR / "traffic_volume_model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
EVALUATION_PATH = MODEL_DIR / "evaluation.json"
FEATURE_DEFINITIONS_PATH = MODEL_DIR / "feature_definitions.json"

LAGS = (1, 2, 3, 6, 12, 24, 48, 168)
FEATURES = (
    *(f"lag_{lag}" for lag in LAGS),
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
)


class MLTrafficError(RuntimeError):
    pass


class MLFeatureError(ValueError):
    pass


@dataclass(frozen=True)
class MLModelBundle:
    model: Any
    metadata: dict[str, Any]
    evaluation: dict[str, Any]
    feature_definitions: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_model_bundle(model_dir: Path = MODEL_DIR) -> MLModelBundle:
    model_dir = Path(model_dir)
    metadata_path = model_dir / METADATA_PATH.name
    evaluation_path = model_dir / EVALUATION_PATH.name
    feature_path = model_dir / FEATURE_DEFINITIONS_PATH.name
    model_path = model_dir / MODEL_PATH.name
    missing = [
        str(path)
        for path in (
            model_path,
            metadata_path,
            evaluation_path,
            feature_path,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing traffic model artifact(s): " + ", ".join(missing)
        )
    model = joblib.load(model_path)
    if not hasattr(model, "predict"):
        raise MLTrafficError("The saved traffic artifact has no predict method.")
    return MLModelBundle(
        model=model,
        metadata=_read_json(metadata_path),
        evaluation=_read_json(evaluation_path),
        feature_definitions=_read_json(feature_path),
    )


def _naive_datetime(value: datetime) -> datetime:
    if value.tzinfo is not None:
        return value.astimezone().replace(tzinfo=None)
    return value


def _is_us_federal_holiday(target: pd.Timestamp) -> int:
    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(
        start=target.normalize(),
        end=target.normalize(),
    )
    return int(len(holidays) > 0)


def build_feature_row(
    history: pd.DataFrame,
    target_time: datetime,
    *,
    is_holiday: bool | None = None,
) -> pd.DataFrame:
    if not isinstance(history, pd.DataFrame):
        raise MLFeatureError("history must be a pandas DataFrame.")
    if "traffic_volume" not in history.columns:
        raise MLFeatureError("history must contain a traffic_volume column.")
    if not isinstance(history.index, pd.DatetimeIndex):
        raise MLFeatureError("history must use a DatetimeIndex.")
    if history.empty:
        raise MLFeatureError("history is empty.")
    if history.index.has_duplicates:
        raise MLFeatureError("history has duplicate timestamps.")
    history = history.sort_index()
    target = pd.Timestamp(target_time)
    if target.tzinfo is not None:
        target = target.tz_convert(None)
    expected_last = target - pd.Timedelta(hours=1)
    if history.index.max() != expected_last:
        raise MLFeatureError(
            "history must end exactly one hour before the target time; "
            f"expected {expected_last}, got {history.index.max()}."
        )

    values: dict[str, float] = {}
    for lag in LAGS:
        timestamp = target - pd.Timedelta(hours=lag)
        if timestamp not in history.index:
            raise MLFeatureError(
                f"history is missing the observation required for lag_{lag}: "
                f"{timestamp}."
            )
        value = history.loc[timestamp, "traffic_volume"]
        if isinstance(value, pd.Series):
            raise MLFeatureError("history has duplicate lag timestamps.")
        numeric = float(value)
        if not np.isfinite(numeric):
            raise MLFeatureError(f"history has a non-finite value at {timestamp}.")
        values[f"lag_{lag}"] = numeric

    hour = target.hour
    dow = target.day_of_week
    month = target.month
    holiday = (
        _is_us_federal_holiday(target)
        if is_holiday is None
        else int(bool(is_holiday))
    )
    values.update(
        {
            "hour_sin": float(np.sin(2.0 * np.pi * hour / 24.0)),
            "hour_cos": float(np.cos(2.0 * np.pi * hour / 24.0)),
            "dow_sin": float(np.sin(2.0 * np.pi * dow / 7.0)),
            "dow_cos": float(np.cos(2.0 * np.pi * dow / 7.0)),
            "month_sin": float(np.sin(2.0 * np.pi * (month - 1) / 12.0)),
            "month_cos": float(np.cos(2.0 * np.pi * (month - 1) / 12.0)),
            "is_weekend": int(dow >= 5),
            "is_holiday": holiday,
        }
    )
    return pd.DataFrame([values], index=[target])[list(FEATURES)]


class MLTrafficForecaster:
    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self.bundle: MLModelBundle | None = None
        self.load_error: str | None = None
        try:
            self.bundle = load_model_bundle(self.model_dir)
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"

    @property
    def model_loaded(self) -> bool:
        return self.bundle is not None

    def predict_volume(
        self,
        history: pd.DataFrame,
        target_time: datetime,
        *,
        is_holiday: bool | None = None,
    ) -> float:
        if self.bundle is None:
            raise MLTrafficError(
                f"Traffic model is unavailable: {self.load_error or 'not loaded'}"
            )
        row = build_feature_row(history, target_time, is_holiday=is_holiday)
        try:
            prediction = self.bundle.model.predict(row)
        except Exception as exc:
            raise MLTrafficError(
                f"Traffic model prediction failed: {type(exc).__name__}: {exc}"
            ) from exc
        value = float(np.asarray(prediction).reshape(-1)[0])
        if not np.isfinite(value):
            raise MLTrafficError("Traffic model returned a non-finite prediction.")
        return max(0.0, value)

    def assess(
        self,
        route_id: str,
        departure: datetime,
        *,
        now: datetime | None = None,
        local_summary: LocalTrafficSummary | None = None,
    ) -> MLTrafficAssessment:
        requested = _naive_datetime(departure)
        current = _naive_datetime(now or datetime.now())
        reasons: list[str] = []
        invalid_request = False
        if not route_id or not route_id.strip():
            reasons.append("No route is selected.")
            invalid_request = True
        if requested <= current:
            reasons.append(
                f"Selected departure {requested:%Y-%m-%d %H:%M} is not in the "
                "future; future-departure validation failed."
            )
            invalid_request = True

        metadata: dict[str, Any] = self.bundle.metadata if self.bundle else {}
        evaluation: dict[str, Any] = self.bundle.evaluation if self.bundle else {}
        dataset = metadata.get("dataset", {})
        coverage = metadata.get("coverage", {})
        route_applicable = bool(metadata.get("route_applicable", False))
        model_kind = str(metadata.get("model_kind", "unavailable"))
        trained = bool(metadata.get("trained", False))
        dataset_name = str(dataset.get("name", "No dataset metadata available"))
        dataset_license = str(dataset.get("license", "Unknown"))
        coverage_text = str(
            coverage.get(
                "description",
                dataset.get(
                    "geographic_coverage",
                    "No geographic coverage metadata available",
                ),
            )
        )
        selected = evaluation.get("selected_model")
        benchmark_validation_mae = self._metric(
            evaluation, "validation", selected, "mae"
        )
        benchmark_test_mae = self._metric(evaluation, "test", selected, "mae")
        benchmark_test_rmse = self._metric(evaluation, "test", selected, "rmse")
        benchmark_test_r2 = self._metric(evaluation, "test", selected, "r2")
        benchmark_baseline_test_mae = self._metric(
            evaluation, "test", "calendar_mean", "mae"
        )

        if self.load_error:
            reasons.append(
                "The trained external benchmark artifact could not be loaded: "
                f"{self.load_error}"
            )
        elif not route_applicable:
            reasons.append(
                "The trained model covers an external Minnesota interstate sensor, "
                "not Karachi."
            )
            reasons.append(
                "The dataset has no Karachi route identifiers, route geometry, "
                "speed, travel-time, or congestion target."
            )
            reasons.append(
                "No sensor-to-road mapping was performed, so the external volume "
                "prediction is not applied to this route."
            )
            reasons.append(
                "No Karachi-specific route model has been trained."
            )

        local_message = (
            local_summary.message
            if local_summary is not None
            else "No opt-in local traffic observations have been imported."
        )
        if local_summary is not None and local_summary.rows:
            if local_summary.ready_for_training:
                reasons.append(
                    "Local observations are present, but they have not yet been used "
                    "to train a Karachi-specific model."
                )
            else:
                reasons.append(
                    "Local observations are present but are below the minimum "
                    "volume for a preliminary route-specific training experiment."
                )

        if invalid_request:
            status = "invalid_request"
        elif self.load_error:
            status = "degraded"
        else:
            status = "unavailable"

        if status == "invalid_request":
            forecast_label = (
                "Unavailable: select a future departure and a valid route."
            )
            explanation = (
                "A selected-route ML forecast requires a future departure time and "
                "a selected route. The current request does not satisfy that "
                "contract, so no prediction is shown."
            )
        elif status == "degraded":
            forecast_label = "Unavailable: model artifact could not be loaded."
            explanation = (
                "A trained external traffic-volume benchmark exists in the "
                "repository, but its artifact could not be loaded. No prediction or "
                "traffic level is shown for the selected route."
            )
        else:
            forecast_label = (
                "Unavailable: the external benchmark is not applicable to this "
                "Karachi route."
            )
            explanation = (
                f"A trained model exists for {dataset_name} ({coverage_text}). "
                "It predicts one-hour-ahead hourly vehicle volume and achieved a "
                f"held-out test MAE of {benchmark_test_mae:.1f} vehicles/hour "
                "against a calendar-mean baseline MAE of "
                f"{benchmark_baseline_test_mae:.1f}. It has no Karachi "
                "observations, route identifiers, route geometry, speed, travel "
                "time, or congestion labels, and no sensor was mapped to a Karachi "
                "road. Therefore it is not used to predict this selected route. The "
                "app does not substitute an external volume forecast for a Karachi "
                "ETA or congestion level."
                if benchmark_test_mae is not None
                and benchmark_baseline_test_mae is not None
                else (
                    f"A trained external model exists for {dataset_name}, but it "
                    "is not applicable to this Karachi route. No route forecast is "
                    "shown."
                )
            )

        limitations = list(metadata.get("limitations", []))
        limitations.extend(
            [
                "The external model predicts vehicle volume, not travel time or "
                "congestion.",
                "No Karachi route-level validation exists.",
                "No route forecast is displayed when coverage or local evidence is "
                "insufficient.",
            ]
        )
        provenance = [
            f"Dataset: {dataset_name}.",
            f"Dataset licence: {dataset_license}.",
            f"Coverage: {coverage_text}.",
            f"Model artifact: {MODEL_PATH.name}.",
            "Model selection used validation MAE; the held-out test split was not "
            "used for selection.",
        ]
        if dataset.get("doi"):
            provenance.append(f"Dataset DOI: {dataset['doi']}.")
        if dataset.get("source_url"):
            provenance.append(f"Dataset source: {dataset['source_url']}.")
        provenance.append(local_message)

        return MLTrafficAssessment(
            route_id=route_id,
            requested_departure=requested,
            status=status,
            forecast_available=False,
            forecast_value=None,
            forecast_label=forecast_label,
            forecast_units="vehicles per hour",
            model_kind=model_kind,
            trained=trained,
            model_loaded=self.model_loaded,
            dataset_name=dataset_name,
            dataset_license=dataset_license,
            coverage=coverage_text,
            route_applicable=route_applicable,
            benchmark_validation_mae=benchmark_validation_mae,
            benchmark_test_mae=benchmark_test_mae,
            benchmark_test_rmse=benchmark_test_rmse,
            benchmark_test_r2=benchmark_test_r2,
            benchmark_baseline_test_mae=benchmark_baseline_test_mae,
            explanation=explanation,
            reasons=tuple(dict.fromkeys(reasons)),
            limitations=tuple(dict.fromkeys(limitations)),
            provenance=tuple(provenance),
            local_data_message=local_message,
        )

    @staticmethod
    def _metric(
        evaluation: dict[str, Any],
        split: str,
        model_name: str | None,
        metric: str,
    ) -> float | None:
        if not model_name:
            return None
        value = (
            evaluation.get(split, {})
            .get(model_name, {})
            .get(metric)
        )
        if value is None:
            return None
        return float(value)
