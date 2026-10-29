from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.local_traffic import (
    LocalTrafficSummary,
    local_traffic_summary,
    summarize_local_observations,
)
from src.ml_traffic import (
    FEATURES,
    METADATA_PATH,
    MLFeatureError,
    MLTrafficError,
    MLTrafficForecaster,
    build_feature_row,
    load_model_bundle,
)

MODEL_AVAILABLE = METADATA_PATH.exists() and (
    Path(METADATA_PATH).parent / "traffic_volume_model.joblib"
).exists()

pytestmark = pytest.mark.skipif(
    not MODEL_AVAILABLE,
    reason="Run scripts/train_traffic_model.py to create the model artifacts.",
)


def empty_local_summary(tmp_path: Path) -> LocalTrafficSummary:
    return local_traffic_summary(tmp_path / "missing.csv")


def history_ending_before(target: datetime) -> pd.DataFrame:
    index = pd.date_range(
        end=pd.Timestamp(target) - pd.Timedelta(hours=1),
        periods=168,
        freq="h",
    )
    return pd.DataFrame(
        {"traffic_volume": np.arange(168, dtype=float) + 1000.0},
        index=index,
    )


def test_model_bundle_loads_and_is_external_only() -> None:
    bundle = load_model_bundle()
    assert hasattr(bundle.model, "predict")
    assert bundle.metadata["route_applicable"] is False
    assert bundle.metadata["coverage"]["karachi_coverage"] is False
    assert bundle.metadata["coverage"]["route_level"] is False
    assert bundle.evaluation["selected_model"] == "random_forest"
    assert bundle.evaluation["test"]["random_forest"]["mae"] > 0
    assert bundle.evaluation["test"]["calendar_mean"]["mae"] > 0
    assert tuple(bundle.metadata["features"]) == FEATURES


def test_feature_row_uses_exact_lags_and_calendar() -> None:
    target = datetime(2019, 1, 15, 12, 0)
    history = history_ending_before(target)
    row = build_feature_row(history, target)
    assert list(row.columns) == list(FEATURES)
    assert row.iloc[0]["lag_1"] == pytest.approx(1167.0)
    assert row.iloc[0]["lag_2"] == pytest.approx(1166.0)
    assert row.iloc[0]["lag_168"] == pytest.approx(1000.0)
    assert row.iloc[0]["hour_cos"] == pytest.approx(-1.0)
    assert row.iloc[0]["is_weekend"] == 0
    assert row.iloc[0]["is_holiday"] == 0


def test_feature_row_rejects_incomplete_history() -> None:
    target = datetime(2019, 1, 15, 12, 0)
    history = history_ending_before(target).iloc[1:]
    with pytest.raises(MLFeatureError, match="lag_168"):
        build_feature_row(history, target)


def test_feature_row_rejects_duplicate_history_index() -> None:
    target = datetime(2019, 1, 15, 12, 0)
    history = history_ending_before(target)
    history = pd.concat([history, history.tail(1)])
    with pytest.raises(MLFeatureError, match="duplicate"):
        build_feature_row(history, target)


def test_external_benchmark_prediction_is_finite() -> None:
    target = datetime(2019, 1, 15, 12, 0)
    forecaster = MLTrafficForecaster()
    prediction = forecaster.predict_volume(history_ending_before(target), target)
    assert np.isfinite(prediction)
    assert prediction >= 0


def test_prediction_failure_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    target = datetime(2019, 1, 15, 12, 0)
    forecaster = MLTrafficForecaster()
    assert forecaster.bundle is not None

    def broken_predict(row):
        raise RuntimeError("forced model failure")

    monkeypatch.setattr(forecaster.bundle.model, "predict", broken_predict)
    with pytest.raises(MLTrafficError, match="forced model failure"):
        forecaster.predict_volume(history_ending_before(target), target)


def test_selected_route_forecast_is_unavailable(tmp_path: Path) -> None:
    forecaster = MLTrafficForecaster()
    assessment = forecaster.assess(
        "osm-1",
        datetime(2030, 1, 15, 12, 0),
        now=datetime(2026, 10, 27, 9, 0),
        local_summary=empty_local_summary(tmp_path),
    )
    assert assessment.route_id == "osm-1"
    assert assessment.status == "unavailable"
    assert assessment.forecast_available is False
    assert assessment.forecast_value is None
    assert assessment.route_applicable is False
    assert assessment.benchmark_test_mae == pytest.approx(136.9203480030724)
    assert assessment.benchmark_baseline_test_mae == pytest.approx(
        253.2519742133373
    )
    assert any("Minnesota" in reason for reason in assessment.reasons)
    assert any(
        "No Karachi-specific route model" in reason
        for reason in assessment.reasons
    )
    assert "held-out test MAE" in assessment.explanation
    assert assessment.forecast_units == "vehicles per hour"


def test_past_departure_is_rejected(tmp_path: Path) -> None:
    forecaster = MLTrafficForecaster()
    assessment = forecaster.assess(
        "osm-1",
        datetime(2025, 1, 15, 12, 0),
        now=datetime(2026, 10, 27, 9, 0),
        local_summary=empty_local_summary(tmp_path),
    )
    assert assessment.status == "invalid_request"
    assert assessment.forecast_available is False
    assert any("not in the future" in reason for reason in assessment.reasons)


def test_route_switch_changes_ml_assessment(tmp_path: Path) -> None:
    forecaster = MLTrafficForecaster()
    now = datetime(2026, 10, 27, 9, 0)
    departure = datetime(2030, 1, 15, 12, 0)
    first = forecaster.assess(
        "osm-1", departure, now=now, local_summary=empty_local_summary(tmp_path)
    )
    second = forecaster.assess(
        "osm-2", departure, now=now, local_summary=empty_local_summary(tmp_path)
    )
    assert first.route_id != second.route_id
    assert first.forecast_available is False
    assert second.forecast_available is False


def test_missing_model_degrades_assessment(tmp_path: Path) -> None:
    forecaster = MLTrafficForecaster(tmp_path / "missing-model-dir")
    assessment = forecaster.assess(
        "osm-1",
        datetime(2030, 1, 15, 12, 0),
        now=datetime(2026, 10, 27, 9, 0),
        local_summary=empty_local_summary(tmp_path),
    )
    assert assessment.status == "degraded"
    assert assessment.model_loaded is False
    assert assessment.forecast_available is False
    assert any("could not be loaded" in reason for reason in assessment.reasons)


def test_local_data_ready_does_not_create_route_model(tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01", periods=168, freq="h")
    frame = pd.DataFrame(
        {
            "timestamp": index,
            "route_id": "osm-1",
            "segment_id": "segment-001",
            "travel_time_min": np.linspace(3.0, 6.0, 168),
        }
    )
    local = summarize_local_observations(frame, path=tmp_path / "local.csv")
    assert local.ready_for_training is True
    forecaster = MLTrafficForecaster()
    assessment = forecaster.assess(
        "osm-1",
        datetime(2030, 1, 15, 12, 0),
        now=datetime(2026, 10, 27, 9, 0),
        local_summary=local,
    )
    assert assessment.forecast_available is False
    assert any(
        "have not yet been used to train" in reason
        for reason in assessment.reasons
    )
    assert "future Karachi-specific training" in assessment.local_data_message
