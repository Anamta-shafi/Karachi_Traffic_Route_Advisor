from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
TRAFFIC_CSV = (
    ROOT
    / "data"
    / "external"
    / "metro_interstate"
    / "extracted"
    / "Metro_Interstate_Traffic_Volume.csv.gz"
)
ACCIDENT_DIR = ROOT / "data" / "external" / "stats19"
MODEL_DIR = ROOT / "models" / "departure_forecast"
REPORT_PATH = ROOT / "reports" / "departure_forecast_evaluation.md"
PROVENANCE_PATH = ACCIDENT_DIR / "provenance.json"

FEATURE_NAMES = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
)
BAND_NAMES = ("low", "moderate", "high")
SEVERITY_WEIGHTS = {1: 10.0, 2: 3.0, 3: 1.0}
RANDOM_STATE = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
EMBARGO_HOURS = 24
MODEL_VERSION = "external-calendar-v1"
DISCLOSURE = (
    "ML model trained on external traffic data — not locally validated for "
    "Gulistan-e-Jauhar."
)
ACCIDENT_DISCLOSURE = (
    "Accident-risk model trained on external UK STATS19 collision data — not "
    "locally validated for Gulistan-e-Jauhar and not corridor-matched to Karachi."
)
TRAFFIC_SOURCE_URL = (
    "https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume"
)
TRAFFIC_DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/492/"
    "metro+interstate+traffic+volume.zip"
)
TRAFFIC_DOI = "10.24432/C5X60B"
TRAFFIC_LICENSE = "CC BY 4.0"
TRAFFIC_ATTRIBUTION = (
    "Hogue, J. (2019). Metro Interstate Traffic Volume [Dataset]. "
    "UCI Machine Learning Repository."
)
TRAFFIC_COVERAGE = (
    "I-94 westbound ATR station 301, Minneapolis-St Paul, Minnesota, USA"
)
ACCIDENT_SOURCE_URL = "https://www.gov.uk/government/statistics/road-safety-data"
ACCIDENT_LICENSE = "Open Government Licence v3.0"
ACCIDENT_ATTRIBUTION = (
    "Department for Transport (UK), Road safety data: collision tables."
)
ACCIDENT_COVERAGE = "Great Britain (England, Scotland, Wales)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calendar_matrix(index: pd.DatetimeIndex) -> pd.DataFrame:
    hours = index.hour.to_numpy(dtype=float)
    dow = index.day_of_week.to_numpy(dtype=float)
    month = index.month.to_numpy(dtype=float)
    values = np.column_stack(
        [
            np.sin(2.0 * np.pi * hours / 24.0),
            np.cos(2.0 * np.pi * hours / 24.0),
            np.sin(2.0 * np.pi * dow / 7.0),
            np.cos(2.0 * np.pi * dow / 7.0),
            np.sin(2.0 * np.pi * (month - 1.0) / 12.0),
            np.cos(2.0 * np.pi * (month - 1.0) / 12.0),
            (dow >= 5).astype(float),
        ]
    )
    return pd.DataFrame(values, index=index, columns=list(FEATURE_NAMES))


def split_chronologically(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = len(frame)
    train_boundary = int(total * TRAIN_FRACTION)
    validation_boundary = int(total * (TRAIN_FRACTION + VALIDATION_FRACTION))
    validation_start = frame.index[train_boundary]
    test_start = frame.index[validation_boundary]
    embargo = pd.Timedelta(hours=EMBARGO_HOURS)
    train = frame.loc[frame.index < validation_start - embargo]
    validation = frame.loc[
        (frame.index >= validation_start) & (frame.index < test_start - embargo)
    ]
    test = frame.loc[frame.index >= test_start]
    return train.copy(), validation.copy(), test.copy()


def split_ranges(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame):
    return {
        name: {
            "rows": f"{len(frame):,}",
            "start": frame.index.min().isoformat(),
            "end": frame.index.max().isoformat(),
        }
        for name, frame in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        )
    }


def band_values(values: pd.Series | np.ndarray, thresholds: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.where(
        array <= float(thresholds[0]),
        0,
        np.where(array <= float(thresholds[1]), 1, 2),
    )


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.zeros_like(probabilities, dtype=float)
    one_hot[np.arange(len(y_true)), np.asarray(y_true, dtype=int)] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def classification_metrics(
    y_true: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, object]:
    labels = [0, 1, 2]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted, labels=labels, zero_division=0
    )
    result: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro")),
        "weighted_f1": float(f1_score(y_true, predicted, average="weighted")),
        "per_class": {
            BAND_NAMES[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in labels
        },
        "confusion_matrix": confusion_matrix(
            y_true, predicted, labels=labels
        ).tolist(),
        "class_counts": {
            BAND_NAMES[index]: int(np.sum(np.asarray(y_true) == index))
            for index in labels
        },
    }
    if probabilities is not None:
        result["multiclass_brier"] = multiclass_brier(y_true, probabilities)
        try:
            result["roc_auc_ovr_macro"] = float(
                roc_auc_score(
                    y_true,
                    probabilities,
                    multi_class="ovr",
                    average="macro",
                    labels=labels,
                )
            )
        except ValueError:
            result["roc_auc_ovr_macro"] = None
    return result


def regression_metrics(
    y_true: pd.Series | np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, predicted)),
        "rmse": float(mean_squared_error(y_true, predicted) ** 0.5),
        "r2": float(r2_score(y_true, predicted)),
    }


def calendar_mean_baseline(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    target: str,
) -> np.ndarray:
    train_keys = pd.MultiIndex.from_arrays(
        [train.index.day_of_week, train.index.hour], names=["dow", "hour"]
    )
    evaluation_keys = pd.MultiIndex.from_arrays(
        [evaluation.index.day_of_week, evaluation.index.hour], names=["dow", "hour"]
    )
    means = train.groupby(train_keys)[target].mean()
    return (
        means.reindex(evaluation_keys)
        .fillna(train[target].mean())
        .to_numpy(dtype=float)
    )


def load_traffic() -> tuple[pd.DataFrame, dict[str, object]]:
    if not TRAFFIC_CSV.exists():
        raise FileNotFoundError(f"Missing traffic dataset: {TRAFFIC_CSV}")
    raw = pd.read_csv(TRAFFIC_CSV, parse_dates=["date_time"])
    raw = raw.sort_values("date_time")
    duplicate_rows = int(raw["date_time"].duplicated().sum())
    conflicting = int(
        raw.groupby("date_time")["traffic_volume"].nunique().gt(1).sum()
    )
    unique = (
        raw.drop_duplicates(subset=["date_time"], keep="first")
        .set_index("date_time")
        .sort_index()
    )
    expected = pd.date_range(unique.index.min(), unique.index.max(), freq="h")
    hourly = unique.reindex(expected)
    hourly = hourly.loc[hourly["traffic_volume"].notna(), ["traffic_volume"]]
    hourly["traffic_volume"] = hourly["traffic_volume"].astype(float)
    quality = {
        "raw_rows": int(len(raw)),
        "unique_timestamp_rows": int(len(unique)),
        "duplicate_timestamp_rows_removed": duplicate_rows,
        "conflicting_duplicate_timestamps": conflicting,
        "expected_hours": int(len(expected)),
        "missing_hours": int(len(expected.difference(unique.index))),
        "usable_rows": int(len(hourly)),
        "start": hourly.index.min().isoformat(),
        "end": hourly.index.max().isoformat(),
        "zero_volume_rows": int((hourly["traffic_volume"] == 0).sum()),
    }
    return hourly, quality


def load_accidents() -> tuple[pd.DataFrame, dict[str, object]]:
    files = sorted(ACCIDENT_DIR.glob("collision-*.csv"))
    if not files:
        raise FileNotFoundError(
            f"Missing STATS19 collision files in {ACCIDENT_DIR}"
        )
    frames = [pd.read_csv(path, low_memory=False) for path in files]
    raw = pd.concat(frames, ignore_index=True)
    timestamps = pd.to_datetime(
        raw["date"], format="%d/%m/%Y", errors="coerce"
    ) + pd.to_timedelta(raw["time"].astype(str) + ":00", errors="coerce")
    raw = raw.assign(timestamp=timestamps)
    invalid_timestamps = int(raw["timestamp"].isna().sum())
    raw = raw.dropna(subset=["timestamp"])
    raw["severity_weight"] = (
        raw["collision_severity"].map(SEVERITY_WEIGHTS).astype(float)
    )
    hourly_index = pd.date_range(
        raw["timestamp"].min().floor("h"),
        raw["timestamp"].max().floor("h"),
        freq="h",
    )
    grouped = raw.groupby(raw["timestamp"].dt.floor("h"))
    severity = grouped["severity_weight"].sum().reindex(hourly_index, fill_value=0.0)
    counts = grouped.size().reindex(hourly_index, fill_value=0)
    frame = pd.DataFrame(
        {
            "severity_weight": severity.astype(float),
            "collision_count": counts.astype(float),
        },
        index=hourly_index,
    )
    quality = {
        "raw_rows": int(len(raw)),
        "files": [path.name for path in files],
        "invalid_timestamps": invalid_timestamps,
        "start": frame.index.min().isoformat(),
        "end": frame.index.max().isoformat(),
        "hourly_cells": int(len(frame)),
        "zero_collision_hours": int((frame["collision_count"] == 0).sum()),
        "severity_counts": {
            str(int(key)): int(value)
            for key, value in raw["collision_severity"]
            .value_counts()
            .sort_index()
            .items()
        },
        "severity_weight_definition": "fatal=10, serious=3, slight=1",
        "raw_collisions": int(frame["collision_count"].sum()),
        "severity_weight_total": float(frame["severity_weight"].sum()),
    }
    return frame, quality


def build_traffic_models() -> tuple[dict[str, object], dict[str, object]]:
    regressors: dict[str, object] = {
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=10,
            max_features=0.8,
            n_jobs=2,
            random_state=RANDOM_STATE,
        ),
    }
    classifiers: dict[str, object] = {
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=10,
            max_features=0.8,
            n_jobs=2,
            random_state=RANDOM_STATE,
        ),
    }
    return regressors, classifiers


def build_accident_models() -> tuple[dict[str, object], dict[str, object]]:
    regressors: dict[str, object] = {
        "hist_gradient_boosting_poisson": HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting_squared": HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=10,
            max_features=0.8,
            n_jobs=2,
            random_state=RANDOM_STATE,
        ),
    }
    classifiers: dict[str, object] = {
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=10,
            max_features=0.8,
            n_jobs=2,
            random_state=RANDOM_STATE,
        ),
    }
    return regressors, classifiers


def evaluate_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    tasks: str,
    thresholds: list[float],
) -> dict[str, object]:
    train_x = calendar_matrix(train.index)
    validation_x = calendar_matrix(validation.index)
    test_x = calendar_matrix(test.index)
    train_y = train[target]
    validation_y = validation[target]
    test_y = test[target]
    train_band = band_values(train_y, thresholds)
    validation_band = band_values(validation_y, thresholds)
    test_band = band_values(test_y, thresholds)

    validation_baseline = calendar_mean_baseline(train, validation, target)
    combined = pd.concat([train, validation])
    test_baseline = calendar_mean_baseline(combined, test, target)

    if tasks == "traffic":
        regressors, classifiers = build_traffic_models()
    else:
        regressors, classifiers = build_accident_models()

    validation_results: dict[str, object] = {
        "calendar_mean": {
            **regression_metrics(validation_y, validation_baseline),
            **classification_metrics(
                validation_band, band_values(validation_baseline, thresholds)
            ),
        }
    }
    fitted_regressors = {}
    for name, model in regressors.items():
        model.fit(train_x, train_y)
        predicted = model.predict(validation_x)
        validation_results[name] = {
            **regression_metrics(validation_y, predicted),
            **classification_metrics(
                validation_band, band_values(predicted, thresholds)
            ),
        }
        fitted_regressors[name] = model

    fitted_classifiers = {}
    for name, model in classifiers.items():
        model.fit(train_x, train_band)
        predicted = model.predict(validation_x)
        probabilities = model.predict_proba(validation_x)
        validation_results[f"{name}_classifier"] = classification_metrics(
            validation_band, predicted, probabilities
        )
        fitted_classifiers[name] = model

    selected_regressor = min(
        fitted_regressors,
        key=lambda name: validation_results[name]["mae"],
    )
    selected_classifier = max(
        fitted_classifiers,
        key=lambda name: validation_results[f"{name}_classifier"]["macro_f1"],
    )

    final_regressor = regressors[selected_regressor]
    final_classifier = classifiers[selected_classifier]
    final_regressor.fit(
        pd.concat([train_x, validation_x]), pd.concat([train_y, validation_y])
    )
    final_classifier.fit(
        pd.concat([train_x, validation_x]),
        np.concatenate([train_band, validation_band]),
    )

    test_prediction = final_regressor.predict(test_x)
    test_probabilities = final_classifier.predict_proba(test_x)
    test_class_prediction = final_classifier.predict(test_x)
    baseline_band = band_values(test_baseline, thresholds)

    test_results: dict[str, object] = {
        "calendar_mean": {
            **regression_metrics(test_y, test_baseline),
            **classification_metrics(test_band, baseline_band),
        },
        f"{selected_regressor}_refit": regression_metrics(
            test_y, test_prediction
        ),
        f"{selected_classifier}_classifier_refit": classification_metrics(
            test_band, test_class_prediction, test_probabilities
        ),
    }

    return {
        "thresholds": thresholds,
        "threshold_basis": "train_split_target_terciles",
        "split": split_ranges(train, validation, test),
        "validation": validation_results,
        "test": test_results,
        "selected_regressor": selected_regressor,
        "selected_classifier": selected_classifier,
        "selection_rule": (
            "Regressor: lowest validation MAE. Classifier: highest validation "
            "macro F1. Test split not used for selection."
        ),
        "test_band_counts_actual": {
            BAND_NAMES[index]: int(np.sum(test_band == index))
            for index in range(3)
        },
        "test_band_counts_predicted": {
            BAND_NAMES[index]: int(np.sum(test_class_prediction == index))
            for index in range(3)
        },
        "test_rows": int(len(test)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
    }


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_feature_definitions(path: Path) -> None:
    descriptions = {
        "hour_sin": "Sine encoding of local hour of day.",
        "hour_cos": "Cosine encoding of local hour of day.",
        "dow_sin": "Sine encoding of day of week.",
        "dow_cos": "Cosine encoding of day of week.",
        "month_sin": "Sine encoding of month.",
        "month_cos": "Cosine encoding of month.",
        "is_weekend": "1 when the target date is Saturday or Sunday.",
    }
    payload = {
        "features": [
            {"name": name, "description": descriptions[name]}
            for name in FEATURE_NAMES
        ],
        "traffic_target": {
            "name": "traffic_volume",
            "units": "vehicles per hour",
            "classes": list(BAND_NAMES),
            "definition": (
                "Hourly westbound I-94 volume at ATR station 301, classified "
                "using train-split terciles."
            ),
        },
        "accident_target": {
            "name": "severity_weighted_collisions_per_hour",
            "units": "severity-weighted collisions per hour",
            "classes": list(BAND_NAMES),
            "severity_weights": SEVERITY_WEIGHTS,
            "definition": (
                "Hourly STATS19 collision count weighted fatal=10, serious=3, "
                "slight=1 across Great Britain."
            ),
        },
        "leakage_controls": [
            "Calendar-only features; no traffic lags, no target-hour weather, "
            "no Mapbox evidence, and no target-derived inputs.",
            "Chronological 70/15/15 split with a 24-hour embargo between splits.",
            "Class thresholds, baselines, and preprocessing are fitted on the "
            "training split only.",
            "The held-out test split is not used for model selection.",
            "Mapbox ETA differences are never used as labels or features.",
            "No sensor-to-Karachi-road or UK-to-Karachi-corridor mapping is used.",
        ],
    }
    save_json(path, payload)


def write_report(
    evaluation: dict[str, object],
    traffic_quality: dict[str, object],
    accident_quality: dict[str, object],
    traffic_metadata: dict[str, object],
    accident_metadata: dict[str, object],
) -> None:
    traffic = evaluation["traffic"]
    accident = evaluation["accident"]

    def cell(metrics: dict[str, object], key: str, spec: str) -> str:
        value = metrics.get(key)
        return "—" if value is None else format(float(value), spec)

    lines = [
        "# Departure-Time External Experimental ML Evaluation",
        "",
        "This report covers two deliberately separate, calendar-only external "
        "models. Neither model is validated for Karachi.",
        "",
        "## Applicability",
        "",
        "**Experimental external-data model — not validated for Gulistan-e-Jauhar.**",
        "",
        f"> {DISCLOSURE}",
        "",
        f"> {ACCIDENT_DISCLOSURE}",
        "",
        "The models predict a calendar-slot traffic band and a calendar-slot "
        "accident-time prior. They do not use Karachi route or crash observations, "
        "and they do not claim a Karachi accident probability or ETA.",
        "",
        "## Traffic Model",
        "",
        f"- Dataset: {traffic_metadata['dataset']['name']}",
        f"- Source: {traffic_metadata['dataset']['source_url']}",
        f"- DOI: {traffic_metadata['dataset']['doi']}",
        f"- Licence: {traffic_metadata['dataset']['license']}",
        f"- Coverage: {traffic_metadata['dataset']['geographic_coverage']}",
        f"- Raw rows: {traffic_quality['raw_rows']:,}",
        f"- Unique hourly rows: {traffic_quality['usable_rows']:,}",
        f"- Time span: {traffic_quality['start']} to {traffic_quality['end']}",
        f"- Target: {traffic_metadata['target']['name']} "
        f"({traffic_metadata['target']['units']})",
        f"- Classes: {', '.join(BAND_NAMES)}; train terciles "
        f"{traffic['thresholds']}",
        f"- Split: train {traffic['train_rows']:,}, validation "
        f"{traffic['validation_rows']:,}, test {traffic['test_rows']:,}",
        f"- Selected regressor: {traffic['selected_regressor']}",
        f"- Selected classifier: {traffic['selected_classifier']}",
        "",
        "### Traffic Held-Out Test",
        "",
        "| Predictor | MAE | RMSE | R2 | Accuracy | Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in traffic["test"].items():
        lines.append(
            f"| {name} | {cell(metrics, 'mae', '.3f')} | "
            f"{cell(metrics, 'rmse', '.3f')} | "
            f"{cell(metrics, 'r2', '.4f')} | "
            f"{cell(metrics, 'accuracy', '.4f')} | "
            f"{cell(metrics, 'macro_f1', '.4f')} |"
        )
    selected_traffic = traffic["test"][
        f"{traffic['selected_classifier']}_classifier_refit"
    ]
    lines.extend(
        [
            "",
            f"Traffic classifier test accuracy: **{selected_traffic['accuracy']:.4f}**; "
            f"macro F1: **{selected_traffic['macro_f1']:.4f}**; "
            f"multiclass Brier: **{selected_traffic['multiclass_brier']:.4f}**; "
            f"ROC-AUC OVR macro: **"
            f"{selected_traffic['roc_auc_ovr_macro']:.4f}**."
            if selected_traffic.get("roc_auc_ovr_macro") is not None
            else "",
            "",
            "## Accident-Time Model",
            "",
            f"- Dataset: {accident_metadata['dataset']['name']}",
            f"- Source: {accident_metadata['dataset']['source_url']}",
            f"- Licence: {accident_metadata['dataset']['license']}",
            f"- Coverage: {accident_metadata['dataset']['geographic_coverage']}",
            f"- Raw collision records: {accident_quality['raw_rows']:,}",
            f"- Hourly cells: {accident_quality['hourly_cells']:,}",
            f"- Time span: {accident_quality['start']} to {accident_quality['end']}",
            f"- Target: severity-weighted collisions/hour "
            f"({accident_quality['severity_weight_definition']})",
            f"- Classes: {', '.join(BAND_NAMES)}; train terciles "
            f"{accident['thresholds']}",
            f"- Split: train {accident['train_rows']:,}, validation "
            f"{accident['validation_rows']:,}, test {accident['test_rows']:,}",
            f"- Selected regressor: {accident['selected_regressor']}",
            f"- Selected classifier: {accident['selected_classifier']}",
            "",
            "### Accident Held-Out Test",
            "",
            "| Predictor | MAE | RMSE | R2 | Accuracy | Macro F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, metrics in accident["test"].items():
        lines.append(
            f"| {name} | {cell(metrics, 'mae', '.3f')} | "
            f"{cell(metrics, 'rmse', '.3f')} | "
            f"{cell(metrics, 'r2', '.4f')} | "
            f"{cell(metrics, 'accuracy', '.4f')} | "
            f"{cell(metrics, 'macro_f1', '.4f')} |"
        )
    selected_accident = accident["test"][
        f"{accident['selected_classifier']}_classifier_refit"
    ]
    lines.extend(
        [
            "",
            f"Accident classifier test accuracy: "
            f"**{selected_accident['accuracy']:.4f}**; macro F1: "
            f"**{selected_accident['macro_f1']:.4f}**; multiclass Brier: "
            f"**{selected_accident['multiclass_brier']:.4f}**; ROC-AUC OVR macro: "
            f"**{selected_accident['roc_auc_ovr_macro']:.4f}**."
            if selected_accident.get("roc_auc_ovr_macro") is not None
            else "",
            "",
            "## Leakage Controls",
            "",
            "- Calendar-only features: hour, weekday, month, weekend.",
            "- No traffic lags, no Mapbox evidence, no target-hour weather, and no "
            "target-derived features.",
            "- Chronological 70/15/15 split with a 24-hour embargo.",
            "- Baselines, class thresholds, and models are fitted on training data; "
            "the test split is untouched until final evaluation.",
            "- The accident model uses collision timestamps only; it does not use "
            "post-departure records.",
            "- No sensor-to-road or UK-to-Karachi corridor mapping is performed.",
            "",
            "## Limitations",
            "",
            "- Both models are trained outside Karachi and are not locally "
            "validated.",
            "- The traffic model is one Minnesota interstate sensor; volume is not "
            "speed, travel time, or congestion.",
            "- The accident model is a Great Britain collision-time prior; it is "
            "not a Karachi crash history, not corridor-matched, and not a "
            "probability that a crash will occur on a Karachi route.",
            "- Calendar-only models cannot distinguish Karachi routes; route "
            "differences are shown separately through OSM context and current "
            "Mapbox evidence.",
            "- Class probabilities are model outputs; they are not calibrated "
            "confidence intervals.",
            "- Mapbox evidence is request-time only and is never used as a label or "
            "feature.",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(line for line in lines if line is not None) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    traffic_frame, traffic_quality = load_traffic()
    accident_frame, accident_quality = load_accidents()

    traffic_train, traffic_validation, traffic_test = split_chronologically(
        traffic_frame
    )
    accident_train, accident_validation, accident_test = split_chronologically(
        accident_frame
    )
    if min(len(traffic_train), len(traffic_validation), len(traffic_test)) < 100:
        raise ValueError("Traffic split is too small for a stable evaluation.")
    if min(len(accident_train), len(accident_validation), len(accident_test)) < 100:
        raise ValueError("Accident split is too small for a stable evaluation.")

    traffic_thresholds = [
        float(traffic_train["traffic_volume"].quantile(1.0 / 3.0)),
        float(traffic_train["traffic_volume"].quantile(2.0 / 3.0)),
    ]
    accident_thresholds = [
        float(accident_train["severity_weight"].quantile(1.0 / 3.0)),
        float(accident_train["severity_weight"].quantile(2.0 / 3.0)),
    ]
    if not traffic_thresholds[0] < traffic_thresholds[1]:
        raise ValueError("Traffic class thresholds are degenerate.")
    if not accident_thresholds[0] < accident_thresholds[1]:
        raise ValueError("Accident class thresholds are degenerate.")

    traffic_evaluation = evaluate_models(
        traffic_train,
        traffic_validation,
        traffic_test,
        "traffic_volume",
        "traffic",
        traffic_thresholds,
    )
    accident_evaluation = evaluate_models(
        accident_train,
        accident_validation,
        accident_test,
        "severity_weight",
        "accident",
        accident_thresholds,
    )

    traffic_regressors, traffic_classifiers = build_traffic_models()
    accident_regressors, accident_classifiers = build_accident_models()

    traffic_train_x = calendar_matrix(
        traffic_train.index.append(traffic_validation.index)
    )
    traffic_train_y = pd.concat(
        [traffic_train["traffic_volume"], traffic_validation["traffic_volume"]]
    )
    traffic_train_band = band_values(traffic_train_y, traffic_thresholds)
    traffic_regressor = traffic_regressors[traffic_evaluation["selected_regressor"]]
    traffic_classifier = traffic_classifiers[
        traffic_evaluation["selected_classifier"]
    ]
    traffic_regressor.fit(traffic_train_x, traffic_train_y)
    traffic_classifier.fit(traffic_train_x, traffic_train_band)

    accident_train_x = calendar_matrix(
        accident_train.index.append(accident_validation.index)
    )
    accident_train_y = pd.concat(
        [accident_train["severity_weight"], accident_validation["severity_weight"]]
    )
    accident_train_band = band_values(accident_train_y, accident_thresholds)
    accident_regressor = accident_regressors[
        accident_evaluation["selected_regressor"]
    ]
    accident_classifier = accident_classifiers[
        accident_evaluation["selected_classifier"]
    ]
    accident_regressor.fit(accident_train_x, accident_train_y)
    accident_classifier.fit(accident_train_x, accident_train_band)

    joblib.dump(
        {
            "model_version": MODEL_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "band_names": list(BAND_NAMES),
            "thresholds": traffic_thresholds,
            "regressor": traffic_regressor,
            "classifier": traffic_classifier,
        },
        MODEL_DIR / "traffic_model.joblib",
        compress=3,
    )
    joblib.dump(
        {
            "model_version": MODEL_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "band_names": list(BAND_NAMES),
            "thresholds": accident_thresholds,
            "severity_weights": SEVERITY_WEIGHTS,
            "regressor": accident_regressor,
            "classifier": accident_classifier,
        },
        MODEL_DIR / "accident_model.joblib",
        compress=3,
    )

    traffic_metadata = {
        "dataset": {
            "name": "Metro Interstate Traffic Volume",
            "source_url": TRAFFIC_SOURCE_URL,
            "download_url": TRAFFIC_DOWNLOAD_URL,
            "doi": TRAFFIC_DOI,
            "license": TRAFFIC_LICENSE,
            "attribution": TRAFFIC_ATTRIBUTION,
            "geographic_coverage": TRAFFIC_COVERAGE,
            "target": "traffic_volume",
            "target_units": "vehicles per hour",
            "time_span": [traffic_quality["start"], traffic_quality["end"]],
            "raw_rows": traffic_quality["raw_rows"],
            "usable_rows": traffic_quality["usable_rows"],
            "missing_hours": traffic_quality["missing_hours"],
            "sha256": sha256(TRAFFIC_CSV),
            "csv_gz": str(TRAFFIC_CSV.relative_to(ROOT)).replace("\\", "/"),
        },
        "target": {
            "name": "traffic_volume",
            "units": "vehicles per hour",
            "classes": list(BAND_NAMES),
            "thresholds": traffic_thresholds,
            "threshold_basis": "train split terciles",
        },
        "model": {
            "regressor": traffic_evaluation["selected_regressor"],
            "classifier": traffic_evaluation["selected_classifier"],
            "features": list(FEATURE_NAMES),
            "feature_set": "calendar_only",
        },
        "metrics_file": "evaluation.json",
        "limitations": [
            "Single Minnesota interstate sensor; not a Karachi route network.",
            "Target is vehicle volume, not speed, travel time, or congestion.",
            "No Karachi observations or route identifiers.",
            "Class probabilities are model outputs, not calibrated intervals.",
        ],
    }
    accident_metadata = {
        "dataset": {
            "name": "STATS19 Road Safety Data (collision tables)",
            "source_url": ACCIDENT_SOURCE_URL,
            "license": ACCIDENT_LICENSE,
            "attribution": ACCIDENT_ATTRIBUTION,
            "geographic_coverage": ACCIDENT_COVERAGE,
            "time_span": [accident_quality["start"], accident_quality["end"]],
            "raw_rows": accident_quality["raw_rows"],
            "hourly_cells": accident_quality["hourly_cells"],
            "files": [
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in sorted(ACCIDENT_DIR.glob("collision-*.csv"))
            ],
            "severity_weights": SEVERITY_WEIGHTS,
            "severity_counts": accident_quality["severity_counts"],
        },
        "target": {
            "name": "severity_weighted_collisions_per_hour",
            "units": "severity-weighted collisions per hour",
            "classes": list(BAND_NAMES),
            "thresholds": accident_thresholds,
            "threshold_basis": "train split terciles",
        },
        "model": {
            "regressor": accident_evaluation["selected_regressor"],
            "classifier": accident_evaluation["selected_classifier"],
            "features": list(FEATURE_NAMES),
            "feature_set": "calendar_only",
        },
        "metrics_file": "evaluation.json",
        "limitations": [
            "Great Britain collision data; not Karachi crash records.",
            "Calendar-only prior; not corridor-matched to Karachi routes.",
            "Not a probability that a crash will occur on a selected route.",
            "Severity weights are a documented policy choice.",
        ],
    }
    metadata = {
        "artifact_version": 1,
        "model_version": MODEL_VERSION,
        "trained": True,
        "model_kind": "calendar_only_external_experimental",
        "route_applicable": False,
        "karachi_validated": False,
        "applicability": "experimental_external",
        "disclosure": DISCLOSURE,
        "accident_disclosure": ACCIDENT_DISCLOSURE,
        "traffic": traffic_metadata,
        "accident": accident_metadata,
        "split": {
            "strategy": "chronological",
            "train_fraction": TRAIN_FRACTION,
            "validation_fraction": VALIDATION_FRACTION,
            "embargo_hours": EMBARGO_HOURS,
            "traffic": traffic_evaluation["split"],
            "accident": accident_evaluation["split"],
        },
        "leakage_controls": [
            "Calendar-only features; no lags, no Mapbox, no target-hour weather.",
            "Chronological split with a 24-hour embargo.",
            "Thresholds and models fit on training data only.",
            "Test split untouched during model selection.",
            "No sensor-to-road or UK-to-Karachi corridor mapping.",
        ],
        "features": list(FEATURE_NAMES),
        "feature_definitions_file": "feature_definitions.json",
        "evaluation_file": "evaluation.json",
        "evaluation_report": "reports/departure_forecast_evaluation.md",
        "limitations": [
            "Experimental external-data model; not validated for Gulistan-e-Jauhar.",
            "Traffic model is one Minnesota sensor; not Karachi traffic.",
            "Accident model is a Great Britain calendar prior; not Karachi crash data.",
            "No route-specific training labels are available.",
            "Mapbox is not a training source and remains request-time evidence.",
        ],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }

    evaluation = {
        "selected_model_version": MODEL_VERSION,
        "traffic": traffic_evaluation,
        "accident": accident_evaluation,
        "traffic_quality": traffic_quality,
        "accident_quality": accident_quality,
    }
    save_json(MODEL_DIR / "metadata.json", metadata)
    save_json(MODEL_DIR / "evaluation.json", evaluation)
    write_feature_definitions(MODEL_DIR / "feature_definitions.json")

    provenance = {
        "dataset": accident_metadata["dataset"],
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "UK Department for Transport road safety open data",
        "license": ACCIDENT_LICENSE,
        "coverage": ACCIDENT_COVERAGE,
        "time_span": accident_quality["start"] + " to " + accident_quality["end"],
        "raw_rows": accident_quality["raw_rows"],
        "hourly_cells": accident_quality["hourly_cells"],
        "features_used": list(FEATURE_NAMES),
        "target": accident_metadata["target"],
        "limitations": accident_metadata["limitations"],
    }
    save_json(PROVENANCE_PATH, provenance)
    write_report(
        evaluation,
        traffic_quality,
        accident_quality,
        traffic_metadata,
        accident_metadata,
    )

    print(
        json.dumps(
            {
                "model_version": MODEL_VERSION,
                "traffic": {
                    "rows": traffic_quality["usable_rows"],
                    "selected_regressor": traffic_evaluation["selected_regressor"],
                    "selected_classifier": traffic_evaluation["selected_classifier"],
                    "test": traffic_evaluation["test"],
                },
                "accident": {
                    "rows": accident_quality["raw_rows"],
                    "selected_regressor": accident_evaluation["selected_regressor"],
                    "selected_classifier": accident_evaluation["selected_classifier"],
                    "test": accident_evaluation["test"],
                },
                "artifacts": [
                    str((MODEL_DIR / name).relative_to(ROOT)).replace("\\", "/")
                    for name in (
                        "traffic_model.joblib",
                        "accident_model.joblib",
                        "metadata.json",
                        "evaluation.json",
                        "feature_definitions.json",
                    )
                ],
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
