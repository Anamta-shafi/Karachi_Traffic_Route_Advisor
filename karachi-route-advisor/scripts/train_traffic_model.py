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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "external" / "metro_interstate"
DATA_ZIP = DATA_DIR / "metro_interstate_traffic_volume.zip"
DATA_CSV = DATA_DIR / "extracted" / "Metro_Interstate_Traffic_Volume.csv.gz"
MODEL_DIR = ROOT / "models" / "traffic_flow"
REPORT_DIR = ROOT / "reports"

LAGS = (1, 2, 3, 6, 12, 24, 48, 168)
FEATURES = [
    *(f"lag_{lag}" for lag in LAGS),
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
]
TARGET = "traffic_volume"
RANDOM_STATE = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
SOURCE_URL = "https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume"
DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/492/"
    "metro+interstate+traffic+volume.zip"
)
DATASET_DOI = "10.24432/C5X60B"
LICENSE = "CC BY 4.0"
ATTRIBUTION = "Hogue, J. (2019). Metro Interstate Traffic Volume [Dataset]. UCI Machine Learning Repository."
GEOGRAPHIC_COVERAGE = (
    "I-94 westbound ATR station 301, Minneapolis-St Paul, Minnesota, USA"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw() -> pd.DataFrame:
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"Missing {DATA_CSV}. Download the UCI archive first."
        )
    frame = pd.read_csv(DATA_CSV, parse_dates=["date_time"])
    frame = frame.sort_values("date_time").reset_index(drop=True)
    return frame


def prepare_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    raw_rows = len(raw)
    duplicate_timestamp_rows = int(raw["date_time"].duplicated().sum())
    unique = (
        raw.drop_duplicates(subset=["date_time"], keep="first")
        .set_index("date_time")
        .sort_index()
    )
    expected_index = pd.date_range(unique.index.min(), unique.index.max(), freq="h")
    missing_hours = int(len(expected_index.difference(unique.index)))
    hourly = unique.reindex(expected_index)
    hourly.index.name = "date_time"

    volume = hourly[TARGET]
    for lag in LAGS:
        hourly[f"lag_{lag}"] = volume.shift(lag)

    index = hourly.index
    hour = index.hour.to_numpy()
    dow = index.day_of_week.to_numpy()
    month = index.month.to_numpy()
    hourly["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    hourly["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    hourly["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    hourly["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    hourly["month_sin"] = np.sin(2.0 * np.pi * (month - 1) / 12.0)
    hourly["month_cos"] = np.cos(2.0 * np.pi * (month - 1) / 12.0)
    hourly["is_weekend"] = (dow >= 5).astype(int)
    hourly["is_holiday"] = hourly["holiday"].notna().astype(int)

    prepared = hourly.dropna(subset=[TARGET, *FEATURES]).copy()
    prepared[FEATURES] = prepared[FEATURES].astype(float)
    prepared[TARGET] = prepared[TARGET].astype(float)

    quality = {
        "raw_rows": raw_rows,
        "unique_timestamp_rows": int(len(unique)),
        "duplicate_timestamp_rows_removed": duplicate_timestamp_rows,
        "duplicate_note": (
            "Duplicate timestamps had identical traffic_volume values; one row "
            "was retained and differing weather descriptions were not used."
        ),
        "raw_start": raw["date_time"].min().isoformat(),
        "raw_end": raw["date_time"].max().isoformat(),
        "hourly_range_start": expected_index.min().isoformat(),
        "hourly_range_end": expected_index.max().isoformat(),
        "expected_hours": int(len(expected_index)),
        "missing_hours": missing_hours,
        "usable_rows": int(len(prepared)),
        "usable_start": prepared.index.min().isoformat(),
        "usable_end": prepared.index.max().isoformat(),
        "zero_volume_rows": int((prepared[TARGET] == 0).sum()),
        "target_min": float(prepared[TARGET].min()),
        "target_max": float(prepared[TARGET].max()),
        "target_mean": float(prepared[TARGET].mean()),
    }
    return prepared, quality


def split_chronologically(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = len(frame)
    train_end = int(total * TRAIN_FRACTION)
    validation_end = int(total * (TRAIN_FRACTION + VALIDATION_FRACTION))
    train = frame.iloc[:train_end].copy()
    validation = frame.iloc[train_end:validation_end].copy()
    test = frame.iloc[validation_end:].copy()
    return train, validation, test


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def calendar_mean_baseline(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> np.ndarray:
    train_keys = pd.MultiIndex.from_arrays(
        [train.index.day_of_week, train.index.hour],
        names=["dow", "hour"],
    )
    evaluation_keys = pd.MultiIndex.from_arrays(
        [evaluation.index.day_of_week, evaluation.index.hour],
        names=["dow", "hour"],
    )
    means = train.groupby(train_keys)[TARGET].mean()
    return means.reindex(evaluation_keys).fillna(train[TARGET].mean()).to_numpy()


def build_models() -> dict[str, Pipeline]:
    return {
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=300,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        min_samples_leaf=20,
                        l2_regularization=0.1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=120,
                        max_depth=16,
                        min_samples_leaf=3,
                        max_features=0.8,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def baseline_predictions(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> dict[str, np.ndarray]:
    return {
        "persistence_lag_1": evaluation["lag_1"].to_numpy(),
        "seasonal_naive_lag_24": evaluation["lag_24"].to_numpy(),
        "calendar_mean": calendar_mean_baseline(train, evaluation),
    }


def fit_and_evaluate(
    models: dict[str, Pipeline],
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> tuple[dict[str, Pipeline], dict[str, dict[str, float]]]:
    fitted: dict[str, Pipeline] = {}
    scores: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        model.fit(train[FEATURES], train[TARGET])
        predicted = model.predict(evaluation[FEATURES])
        scores[name] = metrics(evaluation[TARGET], predicted)
        fitted[name] = model
    return fitted, scores


def residual_summary(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    residual = np.asarray(actual) - np.asarray(predicted)
    absolute = np.abs(residual)
    return {
        "residual_p05": float(np.quantile(residual, 0.05)),
        "residual_p50": float(np.quantile(residual, 0.50)),
        "residual_p95": float(np.quantile(residual, 0.95)),
        "absolute_error_p90": float(np.quantile(absolute, 0.90)),
    }


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_feature_definitions(path: Path) -> None:
    descriptions = {
        **{
            f"lag_{lag}": f"Observed traffic volume {lag} hour(s) before the target hour."
            for lag in LAGS
        },
        "hour_sin": "Sine encoding of local hour of day.",
        "hour_cos": "Cosine encoding of local hour of day.",
        "dow_sin": "Sine encoding of day of week.",
        "dow_cos": "Cosine encoding of day of week.",
        "month_sin": "Sine encoding of month.",
        "month_cos": "Cosine encoding of month.",
        "is_weekend": "1 when the target date is Saturday or Sunday.",
        "is_holiday": "1 when the target date is a listed US/Minnesota holiday.",
    }
    payload = {
        "features": [
            {"name": name, "description": descriptions[name]} for name in FEATURES
        ],
        "target": {
            "name": TARGET,
            "units": "vehicles per hour",
            "definition": "Hourly westbound I-94 volume at ATR station 301.",
        },
        "prediction_horizon": "1 hour",
        "leakage_controls": [
            "All lag features use only observations strictly before the target hour.",
            "Chronological 70/15/15 split; no random shuffling.",
            "The held-out test split is not used for model selection.",
            "Weather at the target hour is excluded because it would not be known at forecast time.",
            "Mapbox ETA differences are not used as labels or features.",
        ],
    }
    save_json(path, payload)


def write_report(
    path: Path,
    quality: dict[str, object],
    validation: dict[str, dict[str, float]],
    test: dict[str, dict[str, float]],
    selected: str,
    selected_test: dict[str, float],
    residuals: dict[str, float],
    split_ranges: dict[str, dict[str, str]],
) -> None:
    baseline_test = test["calendar_mean"]
    lines = [
        "# External Traffic-Volume Model Evaluation",
        "",
        "This is an external-region benchmark, not a Karachi route model.",
        "",
        "## Dataset",
        "",
        f"- Source: UCI Metro Interstate Traffic Volume ({DATASET_DOI})",
        f"- Coverage: {GEOGRAPHIC_COVERAGE}",
        f"- Licence: {LICENSE}",
        f"- Raw rows: {quality['raw_rows']:,}",
        f"- Unique timestamp rows: {quality['unique_timestamp_rows']:,}",
        f"- Missing hourly slots: {quality['missing_hours']:,}",
        f"- Usable lagged rows: {quality['usable_rows']:,}",
        f"- Raw time span: {quality['raw_start']} to {quality['raw_end']}",
        f"- Usable time span: {quality['usable_start']} to {quality['usable_end']}",
        "",
        "Duplicate timestamps were collapsed only because their traffic-volume "
        "values were identical. Target values were never interpolated.",
        "",
        "## Chronological Split",
        "",
        f"- Train: {split_ranges['train']['rows']} rows, "
        f"{split_ranges['train']['start']} to {split_ranges['train']['end']}",
        f"- Validation: {split_ranges['validation']['rows']} rows, "
        f"{split_ranges['validation']['start']} to {split_ranges['validation']['end']}",
        f"- Test: {split_ranges['test']['rows']} rows, "
        f"{split_ranges['test']['start']} to {split_ranges['test']['end']}",
        "",
        "## Validation Comparison",
        "",
        "| Predictor | MAE | RMSE | R2 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, score in validation.items():
        lines.append(
            f"| {name} | {score['mae']:.3f} | {score['rmse']:.3f} | {score['r2']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Validation MAE selected **{selected}**.",
            "",
            "## Held-Out Future Test",
            "",
            "| Predictor | MAE | RMSE | R2 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, score in test.items():
        lines.append(
            f"| {name} | {score['mae']:.3f} | {score['rmse']:.3f} | {score['r2']:.4f} |"
        )
    improvement = (baseline_test["mae"] - selected_test["mae"]) / baseline_test["mae"]
    lines.extend(
        [
            "",
            f"Selected model test MAE: **{selected_test['mae']:.3f} vehicles/hour**.",
            f"Calendar-mean baseline test MAE: **{baseline_test['mae']:.3f} vehicles/hour**.",
            f"MAE improvement over calendar baseline: **{improvement:.1%}**.",
            f"Selected model test RMSE: **{selected_test['rmse']:.3f}**; "
            f"R2: **{selected_test['r2']:.4f}**.",
            "",
            "## Uncertainty",
            "",
            "These are empirical held-out residuals, not a calibrated confidence "
            "interval.",
            "",
            f"- Residual p05: {residuals['residual_p05']:.1f}",
            f"- Residual median: {residuals['residual_p50']:.1f}",
            f"- Residual p95: {residuals['residual_p95']:.1f}",
            f"- Absolute-error p90: {residuals['absolute_error_p90']:.1f}",
            "",
            "## Applicability To Gulistan-e-Jauhar",
            "",
            "This model is **not applicable** to the selected Karachi route. It "
            "predicts hourly vehicle volume at one Minnesota interstate sensor. The "
            "dataset has no Karachi observations, no Karachi road identifiers, no "
            "route geometry, and no speed or travel-time target. No sensor IDs were "
            "mapped to Karachi roads. The application therefore reports the "
            "selected-route ML forecast as unavailable.",
            "",
            "## Limitations",
            "",
            "- Single external sensor, not a route network.",
            "- Target is vehicle volume, not congestion, speed, or travel time.",
            "- Absolute timestamps are local CST and cover 2012-2018.",
            "- Missing hours remain missing; lags are shifted over the true hourly index.",
            "- The model requires recent lag observations and is not a long-horizon "
            "departure forecast.",
            "- The test split is held out, but this is not an independent Karachi "
            "validation.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not DATA_ZIP.exists() or not DATA_CSV.exists():
        raise FileNotFoundError(
            "Download and extract the UCI Metro Interstate archive first."
        )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    prepared, quality = prepare_dataset(raw)
    train, validation, test = split_chronologically(prepared)
    split_ranges = {
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

    baseline_validation = {
        name: metrics(validation[TARGET], predicted)
        for name, predicted in baseline_predictions(train, validation).items()
    }
    models = build_models()
    fitted, validation_scores = fit_and_evaluate(models, train, validation)
    validation_all = {**baseline_validation, **validation_scores}
    selected = min(
        validation_scores,
        key=lambda name: validation_scores[name]["mae"],
    )

    combined = pd.concat([train, validation])
    fitted_final, test_scores = fit_and_evaluate(models, combined, test)
    baseline_test = {
        name: metrics(test[TARGET], predicted)
        for name, predicted in baseline_predictions(combined, test).items()
    }
    test_all = {**baseline_test, **test_scores}
    selected_model = fitted_final[selected]
    selected_prediction = selected_model.predict(test[FEATURES])
    selected_test = test_scores[selected]
    residuals = residual_summary(test[TARGET], selected_prediction)

    model_path = MODEL_DIR / "traffic_volume_model.joblib"
    joblib.dump(selected_model, model_path, compress=3)
    write_feature_definitions(MODEL_DIR / "feature_definitions.json")
    write_report(
        REPORT_DIR / "traffic_model_evaluation.md",
        quality,
        validation_all,
        test_all,
        selected,
        selected_test,
        residuals,
        split_ranges,
    )

    retrieved_at = datetime.fromtimestamp(
        DATA_ZIP.stat().st_mtime, timezone.utc
    ).isoformat()
    evaluation = {
        "selected_model": selected,
        "selection_metric": "validation_mae",
        "selection_rule": "Lowest validation MAE; test split not used for selection.",
        "validation": validation_all,
        "test": test_all,
        "selected_test_residuals": residuals,
        "test_rows": int(len(test)),
        "test_actual_mean": float(test[TARGET].mean()),
        "test_actual_std": float(test[TARGET].std()),
        "prediction_horizon": "1 hour",
        "target_units": "vehicles per hour",
    }
    save_json(MODEL_DIR / "evaluation.json", evaluation)

    metadata = {
        "artifact_version": 1,
        "trained": True,
        "model_kind": selected,
        "model_file": model_path.name,
        "target": {
            "name": TARGET,
            "units": "vehicles per hour",
            "definition": "Hourly westbound I-94 volume at ATR station 301.",
            "supports_travel_time": False,
            "supports_congestion_level": False,
            "supports_volume": True,
        },
        "prediction_horizon": "1 hour",
        "route_applicable": False,
        "route_applicability_reason": (
            "The dataset covers one interstate sensor in Minnesota, USA. It has no "
            "Karachi observations, road identifiers, route geometry, speed, or "
            "travel-time target, so it cannot support a Gulistan-e-Jauhar route "
            "forecast."
        ),
        "coverage": {
            "description": GEOGRAPHIC_COVERAGE,
            "city": "Minneapolis-St Paul",
            "region": "Minnesota",
            "country": "USA",
            "route_level": False,
            "karachi_coverage": False,
        },
        "dataset": {
            "name": "Metro Interstate Traffic Volume",
            "uci_id": 492,
            "doi": DATASET_DOI,
            "source_url": SOURCE_URL,
            "download_url": DOWNLOAD_URL,
            "retrieved_at_utc": retrieved_at,
            "license": LICENSE,
            "attribution": ATTRIBUTION,
            "zip_file": str(DATA_ZIP.relative_to(ROOT)).replace("\\", "/"),
            "zip_size_bytes": DATA_ZIP.stat().st_size,
            "zip_sha256": sha256(DATA_ZIP),
            "csv_gz_file": str(DATA_CSV.relative_to(ROOT)).replace("\\", "/"),
            "csv_gz_size_bytes": DATA_CSV.stat().st_size,
            "csv_gz_sha256": sha256(DATA_CSV),
            "geographic_coverage": GEOGRAPHIC_COVERAGE,
            "target": TARGET,
            "target_units": "vehicles per hour",
            "time_span_raw": [quality["raw_start"], quality["raw_end"]],
            "time_span_usable": [quality["usable_start"], quality["usable_end"]],
            "raw_rows": quality["raw_rows"],
            "unique_timestamp_rows": quality["unique_timestamp_rows"],
            "usable_rows": quality["usable_rows"],
            "missing_hours": quality["missing_hours"],
            "duplicate_timestamp_rows_removed": quality[
                "duplicate_timestamp_rows_removed"
            ],
            "limitations": [
                "Single external interstate sensor, not a Karachi route network.",
                "Target is vehicle volume, not speed, travel time, or congestion.",
                "No Karachi road identifiers or route geometry.",
                "No independent Karachi validation.",
                "Absolute timestamps are local CST, 2012-2018.",
                "Missing hours are not interpolated.",
                "Duplicate timestamp rows were collapsed only when volume was identical.",
            ],
        },
        "features": FEATURES,
        "feature_definitions_file": "feature_definitions.json",
        "evaluation_file": "evaluation.json",
        "evaluation_report": "reports/traffic_model_evaluation.md",
        "split": {
            "strategy": "chronological",
            "fractions": {
                "train": TRAIN_FRACTION,
                "validation": VALIDATION_FRACTION,
                "test": 1.0 - TRAIN_FRACTION - VALIDATION_FRACTION,
            },
            "ranges": split_ranges,
        },
        "quality": quality,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "limitations": [
            "External-region benchmark only; not validated for Karachi.",
            "Not used in custom A* route costs.",
            "Not a congestion, speed, or travel-time model.",
            "Requires recent lag observations for a one-hour-ahead forecast.",
            "No calibrated prediction interval.",
        ],
    }
    save_json(MODEL_DIR / "metadata.json", metadata)
    save_json(
        DATA_DIR / "provenance.json",
        {
            "dataset": metadata["dataset"],
            "selected_model": selected,
            "model_artifact": str(model_path.relative_to(ROOT)).replace("\\", "/"),
            "evaluation": "models/traffic_flow/evaluation.json",
            "route_applicable": False,
        },
    )

    print(json.dumps(
        {
            "dataset": "UCI Metro Interstate Traffic Volume (492)",
            "raw_rows": quality["raw_rows"],
            "unique_timestamp_rows": quality["unique_timestamp_rows"],
            "usable_rows": quality["usable_rows"],
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "selected_model": selected,
            "validation_scores": validation_all,
            "test_scores": test_all,
            "selected_test": selected_test,
            "model_path": str(model_path),
            "model_size_bytes": model_path.stat().st_size,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
