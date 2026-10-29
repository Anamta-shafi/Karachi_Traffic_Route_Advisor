from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT

LOCAL_DATA_DIR = PROJECT_ROOT / "data" / "local"
LOCAL_DATA_PATH = LOCAL_DATA_DIR / "traffic_observations.csv"

REQUIRED_COLUMNS = ("timestamp", "route_id", "segment_id", "travel_time_min")
OPTIONAL_COLUMNS = ("distance_km", "speed_kph", "source", "notes")
KEY_COLUMNS = ("timestamp", "route_id", "segment_id")
MIN_OBSERVATIONS = 168


class LocalTrafficError(ValueError):
    pass


@dataclass(frozen=True)
class LocalTrafficSummary:
    path: str
    exists: bool
    rows: int
    route_count: int
    segment_count: int
    start: str | None
    end: str | None
    ready_for_training: bool
    message: str
    duplicates_removed: int = 0


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise LocalTrafficError("Local observations must be a pandas DataFrame.")
    normalised = [str(column).strip().lower() for column in frame.columns]
    if len(normalised) != len(set(normalised)):
        raise LocalTrafficError(
            "Local-observation columns are duplicated after normalisation."
        )
    missing = [name for name in REQUIRED_COLUMNS if name not in normalised]
    if missing:
        raise LocalTrafficError(
            "Missing required local-observation columns: " + ", ".join(missing)
        )
    frame = frame.rename(
        columns={
            column: str(column).strip().lower() for column in frame.columns
        }
    )
    columns = [
        *REQUIRED_COLUMNS,
        *(name for name in OPTIONAL_COLUMNS if name in frame.columns),
    ]
    return frame[columns].copy()


def validate_local_observations(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _normalise_columns(frame)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any():
        raise LocalTrafficError("timestamp contains invalid or missing values.")
    for column in ("route_id", "segment_id"):
        frame[column] = frame[column].astype("string").str.strip()
        if frame[column].isna().any() or frame[column].eq("").any():
            raise LocalTrafficError(f"{column} contains empty values.")
    frame["travel_time_min"] = pd.to_numeric(
        frame["travel_time_min"], errors="coerce"
    )
    invalid_time = ~np.isfinite(frame["travel_time_min"]) | (
        frame["travel_time_min"] <= 0
    )
    if invalid_time.any():
        raise LocalTrafficError(
            "travel_time_min must contain positive finite minutes."
        )
    for column in ("distance_km", "speed_kph"):
        if column not in frame.columns:
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & (
            ~np.isfinite(frame[column]) | (frame[column] <= 0)
        )
        if invalid.any():
            raise LocalTrafficError(f"{column} must be positive and finite.")
    duplicates = frame.duplicated(subset=list(KEY_COLUMNS), keep=False)
    if duplicates.any():
        count = int(duplicates.sum())
        raise LocalTrafficError(
            "Duplicate timestamp/route/segment observations are not allowed; "
            f"found {count} rows in duplicate groups."
        )
    return frame.sort_values("timestamp").reset_index(drop=True)


def summarize_local_observations(
    frame: pd.DataFrame,
    *,
    path: Path = LOCAL_DATA_PATH,
    duplicates_removed: int = 0,
) -> LocalTrafficSummary:
    valid = validate_local_observations(frame)
    rows = len(valid)
    route_count = int(valid["route_id"].nunique()) if rows else 0
    segment_count = int(valid["segment_id"].nunique()) if rows else 0
    ready = rows >= MIN_OBSERVATIONS and route_count > 0 and segment_count > 0
    if rows == 0:
        message = (
            "No opt-in local traffic observations have been imported. The app "
            "does not collect data automatically."
        )
    elif ready:
        message = (
            f"{rows:,} local observations across {route_count} route(s) and "
            f"{segment_count} segment(s) are available for a future "
            "Karachi-specific training experiment. No model is trained "
            "automatically."
        )
    else:
        message = (
            f"{rows:,} local observations across {route_count} route(s) and "
            f"{segment_count} segment(s) are stored. At least {MIN_OBSERVATIONS} "
            "observations are needed before a preliminary route-specific training "
            "experiment can be considered."
        )
    return LocalTrafficSummary(
        path=str(path),
        exists=Path(path).exists(),
        rows=rows,
        route_count=route_count,
        segment_count=segment_count,
        start=valid["timestamp"].min().isoformat() if rows else None,
        end=valid["timestamp"].max().isoformat() if rows else None,
        ready_for_training=ready,
        message=message,
        duplicates_removed=duplicates_removed,
    )


def load_local_observations(path: Path = LOCAL_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS))
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise LocalTrafficError(
            f"Could not read local observations: {type(exc).__name__}: {exc}"
        ) from exc
    return validate_local_observations(frame)


def local_traffic_summary(path: Path = LOCAL_DATA_PATH) -> LocalTrafficSummary:
    path = Path(path)
    if not path.exists():
        return summarize_local_observations(
            pd.DataFrame(columns=list(REQUIRED_COLUMNS)),
            path=path,
        )
    try:
        frame = load_local_observations(path)
    except Exception as exc:
        return LocalTrafficSummary(
            path=str(path),
            exists=True,
            rows=0,
            route_count=0,
            segment_count=0,
            start=None,
            end=None,
            ready_for_training=False,
            message=(
                "Local observations could not be validated: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
    return summarize_local_observations(frame, path=path)


def append_local_observations(
    frame: pd.DataFrame,
    path: Path = LOCAL_DATA_PATH,
) -> LocalTrafficSummary:
    path = Path(path)
    new_rows = validate_local_observations(frame)
    existing = load_local_observations(path)
    combined = pd.concat([existing, new_rows], ignore_index=True)
    before = len(combined)
    combined = (
        combined.drop_duplicates(subset=list(KEY_COLUMNS), keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    duplicates_removed = before - len(combined)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return summarize_local_observations(
        combined,
        path=path,
        duplicates_removed=duplicates_removed,
    )


def local_traffic_template() -> str:
    return (
        "timestamp,route_id,segment_id,travel_time_min,distance_km,speed_kph,"
        "source,notes\n"
        "2026-10-27T08:00:00+05:00,osm-1,segment-001,4.5,1.2,16.0,manual,\n"
    )
