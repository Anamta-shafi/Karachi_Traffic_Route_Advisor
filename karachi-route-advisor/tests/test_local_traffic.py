from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.local_traffic import (
    MIN_OBSERVATIONS,
    LocalTrafficError,
    append_local_observations,
    load_local_observations,
    local_traffic_summary,
    local_traffic_template,
    summarize_local_observations,
    validate_local_observations,
)


def observations(rows: int = 2) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="h")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "route_id": ["osm-1"] * rows,
            "segment_id": ["segment-001"] * rows,
            "travel_time_min": np.linspace(3.0, 5.0, rows),
        }
    )


def test_validate_accepts_required_columns_and_case_insensitive_headers() -> None:
    frame = observations(2).rename(
        columns={
            "timestamp": " Timestamp ",
            "route_id": "ROUTE_ID",
            "segment_id": "Segment_ID",
            "travel_time_min": "Travel_Time_Min",
        }
    )
    valid = validate_local_observations(frame)
    assert list(valid.columns) == [
        "timestamp",
        "route_id",
        "segment_id",
        "travel_time_min",
    ]
    assert len(valid) == 2


def test_missing_required_column_is_rejected() -> None:
    with pytest.raises(LocalTrafficError, match="segment_id"):
        validate_local_observations(observations().drop(columns=["segment_id"]))


def test_nonpositive_travel_time_is_rejected() -> None:
    frame = observations()
    frame.loc[0, "travel_time_min"] = 0
    with pytest.raises(LocalTrafficError, match="positive finite"):
        validate_local_observations(frame)


def test_duplicate_keys_are_rejected() -> None:
    frame = observations()
    frame = pd.concat([frame, frame.tail(1)], ignore_index=True)
    with pytest.raises(LocalTrafficError, match="Duplicate"):
        validate_local_observations(frame)


def test_optional_numeric_columns_are_validated() -> None:
    frame = observations()
    frame["distance_km"] = 1.0
    frame.loc[0, "distance_km"] = -1.0
    with pytest.raises(LocalTrafficError, match="distance_km"):
        validate_local_observations(frame)


def test_append_deduplicates_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "local.csv"
    first = append_local_observations(observations(2), path)
    assert first.rows == 2
    assert path.exists()

    extra = observations(3)
    second = append_local_observations(extra, path)
    assert second.rows == 3
    assert second.duplicates_removed == 2
    loaded = load_local_observations(path)
    assert len(loaded) == 3


def test_summary_ready_at_minimum_volume(tmp_path: Path) -> None:
    frame = observations(MIN_OBSERVATIONS)
    summary = summarize_local_observations(frame, path=tmp_path / "local.csv")
    assert summary.rows == MIN_OBSERVATIONS
    assert summary.route_count == 1
    assert summary.segment_count == 1
    assert summary.ready_for_training is True
    assert "future Karachi-specific training" in summary.message


def test_missing_file_summary_is_not_ready(tmp_path: Path) -> None:
    summary = local_traffic_summary(tmp_path / "missing.csv")
    assert summary.exists is False
    assert summary.rows == 0
    assert summary.ready_for_training is False
    assert "does not collect data automatically" in summary.message


def test_malformed_file_summary_reports_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("timestamp,route_id\n2026-01-01T00:00:00,osm-1\n")
    summary = local_traffic_summary(path)
    assert summary.exists is True
    assert summary.rows == 0
    assert summary.ready_for_training is False
    assert "could not be validated" in summary.message


def test_template_contains_required_headers() -> None:
    header = local_traffic_template().splitlines()[0]
    for column in (
        "timestamp",
        "route_id",
        "segment_id",
        "travel_time_min",
    ):
        assert column in header
