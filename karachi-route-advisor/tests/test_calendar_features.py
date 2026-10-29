from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
import pytest

from src.calendar_features import (
    FEATURE_NAMES,
    calendar_feature_frame,
    calendar_feature_matrix,
    calendar_features,
)


def test_calendar_features_are_cyclical_and_weekend_aware() -> None:
    friday_evening = calendar_features(datetime(2026, 10, 30, 18, 0))
    assert tuple(friday_evening) == FEATURE_NAMES
    assert friday_evening["hour_sin"] == pytest.approx(math.sin(2 * math.pi * 18 / 24))
    assert friday_evening["hour_cos"] == pytest.approx(math.cos(2 * math.pi * 18 / 24))
    assert friday_evening["dow_sin"] == pytest.approx(
        math.sin(2 * math.pi * 4 / 7)
    )
    assert friday_evening["dow_cos"] == pytest.approx(
        math.cos(2 * math.pi * 4 / 7)
    )
    assert friday_evening["is_weekend"] == 0.0

    saturday = calendar_features(datetime(2026, 10, 31, 0, 0))
    assert saturday["is_weekend"] == 1.0
    assert saturday["hour_sin"] == pytest.approx(0.0)
    assert saturday["hour_cos"] == pytest.approx(1.0)


def test_calendar_feature_frame_matches_single_datetime() -> None:
    departure = datetime(2026, 1, 15, 12, 0)
    frame = calendar_feature_frame(departure)
    assert list(frame.columns) == list(FEATURE_NAMES)
    assert len(frame) == 1
    for name in FEATURE_NAMES:
        assert frame.iloc[0][name] == pytest.approx(
            calendar_features(departure)[name]
        )


def test_calendar_feature_matrix_matches_rows() -> None:
    index = pd.DatetimeIndex(
        ["2026-01-15 12:00", "2026-01-17 03:00", "2026-07-01 20:00"]
    )
    matrix = calendar_feature_matrix(index)
    assert list(matrix.columns) == list(FEATURE_NAMES)
    assert len(matrix) == len(index)
    for timestamp, (_, row) in zip(index, matrix.iterrows()):
        expected = calendar_features(timestamp.to_pydatetime())
        for name in FEATURE_NAMES:
            assert row[name] == pytest.approx(expected[name])
