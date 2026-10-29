from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

FEATURE_NAMES: tuple[str, ...] = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
)


def calendar_features(departure: datetime) -> dict[str, float]:
    hour = int(departure.hour)
    day_of_week = int(departure.weekday())
    month = int(departure.month)
    return {
        "hour_sin": float(math.sin(2.0 * math.pi * hour / 24.0)),
        "hour_cos": float(math.cos(2.0 * math.pi * hour / 24.0)),
        "dow_sin": float(math.sin(2.0 * math.pi * day_of_week / 7.0)),
        "dow_cos": float(math.cos(2.0 * math.pi * day_of_week / 7.0)),
        "month_sin": float(math.sin(2.0 * math.pi * (month - 1) / 12.0)),
        "month_cos": float(math.cos(2.0 * math.pi * (month - 1) / 12.0)),
        "is_weekend": float(day_of_week >= 5),
    }


def calendar_feature_frame(departure: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        [[calendar_features(departure)[name] for name in FEATURE_NAMES]],
        columns=list(FEATURE_NAMES),
    )


def calendar_feature_matrix(index: pd.DatetimeIndex) -> pd.DataFrame:
    hours = index.hour.to_numpy(dtype=float)
    day_of_week = index.day_of_week.to_numpy(dtype=float)
    month = index.month.to_numpy(dtype=float)
    values = np.column_stack(
        [
            np.sin(2.0 * np.pi * hours / 24.0),
            np.cos(2.0 * np.pi * hours / 24.0),
            np.sin(2.0 * np.pi * day_of_week / 7.0),
            np.cos(2.0 * np.pi * day_of_week / 7.0),
            np.sin(2.0 * np.pi * (month - 1.0) / 12.0),
            np.cos(2.0 * np.pi * (month - 1.0) / 12.0),
            (day_of_week >= 5).astype(float),
        ]
    )
    return pd.DataFrame(values, index=index, columns=list(FEATURE_NAMES))
