from __future__ import annotations

from src.config import BBOX, LANDMARKS


def test_landmarks_inside_bbox() -> None:
    for name, (lat, lon) in LANDMARKS.items():
        assert BBOX.contains(lat, lon), name


def test_bbox_orientation() -> None:
    assert BBOX.south < BBOX.north
    assert BBOX.west < BBOX.east
    assert BBOX.osmnx == (67.108, 24.898, 67.168, 24.944)
