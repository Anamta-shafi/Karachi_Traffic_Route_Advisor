from __future__ import annotations

from .ai.optimizer import (
    ADVICE_LABELS,
    BAND_LABELS,
    MAX_DISPLAYED_ROUTES,
    MAX_EDGE_OVERLAP,
    MAX_GEOMETRY_OVERLAP,
    POLICY_DESCRIPTION,
    POLICY_VERSION,
    TRAFFIC_BANDS,
    TRAFFIC_DELAY_FACTOR,
    TRAFFIC_SEVERITY,
    UNKNOWN_RISK_NORM,
    UNKNOWN_RISK_NORM_SAFEST,
    _advice,
    _edge_overlap,
    _expected_severity,
    _normalize,
    _usable_prediction,
    build_final_recommendation,
    build_incident_risk,
)

_overlap_ratio = _edge_overlap

__all__ = [
    "ADVICE_LABELS",
    "BAND_LABELS",
    "MAX_DISPLAYED_ROUTES",
    "MAX_EDGE_OVERLAP",
    "MAX_GEOMETRY_OVERLAP",
    "POLICY_DESCRIPTION",
    "POLICY_VERSION",
    "TRAFFIC_BANDS",
    "TRAFFIC_DELAY_FACTOR",
    "TRAFFIC_SEVERITY",
    "UNKNOWN_RISK_NORM",
    "UNKNOWN_RISK_NORM_SAFEST",
    "build_final_recommendation",
    "build_incident_risk",
]
