from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class RouteCandidate:
    route_id: str
    source: str
    provider: str
    geometry: tuple[tuple[float, float], ...]
    distance_m: float
    eta_min: float
    risk: float
    confidence: float
    congestion: float | None
    incident_count: int
    closure_count: int
    traffic_available: bool
    nodes: tuple[int, ...] = ()
    edges: tuple[tuple[int, int], ...] = ()
    warnings: tuple[str, ...] = ()
    toll: bool = False
    impassable: bool = False
    unlit_ratio: float = 0.0
    risk_available: bool = True
    toll_available: bool = True
    free_flow_eta_min: float | None = None
    eta_evidence_kind: str = "osm_analytical_future"
    eta_evidence_label: str = "Future-departure OSM analytical ETA"


@dataclass(frozen=True)
class RuleTrace:
    rule_id: str
    rule_name: str
    stratum: str
    priority: int
    route_id: str
    evidence: str
    effect: str


@dataclass(frozen=True)
class RouteVerdict:
    route_id: str
    vetoed: bool
    veto_reasons: tuple[str, ...]
    penalty: float
    bonus: float
    trace: tuple[RuleTrace, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RankedRoute:
    route_id: str
    rank: int
    score: int
    final_cost: float
    base_cost: float
    multiplier: float
    confidence_penalty: float
    explanation: str


@dataclass(frozen=True)
class BaselineComparison:
    dijkstra_route: RouteCandidate
    astar_expanded_nodes: int
    dijkstra_expanded_nodes: int
    same_path: bool
    same_cost: bool


@dataclass(frozen=True)
class MapboxRoute:
    route_id: str
    profile: str
    duration_s: float
    duration_typical_s: float | None
    distance_m: float
    geometry: tuple[tuple[float, float], ...]
    congestion_values: tuple[str, ...]
    congestion_numeric: tuple[int | None, ...]
    incident_count: int
    closure_count: int
    traffic_evidence: bool
    code: str = "Ok"
    fetched_at: str = ""
    requested_departure: str = ""
    annotation_distances_m: tuple[float | None, ...] = ()
    annotation_durations_s: tuple[float | None, ...] = ()
    annotation_speeds_kph: tuple[float | None, ...] = ()

    @property
    def duration_delta_s(self) -> float | None:
        if self.duration_typical_s is None:
            return None
        return self.duration_s - self.duration_typical_s

    @property
    def duration_delta_pct(self) -> float | None:
        delta = self.duration_delta_s
        if delta is None or not self.duration_typical_s:
            return None
        return delta / self.duration_typical_s * 100.0

    @property
    def congestion_average(self) -> float | None:
        numeric = [value for value in self.congestion_numeric if value is not None]
        if numeric:
            return sum(numeric) / len(numeric) / 100.0
        mapping = {
            "low": 0.20,
            "moderate": 0.50,
            "heavy": 0.70,
            "severe": 0.90,
        }
        values = [
            mapping[value] for value in self.congestion_values if value in mapping
        ]
        return sum(values) / len(values) if values else None


@dataclass(frozen=True)
class TrafficAssessment:
    route_id: str
    route_source: str
    distance_km: float
    requested_departure: datetime
    ordinary_eta_min: float | None
    ordinary_eta_label: str
    observed_eta_min: float | None
    observed_eta_label: str
    predicted_eta_min: float | None
    predicted_eta_label: str
    baseline_label: str
    baseline_eta_min: float | None
    delay_vs_baseline_min: float | None
    delay_basis: str
    traffic_level: str
    traffic_level_basis: str
    evidence_status: str
    current_evidence: tuple[str, ...]
    confidence_label: str
    explanation: str
    limitations: tuple[str, ...]
    road_composition: tuple[tuple[str, float], ...] = ()
    model_kind: str = "evidence_fallback_no_trained_model"
    trained: bool = False
    historical_data_available: bool = False
    features: tuple[str, ...] = ()
    status: str = "ok"


@dataclass(frozen=True)
class MLTrafficAssessment:
    route_id: str
    requested_departure: datetime
    status: str
    forecast_available: bool
    forecast_value: float | None
    forecast_label: str
    forecast_units: str
    model_kind: str
    trained: bool
    model_loaded: bool
    dataset_name: str
    dataset_license: str
    coverage: str
    route_applicable: bool
    benchmark_validation_mae: float | None
    benchmark_test_mae: float | None
    benchmark_test_rmse: float | None
    benchmark_test_r2: float | None
    benchmark_baseline_test_mae: float | None
    explanation: str
    reasons: tuple[str, ...]
    limitations: tuple[str, ...]
    provenance: tuple[str, ...]
    local_data_message: str = ""


@dataclass(frozen=True)
class RouteContext:
    distance_km: float | None
    ordinary_eta_min: float | None
    free_flow_eta_min: float | None
    delay_ratio: float | None
    risk: float | None
    unlit_ratio: float | None
    road_composition: tuple[tuple[str, float], ...]
    context_band: str
    context_basis: str
    fingerprint: str
    available: bool


@dataclass(frozen=True)
class ForecastComponent:
    available: bool
    band: str
    probability: float | None
    value: float | None
    units: str
    model_kind: str
    model_version: str
    dataset_name: str
    dataset_license: str
    coverage: str
    status: str
    reasons: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class DepartureForecast:
    route_id: str
    route_source: str
    requested_departure: datetime
    status: str
    final_assessment: str
    predicted_traffic: str
    traffic_probability: float | None
    accident_risk: str
    accident_probability: float | None
    current_mapbox_traffic: str
    current_mapbox_basis: str
    current_mapbox_applied: bool
    recommendation: str
    route_context: RouteContext
    traffic: ForecastComponent
    accident: ForecastComponent
    evidence_status: str
    model_version: str
    disclosure: tuple[str, ...]
    explanation: str
    limitations: tuple[str, ...]
    provenance: tuple[str, ...]
    generated_at: datetime


@dataclass(frozen=True)
class RouteTrafficPrediction:
    route_id: str
    route_source: str
    requested_departure: datetime
    status: str
    traffic_band: str
    traffic_probability: float | None
    traffic_probabilities: tuple[tuple[str, float], ...]
    confidence: str
    confidence_note: str
    accident_risk: str
    accident_probability: float | None
    accident_evidence: str
    accident_source: str
    advice_code: str
    advice_label: str
    advice_message: str
    explanation: str
    top_drivers: tuple[str, ...]
    data_provenance: str
    data_provenance_label: str
    model_name: str
    model_version: str
    training_dataset_size: int
    real_records: int
    synthetic_records: int
    evaluation_accuracy: float | None
    evaluation_precision: float | None
    evaluation_recall: float | None
    evaluation_f1: float | None
    confusion_matrix: tuple[tuple[int, ...], ...]
    features: tuple[str, ...]
    limitations: tuple[str, ...]
    provenance: tuple[str, ...]
    generated_at: datetime


@dataclass(frozen=True)
class IncidentRiskAssessment:
    level: str
    probability: float | None
    evidence_type: str
    route_specific: bool
    source: str
    explanation: str
    external_prior_level: str | None = None
    external_prior_probability: float | None = None
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrafficEvidenceMatch:
    route_id: str
    provider_route_id: str
    method: str
    geometry_overlap: float
    segment_coverage: float | None
    provider_eta_min: float
    provider_typical_min: float | None
    provider_delay_ratio: float | None
    adjusted_eta_min: float
    evidence_status: str
    requested_departure: datetime
    fetched_at: str
    incident_count: int
    closure_count: int
    applied: bool
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteAssessment:
    route: RouteCandidate
    requested_departure: datetime
    analytical_eta_min: float
    free_flow_eta_min: float | None
    ml_eta_min: float | None
    traffic_adjusted_eta_min: float | None
    decision_eta_min: float
    decision_eta_basis: str
    eta_kind: str
    eta_label: str
    evidence_source: str
    evidence_timestamp: str | None
    evidence_coverage: float | None
    traffic_evidence: TrafficEvidenceMatch | None
    traffic_prediction: RouteTrafficPrediction | None
    future_traffic_band: str | None
    future_traffic_probability: float | None
    confidence: str
    confidence_note: str
    incident_risk: IncidentRiskAssessment
    expert_verdict: RouteVerdict
    limitations: tuple[str, ...] = ()
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class FinalRouteDecision:
    route_id: str
    rank: int
    recommended: bool
    route: RouteCandidate
    traffic_prediction: RouteTrafficPrediction | None
    incident_risk: IncidentRiskAssessment
    academic_rank: int | None
    academic_score: int | None
    final_score: int
    final_cost: float
    academic_component: float | None
    traffic_component: float | None
    traffic_severity: float | None
    ranking_reason: str
    details: tuple[str, ...] = ()
    evidence_kind: str = "future_ml"
    evidence_label: str = "Future-traffic ML prediction"
    decision_time_min: float = 0.0
    decision_time_basis: str = ""
    risk_available: bool = True
    traffic_evidence_kind: str = "future_ml"
    traffic_evidence_label: str = ""
    assessment: RouteAssessment | None = None


@dataclass(frozen=True)
class FinalRecommendation:
    departure: datetime
    routes: tuple[FinalRouteDecision, ...]
    recommended_route_id: str | None
    advice_code: str
    advice_label: str
    advice_message: str
    policy_version: str
    policy_description: str
    weights: tuple[tuple[str, float], ...]
    fallback_used: bool
    fallback_basis: str
    warnings: tuple[str, ...]
    provenance: tuple[str, ...]
    generated_at: datetime
    participating_sources: tuple[str, ...] = ()
    mapbox_considered: bool = False


@dataclass(frozen=True)
class DepartureAdvisory:
    route_id: str
    final_assessment: str
    warnings: tuple[str, ...]
    positives: tuple[str, ...]
    trace: tuple[RuleTrace, ...]
    evidence_status: str
    model_version: str


@dataclass(frozen=True)
class WeatherSnapshot:
    condition: str
    source: str
    temperature_c: float | None = None
    precipitation_mm: float | None = None
    observed_at: str | None = None
    warning: str | None = None


@dataclass(frozen=True)
class RouteResponse:
    origin: str
    destination: str
    departure: datetime
    preference: str
    weather: WeatherSnapshot
    academic_routes: tuple[RouteCandidate, ...]
    verdicts: tuple[RouteVerdict, ...]
    ranked: tuple[RankedRoute, ...]
    mapbox_routes: tuple[MapboxRoute, ...]
    warnings: tuple[str, ...]
    provenance: tuple[str, ...]
    mapbox_verdicts: tuple[RouteVerdict, ...] = ()
    baseline: BaselineComparison | None = None
    traffic_status: str = "no_mapbox"
    final_recommendation: FinalRecommendation | None = None
    avoid_tolls: bool = False


@dataclass(frozen=True)
class MapboxCheck:
    token_present: bool
    profile: str
    route_count: int
    traffic_evidence: bool
    message: str
