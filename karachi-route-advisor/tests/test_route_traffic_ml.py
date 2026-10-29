from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta

import pytest

from src.config import GRAPH_PATH, PROJECT_ROOT
from src.contracts import RouteTrafficPrediction, WeatherSnapshot
from src.route_traffic_ml import (
    RouteTrafficPredictionService,
    advice_for,
    confidence_for,
)
from src.service import RouteService

pytestmark = pytest.mark.skipif(
    not GRAPH_PATH.exists(),
    reason="Run scripts/fetch_graph.py to create the local graph cache.",
)


def dry_weather() -> WeatherSnapshot:
    return WeatherSnapshot(condition="dry", source="test")


def next_weekday(weekday: int) -> date:
    current = date.today() + timedelta(days=7)
    while current.weekday() != weekday:
        current += timedelta(days=1)
    return current


WEDNESDAY = next_weekday(2)


@pytest.fixture(scope="module")
def service() -> RouteService:
    return RouteService()


@pytest.fixture(scope="module")
def response(service: RouteService):
    return service.route(
        origin_name="Johar Chowk Mosque",
        destination_name="Block 8 bus stop",
        departure=datetime.combine(WEDNESDAY, time(9, 0)),
        preference="Fastest",
        weather=dry_weather(),
        include_mapbox=False,
    )


def predict(
    service: RouteService,
    response,
    departure: datetime,
    route_index: int = 0,
) -> RouteTrafficPrediction:
    route = response.academic_routes[route_index]
    return service.predict_route_traffic(
        response,
        route.route_id,
        departure,
        dry_weather(),
    )


def test_route_traffic_artifacts_are_loaded_and_disclosed(
    service: RouteService,
    response,
) -> None:
    prediction = predict(
        service,
        response,
        datetime.combine(WEDNESDAY, time(9, 0)),
    )

    assert service.route_traffic_service.available is True
    assert service.route_traffic_service.accident_available is True
    assert prediction.status in {"ok", "partial"}
    assert prediction.traffic_band in {"low", "moderate", "high"}
    assert prediction.traffic_probability is not None
    assert 0.0 <= prediction.traffic_probability <= 1.0
    assert prediction.traffic_probabilities
    assert sum(value for _, value in prediction.traffic_probabilities) == pytest.approx(
        1.0
    )
    assert prediction.data_provenance == "MIXED"
    assert prediction.data_provenance_label == "Mixed real + synthetic data"
    assert prediction.synthetic_records == 74_500
    assert prediction.training_dataset_size == prediction.synthetic_records
    assert prediction.real_records == 347_285
    assert prediction.training_dataset_size > 0
    assert prediction.model_version == "route-traffic-sim-v1"
    assert prediction.confidence in {"Low", "Medium"}
    assert "no Karachi real-world validation" in prediction.confidence_note
    assert prediction.evaluation_f1 is not None
    assert prediction.evaluation_f1 > 0.5
    assert len(prediction.confusion_matrix) == 3
    assert prediction.accident_evidence == "REAL"
    assert prediction.accident_risk in {"low", "moderate", "high"}
    assert prediction.accident_probability is not None
    assert "STATS19" in prediction.accident_source
    assert prediction.top_drivers
    assert any("synthetic" in item for item in prediction.provenance)


def test_model_metadata_does_not_claim_karachi_validation() -> None:
    metadata_path = PROJECT_ROOT / "models" / "route_traffic" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["karachi_validated"] is False
    assert metadata["route_applicable"] is True
    assert metadata["data_provenance"] == "MIXED"
    assert metadata["target"]["is_synthetic"] is True
    assert metadata["target"]["classes"] == ["low", "moderate", "high"]
    assert metadata["real_record_breakdown"][
        "karachi_route_level_traffic_records"
    ] == 0
    assert "not observed Karachi traffic" in metadata["disclosure"]


def test_departure_time_changes_the_prediction(
    service: RouteService,
    response,
) -> None:
    morning = predict(
        service,
        response,
        datetime.combine(WEDNESDAY, time(8, 0)),
    )
    night = predict(
        service,
        response,
        datetime.combine(WEDNESDAY, time(22, 0)),
    )

    assert morning.traffic_probabilities != night.traffic_probabilities
    assert (
        morning.traffic_band,
        morning.traffic_probability,
        morning.top_drivers,
    ) != (night.traffic_band, night.traffic_probability, night.top_drivers)


def test_selected_route_changes_the_prediction(
    service: RouteService,
    response,
) -> None:
    departure = datetime.combine(WEDNESDAY, time(8, 0))
    predictions = [
        predict(service, response, departure, route_index=index)
        for index in range(len(response.academic_routes))
    ]

    assert len(predictions) >= 2
    assert len({prediction.route_id for prediction in predictions}) == len(predictions)
    assert len({prediction.traffic_probabilities for prediction in predictions}) > 1


def test_invalid_past_departure_is_not_silently_predicted(
    service: RouteService,
    response,
) -> None:
    prediction = predict(
        service,
        response,
        datetime.now() - timedelta(days=1),
    )

    assert prediction.status == "invalid_request"
    assert prediction.traffic_band == "unknown"
    assert prediction.traffic_probability is None
    assert prediction.advice_code == "unavailable"


def test_advice_mapping_covers_traffic_and_accident_combinations() -> None:
    cases = [
        (("high", "high"), "not_recommended"),
        (("high", "low"), "consider_earlier"),
        (("moderate", "high"), "consider_earlier"),
        (("moderate", "low"), "reasonable_time"),
        (("low", "high"), "reasonable_time"),
        (("low", "low"), "good_time"),
        (("low", "unknown"), "reasonable_time"),
        (("moderate", "unknown"), "reasonable_time"),
        (("high", "unknown"), "consider_earlier"),
        (("unknown", "low"), "unavailable"),
    ]
    for (traffic, accident), expected in cases:
        code, label, message = advice_for(traffic, accident)
        assert code == expected
        assert label
        assert message


def test_confidence_is_capped_for_synthetic_model() -> None:
    assert confidence_for([0.99, 0.005, 0.005])[0] == "Medium"
    assert confidence_for([0.55, 0.30, 0.15])[0] == "Medium"
    assert confidence_for([0.40, 0.35, 0.25])[0] == "Low"
    assert confidence_for([0.30, 0.25, 0.20])[0] == "Low"
    assert "capped at Medium" in confidence_for([0.99, 0.005, 0.005])[1]


def test_missing_artifacts_return_unavailable(tmp_path) -> None:
    prediction_service = RouteTrafficPredictionService(
        model_dir=tmp_path / "missing",
        accident_model_dir=tmp_path / "missing_accident",
    )

    assert prediction_service.available is False
    assert prediction_service.accident_available is False


def test_ml_advisory_does_not_change_academic_route_score(
    service: RouteService,
    response,
) -> None:
    before = tuple(
        (route.route_id, route.score) for route in response.ranked
    )
    predict(
        service,
        response,
        datetime.combine(WEDNESDAY, time(8, 0)),
    )
    after = tuple(
        (route.route_id, route.score) for route in response.ranked
    )

    assert before == after
