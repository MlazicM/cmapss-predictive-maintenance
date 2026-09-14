"""API tests with a stubbed model, so no TensorFlow is required."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app, state
from api.predictor import RULPredictor
from api.schemas import PredictionRequest
from src.config import INFORMATIVE_SENSORS, SEQUENCE_LENGTH


class StubScaler:
    """Subtracts a fixed offset, so scaler use is observable in the output."""

    def transform(self, frame):
        return np.asarray(frame, dtype=float) - 100.0


class StubModel:
    """Returns the mean of the window plus jitter, mimicking active dropout."""

    def __init__(self, noise=1.0):
        self.rng = np.random.default_rng(0)
        self.noise = noise
        self.last_input_shape = None

    def __call__(self, window, training=False):
        self.last_input_shape = window.shape
        return np.array([[window.mean() + self.rng.normal(0, self.noise)]])


def reading(cycle, value=101.0):
    return {"cycle": cycle, "sensors": {s: value for s in INFORMATIVE_SENSORS}}


@pytest.fixture
def predictor():
    return RULPredictor(StubModel(), StubScaler())


@pytest.fixture
def client(predictor):
    state["predictor"] = predictor
    with TestClient(app) as test_client:
        state["predictor"] = predictor   # lifespan startup may have cleared it
        yield test_client
    state["predictor"] = None


# --- predictor ------------------------------------------------------------

def test_window_has_the_shape_the_model_expects(predictor):
    window, padded = predictor.build_window([reading(c) for c in range(1, 41)])
    assert window.shape == (1, SEQUENCE_LENGTH, len(INFORMATIVE_SENSORS))
    assert not padded


def test_window_keeps_the_most_recent_cycles(predictor):
    readings = [reading(c, value=100.0 + c) for c in range(1, 41)]
    window, _ = predictor.build_window(readings)
    # StubScaler subtracts 100, so the last row must be cycle 40's value.
    assert window[0, -1, 0] == pytest.approx(40.0)


def test_short_history_is_padded_and_flagged(predictor):
    window, padded = predictor.build_window([reading(c) for c in range(1, 6)])
    assert window.shape == (1, SEQUENCE_LENGTH, len(INFORMATIVE_SENSORS))
    assert padded, "a 5-cycle prediction must not be presented as a 30-cycle one"


def test_missing_sensor_is_rejected(predictor):
    incomplete = [{"cycle": 1, "sensors": {"sensor_2": 1.0}}]
    with pytest.raises(ValueError, match="missing required sensors"):
        predictor.build_window(incomplete)


def test_prediction_carries_an_interval(predictor):
    result = predictor.predict([reading(c) for c in range(1, 41)], n_samples=30)
    assert result["lower_95"] <= result["predicted_rul"] <= result["upper_95"]
    assert result["std"] > 0, "MC dropout must actually vary between samples"
    assert result["cycles_supplied"] == 40


def test_prediction_is_clamped_to_the_rul_cap():
    """The model cannot distinguish healthy engines, so it must not claim to."""
    huge = RULPredictor(StubModel(noise=0.0), StubScaler(), rul_cap=125)
    result = huge.predict([reading(c, value=100_000.0) for c in range(1, 41)], n_samples=5)
    assert result["predicted_rul"] == 125.0
    assert result["upper_95"] == 125.0


def test_lower_bound_never_goes_negative():
    predictor = RULPredictor(StubModel(noise=50.0), StubScaler())
    result = predictor.predict([reading(c) for c in range(1, 41)], n_samples=30)
    assert result["lower_95"] >= 0.0


# --- schemas --------------------------------------------------------------

def test_out_of_order_cycles_are_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        PredictionRequest(engine_id=1, readings=[reading(5), reading(2)])


def test_duplicate_cycles_are_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        PredictionRequest(engine_id=1, readings=[reading(2), reading(2)])


# --- endpoints ------------------------------------------------------------

def test_health_reports_the_contract(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["expected_sensors"] == INFORMATIVE_SENSORS


def test_predict_returns_a_bounded_estimate(client):
    payload = {"engine_id": 7, "readings": [reading(c) for c in range(1, 41)]}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["engine_id"] == 7
    assert body["lower_95"] <= body["predicted_rul"] <= body["upper_95"]
    assert body["padded"] is False


def test_predict_rejects_a_malformed_payload(client):
    payload = {"engine_id": 7, "readings": [{"cycle": 1, "sensors": {"sensor_2": 1.0}}]}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_is_unavailable_without_a_model():
    state["predictor"] = None
    with TestClient(app) as test_client:
        state["predictor"] = None
        response = test_client.post(
            "/predict", json={"engine_id": 1, "readings": [reading(1)]}
        )
    assert response.status_code == 503
    assert "notebooks" in response.json()["detail"]


# --- artifact loading -------------------------------------------------------

def test_load_artifacts_reports_missing_files_as_not_found(tmp_path):
    """The check runs before TensorFlow is imported, so it holds either way.

    Keras raises ValueError for an absent .keras path, which reads as a corrupt
    file rather than a missing one; callers then write an except clause that
    does not cover it.
    """
    from src.models import load_artifacts

    with pytest.raises(FileNotFoundError, match="no trained artifacts"):
        load_artifacts("does_not_exist", models_dir=tmp_path)


def test_app_starts_even_when_loading_the_model_blows_up(monkeypatch):
    """Startup must degrade to an unhealthy service, never crash the process.

    Regression: with TensorFlow installed and no trained model on disk, Keras
    raised ValueError, the narrow except in the lifespan missed it, and every
    request-level test failed because the app could not boot at all.
    """
    import api.main as main

    def raise_like_keras(*args, **kwargs):
        raise ValueError("File not found: filepath=models/lstm_fd001.keras")

    monkeypatch.setattr(main.RULPredictor, "from_artifacts", staticmethod(raise_like_keras))
    state["predictor"] = None

    with TestClient(app) as test_client:
        assert test_client.get("/health").json()["model_loaded"] is False
        response = test_client.post("/predict", json={"engine_id": 1, "readings": [reading(1)]})

    assert response.status_code == 503
