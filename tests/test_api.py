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


def test_an_uncalibrated_predictor_says_so_in_every_response(predictor):
    """The 62%-coverage band and a guaranteed one come back in the same fields.

    The only thing separating them in the payload is this string, so a caller
    that treats the bounds as a probability has to be able to see which it got.
    """
    result = predictor.predict([reading(c) for c in range(1, 41)], n_samples=10)
    assert "UNCALIBRATED" in result["interval_method"]
    assert result["nominal_coverage"] == 0.95


def test_a_calibrated_predictor_uses_its_own_multiplier():
    """The conformal multiplier must replace 1.96, not sit beside it unused."""
    calibration = {"method": "mc_dropout_split_conformal", "mode": "adaptive",
                   "nominal": 0.95, "multiplier": 6.0}
    readings = [reading(c) for c in range(1, 41)]

    plain = RULPredictor(StubModel(), StubScaler())
    calibrated = RULPredictor(StubModel(), StubScaler(), calibration=calibration)

    wide = calibrated.predict(readings, n_samples=30)
    narrow = plain.predict(readings, n_samples=30)

    assert wide["upper_95"] - wide["lower_95"] > narrow["upper_95"] - narrow["lower_95"]
    assert "split-conformal (adaptive)" in wide["interval_method"]


def test_absolute_mode_ignores_the_models_own_spread():
    """In absolute mode the multiplier is a width in cycles, not a factor on std.

    The readings put the stub's prediction near 60 so that the +/- 12 band sits
    inside [0, 125] and is not clipped -- the width being tested is the one the
    multiplier produced, not the one the cap left behind.
    """
    calibration = {"mode": "absolute", "nominal": 0.9, "multiplier": 12.0}
    calibrated = RULPredictor(StubModel(noise=3.0), StubScaler(), calibration=calibration)

    result = calibrated.predict(
        [reading(c, value=160.0) for c in range(1, 41)], n_samples=30
    )
    assert result["predicted_rul"] == pytest.approx(60.0, abs=5.0)
    assert result["upper_95"] - result["lower_95"] == pytest.approx(24.0)
    assert result["nominal_coverage"] == 0.9


def test_a_saturated_prediction_never_reports_a_bound_above_the_cap():
    """Regression: clipping only the upper bound pushed it back over the cap."""
    calibration = {"mode": "absolute", "nominal": 0.95, "multiplier": 10.0}
    calibrated = RULPredictor(StubModel(noise=0.0), StubScaler(), calibration=calibration)

    result = calibrated.predict(
        [reading(c, value=100_000.0) for c in range(1, 41)], n_samples=5
    )
    assert result["predicted_rul"] == 125.0
    assert result["upper_95"] == 125.0
    assert result["lower_95"] <= result["upper_95"]


class StubQuantileModel:
    """Three heads, deliberately returned out of order."""

    def __init__(self, outputs=(60.0, 40.0, 20.0)):
        self.outputs = outputs
        self.calls = 0

    def __call__(self, window, training=False):
        self.calls += 1
        return np.array([list(self.outputs)])


def test_quantile_predictor_uses_the_heads_as_the_interval():
    calibration = {"method": "quantile_regression_cqr", "nominal": 0.95,
                   "quantiles": [0.025, 0.5, 0.975], "offset": 0.0}
    model = StubQuantileModel()
    predictor = RULPredictor(model, StubScaler(), calibration=calibration)

    result = predictor.predict([reading(c) for c in range(1, 41)], n_samples=50)

    assert (result["lower_95"], result["predicted_rul"], result["upper_95"]) == (20.0, 40.0, 60.0)
    assert model.calls == 1, "one forward pass, not a hundred: n_samples must be ignored"


def test_quantile_predictor_reports_no_std():
    """There is no MC-dropout spread here; inventing one would imply a normal."""
    calibration = {"method": "quantile_regression_cqr", "offset": 0.0}
    predictor = RULPredictor(StubQuantileModel(), StubScaler(), calibration=calibration)

    result = predictor.predict([reading(c) for c in range(1, 41)])
    assert result["std"] is None
    assert "quantile regression" in result["interval_method"]


def test_quantile_predictor_applies_the_cqr_offset():
    calibration = {"method": "quantile_regression_cqr", "offset": 5.0}
    predictor = RULPredictor(StubQuantileModel(), StubScaler(), calibration=calibration)

    result = predictor.predict([reading(c) for c in range(1, 41)])
    assert (result["lower_95"], result["upper_95"]) == (15.0, 65.0)


def test_crossed_quantile_heads_never_produce_an_inverted_interval():
    """The pinball loss does not constrain the heads to stay ordered."""
    calibration = {"method": "quantile_regression_cqr", "offset": 0.0}
    crossed = RULPredictor(
        StubQuantileModel(outputs=(10.0, 55.0, 30.0)), StubScaler(), calibration=calibration
    )

    result = crossed.predict([reading(c) for c in range(1, 41)])
    assert result["lower_95"] <= result["predicted_rul"] <= result["upper_95"]
    assert (result["lower_95"], result["upper_95"]) == (10.0, 55.0)


def test_the_point_estimate_is_never_negative():
    """A model past end-of-life extrapolates below zero; -3 cycles is not an input."""
    calibration = {"method": "quantile_regression_cqr", "offset": 0.0}
    predictor = RULPredictor(
        StubQuantileModel(outputs=(-5.0, -20.0, -40.0)), StubScaler(), calibration=calibration
    )

    result = predictor.predict([reading(c) for c in range(1, 41)])
    assert result["predicted_rul"] == 0.0
    assert result["lower_95"] == 0.0


def test_health_reports_whether_the_interval_is_calibrated(client, predictor):
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["interval_calibrated"] is False, "the stub carries no calibration"


def test_health_reports_no_model_name_when_nothing_is_loaded():
    """A stale name beside model_loaded=false would misread as a working service."""
    state["predictor"] = None
    with TestClient(app) as test_client:
        state["predictor"] = None
        body = test_client.get("/health").json()

    assert body["model_loaded"] is False
    assert body["model_name"] is None
    assert body["interval_calibrated"] is False


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
    assert "run_experiments.py" in response.json()["detail"], (
        "a 503 has to tell the operator how to produce the missing artifact"
    )


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
