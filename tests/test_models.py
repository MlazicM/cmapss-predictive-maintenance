"""Tests for model helpers.

Anything needing TensorFlow is skipped when it is absent, so the suite still
runs on a machine that only has the data stack installed -- which is the whole
point of importing the framework lazily inside :mod:`src.models`.
"""

import json

import numpy as np
import pytest

from src.config import SUBSET_INFO, SUBSETS
from src.models import load_calibration, save_calibration
from src.preprocessing import ConditionScaler, scaler_for_subset


# --- calibration persistence -------------------------------------------------

def test_calibration_round_trips_through_disk(tmp_path):
    calibration = {"method": "mc_dropout_split_conformal", "mode": "adaptive", "multiplier": 4.2}
    path = save_calibration("model", calibration, models_dir=tmp_path)

    assert path.name == "model_calibration.json"
    assert load_calibration("model", models_dir=tmp_path) == calibration


def test_calibration_is_written_as_readable_json(tmp_path):
    """A human deciding whether to trust an interval has to be able to read it."""
    save_calibration("model", {"multiplier": 4.2}, models_dir=tmp_path)
    text = (tmp_path / "model_calibration.json").read_text(encoding="utf-8")

    assert json.loads(text) == {"multiplier": 4.2}
    assert "\n" in text, "indent=2, not a single minified line"


def test_a_missing_calibration_is_not_an_error(tmp_path):
    """An uncalibrated model still serves; the API just has to say which it is."""
    assert load_calibration("never_trained", models_dir=tmp_path) is None


# --- subset metadata ---------------------------------------------------------

def test_every_subset_declares_its_regimes_and_faults():
    assert SUBSETS == ["FD001", "FD002", "FD003", "FD004"]
    for subset in SUBSETS:
        info = SUBSET_INFO[subset]
        assert info["regimes"] in (1, 6)
        assert info["faults"] in (1, 2)


def test_scaler_is_derived_from_the_subset_not_chosen_by_hand():
    """Six-regime subsets must not be pushed through single-regime statistics."""
    assert scaler_for_subset("FD001").n_regimes == 1
    assert scaler_for_subset("FD003").n_regimes == 1
    assert scaler_for_subset("FD002").n_regimes == 6
    assert scaler_for_subset("FD004").n_regimes == 6
    assert isinstance(scaler_for_subset("FD002"), ConditionScaler)


def test_single_regime_scaler_degenerates_to_plain_standardisation(synthetic_run_to_failure):
    df = synthetic_run_to_failure(lifetimes=(60, 80))
    features = ["sensor_1", "sensor_2"]
    scaled = scaler_for_subset("FD001").fit_transform(df, features)

    assert scaled[features].mean().abs().max() < 1e-9
    assert scaled[features].std(ddof=0).sub(1.0).abs().max() < 1e-9


# --- quantile regression (needs TensorFlow) ----------------------------------

@pytest.fixture
def quantiles():
    return [0.025, 0.5, 0.975]


def test_pinball_loss_charges_the_two_directions_asymmetrically(quantiles):
    tf = pytest.importorskip("tensorflow")
    from src.models import pinball_loss

    loss = pinball_loss([0.9])
    y_true = tf.constant([[10.0]])
    # At q = 0.9 under-prediction is charged 0.9 and over-prediction only 0.1,
    # which is what pushes the output up towards the 90th percentile.
    under = float(loss(y_true, tf.constant([[8.0]])))
    over = float(loss(y_true, tf.constant([[12.0]])))

    assert under == pytest.approx(0.9 * 2.0)
    assert over == pytest.approx(0.1 * 2.0)


def test_pinball_loss_at_the_median_is_symmetric():
    tf = pytest.importorskip("tensorflow")
    from src.models import pinball_loss

    loss = pinball_loss([0.5])
    y_true = tf.constant([[10.0]])
    assert float(loss(y_true, tf.constant([[8.0]]))) == pytest.approx(
        float(loss(y_true, tf.constant([[12.0]])))
    )


def test_quantile_model_has_one_output_per_quantile(quantiles):
    pytest.importorskip("tensorflow")
    from src.models import build_quantile_lstm

    model = build_quantile_lstm(input_shape=(30, 11), quantiles=quantiles)
    assert model.output_shape[-1] == len(quantiles)
    assert model.quantiles == quantiles, "the output order must not have to be guessed"


def test_quantile_predict_repairs_crossed_quantiles_and_reports_them(quantiles):
    pytest.importorskip("tensorflow")
    from src.models import quantile_predict

    class CrossingModel:
        """Row 0 has its quantiles inverted; row 1 is well ordered."""

        def predict(self, X, verbose=0):
            return np.array([[50.0, 40.0, 30.0], [10.0, 20.0, 30.0]])

    predictions, crossing_rate = quantile_predict(CrossingModel(), None)

    assert crossing_rate == pytest.approx(0.5)
    assert np.all(np.diff(predictions, axis=1) >= 0), "a valid interval must be ordered"
    assert predictions[1].tolist() == [10.0, 20.0, 30.0], "ordered rows are untouched"


def test_quantile_training_learns_a_band_that_brackets_the_target():
    """End-to-end: a fitted 5/50/95 head must actually straddle the data."""
    pytest.importorskip("tensorflow")
    from src.models import build_quantile_lstm, quantile_predict, set_global_seeds

    set_global_seeds(0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(512, 5, 2)).astype("float32")
    y = (X[:, -1, 0] * 10.0 + rng.normal(scale=3.0, size=512)).astype("float32")

    model = build_quantile_lstm(input_shape=(5, 2), quantiles=[0.05, 0.5, 0.95], units=16)
    model.fit(X, y, epochs=30, batch_size=64, verbose=0)

    predictions, _ = quantile_predict(model, X)
    coverage = ((y >= predictions[:, 0]) & (y <= predictions[:, 2])).mean()
    assert 0.7 < coverage < 1.0, f"a 90% band covered {coverage:.0%} of the training data"


def test_a_pinball_compiled_model_round_trips_through_disk(tmp_path):
    """The riskiest integration in the serving path, so it gets a test.

    ``pinball_loss`` returns a closure, which Keras cannot deserialise without
    being handed it back in ``custom_objects``. Loading with ``compile=False``
    sidesteps that -- inference needs no optimiser state -- and this test is what
    keeps someone from "tidying" that default back to True.
    """
    pytest.importorskip("tensorflow")
    from sklearn.preprocessing import StandardScaler

    from src.models import build_quantile_lstm, load_artifacts, save_artifacts

    quantiles = [0.025, 0.5, 0.975]
    model = build_quantile_lstm(input_shape=(30, 11), quantiles=quantiles, units=8)
    scaler = StandardScaler().fit(np.zeros((4, 11)) + np.arange(11))
    save_artifacts(model, scaler, name="quantile", models_dir=tmp_path)

    restored, restored_scaler = load_artifacts("quantile", models_dir=tmp_path)
    predictions = np.asarray(restored(np.zeros((1, 30, 11), dtype="float32")))

    assert predictions.shape == (1, len(quantiles))
    assert restored_scaler.n_features_in_ == 11
