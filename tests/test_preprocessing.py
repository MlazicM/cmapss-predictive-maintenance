import numpy as np
import pandas as pd
import pytest

from src.data import add_rul
from src.preprocessing import (
    ConditionScaler,
    add_rolling_features,
    augment_with_noise,
    last_sequence_per_engine,
    make_sequences,
)

FEATURES = ["sensor_1", "sensor_2"]


def test_window_count_includes_the_final_window(synthetic_run_to_failure):
    """An engine of length n must yield n - L + 1 windows, not n - L."""
    df = add_rul(synthetic_run_to_failure(lifetimes=(40,)), cap=None)
    X, y, _ = make_sequences(df, FEATURES, sequence_length=30)

    assert X.shape == (11, 30, 2)
    assert y.min() == 0.0, "the window ending at failure must be kept"


def test_target_is_rul_at_last_cycle_of_window(synthetic_run_to_failure):
    df = add_rul(synthetic_run_to_failure(lifetimes=(40,)), cap=None)
    X, y, _ = make_sequences(df, FEATURES, sequence_length=5)

    # sensor_1 equals the cycle number, so the last column identifies the cycle.
    last_cycle_in_window = X[:, -1, 0]
    assert np.allclose(y, 40 - last_cycle_in_window)


def test_windows_never_cross_an_engine_boundary(synthetic_run_to_failure):
    df = add_rul(synthetic_run_to_failure(lifetimes=(40, 40)), cap=None)
    X, y, engine_ids = make_sequences(df, FEATURES, sequence_length=10)

    assert len(X) == 2 * (40 - 10 + 1)
    # Within every window the cycle counter must increase by exactly one.
    steps = np.diff(X[:, :, 0], axis=1)
    assert np.all(steps == 1), "a window spans two engines"
    assert set(np.unique(engine_ids)) == {1, 2}


def test_engines_shorter_than_the_window_are_skipped(synthetic_run_to_failure):
    df = add_rul(synthetic_run_to_failure(lifetimes=(5, 40)), cap=None)
    _, _, engine_ids = make_sequences(df, FEATURES, sequence_length=30)
    assert set(np.unique(engine_ids)) == {2}


def test_make_sequences_handles_unsorted_input(synthetic_run_to_failure):
    df = add_rul(synthetic_run_to_failure(lifetimes=(40,)), cap=None)
    shuffled = df.sample(frac=1.0, random_state=0)
    X, y, _ = make_sequences(shuffled, FEATURES, sequence_length=10)
    assert np.all(np.diff(X[:, :, 0], axis=1) == 1)


def test_last_sequence_per_engine_returns_one_row_per_engine(synthetic_run_to_failure):
    df = add_rul(synthetic_run_to_failure(lifetimes=(40, 60, 100)), cap=None)
    X, y, engine_ids = last_sequence_per_engine(df, FEATURES, sequence_length=30)

    assert X.shape == (3, 30, 2)
    assert engine_ids.tolist() == [1, 2, 3]
    assert np.allclose(y, 0.0)                   # these engines run to failure
    assert np.allclose(X[0, -1, 0], 40)          # window ends at the last cycle


def test_last_sequence_pads_short_engines(synthetic_run_to_failure):
    df = add_rul(synthetic_run_to_failure(lifetimes=(5,)), cap=None)
    X, _, engine_ids = last_sequence_per_engine(df, FEATURES, sequence_length=30)
    assert X.shape == (1, 30, 2)
    assert engine_ids.tolist() == [1], "the benchmark needs a prediction per engine"


def test_augmentation_does_not_create_impossible_windows(synthetic_run_to_failure):
    """Augmenting the tensor keeps every window internally consistent."""
    df = add_rul(synthetic_run_to_failure(lifetimes=(40,)), cap=None)
    X, y, _ = make_sequences(df, FEATURES, sequence_length=10)
    X_aug, y_aug = augment_with_noise(X, y, n_copies=3, noise_level=0.1)

    assert len(X_aug) == 4 * len(X)
    assert np.array_equal(y_aug[: len(y)], y)
    # Copies are jittered versions of the originals, never re-windowed splices.
    assert np.abs(X_aug[len(X):] - np.tile(X, (3, 1, 1))).max() < 1.0


def test_condition_scaler_standardises_within_each_regime():
    """Global scaling across regimes leaves the regime offset in the data."""
    rng = np.random.default_rng(0)
    rows = []
    for regime, offset in enumerate([0.0, 100.0]):
        rows.append(
            pd.DataFrame(
                {
                    "engine_id": 1,
                    "cycle": np.arange(1, 201),
                    "setting_1": float(regime),
                    "setting_2": float(regime),
                    "setting_3": float(regime),
                    "sensor_1": rng.normal(offset, 1.0, 200),
                    "sensor_2": rng.normal(offset, 1.0, 200),
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)

    scaled = ConditionScaler(n_regimes=2).fit_transform(df, ["sensor_1", "sensor_2"])
    per_regime_mean = scaled.groupby("setting_1")["sensor_1"].mean()

    assert np.allclose(per_regime_mean.to_numpy(), 0.0, atol=1e-8)
    assert scaled["sensor_1"].std() == pytest.approx(1.0, abs=0.05)


def test_rolling_features_are_causal(synthetic_run_to_failure):
    """A rolling feature must never see a cycle that has not happened yet."""
    df = add_rul(synthetic_run_to_failure(lifetimes=(40,)), cap=None)
    enriched, names = add_rolling_features(df, FEATURES, windows=(5,))

    assert "sensor_1_mean_5" in names
    row = enriched.iloc[9]
    expected = df["sensor_1"].iloc[5:10].mean()
    assert row["sensor_1_mean_5"] == pytest.approx(expected)
    assert enriched["sensor_1_drift"].iloc[0] == 0.0
