"""End-to-end pipeline check on synthetic files in CMAPSS wire format.

Covers everything the notebooks do except the Keras fit itself, so the loading,
labelling, splitting, scaling and windowing path is verified without the NASA
download and without TensorFlow installed.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import COLUMN_NAMES, SETTING_COLUMNS
from src.data import load_test, load_train, split_by_engine, subsample_engines
from src.evaluate import regression_report
from src.preprocessing import (
    ConditionScaler,
    add_rolling_features,
    apply_scaler,
    fit_scaler,
    last_sequence_per_engine,
    make_sequences,
)

SENSORS = ["sensor_2", "sensor_3", "sensor_4"]


def _engine_frame(engine_id, n_cycles, regimes, rng):
    """One engine whose sensors drift linearly with wear, offset by regime."""
    cycles = np.arange(1, n_cycles + 1)
    wear = cycles / n_cycles
    regime = rng.integers(0, len(regimes), size=n_cycles)
    offsets = np.asarray(regimes)[regime]

    frame = pd.DataFrame({"engine_id": engine_id, "cycle": cycles})
    for i, column in enumerate(SETTING_COLUMNS):
        frame[column] = offsets * (i + 1)
    for i in range(1, 22):
        frame[f"sensor_{i}"] = 500 + offsets * 50 + wear * (i % 7) + rng.normal(0, 0.05, n_cycles)
    return frame


@pytest.fixture
def cmapss_files(tmp_path):
    """Write train_/test_/RUL_ files for a single- and a six-regime subset."""
    rng = np.random.default_rng(0)

    def write(subset, n_engines, regimes):
        train = pd.concat(
            [_engine_frame(e, int(rng.integers(60, 200)), regimes, rng)
             for e in range(1, n_engines + 1)],
            ignore_index=True,
        )
        # Test engines are truncated before failure; remember how much life is left.
        test_parts, remaining = [], []
        for e in range(1, n_engines + 1):
            full = _engine_frame(e, int(rng.integers(60, 200)), regimes, rng)
            cut = int(len(full) * 0.6)
            test_parts.append(full.iloc[:cut])
            remaining.append(len(full) - cut)

        train[COLUMN_NAMES].to_csv(
            tmp_path / f"train_{subset}.txt", sep=" ", header=False, index=False
        )
        pd.concat(test_parts)[COLUMN_NAMES].to_csv(
            tmp_path / f"test_{subset}.txt", sep=" ", header=False, index=False
        )
        pd.Series(remaining).to_csv(
            tmp_path / f"RUL_{subset}.txt", sep=" ", header=False, index=False
        )

    write("FD001", 12, regimes=[0.0])
    write("FD002", 12, regimes=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    return tmp_path


def test_train_loader_reads_wire_format(cmapss_files):
    df = load_train("FD001", cap=125, data_dir=cmapss_files)
    assert list(df.columns) == COLUMN_NAMES + ["RUL"]
    assert df["engine_id"].nunique() == 12
    assert df.groupby("engine_id")["RUL"].min().eq(0).all()
    assert df["RUL"].max() == 125


def test_test_loader_uses_the_official_rul_file(cmapss_files):
    """Test engines stop before failure, so their final RUL must exceed zero."""
    df = load_test("FD001", cap=None, data_dir=cmapss_files)
    final_rul = df.loc[df.groupby("engine_id")["cycle"].idxmax(), "RUL"]

    expected = pd.read_csv(
        cmapss_files / "RUL_FD001.txt", sep=r"\s+", header=None
    )[0].to_numpy()

    assert np.array_equal(np.sort(final_rul.to_numpy()), np.sort(expected))
    assert (final_rul > 0).all(), "a truncated engine must have life remaining"
    # RUL must still decrease by one per cycle within an engine.
    one_engine = df[df["engine_id"] == 1].sort_values("cycle")
    assert np.all(np.diff(one_engine["RUL"]) == -1)


def test_full_single_condition_pipeline(cmapss_files):
    train_all = load_train("FD001", cap=125, data_dir=cmapss_files)
    test_df = load_test("FD001", cap=125, data_dir=cmapss_files)
    train_df, val_df = split_by_engine(train_all, val_fraction=0.25)

    scaler = fit_scaler(train_df, SENSORS)
    train_scaled = apply_scaler(scaler, train_df, SENSORS)
    test_scaled = apply_scaler(scaler, test_df, SENSORS)

    # Scaling statistics come from training only.
    assert abs(train_scaled[SENSORS].mean().max()) < 1e-9

    X_train, y_train, _ = make_sequences(train_scaled, SENSORS, sequence_length=30)
    X_test, y_test, engine_ids = last_sequence_per_engine(test_scaled, SENSORS, sequence_length=30)

    assert X_train.shape[1:] == (30, 3)
    assert len(X_test) == test_df["engine_id"].nunique(), "one prediction per test engine"
    assert len(np.unique(engine_ids)) == len(engine_ids)
    assert y_train.min() == 0.0, "the failure window survives windowing"

    # A mean predictor must score R2 ~ 0; anything else means a metric bug.
    report = regression_report(y_test, np.full_like(y_test, y_test.mean()))
    assert report["r2"] == pytest.approx(0.0, abs=1e-9)


def test_multi_condition_scaling_beats_global_scaling(cmapss_files):
    """The point of ConditionScaler: remove the regime, keep the degradation."""
    fd002 = load_train("FD002", cap=125, data_dir=cmapss_files)

    global_scaled = apply_scaler(fit_scaler(fd002, SENSORS), fd002, SENSORS)
    regime_scaled = ConditionScaler(n_regimes=6).fit_transform(fd002, SENSORS)

    # Spread of per-regime means: how much regime signal is left in the features.
    def regime_spread(df):
        return df.groupby(df["setting_1"].round(0))[SENSORS].mean().std().mean()

    assert regime_spread(regime_scaled) < 0.1 * regime_spread(global_scaled)


def test_scarcity_keeps_the_pipeline_consistent(cmapss_files):
    train_all = load_train("FD001", cap=125, data_dir=cmapss_files)
    scarce = subsample_engines(train_all, fraction=0.5)

    scaler = fit_scaler(scarce, SENSORS)          # refit on what the scenario has
    scaled = apply_scaler(scaler, scarce, SENSORS)
    X, y, _ = make_sequences(scaled, SENSORS, sequence_length=30)

    assert scarce["engine_id"].nunique() == 6
    assert y.min() == 0.0 and y.max() == 125.0    # full target range retained
    assert len(X) == len(y)


def test_rolling_features_survive_the_real_frame_shape(cmapss_files):
    train_all = load_train("FD001", cap=125, data_dir=cmapss_files)
    enriched, names = add_rolling_features(train_all, SENSORS, windows=(5, 20))

    assert len(names) == 1 + len(SENSORS) * 6      # cycle + raw + 2x(mean,std) + drift
    assert enriched[names].isna().sum().sum() == 0
    assert len(enriched) == len(train_all)
