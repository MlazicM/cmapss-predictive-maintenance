"""Headless runner for every experiment phase.

Reproduces the notebooks without Jupyter, writing one CSV per stage into
results/. This is what CI and any re-measurement should call: the notebooks are
for reading, this is for running.

    python scripts/run_experiments.py --stage baselines
    python scripts/run_experiments.py --stage all --quick
"""

from __future__ import annotations

import argparse
import sys
import time

import sys
from pathlib import Path

# Running this file directly puts scripts/ on sys.path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import numpy as np
import pandas as pd

from src.config import (
    INFORMATIVE_SENSORS,
    MODELS_DIR,
    N_OPERATING_REGIMES,
    RANDOM_SEED,
    RESULTS_DIR,
    RUL_CAP,
    SEQUENCE_LENGTH,
)
from src.data import load_test, load_train, split_by_engine, subsample_engines, truncate_engine_prefix
from src.evaluate import format_report, label_distribution, regression_report
from src.models import (
    build_lstm,
    build_xgb,
    compile_for_finetuning,
    freeze_recurrent_layers,
    make_early_stopping,
    mc_dropout_predict,
    save_artifacts,
    set_global_seeds,
)
from src.preprocessing import (
    ConditionScaler,
    add_rolling_features,
    apply_scaler,
    augment_with_noise,
    fit_scaler,
    last_sequence_per_engine,
    make_sequences,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def fit_lstm(X, y, validation, epochs, input_shape=None, model=None, patience=8):
    """Train an LSTM and return it. Shared by every stage so the setup cannot drift."""
    model = model if model is not None else build_lstm(input_shape=input_shape or X.shape[1:])
    model.fit(
        X, y,
        validation_data=validation,
        epochs=epochs,
        batch_size=64,
        callbacks=[make_early_stopping(patience=patience)],
        verbose=2,
    )
    return model


def prepare_fd001(seed=RANDOM_SEED):
    """The splits every stage shares: train / val from train_FD001, official test."""
    full_train = load_train("FD001", cap=RUL_CAP)
    test_df = load_test("FD001", cap=RUL_CAP)
    train_df, val_df = split_by_engine(full_train, val_fraction=0.2, seed=seed)
    return train_df, val_df, test_df


def windows_for(train_df, val_df, test_df, scaler=None):
    """Scale on the given training data only, then build windows for all splits."""
    scaler = scaler or fit_scaler(train_df, INFORMATIVE_SENSORS)
    train_s = apply_scaler(scaler, train_df, INFORMATIVE_SENSORS)
    val_s = apply_scaler(scaler, val_df, INFORMATIVE_SENSORS)
    test_s = apply_scaler(scaler, test_df, INFORMATIVE_SENSORS)

    X_train, y_train, _ = make_sequences(train_s, INFORMATIVE_SENSORS, SEQUENCE_LENGTH)
    X_val, y_val, _ = make_sequences(val_s, INFORMATIVE_SENSORS, SEQUENCE_LENGTH)
    X_test, y_test, engine_ids = last_sequence_per_engine(test_s, INFORMATIVE_SENSORS, SEQUENCE_LENGTH)
    return scaler, (X_train, y_train), (X_val, y_val), (X_test, y_test, engine_ids)


# --- Stage 1: baselines ----------------------------------------------------

def stage_baselines(epochs: int) -> pd.DataFrame:
    log("stage: baselines")
    set_global_seeds()
    train_df, val_df, test_df = prepare_fd001()
    scaler, (X_train, y_train), val, (X_test, y_test, _) = windows_for(train_df, val_df, test_df)

    log(f"train windows {X_train.shape}, target {label_distribution(y_train)}")
    results = {}

    # Tree baselines, scored on the same one-window-per-engine targets.
    train_s = apply_scaler(scaler, train_df, INFORMATIVE_SENSORS)
    test_s = apply_scaler(scaler, test_df, INFORMATIVE_SENSORS)
    test_last = test_s.loc[test_s.groupby("engine_id")["cycle"].idxmax()]

    xgb_plain = build_xgb().fit(train_s[INFORMATIVE_SENSORS], train_s["RUL"])
    results["XGBoost (per-cycle)"] = regression_report(
        test_last["RUL"], xgb_plain.predict(test_last[INFORMATIVE_SENSORS])
    )

    train_feat, names = add_rolling_features(train_s, INFORMATIVE_SENSORS)
    test_feat, _ = add_rolling_features(test_s, INFORMATIVE_SENSORS)
    test_last_feat = test_feat.loc[test_feat.groupby("engine_id")["cycle"].idxmax()]
    xgb_window = build_xgb().fit(train_feat[names], train_feat["RUL"])
    results["XGBoost (rolling features)"] = regression_report(
        test_last_feat["RUL"], xgb_window.predict(test_last_feat[names])
    )

    lstm = fit_lstm(X_train, y_train, val, epochs)
    results["LSTM"] = regression_report(y_test, lstm.predict(X_test, verbose=0).ravel())

    save_artifacts(lstm, scaler, name="lstm_fd001")
    for name, report in results.items():
        log(format_report(name, report))
    return pd.DataFrame(results).T


# --- Stage 2: scarcity and augmentation ------------------------------------

def scenario(train_subset, val_df, test_df, label, epochs, n_copies=0, noise_level=0.1):
    """One scarcity arm: refit the scaler on what this scenario actually has."""
    set_global_seeds()
    _, (X_train, y_train), val, (X_test, y_test, _) = windows_for(train_subset, val_df, test_df)
    if n_copies:
        X_train, y_train = augment_with_noise(X_train, y_train, n_copies, noise_level)

    model = fit_lstm(X_train, y_train, val, epochs)
    report = regression_report(y_test, model.predict(X_test, verbose=0).ravel())
    report["train_engines"] = train_subset["engine_id"].nunique()
    report["train_sequences"] = len(X_train)
    log(format_report(label, report))
    return report


def stage_scarcity(epochs: int, fractions: list[float]) -> pd.DataFrame:
    log("stage: scarcity + augmentation")
    train_df, val_df, test_df = prepare_fd001()
    results = {}

    results["100% engines (reference)"] = scenario(train_df, val_df, test_df, "100% engines", epochs)
    for fraction in fractions:
        label = f"{int(fraction * 100)}% engines"
        subset = subsample_engines(train_df, fraction=fraction, seed=RANDOM_SEED)
        results[label] = scenario(subset, val_df, test_df, label, epochs)

    broken = truncate_engine_prefix(train_df, fraction=0.3)
    log(f"prefix-truncated labels: {label_distribution(broken['RUL'].to_numpy())}")
    results["30% prefix (flawed design)"] = scenario(
        broken, val_df, test_df, "30% prefix (flawed design)", epochs
    )

    scarce = subsample_engines(train_df, fraction=0.3, seed=RANDOM_SEED)
    for noise_level in [0.05, 0.1, 0.25]:
        label = f"30% engines + noise {noise_level}"
        results[label] = scenario(
            scarce, val_df, test_df, label, epochs, n_copies=3, noise_level=noise_level
        )

    return pd.DataFrame(results).T


# --- Stage 3: transfer learning --------------------------------------------

def stage_transfer(epochs: int, fractions: list[float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log("stage: transfer learning")
    import tensorflow as tf

    set_global_seeds()
    train_df, val_df, test_df = prepare_fd001()
    fd002 = load_train("FD002", cap=RUL_CAP)

    # Each fleet standardised within its own operating regimes.
    fd002_scaled = ConditionScaler(n_regimes=N_OPERATING_REGIMES).fit_transform(
        fd002, INFORMATIVE_SENSORS
    )
    fd001_scaler = ConditionScaler(n_regimes=1).fit(train_df, INFORMATIVE_SENSORS)

    X_source, y_source, _ = make_sequences(fd002_scaled, INFORMATIVE_SENSORS, SEQUENCE_LENGTH)
    X_val, y_val, _ = make_sequences(fd001_scaler.transform(val_df), INFORMATIVE_SENSORS, SEQUENCE_LENGTH)
    X_test, y_test, engine_ids = last_sequence_per_engine(
        fd001_scaler.transform(test_df), INFORMATIVE_SENSORS, SEQUENCE_LENGTH
    )
    log(f"source windows {X_source.shape}")

    source_model = fit_lstm(X_source, y_source, (X_val, y_val), epochs)
    source_model.save(MODELS_DIR / "lstm_fd002_pretrained.keras")

    scarce = subsample_engines(train_df, fraction=0.3, seed=RANDOM_SEED)
    X_target, y_target, _ = make_sequences(
        fd001_scaler.transform(scarce), INFORMATIVE_SENSORS, SEQUENCE_LENGTH
    )

    def clone_pretrained(frozen: bool, learning_rate: float):
        model = tf.keras.models.clone_model(source_model)
        model.set_weights(source_model.get_weights())
        freeze_recurrent_layers(model, frozen=frozen)
        compile_for_finetuning(model, learning_rate=learning_rate)  # flag is inert without this
        return model

    arms = {}
    arms["zero-shot (FD002 only)"] = regression_report(
        y_test, source_model.predict(X_test, verbose=0).ravel()
    )
    set_global_seeds()
    scratch = fit_lstm(X_target, y_target, (X_val, y_val), epochs)
    arms["scratch (no transfer)"] = regression_report(
        y_test, scratch.predict(X_test, verbose=0).ravel()
    )

    tuned = {}
    for frozen, learning_rate, label in [
        (True, 1e-3, "fine-tuned, LSTM frozen"),
        (False, 1e-4, "fine-tuned, full network"),
    ]:
        set_global_seeds()
        model = fit_lstm(X_target, y_target, (X_val, y_val), epochs, model=clone_pretrained(frozen, learning_rate))
        arms[label] = regression_report(y_test, model.predict(X_test, verbose=0).ravel())
        tuned[label] = model

    for name, report in arms.items():
        log(format_report(name, report))

    best_label = min(arms, key=lambda k: arms[k]["rmse"])
    log(f"best arm: {best_label}")
    use_frozen = "frozen" in best_label

    # How much target data transfer actually saves, against a from-scratch control.
    sweep = {}
    for fraction in fractions:
        subset = subsample_engines(train_df, fraction=fraction, seed=RANDOM_SEED)
        X_sub, y_sub, _ = make_sequences(
            fd001_scaler.transform(subset), INFORMATIVE_SENSORS, SEQUENCE_LENGTH
        )
        set_global_seeds()
        transferred = fit_lstm(
            X_sub, y_sub, (X_val, y_val), epochs,
            model=clone_pretrained(use_frozen, 1e-3 if use_frozen else 1e-4),
        )
        set_global_seeds()
        baseline = fit_lstm(X_sub, y_sub, (X_val, y_val), epochs)

        sweep[fraction] = {
            "engines": subset["engine_id"].nunique(),
            "transfer_rmse": regression_report(y_test, transferred.predict(X_test, verbose=0).ravel())["rmse"],
            "scratch_rmse": regression_report(y_test, baseline.predict(X_test, verbose=0).ravel())["rmse"],
        }
        log(f"fraction {fraction}: {sweep[fraction]}")

    best_model = tuned.get(best_label, source_model)
    mean, std = mc_dropout_predict(best_model, X_test, n_samples=100)
    uncertainty = pd.DataFrame(
        {"engine_id": engine_ids, "true_rul": y_test, "predicted_rul": mean, "std": std}
    )
    uncertainty["lower_95"] = uncertainty["predicted_rul"] - 1.96 * uncertainty["std"]
    uncertainty["upper_95"] = uncertainty["predicted_rul"] + 1.96 * uncertainty["std"]
    coverage = (
        (uncertainty["true_rul"] >= uncertainty["lower_95"])
        & (uncertainty["true_rul"] <= uncertainty["upper_95"])
    ).mean()
    log(f"95% interval empirical coverage: {coverage:.1%}")

    save_artifacts(best_model, fd001_scaler, name="lstm_transfer_fd001")
    return pd.DataFrame(arms).T, pd.DataFrame(sweep).T, uncertainty


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["baselines", "scarcity", "transfer", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--quick", action="store_true", help="short run for smoke-testing the wiring")
    args = parser.parse_args()

    epochs = 2 if args.quick else args.epochs
    fractions = [0.3] if args.quick else [0.5, 0.3, 0.15]
    sweep_fractions = [0.3] if args.quick else [0.1, 0.2, 0.3, 0.5, 1.0]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if args.stage in ("baselines", "all"):
        stage_baselines(epochs).to_csv(RESULTS_DIR / "02_baselines.csv")
    if args.stage in ("scarcity", "all"):
        stage_scarcity(epochs, fractions).to_csv(RESULTS_DIR / "03_limited_data.csv")
    if args.stage in ("transfer", "all"):
        arms, sweep, uncertainty = stage_transfer(epochs, sweep_fractions)
        arms.to_csv(RESULTS_DIR / "04_transfer_learning.csv")
        sweep.to_csv(RESULTS_DIR / "04_target_data_sweep.csv")
        uncertainty.to_csv(RESULTS_DIR / "04_uncertainty.csv", index=False)

    log(f"done in {(time.time() - started) / 60:.1f} min; results in {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
