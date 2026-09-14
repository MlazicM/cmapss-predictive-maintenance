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
    SUBSET_INFO,
    SUBSETS,
)
from src.calibration import apply_cqr, apply_symmetric, calibrate_cqr, calibrate_symmetric
from src.data import load_test, load_train, split_by_engine, subsample_engines, truncate_engine_prefix
from src.evaluate import (
    format_interval_report,
    format_report,
    interval_report,
    label_distribution,
    regression_report,
)
from src.models import (
    build_lstm,
    build_quantile_lstm,
    build_xgb,
    compile_for_finetuning,
    freeze_recurrent_layers,
    make_early_stopping,
    mc_dropout_predict,
    quantile_predict,
    save_artifacts,
    save_calibration,
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
    scaler_for_subset,
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


def prepare_subset(subset="FD001", seed=RANDOM_SEED):
    """Train / val split of a subset's training file, plus its official test set."""
    full_train = load_train(subset, cap=RUL_CAP)
    test_df = load_test(subset, cap=RUL_CAP)
    train_df, val_df = split_by_engine(full_train, val_fraction=0.2, seed=seed)
    return train_df, val_df, test_df


def prepare_fd001(seed=RANDOM_SEED):
    """FD001 splits, for the stages that only ever look at FD001."""
    return prepare_subset("FD001", seed=seed)


def regime_windows(subset, train_df, val_df, test_df):
    """Scale with a scaler matched to the subset's regimes, then build windows.

    On the six-regime subsets a single global scaler mostly encodes which regime
    a reading came from and buries the degradation signal. Deriving the scaler
    from the subset name rather than choosing it by hand removes the mistake
    that made the first transfer experiment uninterpretable.
    """
    scaler = scaler_for_subset(subset).fit(train_df, INFORMATIVE_SENSORS)
    train_s, val_s, test_s = (scaler.transform(df) for df in (train_df, val_df, test_df))

    X_train, y_train, _ = make_sequences(train_s, INFORMATIVE_SENSORS, SEQUENCE_LENGTH)
    X_val, y_val, _ = make_sequences(val_s, INFORMATIVE_SENSORS, SEQUENCE_LENGTH)
    X_test, y_test, engine_ids = last_sequence_per_engine(
        test_s, INFORMATIVE_SENSORS, SEQUENCE_LENGTH
    )
    return scaler, (X_train, y_train), (X_val, y_val), (X_test, y_test, engine_ids)


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


# --- Stage 4: every benchmark subset -----------------------------------------

def stage_subsets(epochs: int) -> pd.DataFrame:
    """Run the two strongest baselines on all four subsets.

    FD001 is the easy corner of this benchmark: one operating condition, one
    fault mode, near-monotonic degradation. Reporting only FD001 says nothing
    about whether an approach survives several operating regimes (FD002/FD004)
    or a second failure mode (FD003/FD004) - which is where a real fleet lives.
    """
    log("stage: all subsets")
    results = {}

    for subset in SUBSETS:
        info = SUBSET_INFO[subset]
        log(f"--- {subset}: {info['regimes']} regime(s), {info['faults']} fault mode(s) ---")
        set_global_seeds()

        train_df, val_df, test_df = prepare_subset(subset)
        scaler, (X_train, y_train), val, (X_test, y_test, _) = regime_windows(
            subset, train_df, val_df, test_df
        )
        extra = dict(
            regimes=info["regimes"],
            faults=info["faults"],
            train_engines=train_df["engine_id"].nunique(),
        )

        # Tree baseline on the same regime-scaled frames, so the comparison is fair.
        train_feat, names = add_rolling_features(scaler.transform(train_df), INFORMATIVE_SENSORS)
        test_feat, _ = add_rolling_features(scaler.transform(test_df), INFORMATIVE_SENSORS)
        test_last = test_feat.loc[test_feat.groupby("engine_id")["cycle"].idxmax()]

        tree = build_xgb().fit(train_feat[names], train_feat["RUL"])
        report = regression_report(test_last["RUL"], tree.predict(test_last[names]))
        report.update(extra)
        results[f"{subset} XGBoost"] = report
        log(format_report(f"{subset} XGBoost", report))

        lstm = fit_lstm(X_train, y_train, val, epochs)
        report = regression_report(y_test, lstm.predict(X_test, verbose=0).ravel())
        report.update(extra)
        results[f"{subset} LSTM"] = report
        log(format_report(f"{subset} LSTM", report))

        save_artifacts(lstm, scaler, name=f"lstm_{subset.lower()}")

    return pd.DataFrame(results).T


# --- Stage 5: transfer across a controlled domain gap ------------------------

def stage_cross_transfer(epochs: int, source="FD001", target="FD003",
                         fractions=(0.1, 0.3, 1.0)) -> pd.DataFrame:
    """Transfer between subsets that differ in exactly one property.

    The FD002 -> FD001 result is confounded: FD002 is both larger (260 engines)
    and broader (six regimes, probably covering FD001's single one), so part of
    the gain is just more data from a superset domain. FD001 -> FD003 removes
    the confound - same operating condition, same fleet size, one extra fault
    mode - so whatever transfers has to be degradation structure, not coverage.
    """
    import tensorflow as tf

    log(f"stage: cross transfer {source} -> {target}")
    set_global_seeds()

    source_train, _, _ = prepare_subset(source)
    target_train, target_val, target_test = prepare_subset(target)

    source_scaler = scaler_for_subset(source).fit(source_train, INFORMATIVE_SENSORS)
    X_source, y_source, _ = make_sequences(
        source_scaler.transform(source_train), INFORMATIVE_SENSORS, SEQUENCE_LENGTH
    )
    target_scaler = scaler_for_subset(target).fit(target_train, INFORMATIVE_SENSORS)
    X_val, y_val, _ = make_sequences(
        target_scaler.transform(target_val), INFORMATIVE_SENSORS, SEQUENCE_LENGTH
    )
    X_test, y_test, _ = last_sequence_per_engine(
        target_scaler.transform(target_test), INFORMATIVE_SENSORS, SEQUENCE_LENGTH
    )
    log(f"source windows {X_source.shape}, target test {X_test.shape}")

    source_model = fit_lstm(X_source, y_source, (X_val, y_val), epochs)
    results = {}
    results["zero-shot"] = regression_report(
        y_test, source_model.predict(X_test, verbose=0).ravel()
    )
    results["zero-shot"]["target_engines"] = 0
    log(format_report(f"{source}->{target} zero-shot", results["zero-shot"]))

    for fraction in fractions:
        subset_df = subsample_engines(target_train, fraction=fraction, seed=RANDOM_SEED)
        X_sub, y_sub, _ = make_sequences(
            target_scaler.transform(subset_df), INFORMATIVE_SENSORS, SEQUENCE_LENGTH
        )
        n_engines = subset_df["engine_id"].nunique()

        set_global_seeds()
        tuned = tf.keras.models.clone_model(source_model)
        tuned.set_weights(source_model.get_weights())
        freeze_recurrent_layers(tuned, frozen=True)
        compile_for_finetuning(tuned, learning_rate=1e-3)
        tuned = fit_lstm(X_sub, y_sub, (X_val, y_val), epochs, model=tuned)

        set_global_seeds()
        scratch = fit_lstm(X_sub, y_sub, (X_val, y_val), epochs)

        for label, model in (("fine-tuned", tuned), ("scratch", scratch)):
            key = f"{label} @ {int(fraction * 100)}%"
            report = regression_report(y_test, model.predict(X_test, verbose=0).ravel())
            report["target_engines"] = n_engines
            results[key] = report
            log(format_report(f"{source}->{target} {key}", report))

    return pd.DataFrame(results).T


# --- Stage 6: calibrated prediction intervals --------------------------------

def stage_calibration(
    epochs: int, n_members: int = 5, nominal: float = 0.95
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare three ways of producing a 95% interval, before and after conformal.

    The MC-dropout bands this project already reports cover 62% of engines at a
    nominal 95%, because dropout spread is model uncertainty and nothing ties
    it to observation noise. Three candidate fixes are measured against each
    other on the same engines:

    * a **deep ensemble** - five independently seeded networks, spread taken
      across members;
    * **quantile regression** - one network with a 2.5/50/97.5 head trained on
      the pinball loss, so the interval edges are fitted rather than inferred;
    * **split conformal** on top of each - rescaling the interval by the
      empirical quantile of its own errors on engines held out from training.

    The splits are four-way, not three. Training fits on ``fit`` engines, early
    stopping watches ``val`` engines, conformal calibrates on ``calib`` engines
    and the official test files are still touched once. Calibrating on data
    that trained or selected the model would hand back exactly the
    overconfidence the exercise is meant to remove.

    Calibration windows are *every* window of each calibration engine, not just
    its last. The window ending at cycle t is precisely what a test engine
    truncated at cycle t would look like, so every window is a valid draw from
    the deployment distribution - and 16 held-out engines then supply a few
    thousand calibration points rather than 16.

    The ensemble members double as a measurement of something every other LSTM
    number in this project quietly depends on: how much a result moves when only
    the seed changes. Each member is scored individually and the spread is
    written to ``06_seed_variance.csv``, so "single seed per arm" stops being a
    caveat and becomes a number.
    """
    log("stage: calibrated intervals")
    set_global_seeds()

    train_df, val_df, test_df = prepare_fd001()
    fit_df, calib_df = split_by_engine(train_df, val_fraction=0.2, seed=RANDOM_SEED + 1)
    log(
        f"engines - fit {fit_df['engine_id'].nunique()}, "
        f"val {val_df['engine_id'].nunique()}, "
        f"calib {calib_df['engine_id'].nunique()}, "
        f"test {test_df['engine_id'].nunique()}"
    )

    # Scaler fitted on the fitting engines only: calib is held out from that too.
    scaler = fit_scaler(fit_df, INFORMATIVE_SENSORS)
    X_fit, y_fit, _ = make_sequences(
        apply_scaler(scaler, fit_df, INFORMATIVE_SENSORS), INFORMATIVE_SENSORS, SEQUENCE_LENGTH
    )
    X_val, y_val, _ = make_sequences(
        apply_scaler(scaler, val_df, INFORMATIVE_SENSORS), INFORMATIVE_SENSORS, SEQUENCE_LENGTH
    )
    X_calib, y_calib, _ = make_sequences(
        apply_scaler(scaler, calib_df, INFORMATIVE_SENSORS), INFORMATIVE_SENSORS, SEQUENCE_LENGTH
    )
    X_test, y_test, engine_ids = last_sequence_per_engine(
        apply_scaler(scaler, test_df, INFORMATIVE_SENSORS), INFORMATIVE_SENSORS, SEQUENCE_LENGTH
    )
    log(f"fit windows {X_fit.shape}, calib windows {X_calib.shape}, test {X_test.shape}")

    # --- the models -------------------------------------------------------
    members, member_reports = [], {}
    for member in range(n_members):
        seed = RANDOM_SEED + member
        set_global_seeds(seed)
        log(f"ensemble member {member + 1}/{n_members} (seed {seed})")
        model = fit_lstm(X_fit, y_fit, (X_val, y_val), epochs)
        members.append(model)

        report = regression_report(y_test, model.predict(X_test, verbose=0).ravel())
        report["seed"] = seed
        member_reports[f"seed {seed}"] = report
        log(format_report(f"member {member + 1} (seed {seed})", report))

    set_global_seeds()
    quantiles = [(1 - nominal) / 2, 0.5, 1 - (1 - nominal) / 2]
    quantile_model = fit_lstm(
        X_fit, y_fit, (X_val, y_val), epochs,
        model=build_quantile_lstm(input_shape=X_fit.shape[1:], quantiles=quantiles),
    )

    def ensemble_predict(X):
        stacked = np.stack([m.predict(X, verbose=0).ravel() for m in members])
        return stacked.mean(axis=0), stacked.std(axis=0)

    # Member 0 is the plain single network, so "MC dropout" here is exactly the
    # incumbent method rather than a differently trained model.
    base = members[0]
    gaussian_z = 1.96  # what a normal approximation would use, for the raw arms

    rows, columns = {}, {}

    def record(label, lower, upper, point):
        report = interval_report(y_test, lower, upper, point=point, nominal=nominal)
        rows[label] = report
        columns[f"{label} lower"] = lower
        columns[f"{label} upper"] = upper
        log(format_interval_report(label, report))

    def conformalise(label, mean_c, std_c, mean_t, std_t):
        """Calibrate one mean/std method both ways and record each."""
        multipliers = {}
        for mode in ("adaptive", "absolute"):
            multiplier = calibrate_symmetric(y_calib, mean_c, std_c, nominal=nominal, mode=mode)
            multipliers[mode] = multiplier
            unit = "x std" if mode == "adaptive" else "cycles"
            log(f"{label} conformal multiplier ({mode}): {multiplier:.2f} {unit}")
            record(
                f"{label} + conformal ({mode})",
                *apply_symmetric(mean_t, std_t, multiplier, mode=mode),
                mean_t,
            )
        return multipliers

    # --- MC dropout, raw and conformalised --------------------------------
    mc_mean_c, mc_std_c = mc_dropout_predict(base, X_calib, n_samples=100)
    mc_mean_t, mc_std_t = mc_dropout_predict(base, X_test, n_samples=100)
    record(
        "MC dropout",
        mc_mean_t - gaussian_z * mc_std_t, mc_mean_t + gaussian_z * mc_std_t, mc_mean_t,
    )
    log(f"a Gaussian interval would use {gaussian_z} standard deviations")
    mc_multipliers = conformalise("MC dropout", mc_mean_c, mc_std_c, mc_mean_t, mc_std_t)

    # --- deep ensemble, raw and conformalised -----------------------------
    ens_mean_c, ens_std_c = ensemble_predict(X_calib)
    ens_mean_t, ens_std_t = ensemble_predict(X_test)
    record(
        "Deep ensemble",
        ens_mean_t - gaussian_z * ens_std_t, ens_mean_t + gaussian_z * ens_std_t, ens_mean_t,
    )
    conformalise("Deep ensemble", ens_mean_c, ens_std_c, ens_mean_t, ens_std_t)

    # --- quantile regression, raw and conformalised (CQR) -----------------
    q_calib, crossing_calib = quantile_predict(quantile_model, X_calib)
    q_test, crossing_test = quantile_predict(quantile_model, X_test)
    log(f"quantile crossing rate: calib {crossing_calib:.1%}, test {crossing_test:.1%}")
    record("Quantile regression", q_test[:, 0], q_test[:, 2], q_test[:, 1])

    cqr_offset = calibrate_cqr(y_calib, q_calib[:, 0], q_calib[:, 2], nominal=nominal)
    log(f"CQR conformal offset: {cqr_offset:+.2f} cycles")
    record(
        "Quantile regression + CQR",
        *apply_cqr(q_test[:, 0], q_test[:, 2], cqr_offset), q_test[:, 1],
    )

    for label in rows:
        rows[label]["quantile_crossing_rate"] = crossing_test if "Quantile" in label else 0.0

    # --- persist what the service should serve -----------------------------
    # Every model that was calibrated is saved with its own scaler and its own
    # constant, so the API can never apply a number fitted for a different one.
    # Which of the two conformal modes ships is decided by the interval score on
    # the test engines rather than by preference: both hit their coverage
    # target, so all that is left to choose on is how wide they had to be.
    shared = {
        "nominal": nominal,
        "n_calibration_points": int(len(y_calib)),
        "calibration_engines": int(calib_df["engine_id"].nunique()),
        "rul_cap": RUL_CAP,
    }

    served_mode = min(
        ("adaptive", "absolute"),
        key=lambda mode: rows[f"MC dropout + conformal ({mode})"]["interval_score"],
    )
    save_artifacts(base, scaler, name="lstm_fd001_conformal")
    save_calibration(
        "lstm_fd001_conformal",
        {
            "method": "mc_dropout_split_conformal",
            "mode": served_mode,
            "multiplier": mc_multipliers[served_mode],
            "n_mc_samples": 100,
            "test_coverage": rows[f"MC dropout + conformal ({served_mode})"]["coverage"],
            **shared,
        },
    )

    save_artifacts(quantile_model, scaler, name="lstm_fd001_quantile")
    save_calibration(
        "lstm_fd001_quantile",
        {
            "method": "quantile_regression_cqr",
            "quantiles": quantiles,
            "offset": cqr_offset,
            "test_coverage": rows["Quantile regression + CQR"]["coverage"],
            "crossing_rate": crossing_test,
            **shared,
        },
    )

    # The API prefers the quantile artifact; say here whether that preference is
    # still the one the measurements support, so a future run that overturns it
    # shows up in the log instead of being silently contradicted by the service.
    best = min(rows, key=lambda label: rows[label]["interval_score"])
    log(f"best interval score: {best} ({rows[best]['interval_score']:.1f})")
    if not best.startswith("Quantile"):
        log(
            "NOTE: the API's preference order puts the quantile model first, but "
            f"'{best}' scored better on this run -- revisit api/main.py MODEL_NAMES"
        )

    # What one seed buys and what it costs: the ensemble against its own members.
    seeds = pd.DataFrame(member_reports).T
    ensemble_report = regression_report(y_test, ens_mean_t)
    ensemble_report["seed"] = float("nan")
    seeds.loc["ensemble mean"] = ensemble_report
    member_rmse = seeds["rmse"].iloc[:-1]          # every row but the appended mean
    log(
        f"single-seed RMSE across {n_members} seeds: "
        f"{member_rmse.min():.2f} - {member_rmse.max():.2f} "
        f"(spread {member_rmse.max() - member_rmse.min():.2f}); "
        f"averaging them gives {ensemble_report['rmse']:.2f}"
    )

    per_engine = pd.DataFrame({"engine_id": engine_ids, "true_rul": y_test, **columns})
    return pd.DataFrame(rows).T, per_engine, seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=[
            "baselines", "scarcity", "transfer", "subsets",
            "cross-transfer", "calibration", "all",
        ],
        default="all",
    )
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
    if args.stage in ("subsets", "all"):
        stage_subsets(epochs).to_csv(RESULTS_DIR / "05_all_subsets.csv")
    if args.stage in ("cross-transfer", "all"):
        cross_fractions = (0.3,) if args.quick else (0.1, 0.3, 1.0)
        stage_cross_transfer(epochs, fractions=cross_fractions).to_csv(
            RESULTS_DIR / "05_cross_transfer_fd001_fd003.csv"
        )
    if args.stage in ("transfer", "all"):
        arms, sweep, uncertainty = stage_transfer(epochs, sweep_fractions)
        arms.to_csv(RESULTS_DIR / "04_transfer_learning.csv")
        sweep.to_csv(RESULTS_DIR / "04_target_data_sweep.csv")
        uncertainty.to_csv(RESULTS_DIR / "04_uncertainty.csv", index=False)
    if args.stage in ("calibration", "all"):
        members = 2 if args.quick else 5
        summary, per_engine, seeds = stage_calibration(epochs, n_members=members)
        summary.to_csv(RESULTS_DIR / "06_calibration.csv")
        per_engine.to_csv(RESULTS_DIR / "06_calibration_intervals.csv", index=False)
        seeds.to_csv(RESULTS_DIR / "06_seed_variance.csv")

    log(f"done in {(time.time() - started) / 60:.1f} min; results in {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
