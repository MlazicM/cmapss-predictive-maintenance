"""Feature scaling and sequence construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import (
    N_OPERATING_REGIMES,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    SETTING_COLUMNS,
    SUBSET_INFO,
)


# --- Scaling ---------------------------------------------------------------

def fit_scaler(df: pd.DataFrame, features: list[str]) -> StandardScaler:
    """Fit a scaler on the training split only.

    Fitting on the full frame would leak validation/test statistics into
    training. The returned object is what must be persisted alongside the
    model: inference has to reuse these exact statistics.
    """
    return StandardScaler().fit(df[features])


def apply_scaler(scaler, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return a copy of ``df`` with ``features`` transformed by ``scaler``."""
    out = df.copy()
    out[features] = scaler.transform(df[features])
    return out


def scaler_for_subset(subset: str, seed: int = RANDOM_SEED) -> "ConditionScaler":
    """Return a scaler matched to the subset's number of operating regimes.

    Picking this by hand is exactly the mistake that made the first transfer
    experiment uninterpretable: single-condition FD001 was pushed through
    statistics fitted on six-condition FD002. Deriving it from the subset name
    removes the choice.
    """
    regimes = SUBSET_INFO[subset]["regimes"]
    return ConditionScaler(n_regimes=regimes, seed=seed)


class ConditionScaler:
    """Per-operating-regime standardisation.

    FD002 and FD004 are flown under six discrete operating regimes, and the
    regime shifts the sensor readings far more than degradation does. A single
    global :class:`StandardScaler` therefore mostly encodes "which regime is
    this", drowning the degradation signal and making the subset look
    unlearnable. Clustering the three setting columns recovers the regimes and
    standardises within each, which is the standard treatment for the
    multi-condition CMAPSS subsets.

    On single-condition subsets (FD001, FD003) this degenerates to ordinary
    standardisation.
    """

    def __init__(
        self,
        n_regimes: int = N_OPERATING_REGIMES,
        setting_columns: list[str] | None = None,
        seed: int = RANDOM_SEED,
    ):
        self.n_regimes = n_regimes
        self.setting_columns = setting_columns or SETTING_COLUMNS
        self.seed = seed

    def fit(self, df: pd.DataFrame, features: list[str]) -> "ConditionScaler":
        self.features_ = list(features)
        self.kmeans_ = KMeans(
            n_clusters=self.n_regimes, random_state=self.seed, n_init=10
        ).fit(df[self.setting_columns])

        labels = self.kmeans_.labels_
        self.scalers_ = {}
        for regime in range(self.n_regimes):
            mask = labels == regime
            if mask.sum() < 2:
                # Degenerate regime (e.g. single-condition subset): fall back to
                # the global statistics so transform() never divides by zero.
                self.scalers_[regime] = StandardScaler().fit(df[self.features_])
            else:
                self.scalers_[regime] = StandardScaler().fit(df.loc[mask, self.features_])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        regimes = self.kmeans_.predict(df[self.setting_columns])
        out = df.copy()
        values = out[self.features_].to_numpy(dtype=float, copy=True)
        for regime, scaler in self.scalers_.items():
            mask = regimes == regime
            if mask.any():
                values[mask] = scaler.transform(df.loc[mask, self.features_])
        out[self.features_] = values
        return out

    def fit_transform(self, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        return self.fit(df, features).transform(df)


# --- Sequences -------------------------------------------------------------

def make_sequences(
    df: pd.DataFrame,
    features: list[str],
    sequence_length: int = SEQUENCE_LENGTH,
    target: str = "RUL",
    group_col: str = "engine_id",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding windows, never crossing an engine boundary.

    Returns ``(X, y, engine_ids)`` where ``X`` has shape
    ``(n_windows, sequence_length, n_features)`` and ``y[k]`` is the RUL at the
    **last** cycle of window ``k`` -- the quantity actually available at
    inference time, given sensor history up to now.

    An engine of length ``n`` yields ``n - sequence_length + 1`` windows.
    Dropping the ``+ 1`` discards the final window of every engine, which is
    the lowest-RUL and most informative sample in the whole dataset.
    """
    df = df.sort_values([group_col, "cycle"])
    windows, targets, engine_ids = [], [], []

    for engine_id, group in df.groupby(group_col, sort=True):
        if len(group) < sequence_length:
            continue  # too short to form a single window
        values = group[features].to_numpy(dtype=np.float32)
        # (n - L + 1, n_features, L) -> (n - L + 1, L, n_features)
        strided = sliding_window_view(values, sequence_length, axis=0).transpose(0, 2, 1)
        windows.append(strided)
        targets.append(group[target].to_numpy(dtype=np.float32)[sequence_length - 1:])
        engine_ids.append(np.full(len(strided), engine_id))

    if not windows:
        return (
            np.empty((0, sequence_length, len(features)), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )

    return (
        np.concatenate(windows).astype(np.float32),
        np.concatenate(targets).astype(np.float32),
        np.concatenate(engine_ids),
    )


def last_sequence_per_engine(
    df: pd.DataFrame,
    features: list[str],
    sequence_length: int = SEQUENCE_LENGTH,
    target: str = "RUL",
    group_col: str = "engine_id",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One window per engine, ending at its last recorded cycle.

    This is the official CMAPSS test protocol: the benchmark asks for a single
    RUL estimate per test engine, so scoring every intermediate window instead
    would produce numbers that are not comparable to published results.
    Engines shorter than ``sequence_length`` are left-padded with their first
    observation rather than dropped, since the benchmark expects a prediction
    for every engine.
    """
    df = df.sort_values([group_col, "cycle"])
    windows, targets, engine_ids = [], [], []

    for engine_id, group in df.groupby(group_col, sort=True):
        values = group[features].to_numpy(dtype=np.float32)
        if len(values) < sequence_length:
            pad = np.repeat(values[:1], sequence_length - len(values), axis=0)
            values = np.concatenate([pad, values])
        windows.append(values[-sequence_length:])
        targets.append(group[target].to_numpy(dtype=np.float32)[-1])
        engine_ids.append(engine_id)

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(engine_ids),
    )


def augment_with_noise(
    X: np.ndarray,
    y: np.ndarray,
    n_copies: int = 3,
    noise_level: float = 0.1,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Jitter *already-built sequences* with Gaussian noise.

    Augmenting the sequence tensor rather than the dataframe is deliberate.
    Concatenating noisy copies of a dataframe and windowing afterwards makes
    the sliding window straddle the seam between two copies, producing samples
    whose sensors show an engine at end-of-life while the label says it is new.

    ``noise_level`` is in units of standard deviations, because the input is
    already standardised -- the 0.01 that reads like "1% noise" on raw sensors
    is 1% of one sigma here, i.e. nothing.
    """
    rng = np.random.default_rng(seed)
    copies_X = [X]
    copies_y = [y]
    for _ in range(n_copies):
        copies_X.append(X + rng.normal(0.0, noise_level, size=X.shape).astype(np.float32))
        copies_y.append(y)
    return np.concatenate(copies_X), np.concatenate(copies_y)


# --- Tabular features (for the tree baseline) -------------------------------

def add_rolling_features(
    df: pd.DataFrame,
    features: list[str],
    windows: tuple[int, ...] = (5, 20),
    group_col: str = "engine_id",
) -> tuple[pd.DataFrame, list[str]]:
    """Add rolling mean/std and cumulative drift per sensor.

    A tree model fed one cycle at a time cannot see a trend, so comparing it to
    an LSTM that sees thirty cycles measures the input representation, not the
    model class. These features give the tree the same temporal information.
    Returns the enriched frame and the full feature name list.
    """
    df = df.sort_values([group_col, "cycle"]).copy()
    grouped = df.groupby(group_col)
    new_columns = {}

    for window in windows:
        rolled = grouped[features].rolling(window, min_periods=1)
        for name, frame in (("mean", rolled.mean()), ("std", rolled.std())):
            frame = frame.reset_index(level=0, drop=True)
            for feature in features:
                new_columns[f"{feature}_{name}_{window}"] = frame[feature]

    for feature in features:
        new_columns[f"{feature}_drift"] = df[feature] - grouped[feature].transform("first")

    enriched = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    enriched = enriched.fillna(0.0)
    feature_names = ["cycle"] + features + list(new_columns)
    return enriched, feature_names
