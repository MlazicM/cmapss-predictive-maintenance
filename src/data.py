"""Loading, labelling and splitting of the NASA C-MAPSS dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import COLUMN_NAMES, RANDOM_SEED, RAW_DATA_DIR, RUL_CAP


def _read_cmapss_file(path) -> pd.DataFrame:
    """Read one headerless, whitespace-separated CMAPSS file."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)


def add_rul(df: pd.DataFrame, cap: int | None = RUL_CAP) -> pd.DataFrame:
    """Attach a piecewise-linear RUL column to a *training* frame.

    Training engines are run to failure, so the last recorded cycle is the
    failure point and RUL is simply ``max_cycle - cycle``. Values above ``cap``
    are clipped: while the engine is healthy the sensors carry no degradation
    signal, and an uncapped label would ask the model to predict a number it
    cannot possibly infer.
    """
    df = df.copy()
    df["RUL"] = df.groupby("engine_id")["cycle"].transform("max") - df["cycle"]
    if cap is not None:
        df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def load_train(subset: str = "FD001", cap: int | None = RUL_CAP, data_dir=RAW_DATA_DIR) -> pd.DataFrame:
    """Load a training subset (e.g. ``FD001``) with its RUL labels."""
    df = _read_cmapss_file(data_dir / f"train_{subset}.txt")
    return add_rul(df, cap=cap)


def load_test(subset: str = "FD001", cap: int | None = RUL_CAP, data_dir=RAW_DATA_DIR) -> pd.DataFrame:
    """Load a test subset and label it from the official ``RUL_*.txt`` file.

    Test engines are truncated *before* failure. ``RUL_<subset>.txt`` gives the
    remaining life at each engine's last recorded cycle, so for any earlier
    cycle the label is that value plus the number of cycles still to come.
    Using this file is what makes results comparable to published benchmarks.
    """
    df = _read_cmapss_file(data_dir / f"test_{subset}.txt")
    rul_at_last_cycle = pd.read_csv(
        data_dir / f"RUL_{subset}.txt", sep=r"\s+", header=None, names=["rul_at_last_cycle"]
    )
    # The n-th row of the RUL file belongs to the n-th engine id, in order.
    rul_at_last_cycle["engine_id"] = np.sort(df["engine_id"].unique())

    df = df.merge(rul_at_last_cycle, on="engine_id", how="left")
    last_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = df["rul_at_last_cycle"] + (last_cycle - df["cycle"])
    df = df.drop(columns="rul_at_last_cycle")
    if cap is not None:
        df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def split_by_engine(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = RANDOM_SEED,
    shuffle: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on ``engine_id`` so that no engine appears on both sides.

    Splitting on rows would leak: two windows from the same engine, a few
    cycles apart, are almost the same sample. Engine ids are shuffled with a
    fixed seed rather than sliced in file order, which would silently bake in
    whatever ordering the dataset happens to have.
    """
    engines = np.sort(df["engine_id"].unique())
    if shuffle:
        engines = np.random.default_rng(seed).permutation(engines)

    n_val = int(round(len(engines) * val_fraction))
    val_engines = set(engines[:n_val])

    is_val = df["engine_id"].isin(val_engines)
    return df[~is_val].copy(), df[is_val].copy()


def subsample_engines(
    df: pd.DataFrame, fraction: float, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Keep a random subset of engines, each with its **full** life history.

    This is the correct way to simulate label scarcity for RUL: in practice you
    have few engines that were monitored to failure, not a truncated prefix of
    many. Every retained engine still spans the whole RUL range, so the target
    distribution matches the one at inference time.
    """
    engines = np.sort(df["engine_id"].unique())
    n_keep = max(1, int(round(len(engines) * fraction)))
    keep = np.random.default_rng(seed).permutation(engines)[:n_keep]
    return df[df["engine_id"].isin(keep)].copy()


def truncate_engine_prefix(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Keep only the first ``fraction`` of each engine's cycles.

    .. warning::
       This is a **flawed** scarcity simulation, kept so that
       ``notebooks/03`` can demonstrate the failure mode rather than hide it.
       The first cycles of an engine are its healthiest, so after clipping at
       ``RUL_CAP`` every retained label collapses onto the cap for any engine
       living longer than ``cap / (1 - fraction)`` cycles. The model is then
       trained on a near-constant target and evaluated on the full range.
       Use :func:`subsample_engines` for a sound experiment.
    """
    df = df.sort_values(["engine_id", "cycle"])
    position = df.groupby("engine_id").cumcount()
    n_keep = (df.groupby("engine_id")["cycle"].transform("size") * fraction).astype(int)
    return df[position < n_keep].reset_index(drop=True)
