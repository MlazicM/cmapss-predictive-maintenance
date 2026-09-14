import numpy as np
import pandas as pd

from src.data import (
    add_rul,
    split_by_engine,
    subsample_engines,
    truncate_engine_prefix,
)


def test_rul_counts_down_to_zero_at_failure(synthetic_run_to_failure):
    df = add_rul(synthetic_run_to_failure(), cap=None)
    last = df.groupby("engine_id")["RUL"].min()
    assert (last == 0).all()
    first = df.groupby("engine_id").first()["RUL"]
    assert first.tolist() == [39, 59, 99]


def test_rul_cap_is_applied(synthetic_run_to_failure):
    df = add_rul(synthetic_run_to_failure(), cap=50)
    assert df["RUL"].max() == 50


def test_split_by_engine_shares_no_engine(synthetic_run_to_failure):
    df = add_rul(synthetic_run_to_failure(lifetimes=[30] * 10))
    train, val = split_by_engine(df, val_fraction=0.2)
    assert set(train["engine_id"]) & set(val["engine_id"]) == set()
    assert val["engine_id"].nunique() == 2
    assert len(train) + len(val) == len(df)


def test_split_by_engine_is_deterministic(synthetic_run_to_failure):
    df = add_rul(synthetic_run_to_failure(lifetimes=[30] * 10))
    first = split_by_engine(df, seed=7)[1]["engine_id"].unique()
    second = split_by_engine(df, seed=7)[1]["engine_id"].unique()
    assert np.array_equal(first, second)


def test_subsample_engines_keeps_full_lifetimes(synthetic_run_to_failure):
    """The sound scarcity simulation: fewer engines, full RUL range preserved."""
    df = add_rul(synthetic_run_to_failure(lifetimes=[200] * 10), cap=125)
    scarce = subsample_engines(df, fraction=0.3)

    assert scarce["engine_id"].nunique() == 3
    assert scarce["RUL"].min() == 0          # failure is still observed
    assert scarce["RUL"].max() == 125        # and so is the healthy regime
    assert scarce["RUL"].nunique() > 100


def test_truncate_engine_prefix_collapses_labels(synthetic_run_to_failure):
    """Regression guard documenting *why* the prefix approach was abandoned.

    Every engine living longer than cap / (1 - fraction) contributes nothing but
    the clipped cap, so the model is fitted on a constant target.
    """
    df = add_rul(synthetic_run_to_failure(lifetimes=[200] * 10), cap=125)
    truncated = truncate_engine_prefix(df, fraction=0.3)

    assert truncated["RUL"].nunique() == 1
    assert truncated["RUL"].unique()[0] == 125
    assert truncated["RUL"].min() > 100      # failure is never seen at all


def test_truncate_keeps_expected_row_count(synthetic_run_to_failure):
    df = add_rul(synthetic_run_to_failure(lifetimes=(40, 60, 100)))
    truncated = truncate_engine_prefix(df, fraction=0.3)
    assert truncated.groupby("engine_id").size().tolist() == [12, 18, 30]
