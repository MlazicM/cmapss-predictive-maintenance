"""Tests for split-conformal interval calibration.

Pure NumPy, so none of this needs TensorFlow or the dataset. The coverage
claims are checked empirically against draws from a known distribution: a
conformal method whose guarantee does not show up in a simulation does not have
one.
"""

import numpy as np
import pytest

from src.calibration import (
    apply_cqr,
    apply_symmetric,
    calibrate_cqr,
    calibrate_symmetric,
    clip_interval,
    conformal_quantile,
)


@pytest.fixture
def gaussian():
    """Residuals from a model with unit spread and a correct mean."""
    rng = np.random.default_rng(0)
    n = 4000
    return {
        "y": rng.normal(size=n),
        "mean": np.zeros(n),
        "std": np.ones(n),
    }


# --- the corrected quantile -------------------------------------------------

def test_quantile_uses_the_finite_sample_correction():
    """ceil((n+1)*q)/n, not the plain q-th quantile.

    With n = 19 and q = 0.95 the correction lands exactly on the largest score;
    the uncorrected quantile would sit below it and under-cover.
    """
    scores = np.arange(19, dtype=float)
    assert conformal_quantile(scores, nominal=0.95) == 18.0
    assert np.quantile(scores, 0.95) < 18.0


def test_too_few_points_for_the_level_gives_an_infinite_interval():
    """18 points cannot support 95% coverage; say so rather than under-cover."""
    assert conformal_quantile(np.arange(18, dtype=float), nominal=0.95) == float("inf")


def test_empty_calibration_set_is_rejected():
    with pytest.raises(ValueError, match="at least one score"):
        conformal_quantile(np.array([]))


# --- symmetric calibration --------------------------------------------------

@pytest.mark.parametrize("mode", ["adaptive", "absolute"])
def test_calibrated_interval_hits_its_nominal_coverage(gaussian, mode):
    multiplier = calibrate_symmetric(
        gaussian["y"], gaussian["mean"], gaussian["std"], nominal=0.95, mode=mode
    )
    lower, upper = apply_symmetric(gaussian["mean"], gaussian["std"], multiplier, mode=mode)
    coverage = ((gaussian["y"] >= lower) & (gaussian["y"] <= upper)).mean()
    assert coverage == pytest.approx(0.95, abs=0.02)


def test_calibration_recovers_the_gaussian_multiplier(gaussian):
    """On genuinely normal residuals the answer should be ~1.96."""
    multiplier = calibrate_symmetric(
        gaussian["y"], gaussian["mean"], gaussian["std"], nominal=0.95
    )
    assert multiplier == pytest.approx(1.96, abs=0.15)


def test_calibration_widens_an_overconfident_interval():
    """The FD001 situation: the model's spread is a quarter of its real error."""
    rng = np.random.default_rng(1)
    y = rng.normal(scale=4.0, size=3000)
    mean, std = np.zeros(3000), np.ones(3000)

    multiplier = calibrate_symmetric(y, mean, std, nominal=0.95)
    assert multiplier > 1.96, "an overconfident model must be corrected upwards"

    lower, upper = apply_symmetric(mean, std, multiplier)
    assert ((y >= lower) & (y <= upper)).mean() == pytest.approx(0.95, abs=0.02)


def test_adaptive_mode_keeps_intervals_wide_where_the_model_is_unsure():
    """The point of the adaptive score: width must still track the model's std."""
    rng = np.random.default_rng(2)
    std = np.repeat([1.0, 5.0], 2000)
    y = rng.normal(scale=std)
    mean = np.zeros_like(std)

    multiplier = calibrate_symmetric(y, mean, std, nominal=0.95, mode="adaptive")
    lower, upper = apply_symmetric(mean, std, multiplier, mode="adaptive")
    width = upper - lower
    assert width[std == 5.0].mean() > 4 * width[std == 1.0].mean()


def test_absolute_mode_produces_one_constant_width():
    """The robustness trade: no dependence on a spread that may be degenerate."""
    rng = np.random.default_rng(3)
    std = np.repeat([1.0, 5.0], 500)
    y = rng.normal(scale=std)
    mean = np.zeros_like(std)

    multiplier = calibrate_symmetric(y, mean, std, nominal=0.9, mode="absolute")
    lower, upper = apply_symmetric(mean, std, multiplier, mode="absolute")
    assert np.allclose(upper - lower, (upper - lower)[0])


def test_a_collapsed_spread_does_not_divide_by_zero():
    """A deep ensemble whose members agree exactly has std = 0 on some points."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 100.0] * 20)
    mean = np.zeros_like(y)
    std = np.zeros_like(y)

    multiplier = calibrate_symmetric(y, mean, std, nominal=0.9, mode="adaptive")
    assert np.isfinite(multiplier)


def test_unknown_mode_is_rejected():
    y = np.zeros(10)
    with pytest.raises(ValueError, match="unknown mode"):
        calibrate_symmetric(y, y, np.ones(10), mode="sideways")
    with pytest.raises(ValueError, match="unknown mode"):
        apply_symmetric(y, np.ones(10), 1.0, mode="sideways")


# --- conformalised quantile regression --------------------------------------

def test_cqr_widens_a_band_that_was_too_narrow():
    rng = np.random.default_rng(4)
    y = rng.normal(scale=3.0, size=3000)
    lower, upper = np.full(3000, -1.0), np.full(3000, 1.0)

    offset = calibrate_cqr(y, lower, upper, nominal=0.95)
    assert offset > 0

    lower_c, upper_c = apply_cqr(lower, upper, offset)
    assert ((y >= lower_c) & (y <= upper_c)).mean() == pytest.approx(0.95, abs=0.02)


def test_cqr_shrinks_a_band_that_was_needlessly_wide():
    """Negative conformity scores are what make this possible; do not clip them."""
    rng = np.random.default_rng(5)
    y = rng.normal(scale=0.5, size=3000)
    lower, upper = np.full(3000, -50.0), np.full(3000, 50.0)

    offset = calibrate_cqr(y, lower, upper, nominal=0.95)
    assert offset < 0, "an over-wide band should be tightened, not left alone"

    lower_c, upper_c = apply_cqr(lower, upper, offset)
    assert (upper_c - lower_c)[0] < (upper - lower)[0]
    assert ((y >= lower_c) & (y <= upper_c)).mean() == pytest.approx(0.95, abs=0.02)


# --- clipping ---------------------------------------------------------------

def test_clipping_holds_the_interval_inside_the_rul_range():
    lower, upper = clip_interval(np.array([-30.0]), np.array([200.0]), low=0.0, high=125.0)
    assert lower[0] == 0.0
    assert upper[0] == 125.0


def test_clipping_a_saturated_prediction_keeps_both_bounds_at_the_cap():
    """Regression: clipping only the upper bound pushed it back above the cap.

    A prediction far above the cap has a *lower* bound above it too. Repairing
    the resulting inverted interval by widening the upper bound undid exactly
    the clipping that was asked for.
    """
    lower, upper = clip_interval(np.array([900.0]), np.array([1100.0]), low=0.0, high=125.0)
    assert upper[0] == 125.0
    assert lower[0] <= upper[0]


def test_clipping_never_inverts_an_interval():
    rng = np.random.default_rng(6)
    lower = rng.normal(scale=200.0, size=500)
    upper = lower + np.abs(rng.normal(scale=50.0, size=500))
    lower_c, upper_c = clip_interval(lower, upper, low=0.0, high=125.0)
    assert np.all(lower_c <= upper_c)


def test_clipping_without_an_upper_limit_leaves_the_top_alone():
    lower, upper = clip_interval(np.array([-5.0]), np.array([1e6]), low=0.0, high=None)
    assert lower[0] == 0.0
    assert upper[0] == 1e6
