"""Split-conformal calibration of prediction intervals.

MC dropout and deep ensembles produce a *spread*, not a probability: nothing
forces a nominal 95% band to contain 95% of the observations, and on this
project's FD001 model it contains 62%. Split conformal prediction fixes that
without retraining. It holds out a calibration set the model never saw, measures
how wrong the model's own uncertainty estimate actually was on it, and rescales
the interval by that empirical quantile.

The guarantee is distribution-free and finite-sample: with ``n`` exchangeable
calibration points the calibrated interval covers a new exchangeable point with
probability at least ``nominal``. Exchangeability is the whole assumption, so
the calibration points have to look like the points served at inference time --
see ``scripts/run_experiments.py`` for how that is arranged here.

Everything in this module is plain NumPy, so it is testable without TensorFlow.
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, nominal: float = 0.95) -> float:
    """The finite-sample corrected ``nominal`` quantile of conformity scores.

    The correction is the ``ceil((n + 1) * nominal) / n`` empirical quantile
    rather than the plain ``nominal`` one. Taking the uncorrected quantile
    under-covers by roughly ``1 / n``, which matters exactly when the
    calibration set is small -- which is when anyone reaches for this.

    With too few calibration points to reach the requested level at all
    (``ceil((n + 1) * nominal) > n``), no finite interval can carry the
    guarantee and the score is infinite. Returning ``inf`` rather than the
    maximum score keeps that honest: the caller sees an unbounded interval
    instead of a bound that quietly does not hold.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    n = scores.size
    if n == 0:
        raise ValueError("conformal calibration needs at least one score")

    rank = int(np.ceil((n + 1) * nominal))
    if rank > n:
        return float("inf")
    return float(np.sort(scores)[rank - 1])


def calibrate_symmetric(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    nominal: float = 0.95,
    mode: str = "adaptive",
    floor: float = 1e-6,
) -> float:
    """Conformity score for a symmetric ``mean +/- k * scale`` interval.

    Two ways of scoring, because they fail in opposite directions:

    ``adaptive``
        score ``|y - mean| / std``. The returned multiplier replaces the 1.96 a
        Gaussian assumption would use, and the interval stays *narrow where the
        model is confident* -- it preserves the one part of MC dropout that was
        already working, its ranking of which engines are uncertain. The
        weakness is the denominator: a model whose spread collapses towards
        zero on some calibration points produces enormous scores, and the
        multiplier needed to cover 95% of them inflates every interval. A deep
        ensemble of five members does exactly this.

    ``absolute``
        score ``|y - mean|``, so the multiplier is a width in cycles and every
        interval is the same width. It throws away the model's own uncertainty
        ranking, and in exchange it cannot be destabilised by it.

    Neither is universally better, which is why both are measured. Compare them
    on the interval score from :func:`src.evaluate.interval_report` rather than
    on coverage, which both hit by construction.
    """
    mean = np.asarray(mean, float)
    residual = np.abs(np.asarray(y_true, float) - mean)
    if mode == "absolute":
        return conformal_quantile(residual, nominal)
    if mode != "adaptive":
        raise ValueError(f"unknown mode {mode!r}; expected 'adaptive' or 'absolute'")
    std = np.maximum(np.asarray(std, float), floor)
    return conformal_quantile(residual / std, nominal)


def apply_symmetric(
    mean: np.ndarray,
    std: np.ndarray | None,
    multiplier: float,
    mode: str = "adaptive",
    floor: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a mean (and, in adaptive mode, a std) into a calibrated interval.

    ``mode`` must match the one :func:`calibrate_symmetric` produced the
    multiplier with -- an adaptive multiplier applied as an absolute width, or
    the reverse, is off by the scale of the model's spread.
    """
    mean = np.asarray(mean, float)
    if mode == "absolute":
        half_width = float(multiplier)
    elif mode == "adaptive":
        half_width = multiplier * np.maximum(np.asarray(std, float), floor)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'adaptive' or 'absolute'")
    return mean - half_width, mean + half_width


def calibrate_cqr(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, nominal: float = 0.95
) -> float:
    """Conformity offset for conformalised quantile regression (CQR).

    The score is ``max(lower - y, y - upper)``: how far outside the predicted
    band the observation fell, and *negative* when it fell comfortably inside.
    Keeping the negative scores is what lets CQR shrink an over-wide band as
    well as widen an over-narrow one, so a quantile model that is already close
    to calibrated is not punished with a needlessly vague interval.
    """
    y_true = np.asarray(y_true, float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    return conformal_quantile(np.maximum(lower - y_true, y_true - upper), nominal)


def apply_cqr(
    lower: np.ndarray, upper: np.ndarray, offset: float
) -> tuple[np.ndarray, np.ndarray]:
    """Widen (or shrink) a quantile band by the conformity offset."""
    return np.asarray(lower, float) - offset, np.asarray(upper, float) + offset


def clip_interval(
    lower: np.ndarray, upper: np.ndarray, low: float = 0.0, high: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Clip an interval to the physically possible RUL range.

    Remaining life is never negative and, under a piecewise-linear label, never
    above the cap. Clipping to that range can only *improve* coverage, since it
    never moves a bound past the truth, and it keeps the API from reporting a
    negative lower bound that a planner would rightly not trust.

    **Both** bounds are clipped, not just the one that looks out of range. A
    prediction saturated at the cap has a lower bound above the cap too, and
    clipping only the upper one produces an inverted interval -- or, if that is
    then repaired by widening, an upper bound back above the cap that the
    clipping was there to prevent. Clipping is monotone, so applying it to both
    ends preserves their order.
    """
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    lower = np.clip(lower, low, high)
    upper = np.clip(upper, low, high)
    return lower, upper
