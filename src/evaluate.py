"""Metrics and reporting for RUL regression.

Metrics are implemented in NumPy rather than imported from scikit-learn: they
are four lines each, and it removes a hard floor on the scikit-learn version
(``root_mean_squared_error`` only exists from 1.4).
"""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The official C-MAPSS asymmetric score (lower is better).

    ``d = pred - true``. Overestimating remaining life (``d > 0``) means the
    engine fails before the scheduled maintenance, so it is penalised with a
    time constant of 10 against 13 for an early warning. RMSE treats the two
    as equal; in maintenance planning they are not, which is why the benchmark
    reports this number alongside it.
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    penalty = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(penalty))


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """All headline metrics for one model in a single dict."""
    return {
        "n": int(len(y_true)),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
    }


def format_report(name: str, report: dict[str, float]) -> str:
    return (
        f"{name:<34} n={report['n']:>6}  RMSE={report['rmse']:6.2f}  "
        f"MAE={report['mae']:6.2f}  R2={report['r2']:5.2f}  "
        f"NASA={report['nasa_score']:10.1f}"
    )


def interval_report(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    point: np.ndarray | None = None,
    nominal: float = 0.95,
) -> dict[str, float]:
    """Score a set of prediction intervals.

    Coverage alone is not enough: an interval spanning 0-125 covers everything
    and tells a planner nothing. Report it against mean width, and against the
    interval score, which adds a penalty proportional to how far a missed
    observation fell outside - the standard proper scoring rule for intervals,
    so a model cannot win by being vague.
    """
    y_true = np.asarray(y_true, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    alpha = 1.0 - nominal

    covered = (y_true >= lower) & (y_true <= upper)
    width = upper - lower
    penalty = (2 / alpha) * (
        np.maximum(lower - y_true, 0.0) + np.maximum(y_true - upper, 0.0)
    )

    report = {
        "n": int(y_true.size),
        "nominal": nominal,
        "coverage": float(covered.mean()),
        "mean_width": float(width.mean()),
        "median_width": float(np.median(width)),
        "interval_score": float((width + penalty).mean()),
        "calibration_error": float(abs(covered.mean() - nominal)),
    }
    if point is not None:
        report["rmse"] = rmse(y_true, point)
    return report


def format_interval_report(name: str, report: dict[str, float]) -> str:
    return (
        f"{name:<30} coverage={report['coverage']:6.1%} "
        f"(nominal {report['nominal']:.0%})  width={report['mean_width']:6.1f}  "
        f"interval_score={report['interval_score']:7.1f}"
    )


def label_distribution(y: np.ndarray) -> dict[str, float]:
    """Summarise a target vector.

    Call this after every transformation of the training set. A collapsed or
    truncated target range is invisible in the loss curve but silently caps
    what the model can ever learn -- print it and the problem is obvious.
    """
    y = np.asarray(y, float)
    return {
        "n": int(y.size),
        "min": float(y.min()) if y.size else float("nan"),
        "max": float(y.max()) if y.size else float("nan"),
        "mean": float(y.mean()) if y.size else float("nan"),
        "std": float(y.std()) if y.size else float("nan"),
        "n_unique": int(np.unique(y).size),
    }
