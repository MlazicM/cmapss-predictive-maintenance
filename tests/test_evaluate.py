import numpy as np
import pytest

from src.evaluate import (
    interval_report,
    label_distribution,
    nasa_score,
    r2,
    regression_report,
    rmse,
)


def test_rmse_and_r2_match_known_values():
    y_true = np.array([0.0, 10.0, 20.0, 30.0])
    assert rmse(y_true, y_true) == 0.0
    assert r2(y_true, y_true) == pytest.approx(1.0)
    assert rmse(y_true, y_true + 2.0) == pytest.approx(2.0)


def test_r2_of_the_mean_predictor_is_zero():
    y_true = np.array([0.0, 10.0, 20.0, 30.0])
    assert r2(y_true, np.full_like(y_true, y_true.mean())) == pytest.approx(0.0)


def test_nasa_score_penalises_late_predictions_harder():
    """Overestimating remaining life means the engine fails unattended."""
    y_true = np.array([50.0])
    early = nasa_score(y_true, np.array([40.0]))   # warns 10 cycles too soon
    late = nasa_score(y_true, np.array([60.0]))    # 10 cycles too late

    assert late > early
    assert early == pytest.approx(np.exp(10 / 13) - 1)
    assert late == pytest.approx(np.exp(10 / 10) - 1)


def test_nasa_score_is_zero_for_a_perfect_prediction():
    y = np.array([10.0, 50.0, 125.0])
    assert nasa_score(y, y) == pytest.approx(0.0)


def test_regression_report_exposes_every_headline_metric():
    report = regression_report(np.array([10.0, 20.0]), np.array([12.0, 18.0]))
    assert set(report) == {"n", "rmse", "mae", "r2", "nasa_score"}
    assert report["n"] == 2


def test_label_distribution_exposes_a_collapsed_target():
    """The one-line check that would have caught the prefix-truncation bug."""
    collapsed = label_distribution(np.full(500, 125.0))
    assert collapsed["n_unique"] == 1
    assert collapsed["std"] == 0.0

    healthy = label_distribution(np.arange(0.0, 126.0))
    assert healthy["min"] == 0.0 and healthy["max"] == 125.0


# --- interval metrics -------------------------------------------------------

def test_interval_report_measures_coverage_against_width():
    """Coverage alone is gameable; the report has to carry the width beside it."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    report = interval_report(y_true, y_true - 5.0, y_true + 5.0)

    assert report["coverage"] == 1.0
    assert report["mean_width"] == pytest.approx(10.0)
    assert report["calibration_error"] == pytest.approx(0.05)


def test_a_vacuous_interval_is_not_rewarded():
    """0-125 covers everything and tells a planner nothing: the score must say so."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    vacuous = interval_report(y_true, np.zeros(4), np.full(4, 125.0))
    tight = interval_report(y_true, y_true - 6.0, y_true + 6.0)

    assert vacuous["coverage"] == tight["coverage"] == 1.0
    assert tight["interval_score"] < vacuous["interval_score"]


def test_interval_score_charges_for_missing_far():
    """Missing by 50 cycles must cost more than missing by 1."""
    y_true = np.array([100.0])
    near = interval_report(y_true, np.array([0.0]), np.array([99.0]))
    far = interval_report(y_true, np.array([0.0]), np.array([50.0]))

    assert far["interval_score"] > near["interval_score"]
    assert near["coverage"] == far["coverage"] == 0.0


def test_interval_report_adds_point_error_when_given_a_point_estimate():
    y_true = np.array([10.0, 20.0])
    without = interval_report(y_true, y_true - 1.0, y_true + 1.0)
    with_point = interval_report(y_true, y_true - 1.0, y_true + 1.0, point=y_true + 2.0)

    assert "rmse" not in without
    assert with_point["rmse"] == pytest.approx(2.0)
