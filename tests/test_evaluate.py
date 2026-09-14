import numpy as np
import pytest

from src.evaluate import label_distribution, nasa_score, r2, regression_report, rmse


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
