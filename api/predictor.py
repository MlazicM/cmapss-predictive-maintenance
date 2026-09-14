"""Loading the trained artifacts and turning raw readings into a prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration import apply_symmetric, clip_interval
from src.config import INFORMATIVE_SENSORS, MODELS_DIR, RUL_CAP, SEQUENCE_LENGTH


class RULPredictor:
    """Wraps a trained model together with the scaler it was trained with.

    The two are inseparable: serving a model with different scaling statistics
    than it saw in training produces confident nonsense, with nothing in the
    output to signal that anything is wrong.

    The model and scaler are injected rather than loaded in ``__init__`` so the
    request-handling logic can be tested without TensorFlow.
    """

    def __init__(
        self,
        model,
        scaler,
        features: list[str] | None = None,
        sequence_length: int = SEQUENCE_LENGTH,
        rul_cap: int = RUL_CAP,
        calibration: dict | None = None,
    ):
        self.model = model
        self.scaler = scaler
        self.features = features or INFORMATIVE_SENSORS
        self.sequence_length = sequence_length
        self.rul_cap = rul_cap
        self.calibration = calibration

    @classmethod
    def from_artifacts(cls, name: str = "lstm_fd001", models_dir: Path = MODELS_DIR):
        """Load artifacts written by :func:`src.models.save_artifacts`.

        A conformal calibration is picked up when one was saved next to the
        model, and its absence is not an error -- it downgrades the interval to
        the uncalibrated Gaussian one and says so in every response.
        """
        from src.models import load_artifacts, load_calibration

        model, scaler = load_artifacts(name, models_dir=models_dir)
        return cls(model, scaler, calibration=load_calibration(name, models_dir=models_dir))

    @property
    def method(self) -> str:
        """Which interval strategy this predictor runs."""
        if self.calibration is None:
            return "mc_dropout"
        return self.calibration.get("method", "mc_dropout")

    @property
    def interval_method(self) -> str:
        """How the returned bounds were produced, in words a caller can act on."""
        if self.calibration is None:
            return "mc-dropout spread, Gaussian 1.96 sigma (UNCALIBRATED)"
        if self.method.startswith("quantile"):
            return "quantile regression, conformalised (CQR)"
        mode = self.calibration.get("mode", "adaptive")
        return f"mc-dropout spread, split-conformal ({mode})"

    @property
    def nominal_coverage(self) -> float:
        if self.calibration is None:
            return 0.95
        return float(self.calibration.get("nominal", 0.95))

    def build_window(self, readings: list[dict]) -> tuple[np.ndarray, bool]:
        """Scale the readings and shape them into one model input window.

        Returns the window and whether it had to be left-padded. Padding is
        reported rather than hidden: a prediction from six cycles of history is
        not the same product as one from thirty.
        """
        frame = pd.DataFrame([reading["sensors"] for reading in readings])
        missing = [feature for feature in self.features if feature not in frame.columns]
        if missing:
            raise ValueError(f"missing required sensors: {missing}")

        values = self.scaler.transform(frame[self.features]).astype(np.float32)

        padded = len(values) < self.sequence_length
        if padded:
            pad = np.repeat(values[:1], self.sequence_length - len(values), axis=0)
            values = np.concatenate([pad, values])

        return values[-self.sequence_length :][np.newaxis, ...], padded

    def _predict_quantile(self, window) -> tuple[float, float, float, float | None]:
        """One forward pass through a model with a quantile head.

        Cheaper than MC dropout by two orders of magnitude -- one pass against a
        hundred -- and it was also the better interval in the Phase 6
        comparison, so the service is not trading accuracy for latency here.

        The heads are sorted before use: nothing in the pinball loss forces the
        2.5% output to stay below the 97.5% one, and an inverted interval is
        worse than a wide one. There is no meaningful ``std`` for this method,
        so it is reported as null rather than back-solved from the width.
        """
        outputs = np.sort(np.asarray(self.model(window), dtype=float).ravel())
        offset = float(self.calibration.get("offset", 0.0))
        lower, median, upper = outputs[0], outputs[len(outputs) // 2], outputs[-1]
        return median, lower - offset, upper + offset, None

    def _predict_mc_dropout(self, window, n_samples: int) -> tuple[float, float, float, float]:
        """Repeated passes with dropout active, scaled by the conformal multiplier."""
        samples = np.stack(
            [np.asarray(self.model(window, training=True)).ravel() for _ in range(n_samples)]
        )
        mean, std = float(samples.mean()), float(samples.std())

        if self.calibration is None:
            mode, multiplier = "adaptive", 1.96
        else:
            mode = self.calibration.get("mode", "adaptive")
            multiplier = float(self.calibration["multiplier"])

        lower, upper = apply_symmetric(mean, std, multiplier, mode=mode)
        return mean, float(lower), float(upper), std

    def predict(self, readings: list[dict], n_samples: int = 50) -> dict:
        """Point estimate plus an interval at the calibrated confidence level.

        Without a calibration the bounds are the raw MC-dropout spread at 1.96
        sigma, which on FD001 covers 63% of engines at a nominal 95%. That is
        why the method is reported in the response rather than left implicit:
        the caller has to be able to tell a guaranteed band from a suggestive
        one, and both come back through the same two fields.
        """
        window, padded = self.build_window(readings)

        if self.method.startswith("quantile"):
            point, lower, upper, std = self._predict_quantile(window)
        else:
            point, lower, upper, std = self._predict_mc_dropout(window, n_samples)

        lower, upper = clip_interval(lower, upper, low=0.0, high=float(self.rul_cap))

        return {
            # Clamped at both ends, not just the cap: a model extrapolating past
            # end-of-life can return a negative RUL, and "-3 cycles remaining" is
            # not a maintenance input.
            "predicted_rul": float(np.clip(point, 0.0, float(self.rul_cap))),
            "std": std,
            "lower_95": float(lower),
            "upper_95": float(upper),
            "cycles_supplied": len(readings),
            "padded": padded,
            "interval_method": self.interval_method,
            "nominal_coverage": self.nominal_coverage,
        }
