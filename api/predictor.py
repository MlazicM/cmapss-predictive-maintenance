"""Loading the trained artifacts and turning raw readings into a prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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
    ):
        self.model = model
        self.scaler = scaler
        self.features = features or INFORMATIVE_SENSORS
        self.sequence_length = sequence_length
        self.rul_cap = rul_cap

    @classmethod
    def from_artifacts(cls, name: str = "lstm_fd001", models_dir: Path = MODELS_DIR):
        """Load artifacts written by :func:`src.models.save_artifacts`."""
        from src.models import load_artifacts

        model, scaler = load_artifacts(name, models_dir=models_dir)
        return cls(model, scaler)

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

    def predict(self, readings: list[dict], n_samples: int = 50) -> dict:
        """Point estimate plus a 95% MC-dropout interval."""
        window, padded = self.build_window(readings)

        samples = np.stack(
            [np.asarray(self.model(window, training=True)).ravel() for _ in range(n_samples)]
        )
        mean = float(samples.mean())
        std = float(samples.std())

        return {
            "predicted_rul": min(mean, float(self.rul_cap)),
            "std": std,
            "lower_95": max(0.0, mean - 1.96 * std),
            "upper_95": min(mean + 1.96 * std, float(self.rul_cap)),
            "cycles_supplied": len(readings),
            "padded": padded,
        }
