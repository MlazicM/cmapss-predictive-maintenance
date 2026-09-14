"""Model definitions and persistence.

TensorFlow and XGBoost are imported lazily inside the functions that need
them: importing this module must stay cheap, and the data/eval utilities have
to remain usable in environments where neither is installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import MODELS_DIR, RANDOM_SEED


def set_global_seeds(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy and TensorFlow.

    Without this, reporting results to two decimals is meaningless -- rerunning
    the same notebook gives a different number.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def build_lstm(
    input_shape: tuple[int, int],
    units: int = 64,
    dropout: float = 0.2,
    dense_units: int = 32,
    learning_rate: float = 1e-3,
):
    """LSTM(units) -> Dropout -> Dense -> Dense(1).

    ``input_shape`` is passed in (derive it from ``X.shape[1:]``) instead of
    being hard-coded, so changing the sensor list or the window length cannot
    silently desynchronise the model from the data.
    """
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM

    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(units),
            Dropout(dropout),
            Dense(dense_units, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def make_early_stopping(patience: int = 5, monitor: str = "val_loss"):
    """A *fresh* EarlyStopping callback.

    Always build a new one per ``fit`` call rather than reusing a module-level
    instance across models -- the callback carries the best weights it has seen,
    and sharing it between unrelated trainings is a class of bug that is very
    hard to spot after the fact.
    """
    from tensorflow.keras.callbacks import EarlyStopping

    return EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True)


def pinball_loss(quantiles: list[float]):
    """Multi-quantile pinball (check) loss.

    For quantile q the loss is ``max(q*e, (q-1)*e)`` with ``e = y - y_hat``.
    Under-predicting is charged ``q`` and over-predicting ``1-q``, so minimising
    it drives the output to the q-th conditional quantile. Unlike MC dropout,
    which infers a spread from weight noise, this learns the interval edges
    directly from the data - which is what it takes for the stated confidence
    level to mean anything.
    """
    import tensorflow as tf

    quantiles_tensor = tf.constant(quantiles, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, (-1, 1)), tf.float32)
        error = y_true - y_pred
        return tf.reduce_mean(
            tf.maximum(quantiles_tensor * error, (quantiles_tensor - 1.0) * error)
        )

    return loss


def build_quantile_lstm(
    input_shape: tuple[int, int],
    quantiles: list[float] | None = None,
    units: int = 64,
    dropout: float = 0.2,
    dense_units: int = 32,
    learning_rate: float = 1e-3,
):
    """LSTM with one output per quantile, trained on the pinball loss.

    Returns the model; ``model.quantiles`` records the order of the outputs so
    a caller never has to guess which column is which.
    """
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM

    quantiles = quantiles or [0.025, 0.5, 0.975]
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(units),
            Dropout(dropout),
            Dense(dense_units, activation="relu"),
            Dense(len(quantiles)),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=pinball_loss(quantiles),
    )
    model.quantiles = quantiles
    return model


def quantile_predict(model, X, sort_outputs: bool = True):
    """Predict every quantile at once.

    Nothing in the loss forces the outputs to stay ordered, so on hard inputs
    the 2.5% head can cross above the 97.5% head. Sorting each row restores a
    valid interval; the crossing rate is worth reporting rather than hiding,
    since it measures how strained the fit is.
    """
    import numpy as np

    predictions = np.asarray(model.predict(X, verbose=0), dtype=float)
    crossings = float(np.mean(np.any(np.diff(predictions, axis=1) < 0, axis=1)))
    if sort_outputs:
        predictions = np.sort(predictions, axis=1)
    return predictions, crossings


def build_xgb(**kwargs):
    """Gradient-boosted tree baseline."""
    import xgboost as xgb

    params = dict(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    params.update(kwargs)
    return xgb.XGBRegressor(**params)


def freeze_recurrent_layers(model, frozen: bool) -> None:
    """Freeze or unfreeze every recurrent layer.

    Keras only picks up a ``trainable`` change on the next ``compile``, so the
    caller **must** follow this with :func:`compile_for_finetuning`. Flipping
    the flag without recompiling is a silent no-op -- which is exactly how a
    "frozen" and an "unfrozen" fine-tuning run end up being the same run.
    """
    from tensorflow.keras.layers import LSTM

    for layer in model.layers:
        if isinstance(layer, LSTM):
            layer.trainable = not frozen


def compile_for_finetuning(model, learning_rate: float = 1e-4) -> None:
    """Recompile with a lower learning rate before fine-tuning."""
    import tensorflow as tf

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")


def mc_dropout_predict(model, X: np.ndarray, n_samples: int = 50, batch_size: int = 256):
    """Predict with dropout left active to obtain an uncertainty estimate.

    Running the network ``n_samples`` times with dropout on approximates
    sampling from the posterior predictive distribution. The spread is what the
    API reports as confidence bounds: a point RUL estimate with no uncertainty
    is not actionable for maintenance planning.

    Returns ``(mean, std)`` over the sampled predictions.
    """
    samples = []
    for _ in range(n_samples):
        chunks = [
            model(X[i : i + batch_size], training=True).numpy().ravel()
            for i in range(0, len(X), batch_size)
        ]
        samples.append(np.concatenate(chunks))
    predictions = np.stack(samples)
    return predictions.mean(axis=0), predictions.std(axis=0)


# --- Persistence ------------------------------------------------------------

def save_artifacts(model, scaler, name: str, models_dir: Path = MODELS_DIR) -> dict[str, Path]:
    """Persist a Keras model and its scaler together.

    They are a single unit: a model served with a different scaler than it was
    trained with produces confident nonsense. Nothing downstream (the API in
    particular) can exist until both are on disk.
    """
    import joblib

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{name}.keras"
    scaler_path = models_dir / f"{name}_scaler.joblib"
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    return {"model": model_path, "scaler": scaler_path}


def save_calibration(
    name: str, calibration: dict, models_dir: Path = MODELS_DIR
) -> Path:
    """Write the conformal calibration of a model next to its weights.

    JSON rather than joblib: this is four numbers and a method name, and it
    should stay readable by a human deciding whether to trust the interval a
    service just returned.
    """
    import json

    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}_calibration.json"
    path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    return path


def load_calibration(name: str, models_dir: Path = MODELS_DIR) -> dict | None:
    """Load a calibration written by :func:`save_calibration`, or ``None``.

    Absence is a normal state, not an error: a model trained before this
    existed still serves, it just serves an uncalibrated interval. The caller
    is expected to say which of the two it is rather than pass off one as the
    other.
    """
    import json

    path = models_dir / f"{name}_calibration.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_artifacts(name: str, models_dir: Path = MODELS_DIR, compile_model: bool = False):
    """Load a model and its scaler saved by :func:`save_artifacts`.

    Missing files are reported as :class:`FileNotFoundError` before any
    framework is imported. Keras raises ``ValueError`` for an absent ``.keras``
    path, which reads as a corrupt-file error rather than a missing one and is
    easy to forget when writing the caller's ``except`` clause.

    ``compile_model`` defaults to False because serving only ever runs a forward
    pass. It also removes a real failure mode: a quantile model is compiled with
    :func:`pinball_loss`, which is a closure and cannot be deserialised without
    being handed back in ``custom_objects``. Loading uncompiled sidesteps that
    entirely -- there is no optimiser state to restore for inference. Pass True
    to resume training.
    """
    model_path = models_dir / f"{name}.keras"
    scaler_path = models_dir / f"{name}_scaler.joblib"
    missing = [str(p) for p in (model_path, scaler_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"no trained artifacts for '{name}': missing {', '.join(missing)}. "
            "Run scripts/run_experiments.py --stage baselines to create them."
        )

    import joblib
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=compile_model)
    scaler = joblib.load(scaler_path)
    return model, scaler
