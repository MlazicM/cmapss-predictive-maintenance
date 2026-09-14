"""FastAPI service exposing RUL predictions.

Run with:  uvicorn api.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.predictor import RULPredictor
from api.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.config import INFORMATIVE_SENSORS, SEQUENCE_LENGTH

logger = logging.getLogger(__name__)

# Populated at startup; stays None when no artifacts exist so that /health can
# report the situation instead of the process refusing to boot.
state: dict[str, RULPredictor | None] = {"predictor": None, "model_name": None}

# Preference order, best first, and it is the order Phase 6 measured rather than
# a guess: the quantile model had the best interval score and needs one forward
# pass instead of a hundred; conformalised MC dropout is calibrated but wider;
# the plain baseline has only the raw MC-dropout spread, which covers 63% of
# engines at a nominal 95%. The last is a fallback, not an equivalent, so
# /health reports which one is actually loaded.
MODEL_NAMES = ("lstm_fd001_quantile", "lstm_fd001_conformal", "lstm_fd001")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the best available model once at startup rather than per request."""
    for name in MODEL_NAMES:
        try:
            predictor = RULPredictor.from_artifacts(name)
        except Exception as exc:  # noqa: BLE001
            # Deliberately broad: every loading failure must degrade to a 503
            # rather than take the process down. A narrow tuple missed Keras's
            # ValueError for an absent .keras path and crashed startup instead.
            logger.warning("could not load artifacts for %s (%s)", name, exc)
            continue

        state["predictor"] = predictor
        state["model_name"] = name
        logger.info("loaded %s (%s)", name, predictor.interval_method)
        break
    else:
        logger.warning("no model artifacts available; /predict will return 503")

    yield
    state["predictor"] = None
    state["model_name"] = None


app = FastAPI(
    title="CMAPSS RUL Prediction",
    description="Remaining useful life estimation for jet engines, with uncertainty bounds.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    predictor = state["predictor"]
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        sequence_length=SEQUENCE_LENGTH,
        expected_sensors=INFORMATIVE_SENSORS,
        # Both derived from the predictor, so a name can never be reported for a
        # model that is not actually loaded.
        model_name=state["model_name"] if predictor is not None else None,
        interval_calibrated=predictor is not None and predictor.calibration is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    predictor = state["predictor"]
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained model available. Run "
                "'python scripts/run_experiments.py --stage calibration' to train and "
                "persist a calibrated one, or --stage baselines for an uncalibrated one."
            ),
        )

    try:
        result = predictor.predict([reading.model_dump() for reading in request.readings])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictionResponse(engine_id=request.engine_id, **result)
