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
state: dict[str, RULPredictor | None] = {"predictor": None}

MODEL_NAME = "lstm_fd001"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup rather than per request."""
    try:
        state["predictor"] = RULPredictor.from_artifacts(MODEL_NAME)
        logger.info("loaded artifacts for %s", MODEL_NAME)
    except (FileNotFoundError, OSError, ImportError) as exc:
        # Train and save a model first: notebooks/02_baseline_models.ipynb.
        logger.warning("no model artifacts available (%s); /predict will return 503", exc)
    yield
    state["predictor"] = None


app = FastAPI(
    title="CMAPSS RUL Prediction",
    description="Remaining useful life estimation for jet engines, with uncertainty bounds.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=state["predictor"] is not None,
        sequence_length=SEQUENCE_LENGTH,
        expected_sensors=INFORMATIVE_SENSORS,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    predictor = state["predictor"]
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained model available. Run notebooks/02_baseline_models.ipynb "
                "to train and persist one."
            ),
        )

    try:
        result = predictor.predict([reading.model_dump() for reading in request.readings])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictionResponse(engine_id=request.engine_id, **result)
