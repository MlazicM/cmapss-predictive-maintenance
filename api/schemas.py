"""Request and response models for the inference API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from src.config import INFORMATIVE_SENSORS, RUL_CAP, SEQUENCE_LENGTH


class SensorReading(BaseModel):
    """One cycle of readings for a single engine."""

    cycle: int = Field(..., ge=1, description="Operating cycle number")
    sensors: dict[str, float] = Field(
        ..., description=f"Readings keyed by sensor name; must contain {INFORMATIVE_SENSORS}"
    )

    @field_validator("sensors")
    @classmethod
    def require_informative_sensors(cls, value: dict[str, float]) -> dict[str, float]:
        missing = [s for s in INFORMATIVE_SENSORS if s not in value]
        if missing:
            raise ValueError(f"missing required sensors: {missing}")
        return value


class PredictionRequest(BaseModel):
    """A window of consecutive cycles for one engine."""

    engine_id: int | str = Field(..., description="Identifier echoed back in the response")
    readings: list[SensorReading] = Field(..., min_length=1)

    @field_validator("readings")
    @classmethod
    def require_strictly_increasing_cycles(cls, value: list[SensorReading]) -> list[SensorReading]:
        cycles = [reading.cycle for reading in value]
        if cycles != sorted(cycles) or len(set(cycles)) != len(cycles):
            raise ValueError("readings must be ordered by strictly increasing cycle")
        return value


class PredictionResponse(BaseModel):
    """A point estimate with an uncertainty band.

    A bare number is not actionable for maintenance planning: RUL 40 +/- 5 and
    RUL 40 +/- 35 justify very different decisions.
    """

    engine_id: int | str
    predicted_rul: float = Field(..., description="Cycles of remaining useful life")
    lower_95: float
    upper_95: float
    std: float
    cycles_supplied: int
    padded: bool = Field(
        ...,
        description=(
            f"True when fewer than {SEQUENCE_LENGTH} cycles were supplied and the "
            "window was left-padded; treat such predictions as provisional"
        ),
    )
    rul_cap: int = Field(
        default=RUL_CAP,
        description="Predictions saturate at this value: the model cannot distinguish healthy engines",
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    sequence_length: int
    expected_sensors: list[str]
