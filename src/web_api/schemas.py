"""Response models for the web API."""

import math
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_loaded: bool
    classes: int
    version: str


class PredictResponse(BaseModel):
    sign: Optional[str] = Field(None, description="Smoothed/accepted sign")
    confidence: float = Field(..., ge=0.0, le=1.0)
    hand_detected: bool
    raw_sign: Optional[str] = Field(None, description="This frame's unsmoothed prediction")
    emitted: bool = Field(..., description="True only on the frame a sign is accepted")
    landmarks: Optional[List[List[float]]] = Field(
        None, description="21 (x,y) pairs in [0,1] within the square crop")


class ModelInfoResponse(BaseModel):
    classes: List[str]
    num_classes: int
    offline_test_accuracy: float
    offline_macro_f1: float
    accuracy_note: str
    known_limitations: List[str]
    smoothing_window: int
    confidence_threshold: float
    # MediaPipe settings the browser must mirror so the landmarks it produces
    # match what the training features were extracted with.
    min_hand_detection_confidence: float
    min_tracking_confidence: float
    num_hands: int


# ---------- landmark based prediction ----------

# MediaPipe returns x and y normalized to the input image, nominally [0, 1],
# but it can extrapolate slightly past the edges for a partly visible hand.
# z is a relative depth around the wrist and is unbounded in principle.
# These bounds reject nonsense without rejecting legitimate detections.
_XY_MIN, _XY_MAX = -2.0, 3.0
_Z_MIN, _Z_MAX = -10.0, 10.0


class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float = 0.0

    @field_validator("x", "y", "z")
    @classmethod
    def finite(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v):
            raise ValueError("landmark coordinates must be finite numbers")
        return v

    @field_validator("x", "y")
    @classmethod
    def xy_in_range(cls, v: float) -> float:
        if not (_XY_MIN <= v <= _XY_MAX):
            raise ValueError(f"x and y must lie within [{_XY_MIN}, {_XY_MAX}]")
        return v

    @field_validator("z")
    @classmethod
    def z_in_range(cls, v: float) -> float:
        if not (_Z_MIN <= v <= _Z_MAX):
            raise ValueError(f"z must lie within [{_Z_MIN}, {_Z_MAX}]")
        return v


class PredictLandmarksRequest(BaseModel):
    landmarks: Optional[List[LandmarkPoint]] = Field(
        None,
        description="Exactly 21 MediaPipe hand landmarks, or null when no hand "
                    "was detected in this frame.",
    )
    session_id: str = Field("", description="Per browser smoothing session")

    @field_validator("landmarks")
    @classmethod
    def exactly_21(cls, v):
        if v is not None and len(v) != 21:
            raise ValueError(f"expected exactly 21 landmarks, received {len(v)}")
        return v
