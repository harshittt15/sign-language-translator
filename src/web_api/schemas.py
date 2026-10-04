"""Response models for the web API."""

from typing import List, Optional

from pydantic import BaseModel, Field


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
