"""
Inference from raw hand landmarks, without MediaPipe.

The browser now runs HandLandmarker and posts the 21 raw landmarks, so the
server only has to normalize, scale and classify. That keeps MediaPipe, OpenCV,
jax and matplotlib out of the deployed bundle.

Normalization is NOT reimplemented here: it calls landmarks.normalize_landmarks,
the same function HandTracker.normalize_landmarks delegates to, so the desktop
and web paths produce identical features.
"""

import os
import sys
import threading
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from landmarks import normalize_landmarks
from model import SignLanguageModel


@dataclass
class InferenceResult:
    """Outcome of running one frame or one landmark set through the pipeline."""

    hand_detected: bool
    raw_sign: Optional[str] = None       # this frame's prediction, unsmoothed
    confidence: float = 0.0
    sign: Optional[str] = None           # smoothed/accepted sign (raw if no smoother)
    emitted: bool = False                # True only on the frame a sign is accepted
    crop_offset: tuple = (0, 0, 0)       # (y, x, side) of the square crop
    landmarks: Optional[List] = None     # raw landmarks, for drawing the overlay

    def as_dict(self):
        return {
            "sign": self.sign,
            "confidence": round(float(self.confidence), 4),
            "hand_detected": self.hand_detected,
            "raw_sign": self.raw_sign,
            "emitted": self.emitted,
        }


class LandmarkInference:
    """Classifies raw hand landmarks. Loads the model once."""

    def __init__(self, model=None):
        self._lock = threading.Lock()
        if model is not None:
            self.model = model
        else:
            self.model = SignLanguageModel()
            self.model.load_model()
            self.model.load_metadata()

    @property
    def classes(self):
        """Sorted list of letters the model can predict."""
        return [self.model.label_decoder[i]
                for i in sorted(self.model.label_decoder)]

    def predict(self, raw_landmarks, smoother=None):
        """
        Classify one hand.

        Args:
            raw_landmarks: flat sequence of 63 values (21 x,y,z triples), or
                None when the browser detected no hand
            smoother: optional PredictionSmoother holding per-viewer state

        Returns:
            InferenceResult
        """
        normalized = normalize_landmarks([raw_landmarks]) if raw_landmarks is not None else None

        with self._lock:
            raw_sign, confidence = None, 0.0
            if normalized:
                raw_sign, confidence = self.model.predict(normalized)

            if smoother is not None:
                accepted, emitted = smoother.update(raw_sign, confidence)
            else:
                accepted, emitted = raw_sign, False

        return InferenceResult(
            hand_detected=normalized is not None,
            raw_sign=raw_sign,
            confidence=confidence,
            sign=accepted,
            emitted=emitted,
            landmarks=[raw_landmarks] if raw_landmarks is not None else None,
        )
