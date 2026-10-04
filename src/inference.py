"""
Shared inference pipeline.

Single source of truth for the production path, so the desktop application
and the web API cannot drift apart:

    frame -> center_square_crop -> HandLandmarker -> normalize_landmarks
          -> scaler -> Random Forest -> (optional) PredictionSmoother

The expensive objects (HandTracker, model, scaler) are loaded once and
guarded by a lock, because MediaPipe's HandLandmarker is not thread-safe and
the web API serves concurrent requests.

PredictionSmoother is deliberately NOT owned here: it is per-viewer state.
The desktop app keeps one; the web API keeps one per browser session.
"""

import os
import sys
import threading
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from hand_tracker import HandTracker, center_square_crop
from model import SignLanguageModel


def landmarks_to_points(landmarks):
    """
    Convert raw landmark vectors into (x, y) pairs in [0,1] within the square
    crop, for drawing in a browser. Takes landmarks that were already
    extracted, so no second MediaPipe pass is needed.

    Returns a list of hands, each a list of 21 (x, y) pairs.
    """
    if not landmarks:
        return []
    return [[(float(h[i]), float(h[i + 1])) for i in range(0, len(h), 3)]
            for h in landmarks]


@dataclass
class InferenceResult:
    """Outcome of running one frame through the pipeline."""

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


class SignLanguageInference:
    """Loads the trained model once and runs frames through the pipeline."""

    def __init__(self, tracker=None, model=None):
        """
        Args:
            tracker: optional pre-built HandTracker (mainly for tests)
            model: optional pre-loaded SignLanguageModel (mainly for tests)
        """
        self._lock = threading.Lock()

        if tracker is not None:
            self.tracker = tracker
        else:
            self.tracker = HandTracker()

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

    def process_frame(self, frame, smoother=None):
        """
        Run one frame through the full production pipeline.

        Args:
            frame: BGR frame of any aspect ratio
            smoother: optional PredictionSmoother holding per-viewer state

        Returns:
            InferenceResult
        """
        frame_square, crop_y, crop_x, side = center_square_crop(frame)

        with self._lock:
            landmarks = self.tracker.extract_landmarks(frame_square)
            normalized = self.tracker.normalize_landmarks(landmarks)

            raw_sign, confidence = None, 0.0
            if normalized:
                raw_sign, confidence = self.model.predict(normalized)

            if smoother is not None:
                accepted, emitted = smoother.update(raw_sign, confidence)
            else:
                accepted, emitted = raw_sign, False

        return InferenceResult(
            hand_detected=bool(normalized),
            raw_sign=raw_sign,
            confidence=confidence,
            sign=accepted,
            emitted=emitted,
            crop_offset=(crop_y, crop_x, side),
            landmarks=landmarks,
        )

    def close(self):
        self.tracker.close()
