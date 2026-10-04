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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from hand_tracker import HandTracker, center_square_crop
from model import SignLanguageModel


from landmarks import landmarks_to_points                      # noqa: F401
from landmark_inference import InferenceResult, LandmarkInference  # noqa: F401


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
