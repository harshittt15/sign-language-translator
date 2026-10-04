"""
Temporal smoothing for real-time sign predictions.

Static ASL alphabet signs are held for many frames, so a single frame's
prediction is weak evidence. This accumulates a short rolling window and
only accepts a letter once a majority of recent frames agree, which stops
transient misclassifications from being emitted.

Kept free of camera/model/TTS imports so it can be unit tested directly.
"""

import os
import sys
from collections import Counter, deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config


class PredictionSmoother:
    """Rolling-majority debounce over frame-level predictions."""

    def __init__(self, window=None, min_confidence=None, reset_after_missing=None,
                 required_votes=None):
        """
        Args:
            window: Frames in the rolling window (default config.PREDICTION_SMOOTHING)
            min_confidence: Minimum per-frame confidence to count as a vote
                (default config.CONFIDENCE_THRESHOLD)
            reset_after_missing: Consecutive no-hand frames before the window
                is cleared (default config.NO_HAND_RESET_FRAMES)
            required_votes: Votes needed to accept (default window - 1)
        """
        self.window = window if window is not None else config.PREDICTION_SMOOTHING
        self.min_confidence = (min_confidence if min_confidence is not None
                               else config.CONFIDENCE_THRESHOLD)
        self.reset_after_missing = (reset_after_missing if reset_after_missing is not None
                                    else config.NO_HAND_RESET_FRAMES)

        # A bare majority is not enough: with window=5 a perfect A/M/A/M
        # alternation gives the leader 3 votes every frame, so the accepted
        # letter would flip on every frame. Requiring window-1 means at most
        # one dissenting frame per window, which rejects alternation.
        if required_votes is not None:
            self.required_votes = required_votes
        else:
            self.required_votes = max(self.window - 1, (self.window // 2) + 1)

        self._recent = deque(maxlen=self.window)
        self._missing = 0
        self._accepted = None

    @property
    def accepted(self):
        """Currently accepted letter, or None."""
        return self._accepted

    @property
    def missing_frames(self):
        """Consecutive frames with no hand."""
        return self._missing

    def reset(self):
        """Clear all smoothing state."""
        self._recent.clear()
        self._missing = 0
        self._accepted = None

    def update(self, prediction, confidence=0.0):
        """
        Feed one frame's result.

        Args:
            prediction: Frame-level predicted letter, or None if no hand
            confidence: Frame-level confidence

        Returns:
            (accepted, emitted) where accepted is the currently accepted
            letter (or None) and emitted is True only on the frame where
            the accepted letter changes.
        """
        if prediction is None:
            self._missing += 1
            if self._missing >= self.reset_after_missing:
                self._recent.clear()
                self._accepted = None
            return self._accepted, False

        self._missing = 0

        # Sub-threshold frames still occupy a slot, so a run of weak
        # predictions cannot accumulate into a consensus.
        self._recent.append(prediction if confidence >= self.min_confidence else None)

        if len(self._recent) < self.window:
            return self._accepted, False

        votes = Counter(p for p in self._recent if p is not None)
        if not votes:
            return self._accepted, False

        candidate, count = votes.most_common(1)[0]
        if count < self.required_votes:
            return self._accepted, False

        if candidate != self._accepted:
            self._accepted = candidate
            return self._accepted, True

        return self._accepted, False
