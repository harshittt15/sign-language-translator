"""
Exercises the exact frame-handling wiring in main.py's run loop.

main.py needs a webcam, so the loop cannot be run under pytest directly.
These tests replicate its per-frame body line-for-line against a synthetic
widescreen frame, which catches shape and unpacking errors that would
otherwise only show up when a human starts the desktop app.
"""

import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from inference import SignLanguageInference
from prediction_smoother import PredictionSmoother

MODEL_PKL = os.path.join(ROOT, "models", "sign_language_model.pkl")
DATASET = os.path.expanduser(
    "~/Downloads/kaggle_asl_temp/asl_alphabet_train/asl_alphabet_train")

needs_model = pytest.mark.skipif(
    not os.path.exists(MODEL_PKL), reason="trained model not present")


@pytest.fixture(scope="module")
def engine():
    eng = SignLanguageInference()
    yield eng
    eng.close()


# MediaPipe only finds the composited hand when it fills most of the square
# crop; smaller insets are not detected at all. Measured: 400/480/540/600 all
# fail, 660 succeeds.
_COMPOSITE_SIDE = 660


def _widescreen_with_hand():
    """1280x720 frame with a real dataset hand composited in, like a webcam."""
    frame = np.full((720, 1280, 3), 220, np.uint8)
    if os.path.isdir(DATASET):
        from pathlib import Path
        p = sorted((Path(DATASET) / "A").glob("*.jpg"))[0]
        hand = cv2.imread(str(p))
        if hand is not None:
            side = _COMPOSITE_SIDE
            y0, x0 = (720 - side) // 2, (1280 - side) // 2
            frame[y0:y0 + side, x0:x0 + side] = cv2.resize(hand, (side, side))
    return frame


def _hand_is_detectable(engine):
    return engine.process_frame(_widescreen_with_hand()).hand_detected


def _run_loop_body(engine, frame, smoother):
    """Verbatim copy of main.py's per-frame body."""
    frame_display = frame.copy()

    result = engine.process_frame(frame, smoother=smoother)

    predicted_sign = result.raw_sign
    confidence = result.confidence
    accepted, emitted = result.sign, result.emitted
    normalized = result.hand_detected

    crop_y, crop_x, crop_side = result.crop_offset
    frame_square = frame[crop_y:crop_y + crop_side,
                         crop_x:crop_x + crop_side]
    frame_display[crop_y:crop_y + crop_side,
                  crop_x:crop_x + crop_side] = \
        engine.tracker.draw_landmarks(frame_square, result.landmarks)

    return frame_display, predicted_sign, confidence, accepted, emitted, normalized


@needs_model
def test_loop_body_runs_on_widescreen_frame(engine):
    frame = _widescreen_with_hand()
    disp, raw, conf, accepted, emitted, detected = _run_loop_body(
        engine, frame, PredictionSmoother())
    assert disp.shape == (720, 1280, 3)
    assert disp.dtype == np.uint8
    assert isinstance(conf, float)
    assert emitted is False          # first frame can never fill the window


@needs_model
def test_loop_body_runs_with_no_hand(engine):
    frame = np.zeros((720, 1280, 3), np.uint8)
    disp, raw, conf, accepted, emitted, detected = _run_loop_body(
        engine, frame, PredictionSmoother())
    assert disp.shape == (720, 1280, 3)
    assert detected is False and raw is None and conf == 0.0


@needs_model
def test_overlay_lands_inside_the_crop_region(engine):
    """The overlay must not be written outside the square crop."""
    frame = _widescreen_with_hand()
    disp, *_ = _run_loop_body(engine, frame, PredictionSmoother())
    # columns outside the centred 720-wide crop must be untouched
    assert np.array_equal(disp[:, :280], frame[:, :280])
    assert np.array_equal(disp[:, 1000:], frame[:, 1000:])
    assert disp[:, 280:1000].shape == (720, 720, 3)


@needs_model
def test_loop_body_is_stable_over_many_frames(engine):
    """Sustained signing must emit exactly once, as the desktop app expects."""
    if not _hand_is_detectable(engine):
        pytest.skip("composited fixture hand not detected by MediaPipe")
    frame = _widescreen_with_hand()
    smoother = PredictionSmoother()
    emissions = 0
    for _ in range(25):
        *_, emitted, _ = _run_loop_body(engine, frame, smoother)
        emissions += bool(emitted)
    assert emissions == 1


@needs_model
def test_various_camera_resolutions(engine):
    for h, w in [(720, 1280), (480, 640), (1080, 1920), (720, 720)]:
        frame = np.zeros((h, w, 3), np.uint8)
        disp, *_ = _run_loop_body(engine, frame, PredictionSmoother())
        assert disp.shape == (h, w, 3), f"broke at {h}x{w}"
