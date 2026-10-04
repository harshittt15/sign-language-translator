"""
Tests for the shared inference pipeline.

The important ones are the equivalence tests: they prove SignLanguageInference
produces byte-identical results to the inline pipeline main.py used before the
refactor, so the desktop app's behavior is unchanged.
"""

import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from hand_tracker import HandTracker, center_square_crop
from model import SignLanguageModel
from inference import SignLanguageInference, InferenceResult, landmarks_to_points
from prediction_smoother import PredictionSmoother

DATASET = os.path.expanduser(
    "~/Downloads/kaggle_asl_temp/asl_alphabet_train/asl_alphabet_train")
MODEL_PKL = os.path.join(ROOT, "models", "sign_language_model.pkl")

needs_model = pytest.mark.skipif(
    not os.path.exists(MODEL_PKL), reason="trained model not present")
needs_dataset = pytest.mark.skipif(
    not os.path.isdir(DATASET), reason="ASL dataset not present")


@pytest.fixture(scope="module")
def engine():
    eng = SignLanguageInference()
    yield eng
    eng.close()


def _sample(letter, n=1):
    from pathlib import Path
    paths = sorted((Path(DATASET) / letter).glob("*.jpg"))[:n]
    return [cv2.imread(str(p)) for p in paths]


@needs_model
def test_engine_exposes_26_classes(engine):
    assert len(engine.classes) == 26
    assert engine.classes[0] == "A" and engine.classes[-1] == "Z"


@needs_model
def test_no_hand_returns_clean_result(engine):
    blank = np.zeros((720, 1280, 3), np.uint8)
    r = engine.process_frame(blank)
    assert isinstance(r, InferenceResult)
    assert r.hand_detected is False
    assert r.raw_sign is None and r.sign is None
    assert r.confidence == 0.0 and r.emitted is False


@needs_model
def test_result_serializes_for_json(engine):
    blank = np.zeros((720, 1280, 3), np.uint8)
    d = engine.process_frame(blank).as_dict()
    assert set(d) == {"sign", "confidence", "hand_detected", "raw_sign", "emitted"}
    import json
    json.loads(json.dumps(d))          # must be JSON-serializable


@needs_model
@needs_dataset
def test_matches_inline_pipeline_exactly(engine):
    """
    Equivalence: SignLanguageInference must agree with the hand-rolled
    crop -> extract -> normalize -> predict sequence main.py used.
    """
    tracker = HandTracker()
    model = SignLanguageModel()
    model.load_model()
    model.load_metadata()

    checked = 0
    for letter in ["A", "B", "F", "L"]:
        for frame in _sample(letter, 3):
            if frame is None:
                continue

            square, _, _, _ = center_square_crop(frame)
            lm = tracker.extract_landmarks(square)
            nz = tracker.normalize_landmarks(lm)
            exp_sign, exp_conf = (model.predict(nz) if nz else (None, 0.0))

            got = engine.process_frame(frame)
            assert got.hand_detected == bool(nz)
            assert got.raw_sign == exp_sign
            assert got.confidence == pytest.approx(exp_conf, abs=1e-9)
            checked += 1

    tracker.close()
    assert checked > 0, "no frames were compared"


@needs_model
@needs_dataset
def test_smoother_state_is_caller_owned(engine):
    """Two smoothers must not interfere -- this is what makes web sessions safe."""
    frames = _sample("A", 1)
    if not frames or frames[0] is None:
        pytest.skip("no sample frame")
    frame = frames[0]

    a = PredictionSmoother()
    b = PredictionSmoother()

    for _ in range(5):
        engine.process_frame(frame, smoother=a)

    assert a.accepted is not None
    assert b.accepted is None            # untouched by a's updates


@needs_model
@needs_dataset
def test_smoother_emits_once_through_engine(engine):
    frames = _sample("A", 1)
    if not frames or frames[0] is None:
        pytest.skip("no sample frame")
    frame = frames[0]

    sm = PredictionSmoother()
    emissions = [engine.process_frame(frame, smoother=sm).emitted
                 for _ in range(20)]
    assert sum(emissions) == 1


@needs_model
def test_without_smoother_sign_is_raw(engine):
    blank = np.zeros((400, 400, 3), np.uint8)
    r = engine.process_frame(blank)
    assert r.sign == r.raw_sign


@needs_model
def test_crop_offset_reported_for_widescreen(engine):
    blank = np.zeros((720, 1280, 3), np.uint8)
    r = engine.process_frame(blank)
    assert r.crop_offset == (0, 280, 720)


@needs_model
@needs_dataset
def test_result_carries_landmarks_for_overlay(engine):
    """main.py draws from these, so one frame must mean one MediaPipe pass."""
    frames = _sample("A", 1)
    if not frames or frames[0] is None:
        pytest.skip("no sample frame")
    r = engine.process_frame(frames[0])
    assert r.hand_detected is True
    assert r.landmarks is not None and len(r.landmarks) >= 1
    assert len(r.landmarks[0]) == 63


@needs_model
@needs_dataset
def test_landmarks_to_points(engine):
    frames = _sample("A", 1)
    if not frames or frames[0] is None:
        pytest.skip("no sample frame")
    r = engine.process_frame(frames[0])
    hands = landmarks_to_points(r.landmarks)
    assert len(hands) >= 1
    assert len(hands[0]) == 21
    assert all(len(p) == 2 for p in hands[0])
    assert landmarks_to_points(None) == []
    assert landmarks_to_points([]) == []


@needs_model
def test_concurrent_frames_do_not_corrupt_state(engine):
    """FastAPI serves concurrently; the lock must keep MediaPipe safe."""
    import concurrent.futures as cf
    blank = np.zeros((720, 1280, 3), np.uint8)
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        out = list(ex.map(lambda _: engine.process_frame(blank), range(24)))
    assert len(out) == 24
    assert all(r.hand_detected is False for r in out)
