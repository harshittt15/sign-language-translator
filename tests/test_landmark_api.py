"""
Tests for the landmark based prediction path.

The load-bearing one is test_browser_landmarks_match_python_pipeline: it proves
that posting raw landmarks to /predict-landmarks yields the same 63 normalized
features and the same prediction as running the image through the original
server side pipeline. That is what makes moving MediaPipe into the browser safe.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from fastapi.testclient import TestClient

from landmarks import normalize_landmarks, flatten_landmark_dicts
from landmark_inference import LandmarkInference
from prediction_smoother import PredictionSmoother
from web_api.app import app, _sessions

MODEL_PKL = os.path.join(ROOT, "models", "sign_language_model.pkl")
DATASET = os.path.expanduser(
    "~/Downloads/kaggle_asl_temp/asl_alphabet_train/asl_alphabet_train")

needs_model = pytest.mark.skipif(
    not os.path.exists(MODEL_PKL), reason="trained model not present")
needs_dataset = pytest.mark.skipif(
    not os.path.isdir(DATASET), reason="ASL dataset not present")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _browser_payload(flat):
    """Turn a flat 63-value vector into the browser's {x,y,z} JSON shape."""
    return [{"x": float(flat[i]), "y": float(flat[i + 1]), "z": float(flat[i + 2])}
            for i in range(0, 63, 3)]


def _raw_landmarks_from_dataset(letter="A"):
    """Raw landmarks straight out of the server side MediaPipe pipeline."""
    import cv2
    from hand_tracker import HandTracker, center_square_crop
    img = cv2.imread(str(sorted((Path(DATASET) / letter).glob("*.jpg"))[0]))
    t = HandTracker()
    try:
        square, _, _, _ = center_square_crop(img)
        lm = t.extract_landmarks(square)
    finally:
        t.close()
    return (lm[0] if lm else None), img


# ---------------- the regression test ----------------

@needs_model
@needs_dataset
def test_browser_landmarks_match_python_pipeline(client):
    """
    Raw landmarks -> /predict-landmarks must reproduce the original pipeline's
    63 features and prediction exactly.
    """
    raw, img = _raw_landmarks_from_dataset("A")
    assert raw is not None, "no hand detected in the fixture image"

    # Reference: the original server side path.
    from inference import SignLanguageInference
    eng = SignLanguageInference()
    try:
        reference = eng.process_frame(img)
        ref_features = eng.tracker.normalize_landmarks([raw])[0]
    finally:
        eng.close()

    # Browser path: same landmarks, shipped as JSON, normalized server side.
    payload = _browser_payload(raw)
    flat = flatten_landmark_dicts(payload)
    api_features = normalize_landmarks([flat])[0]

    assert api_features.shape == (63,)
    assert np.allclose(ref_features, api_features, atol=1e-12), \
        f"feature mismatch, max diff {np.max(np.abs(ref_features - api_features)):.3e}"

    body = client.post("/predict-landmarks", json={
        "landmarks": payload, "session_id": "regression-parity"}).json()

    assert body["raw_sign"] == reference.raw_sign
    assert body["confidence"] == pytest.approx(reference.confidence, abs=1e-9)
    assert body["hand_detected"] is True


@needs_model
@needs_dataset
def test_flatten_preserves_values():
    raw, _ = _raw_landmarks_from_dataset("A")
    assert raw is not None
    flat = flatten_landmark_dicts(_browser_payload(raw))
    assert flat.shape == (63,)
    assert np.allclose(flat, np.asarray(raw, dtype=float))


# ---------------- request validation ----------------

@needs_model
def test_valid_21_landmark_request(client):
    pts = [{"x": 0.5 + i * 0.01, "y": 0.5, "z": 0.0} for i in range(21)]
    r = client.post("/predict-landmarks", json={"landmarks": pts, "session_id": "v"})
    assert r.status_code == 200
    body = r.json()
    assert body["hand_detected"] is True
    assert 0.0 <= body["confidence"] <= 1.0


@needs_model
@pytest.mark.parametrize("n", [0, 1, 20, 22, 42])
def test_wrong_landmark_count_is_4xx(client, n):
    pts = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(n)]
    r = client.post("/predict-landmarks", json={"landmarks": pts})
    assert 400 <= r.status_code < 500, f"n={n} returned {r.status_code}"


@needs_model
@pytest.mark.parametrize("bad", [
    {"x": "abc", "y": 0.5, "z": 0.0},
    {"x": None, "y": 0.5, "z": 0.0},
    {"y": 0.5, "z": 0.0},
    {"x": 1e9, "y": 0.5, "z": 0.0},
    {"x": 0.5, "y": -500.0, "z": 0.0},
    {"x": 0.5, "y": 0.5, "z": 99.0},
])
def test_malformed_coordinates_are_4xx(client, bad):
    pts = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(20)] + [bad]
    r = client.post("/predict-landmarks", json={"landmarks": pts})
    assert 400 <= r.status_code < 500, f"{bad} returned {r.status_code}"


@needs_model
@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_nan_and_infinity_rejected(bad):
    """
    NaN and Infinity are not valid JSON but Python's parser accepts them, so
    the schema has to reject them. A real server turns the resulting
    RequestValidationError into a 422; TestClient re-raises it by default, so
    this client is configured to return the response instead.
    """
    strict = TestClient(app, raise_server_exceptions=False)
    pts = ", ".join(['{"x": 0.5, "y": 0.5, "z": 0.0}'] * 20)
    body = '{"landmarks": [%s, {"x": %s, "y": 0.5, "z": 0.0}]}' % (pts, bad)
    r = strict.post("/predict-landmarks", content=body,
                    headers={"Content-Type": "application/json"})
    assert 400 <= r.status_code < 500, f"{bad} returned {r.status_code}"


@needs_model
def test_null_landmarks_means_no_hand(client):
    r = client.post("/predict-landmarks",
                    json={"landmarks": None, "session_id": "nohand"})
    assert r.status_code == 200
    body = r.json()
    assert body["hand_detected"] is False
    assert body["sign"] is None and body["confidence"] == 0.0


@needs_model
def test_response_shape(client):
    pts = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(21)]
    body = client.post("/predict-landmarks", json={"landmarks": pts}).json()
    assert set(body) == {"sign", "confidence", "hand_detected", "raw_sign",
                         "emitted", "landmarks"}
    import json
    json.loads(json.dumps(body))


# ---------------- smoothing ----------------

@needs_model
@needs_dataset
def test_smoothing_emits_once_per_held_sign(client):
    raw, _ = _raw_landmarks_from_dataset("A")
    assert raw is not None
    pts = _browser_payload(raw)
    sid = "smooth-landmarks"
    client.post("/reset", data={"session_id": sid})

    emits = []
    for _ in range(12):
        body = client.post("/predict-landmarks",
                           json={"landmarks": pts, "session_id": sid}).json()
        emits.append(body["emitted"])

    assert emits[:4] == [False] * 4, "emitted before the window filled"
    assert sum(emits) == 1, f"expected one emission, got {sum(emits)}"


@needs_model
@needs_dataset
def test_sessions_are_isolated(client):
    raw, _ = _raw_landmarks_from_dataset("A")
    pts = _browser_payload(raw)
    for sid in ("lm-a", "lm-b"):
        client.post("/reset", data={"session_id": sid})
    for _ in range(5):
        client.post("/predict-landmarks", json={"landmarks": pts, "session_id": "lm-a"})
    body = client.post("/predict-landmarks",
                       json={"landmarks": pts, "session_id": "lm-b"}).json()
    assert body["emitted"] is False, "lm-b inherited lm-a's smoothing state"


@needs_model
def test_reset_clears_landmark_session(client):
    pts = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(21)]
    client.post("/predict-landmarks", json={"landmarks": pts, "session_id": "lm-reset"})
    assert "lm-reset" in _sessions
    client.post("/reset", data={"session_id": "lm-reset"})
    assert "lm-reset" not in _sessions


@needs_model
@needs_dataset
def test_no_hand_frames_reset_the_window():
    """
    Losing the hand must clear smoothing state, as on the desktop.

    Uses real landmarks: a synthetic vector predicts at around 0.4 confidence,
    below the 0.6 threshold, so it would never vote and the window would never
    fill. That is correct behaviour, but it cannot exercise the reset.
    """
    raw, _ = _raw_landmarks_from_dataset("A")
    assert raw is not None
    flat = np.asarray(raw, dtype=float)

    sm = PredictionSmoother()
    eng = LandmarkInference()

    for _ in range(6):
        eng.predict(flat, smoother=sm)
    assert sm.accepted is not None, "a confidently detected hand should be accepted"

    for _ in range(6):
        eng.predict(None, smoother=sm)
    assert sm.accepted is None, "no-hand frames should have reset the window"


# ---------------- deployment shape ----------------

def test_app_imports_without_mediapipe_or_opencv():
    """
    The deployed bundle must not need MediaPipe or OpenCV. Importing the app
    must not drag them in, even though /predict can still use them locally.
    """
    import subprocess
    code = (
        "import sys; sys.path.insert(0, %r); "
        "from src.web_api.app import app; "
        "bad=[m for m in ('mediapipe','cv2','jax','jaxlib','matplotlib') "
        "if any(k==m or k.startswith(m+'.') for k in sys.modules)]; "
        "print(','.join(bad))" % ROOT
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=ROOT)
    heavy = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    assert heavy == "", f"heavy modules imported by the app: {heavy}"


@needs_model
def test_model_task_is_served(client):
    """The browser fetches the same landmarker bundle the features came from."""
    r = client.get("/model/hand_landmarker.task")
    assert r.status_code == 200
    assert len(r.content) > 1_000_000


@needs_model
def test_model_info_exposes_mediapipe_settings(client):
    b = client.get("/model-info").json()
    for k in ("min_hand_detection_confidence", "min_tracking_confidence", "num_hands"):
        assert k in b, f"{k} missing; the browser needs it to match the server"
    assert b["num_hands"] == 2
    assert b["min_hand_detection_confidence"] == 0.7
