"""Tests for the FastAPI inference layer."""

import io
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from fastapi.testclient import TestClient

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


def _jpeg(img):
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return io.BytesIO(buf.tobytes())


def _blank(h=720, w=1280):
    return np.zeros((h, w, 3), np.uint8)


def _letter_frame(letter):
    from pathlib import Path
    p = sorted((Path(DATASET) / letter).glob("*.jpg"))[0]
    return cv2.imread(str(p))


@needs_model
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["classes"] == 26


@needs_model
def test_model_info_labels_accuracy_as_offline(client):
    body = client.get("/model-info").json()
    assert body["num_classes"] == 26
    assert body["offline_test_accuracy"] == 0.979
    # the note must not let 97.9% be read as a webcam figure
    assert "NOT" in body["accuracy_note"]
    assert "webcam" in body["accuracy_note"].lower()
    assert any("M and N" in lim for lim in body["known_limitations"])


@needs_model
def test_predict_no_hand(client):
    r = client.post("/predict", files={"frame": ("f.jpg", _jpeg(_blank()), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["hand_detected"] is False
    assert body["sign"] is None
    assert body["confidence"] == 0.0


@needs_model
@needs_dataset
def test_predict_detects_hand(client):
    r = client.post("/predict",
                    files={"frame": ("f.jpg", _jpeg(_letter_frame("A")), "image/jpeg")},
                    data={"session_id": "t-detect"})
    body = r.json()
    assert body["hand_detected"] is True
    assert body["raw_sign"] == "A"
    assert 0.0 < body["confidence"] <= 1.0


@needs_model
@needs_dataset
def test_smoothing_suppresses_first_frames_then_emits_once(client):
    """The API must return smoothed output, not raw per-frame predictions."""
    img = _letter_frame("A")
    sid = "t-smooth"
    client.post("/reset", data={"session_id": sid})

    emits = []
    for _ in range(12):
        body = client.post(
            "/predict",
            files={"frame": ("f.jpg", _jpeg(img), "image/jpeg")},
            data={"session_id": sid}).json()
        emits.append(body["emitted"])

    assert emits[:4] == [False] * 4, "emitted before the smoothing window filled"
    assert sum(emits) == 1, f"expected exactly one emission, got {sum(emits)}"


@needs_model
@needs_dataset
def test_sessions_are_isolated(client):
    img = _letter_frame("A")
    for sid in ("sess-a", "sess-b"):
        client.post("/reset", data={"session_id": sid})

    for _ in range(5):
        client.post("/predict", files={"frame": ("f.jpg", _jpeg(img), "image/jpeg")},
                    data={"session_id": "sess-a"})

    body = client.post("/predict",
                       files={"frame": ("f.jpg", _jpeg(img), "image/jpeg")},
                       data={"session_id": "sess-b"}).json()
    assert body["emitted"] is False, "sess-b inherited sess-a's smoothing state"


@needs_model
def test_reset_clears_session(client):
    client.post("/predict", files={"frame": ("f.jpg", _jpeg(_blank()), "image/jpeg")},
                data={"session_id": "t-reset"})
    assert "t-reset" in _sessions
    client.post("/reset", data={"session_id": "t-reset"})
    assert "t-reset" not in _sessions


@needs_model
def test_rejects_empty_and_undecodable(client):
    r = client.post("/predict", files={"frame": ("f.jpg", io.BytesIO(b""), "image/jpeg")})
    assert r.status_code == 400
    r = client.post("/predict",
                    files={"frame": ("f.jpg", io.BytesIO(b"not an image"), "image/jpeg")})
    assert r.status_code == 400


@needs_model
@needs_dataset
def test_landmarks_optional(client):
    img = _letter_frame("A")
    off = client.post("/predict", files={"frame": ("f.jpg", _jpeg(img), "image/jpeg")},
                      data={"session_id": "t-lm"}).json()
    assert off["landmarks"] is None

    on = client.post("/predict", files={"frame": ("f.jpg", _jpeg(img), "image/jpeg")},
                     data={"session_id": "t-lm", "want_landmarks": "true"}).json()
    assert on["landmarks"] is not None
    assert len(on["landmarks"]) == 21
    assert all(len(p) == 2 for p in on["landmarks"])


@needs_model
def test_widescreen_and_square_both_accepted(client):
    for shape in [(720, 1280), (480, 640), (400, 400)]:
        r = client.post("/predict",
                        files={"frame": ("f.jpg", _jpeg(_blank(*shape)), "image/jpeg")})
        assert r.status_code == 200, f"failed for {shape}"


# ---------------- static frontend ----------------

@needs_model
def test_pages_served(client):
    for path, marker in [("/", "start translating"),
                         ("/translator", "current sign"),
                         ("/about", "how it works")]:
        r = client.get(path)
        assert r.status_code == 200, path
        assert "text/html" in r.headers["content-type"]
        # casing is a copy decision, presence is the contract
        assert marker in r.text.lower(), f"{marker!r} missing from {path}"


@needs_model
def test_static_assets_served(client):
    for path, ctype in [("/static/css/style.css", "css"),
                        ("/static/js/translator.js", "javascript")]:
        r = client.get(path)
        assert r.status_code == 200, path
        assert ctype in r.headers["content-type"]


@needs_model
def test_accuracy_is_never_presented_as_a_webcam_figure(client):
    """
    Guards an explicit requirement: 97.9% must always be labelled as an
    offline dataset result, never as live webcam accuracy.
    """
    import re
    for path in ("/", "/about"):
        html = client.get(path).text
        assert "97.9%" in html
        # strip tags: the disclaimer wording spans <b> elements
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).lower()
        assert "offline" in text or "dataset result" in text
        assert ("not a webcam" in text or "not a measurement" in text), \
            f"{path} does not disclaim the figure as non-webcam"


@needs_model
def test_pages_disclose_mn_limitation_and_privacy(client):
    about = client.get("/about").text
    assert "M and N" in about
    translator = client.get("/translator").text
    assert "never stored" in translator or "discarded" in translator
