"""
FastAPI layer around the existing inference pipeline.

Reuses src/inference.SignLanguageInference, so the web path and the desktop
app run identical preprocessing and the same trained model. No frame is ever
written to disk; each upload is decoded in memory, used for one prediction,
and discarded.
"""

import os
import sys
import time
import uuid
from collections import OrderedDict

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import config
from inference import SignLanguageInference, landmarks_to_points
from prediction_smoother import PredictionSmoother
from web_api.schemas import HealthResponse, PredictResponse, ModelInfoResponse

VERSION = "1.0.0"
WEB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "web")

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_SESSIONS = 64
SESSION_TTL_SECONDS = 30 * 60

app = FastAPI(
    title="Real-Time Sign Language Translator",
    description="Web API around the Random Forest ASL alphabet classifier.",
    version=VERSION,
)

_engine = None
_sessions = OrderedDict()          # session_id -> (PredictionSmoother, last_seen)


def get_engine():
    """Load the model lazily so importing the module stays cheap for tests."""
    global _engine
    if _engine is None:
        _engine = SignLanguageInference()
    return _engine


def _smoother_for(session_id):
    """
    Per-browser smoothing state. Each viewer needs its own temporal window;
    sharing one would let separate users contaminate each other's predictions.
    """
    now = time.time()
    for sid in [s for s, (_, seen) in _sessions.items()
                if now - seen > SESSION_TTL_SECONDS]:
        _sessions.pop(sid, None)

    if session_id in _sessions:
        sm, _ = _sessions.pop(session_id)
    else:
        sm = PredictionSmoother()
    _sessions[session_id] = (sm, now)

    while len(_sessions) > MAX_SESSIONS:
        _sessions.popitem(last=False)
    return sm


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness probe; also reports whether the model actually loaded."""
    try:
        eng = get_engine()
        return HealthResponse(status="ok", model_loaded=True,
                              classes=len(eng.classes), version=VERSION)
    except Exception:
        return HealthResponse(status="degraded", model_loaded=False,
                              classes=0, version=VERSION)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """Metrics for the UI. Figures are offline dataset results, not webcam."""
    eng = get_engine()
    return ModelInfoResponse(
        classes=eng.classes,
        num_classes=len(eng.classes),
        offline_test_accuracy=0.979,
        offline_macro_f1=0.97,
        accuracy_note=(
            "97.9% is the held-out test accuracy on the public ASL Alphabet "
            "image dataset. It is NOT a measurement of live webcam accuracy, "
            "which is lower."
        ),
        known_limitations=[
            "M and N are frequently confused under webcam conditions; the "
            "distinguishing thumb position is occluded in both signs.",
            "Live webcam accuracy is lower than the offline dataset result.",
            "Static alphabet letters only - no motion-based signs or words.",
        ],
        smoothing_window=config.PREDICTION_SMOOTHING,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    frame: UploadFile = File(..., description="Single webcam frame (JPEG/PNG)"),
    session_id: str = Form(default=""),
    want_landmarks: bool = Form(default=False),
):
    """
    Run one frame through the production pipeline.

    The frame is decoded in memory and discarded when this call returns; it is
    never written to disk or logged.
    """
    raw = await frame.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty frame upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Frame too large")

    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    del raw
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    eng = get_engine()
    smoother = _smoother_for(session_id or str(uuid.uuid4()))
    result = eng.process_frame(img, smoother=smoother)

    landmarks = None
    if want_landmarks and result.hand_detected:
        hands = landmarks_to_points(result.landmarks)
        if hands:
            landmarks = [[x, y] for x, y in hands[0]]

    payload = result.as_dict()
    payload["landmarks"] = landmarks
    return PredictResponse(**payload)


@app.post("/reset")
def reset(session_id: str = Form(default="")):
    """Drop a session's smoothing state (used by the Clear button)."""
    _sessions.pop(session_id, None)
    return {"status": "reset"}


if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=os.path.join(WEB_DIR, "static")),
              name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(WEB_DIR, "index.html"))

    @app.get("/translator")
    def translator():
        return FileResponse(os.path.join(WEB_DIR, "translator.html"))

    @app.get("/about")
    def about():
        return FileResponse(os.path.join(WEB_DIR, "about.html"))
