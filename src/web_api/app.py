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

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import config
from landmark_inference import LandmarkInference
from landmarks import flatten_landmark_dicts
from prediction_smoother import PredictionSmoother
from web_api.schemas import (
    HealthResponse,
    PredictResponse,
    ModelInfoResponse,
    PredictLandmarksRequest,
)

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

def _json_safe(value):
    """
    Make a validation error payload serializable.

    FastAPI echoes the rejected input back in the 422 body. A payload
    containing NaN or Infinity cannot be serialized to valid JSON, so the
    error response itself fails and the client sees a 500 instead of the 4xx
    it should get. Replace anything non-finite with its string form.
    """
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return repr(value)
        return value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Return 422 for malformed requests, never 500."""
    details = []
    for err in exc.errors():
        details.append({
            "loc": [str(p) for p in err.get("loc", [])],
            "msg": str(err.get("msg", "invalid value")),
            "type": str(err.get("type", "value_error")),
            "input": _json_safe(err.get("input")),
        })
    return JSONResponse(status_code=422, content={"detail": details})


_engine = None
_frame_engine = None
_sessions = OrderedDict()          # session_id -> (PredictionSmoother, last_seen)


def get_engine():
    """
    Landmark classifier. Needs only numpy and scikit-learn, so the deployed
    bundle does not have to carry MediaPipe or OpenCV.
    """
    global _engine
    if _engine is None:
        _engine = LandmarkInference()
    return _engine


def get_frame_engine():
    """
    Frame based pipeline, which runs MediaPipe server side.

    Imported lazily and only when /predict is called. The browser now does
    landmark detection, so production deployments do not install MediaPipe;
    this path stays available locally for regression testing against the
    desktop pipeline.
    """
    global _frame_engine
    if _frame_engine is None:
        from inference import SignLanguageInference
        _frame_engine = SignLanguageInference()
    return _frame_engine


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
        min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        num_hands=config.MAX_NUM_HANDS,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    frame: UploadFile = File(..., description="Single webcam frame (JPEG/PNG)"),
    session_id: str = Form(default=""),
    want_landmarks: bool = Form(default=False),
):
    """
    Run one image frame through the server side MediaPipe pipeline.

    Retained for regression testing against the desktop pipeline. The deployed
    web client uses /predict-landmarks instead, and production deployments do
    not install MediaPipe, so this returns 503 there.

    The frame is decoded in memory and discarded when this call returns; it is
    never written to disk or logged.
    """
    try:
        import cv2
        import numpy as np
        eng = get_frame_engine()
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=("Frame based prediction needs MediaPipe and OpenCV, which are "
                    "not installed in this deployment. Use /predict-landmarks."),
        ) from exc

    raw = await frame.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty frame upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Frame too large")

    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    del raw
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    smoother = _smoother_for(session_id or str(uuid.uuid4()))
    result = eng.process_frame(img, smoother=smoother)

    landmarks = None
    if want_landmarks and result.hand_detected:
        from landmarks import landmarks_to_points
        hands = landmarks_to_points(result.landmarks)
        if hands:
            landmarks = [[x, y] for x, y in hands[0]]

    payload = result.as_dict()
    payload["landmarks"] = landmarks
    return PredictResponse(**payload)


@app.post("/predict-landmarks", response_model=PredictResponse)
def predict_landmarks(body: PredictLandmarksRequest):
    """
    Classify 21 raw hand landmarks produced by the browser.

    The browser runs MediaPipe HandLandmarker on the same centred square region
    the desktop pipeline crops to, so these coordinates mean the same thing as
    the ones the Python pipeline produces. Normalization, scaling, prediction
    and temporal smoothing all stay here on the server.

    A null `landmarks` value means the browser saw no hand in that frame, which
    is valid: it advances the smoother's no-hand counter.
    """
    eng = get_engine()
    smoother = _smoother_for(body.session_id or str(uuid.uuid4()))

    flat = None
    if body.landmarks is not None:
        flat = flatten_landmark_dicts([p.model_dump() for p in body.landmarks])

    result = eng.predict(flat, smoother=smoother)

    payload = result.as_dict()
    payload["landmarks"] = None
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


MODEL_TASK = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "hand_landmarker.task")


@app.get("/model/hand_landmarker.task")
def hand_landmarker_model():
    """
    Serve the HandLandmarker bundle to the browser.

    The training features were extracted with this exact model file. Letting
    the browser fetch a different build from a public CDN risks subtly
    different landmarks, so the same artifact is served here.
    """
    if not os.path.exists(MODEL_TASK):
        raise HTTPException(status_code=404, detail="Landmark model not found")
    return FileResponse(MODEL_TASK, media_type="application/octet-stream")
