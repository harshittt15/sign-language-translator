"""
Landmark geometry: the normalization that turns raw hand landmarks into the
63 features the classifier was trained on.

This module deliberately imports nothing beyond numpy. MediaPipe is a very
heavy dependency (it pulls in jax, jaxlib, OpenCV and matplotlib, around 1 GB
unzipped on Linux), and the deployed web API does not need it because the
browser now performs landmark detection. Keeping the normalization here lets
the server stay the single source of truth for feature construction without
dragging MediaPipe into the serverless bundle.

`HandTracker.normalize_landmarks` delegates to `normalize_landmarks` below, so
the desktop and web paths run the exact same code.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config

NUM_LANDMARKS = config.NUM_LANDMARKS
LANDMARK_DIMENSIONS = config.LANDMARK_DIMENSIONS
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9

TOTAL_FEATURES = NUM_LANDMARKS * LANDMARK_DIMENSIONS


def normalize_landmarks(landmarks):
    """
    Normalize landmarks for model input.

    Args:
        landmarks: list of raw landmark vectors, each 63 values laid out as
            21 consecutive (x, y, z) triples, or None

    Returns:
        List of normalized (63,) arrays, or None when there is nothing to
        normalize.
    """
    if landmarks is None:
        return None

    normalized = []
    for hand_landmarks in landmarks:
        # Convert to numpy array if needed
        if not isinstance(hand_landmarks, np.ndarray):
            hand_landmarks = np.array(hand_landmarks)

        # Reshape flat 63 values into 21 (x, y, z) points so the wrist
        # offset and scale apply per-landmark rather than broadcasting.
        points = hand_landmarks.reshape(NUM_LANDMARKS, LANDMARK_DIMENSIONS)

        # Normalize: use the wrist (landmark 0) as reference point
        normalized_points = points - points[WRIST_IDX]

        # Scale by hand size (distance from wrist to middle finger MCP)
        scale = np.linalg.norm(points[MIDDLE_MCP_IDX] - points[WRIST_IDX])
        if scale > 0:
            normalized_points = normalized_points / scale

        normalized.append(normalized_points.reshape(-1))

    return normalized if len(normalized) > 0 else None


def landmarks_to_points(landmarks):
    """
    Convert raw landmark vectors into (x, y) pairs in [0,1] within the square
    crop, for drawing in a browser.

    Returns a list of hands, each a list of 21 (x, y) pairs.
    """
    if not landmarks:
        return []
    return [[(float(h[i]), float(h[i + 1])) for i in range(0, len(h), 3)]
            for h in landmarks]


def flatten_landmark_dicts(items):
    """
    Turn a browser payload of 21 {x, y, z} objects into the flat 63-value
    vector the rest of the pipeline expects.

    The browser runs MediaPipe on the same centred square region the Python
    pipeline cropped to, so these coordinates carry the same meaning as the
    ones `HandTracker.extract_landmarks` produces.
    """
    flat = []
    for p in items:
        flat.extend([float(p["x"]), float(p["y"]), float(p["z"])])
    return np.array(flat, dtype=float)
