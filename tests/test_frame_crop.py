"""Tests for the aspect-ratio compatibility crop."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from hand_tracker import center_square_crop


def test_webcam_frame_becomes_square():
    frame = np.zeros((720, 1280, 3), np.uint8)
    crop, y0, x0, side = center_square_crop(frame)
    assert crop.shape == (720, 720, 3)
    assert side == 720
    assert (y0, x0) == (0, 280)


def test_square_frame_is_unchanged():
    """Dataset images are already square and must pass through untouched."""
    frame = np.zeros((200, 200, 3), np.uint8)
    crop, y0, x0, side = center_square_crop(frame)
    assert crop.shape == frame.shape
    assert (y0, x0, side) == (0, 0, 200)
    assert np.array_equal(crop, frame)


def test_crop_is_centred():
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[:, 640] = 255                       # vertical line at frame centre
    crop, _, x0, side = center_square_crop(frame)
    assert np.all(crop[:, 640 - x0] == 255)   # still the centre of the crop
    assert 640 - x0 == side // 2


def test_offsets_reconstruct_region():
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    crop, y0, x0, side = center_square_crop(frame)
    assert np.array_equal(frame[y0:y0 + side, x0:x0 + side], crop)


def test_portrait_frame():
    frame = np.zeros((1280, 720, 3), np.uint8)
    crop, y0, x0, side = center_square_crop(frame)
    assert crop.shape == (720, 720, 3)
    assert (y0, x0) == (280, 0)


def test_odd_dimensions():
    frame = np.zeros((721, 1281, 3), np.uint8)
    crop, _, _, side = center_square_crop(frame)
    assert crop.shape == (721, 721, 3) and side == 721
