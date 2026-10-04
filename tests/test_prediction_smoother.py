"""Unit tests for PredictionSmoother temporal debouncing."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import config
from prediction_smoother import PredictionSmoother


def feed(sm, seq, conf=0.99):
    """Feed (prediction, confidence) pairs; return list of emitted letters."""
    emitted = []
    for item in seq:
        pred, c = item if isinstance(item, tuple) else (item, conf)
        accepted, did_emit = sm.update(pred, c)
        if did_emit:
            emitted.append(accepted)
    return emitted


def test_config_defaults():
    assert config.PREDICTION_SMOOTHING == 5
    assert config.NO_HAND_RESET_FRAMES == 5
    sm = PredictionSmoother()
    assert sm.window == 5
    assert sm.required_votes == 4
    assert sm.min_confidence == config.CONFIDENCE_THRESHOLD == 0.6


def test_no_emission_before_window_is_full():
    sm = PredictionSmoother()
    assert feed(sm, ["A"] * 4) == []
    assert sm.accepted is None
    assert feed(sm, ["A"]) == ["A"]


def test_held_sign_emits_exactly_once():
    sm = PredictionSmoother()
    assert feed(sm, ["A"] * 60) == ["A"]


def test_transient_single_frame_flip_does_not_emit():
    """Requirement 5: A -> M -> A must emit only A."""
    sm = PredictionSmoother()
    emitted = feed(sm, ["A"] * 5 + ["M"] + ["A"] * 5)
    assert emitted == ["A"], f"expected only A, got {emitted}"


def test_transient_two_frame_flip_does_not_emit():
    sm = PredictionSmoother()
    emitted = feed(sm, ["A"] * 5 + ["M", "M"] + ["A"] * 5)
    assert emitted == ["A"], f"expected only A, got {emitted}"


def test_sustained_change_does_emit():
    sm = PredictionSmoother()
    emitted = feed(sm, ["A"] * 5 + ["M"] * 5)
    assert emitted == ["A", "M"]


def test_oscillation_does_not_emit_every_flip():
    """Old logic emitted on every change; this must not."""
    sm = PredictionSmoother()
    emitted = feed(sm, ["A", "M"] * 15)
    assert len(emitted) <= 1, f"oscillation produced {len(emitted)} emissions: {emitted}"


def test_low_confidence_frames_do_not_vote():
    sm = PredictionSmoother()
    assert feed(sm, [("A", 0.59)] * 20) == []
    assert sm.accepted is None


def test_confidence_exactly_at_threshold_counts():
    sm = PredictionSmoother()
    assert feed(sm, [("A", 0.60)] * 5) == ["A"]


def test_mixed_confidence_blocks_weak_consensus():
    """4 strong + 1 weak reaches the 4-of-5 bar; 3 strong + 2 weak does not."""
    sm = PredictionSmoother()
    assert feed(sm, [("A", 0.9), ("A", 0.9), ("A", 0.9),
                     ("A", 0.9), ("A", 0.3)]) == ["A"]

    sm2 = PredictionSmoother()
    assert feed(sm2, [("A", 0.9), ("A", 0.9), ("A", 0.9),
                      ("A", 0.3), ("A", 0.3)]) == []


def test_no_hand_resets_after_configured_frames():
    sm = PredictionSmoother()
    feed(sm, ["A"] * 5)
    assert sm.accepted == "A"
    feed(sm, [None] * config.NO_HAND_RESET_FRAMES)
    assert sm.accepted is None
    assert sm.missing_frames == config.NO_HAND_RESET_FRAMES


def test_brief_dropout_does_not_reset():
    sm = PredictionSmoother()
    feed(sm, ["A"] * 5)
    feed(sm, [None] * (config.NO_HAND_RESET_FRAMES - 1))
    assert sm.accepted == "A"


def test_reacquisition_after_dropout_needs_full_window():
    """A flicker on reacquisition must not emit a spurious letter."""
    sm = PredictionSmoother()
    feed(sm, ["F"] * 5)
    feed(sm, [None] * 10)              # hand lost, state reset
    emitted = feed(sm, ["P"] + ["F"] * 6)   # one bad frame then steady F
    assert emitted == ["F"], f"expected only F, got {emitted}"


def test_detection_recovers_and_reemits_same_letter():
    sm = PredictionSmoother()
    assert feed(sm, ["A"] * 5) == ["A"]
    feed(sm, [None] * 10)
    assert feed(sm, ["A"] * 5) == ["A"]


def test_reset_clears_state():
    sm = PredictionSmoother()
    feed(sm, ["A"] * 5)
    sm.reset()
    assert sm.accepted is None
    assert sm.missing_frames == 0
    assert feed(sm, ["A"] * 4) == []


def test_custom_parameters_are_honoured():
    sm = PredictionSmoother(window=3, min_confidence=0.8, reset_after_missing=2)
    assert sm.required_votes == 2
    assert feed(sm, [("B", 0.85)] * 3) == ["B"]
    feed(sm, [None, None])
    assert sm.accepted is None


def test_update_returns_accepted_between_emissions():
    sm = PredictionSmoother()
    feed(sm, ["A"] * 5)
    accepted, emitted = sm.update("A", 0.99)
    assert accepted == "A" and emitted is False
