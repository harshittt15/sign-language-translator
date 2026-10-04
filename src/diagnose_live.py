"""
TEMPORARY diagnostic harness for the live inference path.

Exercises exactly the chain main.py uses:
    frame -> extract_landmarks -> normalize_landmarks -> model.predict

Reports per-frame feature statistics, detection consistency and prediction
stability, and compares live feature distributions against the training set
so domain mismatch is visible.

Nothing here modifies the model, scaler, thresholds or preprocessing.

Usage:
    python src/diagnose_live.py                 # continuous stream diagnostics
    python src/diagnose_live.py --per-sign      # one sign at a time, 30 frames each
    python src/diagnose_live.py --source DIR    # replay images instead of webcam

Delete this file once the live accuracy issue is resolved.
"""

import os
import sys
import argparse
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
from hand_tracker import HandTracker, center_square_crop
from model import SignLanguageModel

WINDOW = 10          # frames for rolling detection/stability stats
FRAMES_PER_SIGN = 30


def training_reference():
    """Per-letter feature stats from the training CSV, for domain comparison."""
    csv_path = os.path.join(config.PROCESSED_DATA_DIR, "asl_alphabet_landmarks.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    feat = [f"feature_{i}" for i in range(config.TOTAL_FEATURES)]
    ref = {"__all__": (df[feat].values.min(), df[feat].values.max(), df[feat].values.mean())}
    for letter, grp in df.groupby("label"):
        v = grp[feat].values
        ref[letter] = (v.min(), v.max(), v.mean())
    return ref


class ImageFolderSource:
    """Stand-in for CameraHandler so the harness can run without a webcam."""

    def __init__(self, directory):
        self.paths = sorted(Path(directory).rglob("*.jpg"))
        if not self.paths:
            raise RuntimeError(f"No .jpg images under {directory}")
        self.i = 0

    def read(self):
        if self.i >= len(self.paths):
            raise RuntimeError("end of image source")
        p = self.paths[self.i]
        self.i += 1
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"failed to read {p}")
        return img

    def release(self):
        pass


def feature_stats(vec):
    return float(vec.min()), float(vec.max()), float(vec.mean())


def run_continuous(source, tracker, model, ref, show):
    detect_hist = deque(maxlen=WINDOW)
    pred_hist = deque(maxlen=WINDOW)
    frame_no = 0

    print(f"\n{'frame':>6} {'hand':>5} {'pred':>5} {'conf':>6} "
          f"{'feat_min':>9} {'feat_max':>9} {'feat_mean':>10} "
          f"{'det':>7} {'stable':>8}")
    print("-" * 78)

    while True:
        try:
            frame = source.read()
        except RuntimeError:
            break
        frame_no += 1

        # Same aspect-ratio compatibility fix as main.py, so these
        # diagnostics measure the production path exactly.
        frame_square, _, _, _ = center_square_crop(frame)
        landmarks = tracker.extract_landmarks(frame_square)
        normalized = tracker.normalize_landmarks(landmarks)
        detect_hist.append(bool(normalized))

        pred, conf, stats = "-", 0.0, (float("nan"),) * 3
        if normalized:
            vec = np.asarray(normalized[0]).flatten()
            stats = feature_stats(vec)
            pred, conf = model.predict(normalized)
            pred_hist.append(pred)
        else:
            pred_hist.append(None)

        det_rate = sum(detect_hist) / len(detect_hist)
        valid = [p for p in pred_hist if p]
        stability = (Counter(valid).most_common(1)[0][1] / len(valid)) if valid else 0.0

        print(f"{frame_no:>6} {'Y' if normalized else 'n':>5} {str(pred):>5} "
              f"{conf:>6.3f} {stats[0]:>9.3f} {stats[1]:>9.3f} {stats[2]:>10.3f} "
              f"{det_rate:>6.0%} {stability:>7.0%}")

        if show:
            disp = tracker.draw_landmarks(frame_square, landmarks)
            cv2.putText(disp, f"{pred} {conf:.2f}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.imshow("live diagnostics - q to quit", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if ref:
        lo, hi, mu = ref["__all__"]
        print(f"\ntraining-set reference (all classes): "
              f"min {lo:.3f}  max {hi:.3f}  mean {mu:.3f}")


def run_per_sign(source, tracker, model, ref, show):
    print("\nOne sign at a time. Hold the sign steady; "
          f"{FRAMES_PER_SIGN} frames are captured per round.")
    print("Enter the letter you are signing, or 'q' to finish.\n")

    while True:
        expected = input("Sign you will make (A-Z, or q): ").strip().upper()
        if expected == "Q":
            break
        if len(expected) != 1 or not expected.isalpha():
            print("  Enter a single letter.\n")
            continue

        input(f"  Hold '{expected}' steady, then press Enter...")

        preds, confs, mins, maxs, means = [], [], [], [], []
        detected = 0
        for _ in range(FRAMES_PER_SIGN):
            try:
                frame = source.read()
            except RuntimeError:
                break
            # Same aspect-ratio compatibility fix as main.py, so these
            # diagnostics measure the production path exactly.
            frame_square, _, _, _ = center_square_crop(frame)
            landmarks = tracker.extract_landmarks(frame_square)
            normalized = tracker.normalize_landmarks(landmarks)
            if not normalized:
                preds.append(None)
                continue
            detected += 1
            vec = np.asarray(normalized[0]).flatten()
            lo, hi, mu = feature_stats(vec)
            mins.append(lo); maxs.append(hi); means.append(mu)
            p, c = model.predict(normalized)
            preds.append(p); confs.append(c)

            if show:
                disp = tracker.draw_landmarks(frame_square, landmarks)
                cv2.putText(disp, f"{p} {c:.2f}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                cv2.imshow("per-sign diagnostics", disp)
                cv2.waitKey(1)

        total = len(preds)
        if total == 0:
            print("  No frames captured.\n")
            continue

        print(f"\n  === '{expected}' over {total} frames ===")
        print(f"  hand detected      : {detected}/{total} ({detected/total:.0%})")

        valid = [p for p in preds if p]
        if not valid:
            print("  no hand detected in any frame\n")
            continue

        dist = Counter(valid).most_common()
        top, top_n = dist[0]
        print("  prediction distribution:")
        for letter, n in dist:
            mark = " <-- expected" if letter == expected else ""
            print(f"      {letter}: {n:>3}/{len(valid)}  {n/len(valid):>4.0%}  "
                  f"{'#' * int(20 * n / len(valid))}{mark}")
        print(f"  stability (modal)  : {top_n}/{len(valid)} ({top_n/len(valid):.0%}) -> '{top}'")
        correct = sum(1 for p in valid if p == expected)
        print(f"  matches expected   : {correct}/{len(valid)} ({correct/len(valid):.0%})")
        print(f"  confidence         : mean {np.mean(confs):.3f}  "
              f"min {np.min(confs):.3f}  max {np.max(confs):.3f}")
        print(f"  live feature stats : min {np.mean(mins):.3f}  "
              f"max {np.mean(maxs):.3f}  mean {np.mean(means):.3f}")
        if ref and expected in ref:
            lo, hi, mu = ref[expected]
            print(f"  training '{expected}' stats  : min {lo:.3f}  max {hi:.3f}  mean {mu:.3f}")
            drift = abs(np.mean(means) - mu)
            print(f"  mean drift         : {drift:.3f}"
                  f"{'   <-- large, suggests domain mismatch' if drift > 0.15 else ''}")
        print()


def run_hold(source, tracker, model, ref, show, seconds, expected, dump_path=None):
    """
    Hold one sign continuously. Replicates main.py's acceptance/debounce
    logic exactly so emitted-prediction counts are directly comparable.
    """
    import time

    preds, confs = [], []
    dumped = []
    detected = frames = 0
    emitted = []                 # what main.py would have spoken/appended
    recognition_history = []     # mirrors main.py's variable
    accepted_above_thresh = 0

    print(f"\nHold '{expected}' steady for {seconds}s. Starting in 3s...")
    time.sleep(3)
    print("GO\n")

    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            frame = source.read()
        except RuntimeError:
            break
        frames += 1

        # Same aspect-ratio compatibility fix as main.py, so these
        # diagnostics measure the production path exactly.
        frame_square, _, _, _ = center_square_crop(frame)
        landmarks = tracker.extract_landmarks(frame_square)
        normalized = tracker.normalize_landmarks(landmarks)

        if not normalized:
            preds.append(None)
            if show:
                cv2.imshow("hold test", frame_square)
                cv2.waitKey(1)
            continue

        detected += 1
        pred, conf = model.predict(normalized)
        preds.append(pred)
        confs.append(conf)
        if dump_path:
            vec = np.asarray(normalized[0]).flatten()
            dumped.append([expected, frames, pred, conf] + vec.tolist())

        # --- replicate main.py:93-97 exactly ---
        if conf >= config.CONFIDENCE_THRESHOLD:
            accepted_above_thresh += 1
            if not recognition_history or recognition_history[-1] != pred:
                recognition_history.append(pred)
                emitted.append((frames, pred, conf))

        if show:
            disp = tracker.draw_landmarks(frame_square, landmarks)
            cv2.putText(disp, f"{pred} {conf:.2f}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.imshow("hold test", disp)
            cv2.waitKey(1)

    elapsed = time.time() - t0
    print("=" * 66)
    print(f"HOLD TEST: '{expected}'  |  {frames} frames in {elapsed:.1f}s "
          f"({frames/elapsed:.1f} fps)")
    print("=" * 66)

    print(f"\nhand detection : {detected}/{frames} ({detected/max(frames,1):.0%})")
    gaps = sum(1 for i in range(1, len(preds))
               if preds[i] is None and preds[i-1] is not None)
    print(f"dropout events : {gaps}  (hand present -> absent transitions)")

    valid = [p for p in preds if p]
    if not valid:
        print("\nno hand detected at any point")
        return

    print(f"\nprediction distribution ({len(valid)} frames with a hand):")
    for letter, n in Counter(valid).most_common():
        mark = " <-- expected" if letter == expected else ""
        print(f"    {letter}: {n:>4}/{len(valid)}  {n/len(valid):>4.0%}  "
              f"{'#' * int(30 * n / len(valid))}{mark}")

    arr = np.array(confs)
    print("\nconfidence distribution:")
    print(f"    mean {arr.mean():.3f}  median {np.median(arr):.3f}  "
          f"min {arr.min():.3f}  max {arr.max():.3f}")
    for lo, hi in [(0.0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]:
        n = int(((arr >= lo) & (arr < hi)).sum())
        tag = "  (below threshold, rejected)" if hi <= config.CONFIDENCE_THRESHOLD else ""
        print(f"    [{lo:.1f},{hi:.1f}) {n:>4}  {'#' * int(30*n/len(arr))}{tag}")

    print(f"\nacceptance / emission (main.py logic, "
          f"threshold={config.CONFIDENCE_THRESHOLD}):")
    print(f"    frames passing confidence gate : {accepted_above_thresh}")
    print(f"    predictions EMITTED            : {len(emitted)}")
    print("    ideal for one held sign        : 1")
    if emitted:
        print(f"    emitted sequence: "
              f"{' -> '.join(p for _, p, _ in emitted[:25])}"
              f"{' ...' if len(emitted) > 25 else ''}")
        wrong = sum(1 for _, p, _ in emitted if p != expected)
        print(f"    spurious emissions (!= '{expected}'): {wrong}/{len(emitted)}")

    if ref and expected in ref:
        lo, hi, mu = ref[expected]
        print(f"\ntraining '{expected}' feature mean: {mu:.3f}")

    if dump_path and dumped:
        cols = ["expected", "frame", "predicted", "confidence"] + \
               [f"feature_{i}" for i in range(config.TOTAL_FEATURES)]
        new = pd.DataFrame(dumped, columns=cols)
        if os.path.exists(dump_path):
            new = pd.concat([pd.read_csv(dump_path), new], ignore_index=True)
        new.to_csv(dump_path, index=False)
        print(f"\n✓ {len(dumped)} feature vectors appended to {dump_path} "
              f"({len(new)} total rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-sign", action="store_true",
                    help="one sign at a time, 30 frames each")
    ap.add_argument("--source", default=None,
                    help="directory of .jpg images to replay instead of the webcam")
    ap.add_argument("--no-window", action="store_true", help="suppress the video window")
    ap.add_argument("--hold", metavar="LETTER", default=None,
                    help="hold one sign continuously and measure emissions")
    ap.add_argument("--seconds", type=float, default=8.0,
                    help="duration for --hold (default 8)")
    ap.add_argument("--dump", metavar="CSV", default=None,
                    help="with --hold, write every frame's 63 normalized features to CSV")
    args = ap.parse_args()

    tracker = HandTracker()
    model = SignLanguageModel()
    model.load_model()
    model.load_metadata()
    ref = training_reference()

    if args.source:
        source = ImageFolderSource(args.source)
        show = False
    else:
        from camera import CameraHandler
        source = CameraHandler()
        show = not args.no_window

    try:
        if args.hold:
            run_hold(source, tracker, model, ref, show,
                     args.seconds, args.hold.strip().upper(), args.dump)
        elif args.per_sign:
            run_per_sign(source, tracker, model, ref, show)
        else:
            run_continuous(source, tracker, model, ref, show)
    finally:
        source.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
