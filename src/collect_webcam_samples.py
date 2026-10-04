"""
Collect webcam training samples for specific signs.

Runs the EXACT production pipeline so collected features are directly
comparable with what inference produces:

    camera -> center_square_crop -> extract_landmarks -> normalize_landmarks
           -> 63 normalized features

Samples are written to their own CSV. The public ASL Alphabet dataset
(data/processed/asl_alphabet_landmarks.csv) is never modified.

Each capture round is recorded with a round id so that train/test splits can
be grouped by round. Consecutive video frames are highly correlated, so a
random split would leak near-duplicates across the split and inflate scores.

Usage:
    python src/collect_webcam_samples.py                 # collect M and N
    python src/collect_webcam_samples.py --letters M     # one letter
    python src/collect_webcam_samples.py --inspect       # counts + plots
"""

import os
import sys
import time
import argparse

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
from hand_tracker import HandTracker, center_square_crop

OUTPUT_CSV = os.path.join(config.PROCESSED_DATA_DIR, "webcam_samples.csv")
PUBLIC_CSV = os.path.join(config.PROCESSED_DATA_DIR, "asl_alphabet_landmarks.csv")

# Each round deliberately varies one factor so the collected set spans the
# conditions the model will actually see.
ROUNDS = [
    ("centred",        "Hold the sign centred, arm's length from the camera"),
    ("close",          "Move noticeably CLOSER to the camera"),
    ("far",            "Move noticeably FURTHER from the camera"),
    ("rotate_left",    "Rotate your wrist slightly LEFT (~20 degrees)"),
    ("rotate_right",   "Rotate your wrist slightly RIGHT (~20 degrees)"),
    ("shift_left",     "Move your hand to the LEFT side of the frame"),
    ("shift_right",    "Move your hand to the RIGHT side of the frame"),
    ("tilt",           "Tilt your hand slightly toward and away from the camera"),
    ("natural",        "Sign naturally, drifting slowly as you would in use"),
]

FRAMES_PER_ROUND = 40
FRAME_SKIP = 2          # keep every Nth detected frame to reduce correlation


def collect(letters, frames_per_round, out_csv):
    from camera import CameraHandler

    tracker = HandTracker()
    camera = CameraHandler()
    rows = []

    print("\n" + "=" * 66)
    print("WEBCAM SAMPLE COLLECTION")
    print("=" * 66)
    print(f"letters          : {', '.join(letters)}")
    print(f"rounds per letter: {len(ROUNDS)}")
    print(f"frames per round : {frames_per_round}")
    print(f"target per letter: {len(ROUNDS) * frames_per_round}")
    print(f"output           : {out_csv}")
    print("\nThe public dataset is not touched. Press Ctrl-C to abort.\n")

    try:
        for letter in letters:
            print("\n" + "#" * 66)
            print(f"#  LETTER: {letter}")
            print("#" * 66)

            for round_idx, (round_name, instruction) in enumerate(ROUNDS):
                print(f"\n[{letter} {round_idx + 1}/{len(ROUNDS)}] {round_name}")
                print(f"  {instruction}")
                input("  Press Enter when ready, then hold steady...")

                for n in (3, 2, 1):
                    print(f"  {n}...", end="", flush=True)
                    time.sleep(1)
                print(" GO")

                captured = skipped = 0
                no_hand = 0
                while captured < frames_per_round:
                    frame = camera.read()
                    frame_square, cy, cx, side = center_square_crop(frame)
                    landmarks = tracker.extract_landmarks(frame_square)
                    normalized = tracker.normalize_landmarks(landmarks)

                    disp = frame.copy()
                    if normalized:
                        disp[cy:cy + side, cx:cx + side] = \
                            tracker.draw_landmarks(frame_square, landmarks)
                        skipped += 1
                        if skipped % FRAME_SKIP == 0:
                            vec = np.asarray(normalized[0]).flatten()
                            rows.append([letter, round_idx, round_name,
                                         captured] + vec.tolist())
                            captured += 1
                    else:
                        no_hand += 1

                    cv2.putText(disp, f"{letter} / {round_name}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    cv2.putText(disp, f"{captured}/{frames_per_round}", (10, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 255, 0) if normalized else (0, 0, 255), 2)
                    cv2.imshow("collection - q aborts round", disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("  round aborted")
                        break

                total = captured + no_hand
                rate = captured / total if total else 0
                print(f"  captured {captured}  (no-hand frames: {no_hand}, "
                      f"detection {rate:.0%})")
    except KeyboardInterrupt:
        print("\n\nAborted by user - saving what was collected so far.")
    finally:
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()

    if not rows:
        print("\nNo samples collected.")
        return None

    cols = ["label", "round", "round_name", "frame"] + \
           [f"feature_{i}" for i in range(config.TOTAL_FEATURES)]
    new = pd.DataFrame(rows, columns=cols)
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        new["round"] = new["round"] + prev["round"].max() + 1      # keep groups distinct
        new = pd.concat([prev, new], ignore_index=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    new.to_csv(out_csv, index=False)

    print(f"\n✓ saved {len(rows)} new samples -> {out_csv} ({len(new)} total)")
    return out_csv


def inspect(out_csv):
    if not os.path.exists(out_csv):
        print(f"No collected samples at {out_csv}")
        return

    df = pd.read_csv(out_csv)
    feat = [f"feature_{i}" for i in range(config.TOTAL_FEATURES)]

    print("\n" + "=" * 66)
    print(f"COLLECTED WEBCAM SAMPLES  ({out_csv})")
    print("=" * 66)
    print(f"\ntotal rows: {len(df)}   features: {len(feat)}")

    print("\nper letter:")
    for letter, grp in df.groupby("label"):
        print(f"  {letter}: {len(grp):>4} samples across "
              f"{grp['round'].nunique()} rounds")

    print("\nper letter x round:")
    pivot = df.pivot_table(index="round_name", columns="label",
                           values="frame", aggfunc="count", fill_value=0)
    print(pivot.to_string())

    # how far are webcam samples from the public-dataset distribution?
    if os.path.exists(PUBLIC_CSV):
        pub = pd.read_csv(PUBLIC_CSV)
        print("\nmean |z| of webcam samples vs PUBLIC distribution (same letter):")
        for letter, grp in df.groupby("label"):
            ref = pub[pub.label == letter][feat].values
            if len(ref) == 0:
                continue
            sd = ref.std(0)
            # Features 0-2 are the wrist, forced to exactly [0,0,0] by the
            # normalization, so their std is 0. Including them would divide
            # by ~0 and swamp the metric; they carry no information anyway.
            informative = sd > 1e-8
            z = np.abs((grp[feat].values[:, informative] - ref.mean(0)[informative])
                       / sd[informative]).mean()
            print(f"  {letter}: {z:.2f}   (1.0 = one public-data std away, "
                  f"{int(informative.sum())}/{len(feat)} informative features)")

    # separability of the collected letters among themselves
    letters = sorted(df.label.unique())
    if len(letters) == 2:
        a, b = letters
        A = df[df.label == a][feat].values
        B = df[df.label == b][feat].values
        pooled = np.sqrt((A.var(0) + B.var(0)) / 2)
        d = np.abs(A.mean(0) - B.mean(0)) / np.where(pooled == 0, 1e-9, pooled)
        print(f"\n{a}-vs-{b} separation in the WEBCAM samples:")
        print(f"  max Cohen's d = {d.max():.2f}   features |d|>0.8: {int((d>0.8).sum())}/63")
        if os.path.exists(PUBLIC_CSV):
            pub = pd.read_csv(PUBLIC_CSV)
            PA = pub[pub.label == a][feat].values
            PB = pub[pub.label == b][feat].values
            pp = np.sqrt((PA.var(0) + PB.var(0)) / 2)
            dp = np.abs(PA.mean(0) - PB.mean(0)) / np.where(pp == 0, 1e-9, pp)
            print(f"  (public dataset, same pair: max d = {dp.max():.2f})")

    _plot(df, feat, out_csv)


def _plot(df, feat, out_csv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    out_png = out_csv.replace(".csv", "_distribution.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    if os.path.exists(PUBLIC_CSV):
        pub = pd.read_csv(PUBLIC_CSV)
        letters = sorted(df.label.unique())
        pub = pub[pub.label.isin(letters)]
        pca = PCA(n_components=2).fit(pub[feat].values)
        for letter in letters:
            p = pca.transform(pub[pub.label == letter][feat].values)
            axes[0].scatter(p[:, 0], p[:, 1], s=14, alpha=.5, label=f"public {letter}")
            w = pca.transform(df[df.label == letter][feat].values)
            axes[0].scatter(w[:, 0], w[:, 1], s=14, alpha=.5, marker="x",
                            label=f"webcam {letter}")
        axes[0].set_title("PCA (fit on public data): public vs webcam")
        axes[0].legend(fontsize=8)
        axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")

    pca2 = PCA(n_components=2).fit(df[feat].values)
    for letter in sorted(df.label.unique()):
        w = pca2.transform(df[df.label == letter][feat].values)
        axes[1].scatter(w[:, 0], w[:, 1], s=16, alpha=.6, label=f"webcam {letter}")
    axes[1].set_title("PCA (fit on webcam data): are the letters separable?")
    axes[1].legend(fontsize=8)
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")

    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    print(f"\n✓ distribution plot -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letters", default="M,N",
                    help="comma-separated letters to collect (default M,N)")
    ap.add_argument("--frames", type=int, default=FRAMES_PER_ROUND,
                    help=f"frames per round (default {FRAMES_PER_ROUND})")
    ap.add_argument("--out", default=OUTPUT_CSV, help="output CSV")
    ap.add_argument("--inspect", action="store_true",
                    help="report counts and plot distributions, collect nothing")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.out)
        return

    letters = [c.strip().upper() for c in args.letters.split(",") if c.strip()]
    if collect(letters, args.frames, args.out):
        inspect(args.out)


if __name__ == "__main__":
    main()
