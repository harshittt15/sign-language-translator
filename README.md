# Real-Time Sign Language Translator

Translates static ASL alphabet signs (A to Z) into text in real time. Hand
landmarks are detected with MediaPipe, normalized, and classified by a Random
Forest. Ships as both a desktop application and a web platform, which share a
single inference pipeline.

## Results

All figures below come from the held-out split of the **public ASL Alphabet
image dataset**. They are offline dataset measurements, not live webcam
measurements.

| Metric | Value |
|---|---|
| Processed landmark samples | 5,712 |
| Classes | 26 (A to Z) |
| Features per sample | 63 (21 landmarks x 3 coordinates) |
| Offline held-out test accuracy | **97.9%** (1,143 test samples) |
| Macro F1 | **0.97** |
| Classifier | Random Forest, 200 trees (scikit-learn) |
| Landmark model | MediaPipe 0.10.35, Tasks API HandLandmarker |

Per-class results are in `models/confusion_matrix.csv`.

### Offline evaluation vs live webcam testing

These are two different things and this project keeps them separate.

**Offline evaluation** is the 97.9% above: a stratified held-out split of the
public dataset images, which were captured under consistent lighting at a fixed
distance.

**Live webcam testing** has been done informally during development, but **no
webcam accuracy metric has been established**. The 97.9% figure should not be
read as live performance. Live accuracy is lower, and quantifying it would need
a labelled webcam test set that does not currently exist.

## Known limitations

- **M and N confusion.** The two signs differ only by whether the thumb tucks
  under two fingers or three, and the thumb is occluded in both. The landmark
  model infers its position rather than observing it, so the two letters sit
  close together in feature space. M versus N is the weakest pair in the model,
  and under webcam conditions N is frequently read as M.
- **Detection depends on hand size in frame.** MediaPipe needs the hand to
  occupy a reasonable share of the frame. Extraction detection rates varied
  widely by letter on the source dataset, from 29% (N) to 88% (F).
- **Domain gap.** The training images differ from a typical webcam in lighting,
  distance and framing, so live behaviour differs from the offline result.
- **Static letters only.** J and Z involve motion in real ASL but are treated
  here as fixed poses. Words and continuous signing are out of scope.
- **Single hand used for prediction.** If two hands are detected, the first is
  classified.

## Requirements

- Python 3.10 or newer (developed on 3.14)
- A webcam
- macOS, Windows or Linux

## Installation

```bash
cd sign_language_translator
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Verify:

```bash
python -c "import cv2, mediapipe, sklearn; print('dependencies OK')"
```

The trained model, scaler and extracted landmark dataset are committed, so the
project runs immediately after a clone. No training step is required.

## Running

### Desktop application

```bash
PYTHONPATH=. python3 src/main.py
```

Keys: `q` quit, `a` toggle audio, `c` clear history, `d` toggle per-frame
diagnostics.

### Web platform

```bash
PYTHONPATH=src python3 -m uvicorn web_api.app:app --port 8000
```

Then open http://127.0.0.1:8000. The site has a landing page, a live translator
and a "how it works" page. Webcam frames are posted to the local backend, used
for a single prediction and discarded in memory. No frame is written to disk or
logged.

### Tests

```bash
python -m pytest tests/ -q
```

## How a frame is processed

The desktop app and the web API both call `SignLanguageInference.process_frame`,
so they cannot drift apart.

1. **Capture** a webcam frame.
2. **Square crop** (`center_square_crop`). MediaPipe scales x by frame width and
   y by frame height, so a 16:9 frame stretches the hand horizontally relative
   to the 200x200 training images. Measured drift without this step was 0.155 on
   x against 0.021 on y, which displaces every feature. This step is required
   for accuracy, not cosmetic.
3. **Landmarks.** MediaPipe HandLandmarker returns 21 joints, each with x, y and
   z, giving 63 numbers.
4. **Normalization.** Landmarks are re-centred on the wrist and divided by the
   wrist to middle-knuckle distance, making features invariant to position and
   distance.
5. **Classification.** Features are standardised and passed to the Random
   Forest, which returns a letter and the share of trees that agreed.
6. **Temporal smoothing.** A letter is committed only when at least 4 of the
   last 5 frames agree and clear the confidence threshold. Losing the hand for
   several frames resets the window, so reacquiring it cannot inject a spurious
   letter.
7. **Output.** Committed letters append to the transcript and, optionally, are
   spoken aloud.

## Project structure

```
sign_language_translator/
├── config.py                       # thresholds, paths, smoothing parameters
├── requirements.txt
├── data/processed/
│   └── asl_alphabet_landmarks.csv  # 5,712 extracted samples (committed)
├── models/
│   ├── sign_language_model.pkl     # trained Random Forest (committed)
│   ├── feature_scaler.pkl
│   ├── metadata.json               # label encoder/decoder
│   ├── confusion_matrix.csv        # per-class offline results
│   └── hand_landmarker.task        # MediaPipe model
├── src/
│   ├── inference.py                # shared pipeline, single source of truth
│   ├── hand_tracker.py             # MediaPipe wrapper, crop, normalization
│   ├── model.py                    # training and inference
│   ├── prediction_smoother.py      # temporal debouncing
│   ├── camera.py
│   ├── main.py                     # desktop application
│   ├── gui.py
│   ├── tts.py
│   ├── data_collector.py           # manual webcam collection
│   ├── collect_webcam_samples.py   # guided per-letter collection
│   ├── extract_landmarks_from_dataset.py
│   ├── diagnose_live.py            # live diagnostics harness
│   └── web_api/                    # FastAPI backend
├── web/                            # static frontend
└── tests/                          # 53 tests
```

## Configuration

Edit `config.py`:

| Setting | Default | Effect |
|---|---|---|
| `MIN_DETECTION_CONFIDENCE` | 0.7 | MediaPipe hand detection threshold |
| `MAX_NUM_HANDS` | 2 | Hands MediaPipe will look for |
| `CONFIDENCE_THRESHOLD` | 0.6 | Minimum confidence for a frame to vote |
| `PREDICTION_SMOOTHING` | 5 | Rolling window size |
| `NO_HAND_RESET_FRAMES` | 5 | No-hand frames before the window resets |

## Rebuilding the dataset (optional)

The extracted features are committed, so this is only needed to regenerate them
or to change the sample count.

The source images are the [Kaggle ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet),
about 215 MB. They are not committed here.

**1. Set up Kaggle API access.** Create an API token at
https://www.kaggle.com/account (Settings, then API, then "Create New Token").
This downloads `kaggle.json`.

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
pip install kaggle
```

The newer `~/.kaggle/access_token` method also works; the script defers to
whatever the Kaggle CLI is configured with.

**2. Run the extraction.**

```bash
python src/extract_landmarks_from_dataset.py
```

It iterates the A to Z folders, takes up to 300 images per letter, runs each
through the same landmark and normalization pipeline used at inference time, and
writes `data/processed/asl_alphabet_landmarks.csv`. Images where no hand is
detected are skipped and reported per class.

Detection is not uniform across letters, which is why 7,800 attempted images
yielded 5,712 usable samples.

**3. Output format.** One row per sample: a `label` column (A to Z) followed by
`feature_0` through `feature_62`. Features 0 to 2 are the wrist and are always
`[0, 0, 0]` by construction of the normalization.

## Diagnostics

`src/diagnose_live.py` runs the production pipeline with instrumentation:

```bash
python src/diagnose_live.py                      # per-frame table
python src/diagnose_live.py --per-sign           # one sign at a time
python src/diagnose_live.py --hold A --seconds 8 # hold one sign, measure
```

It reports detection rate, prediction distribution, confidence distribution,
prediction stability and how many letters would actually be emitted, and can
dump the normalized feature vectors to CSV with `--dump`.

## Acknowledgments

- **MediaPipe** (Google) for hand landmark detection
- **scikit-learn** for the Random Forest classifier
- **OpenCV** for camera capture and image handling
- **FastAPI** for the web backend
- **pyttsx3** for desktop text to speech
- The public **ASL Alphabet** dataset on Kaggle

## License

MIT.
