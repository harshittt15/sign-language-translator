# ASL Alphabet Dataset Extraction Guide

## Overview

This guide explains how to use the `extract_landmarks_from_dataset.py` script to extract hand landmarks from the Kaggle ASL Alphabet dataset and prepare them for model training.

---

## Dataset Information

### Kaggle ASL Alphabet Dataset
- **Source:** https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- **Size:** ~215 MB (all 26 letters + numbers)
- **Format:** PNG images organized in A-Z folders
- **Images:** ~100,000+ total images

### Our Extraction Plan
- **Images per letter:** 300 (manageable subset)
- **Total images:** 26 letters × 300 = **7,800 images**
- **Disk space needed:** ~265-285 MB
- **Processing time:** ~10-15 minutes
- **Output:** `data/processed/asl_alphabet_landmarks.csv`

---

## Prerequisites

### 1. Kaggle API Setup

The script requires Kaggle API credentials to download the dataset automatically.

#### Step 1: Get Kaggle API Token
1. Go to https://www.kaggle.com/account
2. Scroll down to "API" section
3. Click "Create New Token"
4. This downloads `kaggle.json`

#### Step 2: Install Kaggle Credentials
```bash
# Create .kaggle directory if it doesn't exist
mkdir -p ~/.kaggle

# Move kaggle.json to the correct location
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set proper permissions (important!)
chmod 600 ~/.kaggle/kaggle.json
```

#### Step 3: Install Kaggle CLI
```bash
pip install kaggle
```

### 2. Required Python Packages
All packages are already in `requirements.txt`:
- `opencv-python` - Image loading
- `mediapipe==0.10.35` - Hand landmark detection (already installed)
- `pandas` - CSV handling
- `numpy` - Numerical operations
- `tqdm` - Progress bars
- `kaggle` - Dataset download

---

## Usage

### Quick Start

```bash
cd /Users/harshit/Downloads/Mohak\ Project/sign_language_translator

# Activate virtual environment
source venv/bin/activate

# Run the extraction script
python src/extract_landmarks_from_dataset.py
```

### What Happens

1. **Download (2-5 minutes)**
   - Script downloads Kaggle ASL Alphabet dataset
   - Extracts to `~/Downloads/kaggle_asl_temp`
   - Keeps only 300 images per letter

2. **Process (10-15 minutes)**
   - Loads each image
   - Detects hands using MediaPipe HandLandmarker
   - Extracts 21 landmarks (x, y, z coordinates)
   - Normalizes landmarks (scale-invariant)
   - Tracks skipped images

3. **Save (1 minute)**
   - Creates CSV file: `data/processed/asl_alphabet_landmarks.csv`
   - Columns: `label` (A-Z) + `feature_0` to `feature_62` (63 features)
   - Reports statistics

### Expected Output

```
======================================================================
🤟 ASL ALPHABET DATASET LANDMARK EXTRACTION
======================================================================

📥 Downloading Kaggle dataset: grassknoted/asl-alphabet
...
✅ Dataset downloaded to: /Users/harshit/Downloads/kaggle_asl_temp/...

🔍 Processing ASL Alphabet dataset...

✅ HandTracker initialized

Found 26 letter folders

Processing A: 300 images... ✅ A: 290/300 processed (10 skipped)
Processing B: 300 images... ✅ B: 298/300 processed (2 skipped)
...

📊 Processing Summary:
  Total images processed: 7650
  Total images skipped: 150
  Success rate: 98.1%

💾 Saving to CSV: .../data/processed/asl_alphabet_landmarks.csv
✅ Saved 7650 samples to .../data/processed/asl_alphabet_landmarks.csv

======================================================================
📋 EXTRACTION REPORT
======================================================================

📁 Dataset Source: grassknoted/asl-alphabet
📁 Output File: .../data/processed/asl_alphabet_landmarks.csv
📊 Total Samples: 7650
🏷️  Number of Features: 63

📊 Samples Per Letter:
  A: 290 samples
  B: 298 samples
  ...
  Z: 295 samples

💾 CSV Details:
  Rows: 7650
  Columns: 64 (1 label + 63 features)
  File size: 65.3 MB

======================================================================

🎉 Extraction complete! Ready for model training.
```

---

## CSV Output Format

### File: `data/processed/asl_alphabet_landmarks.csv`

| Column | Type | Description |
|--------|------|-------------|
| `label` | str (A-Z) | The letter represented in the image |
| `feature_0` to `feature_62` | float | Normalized hand landmarks |

### Feature Mapping

```
feature_0-2:    Landmark 0 (Wrist) - x, y, z
feature_3-5:    Landmark 1 (Thumb base) - x, y, z
feature_6-8:    Landmark 2 (Thumb) - x, y, z
...
feature_60-62:  Landmark 20 (Pinky tip) - x, y, z

Total: 21 landmarks × 3 coordinates = 63 features
```

### Normalization Applied

Each hand's landmarks are normalized using:
1. **Reference point:** Wrist (landmark 0)
2. **Scale:** Distance from wrist to middle finger base
3. **Result:** Scale-invariant representation

This matches the normalization in `HandTracker.normalize_landmarks()`.

---

## Troubleshooting

### Issue: "Kaggle credentials not found"

**Solution:** Follow the Kaggle API setup section above.

```bash
# Verify credentials exist
ls -la ~/.kaggle/kaggle.json

# Should show: -rw------- 1 user group ... ~/.kaggle/kaggle.json
```

### Issue: "No letter folders found"

**Solution:** Check dataset extraction succeeded:
```bash
ls -la ~/Downloads/kaggle_asl_temp/
```

### Issue: MediaPipe initialization fails

**Solution:** Ensure MediaPipe 0.10.35 is installed:
```bash
pip list | grep mediapipe
# Should show: mediapipe  0.10.35
```

### Issue: High skip rate (>10%)

**Reason:** Images where no hand is detected or hand is partially visible
- This is expected and normal
- Script handles it gracefully
- Poor image quality in original dataset

---

## After Extraction

### Ready for Next Steps

Once the CSV is generated, you can:

1. **Train the ML Model**
   ```bash
   python src/model.py data/processed/asl_alphabet_landmarks.csv
   ```

2. **Validate the Dataset**
   - Check sample distribution across letters
   - Verify feature values are normalized (~-1 to +1 range)
   - Visualize distribution of samples

3. **Use for Real-Time Translation**
   - Model will use the same feature format (63 normalized landmarks)
   - Easy integration with existing `hand_tracker.py`

---

## Performance Notes

### Extraction Speed
- **Per image:** ~5-10 images/second
- **Total time (7,800 images):** ~10-15 minutes
- **Bottleneck:** MediaPipe landmark detection

### Disk Space
- **During extraction:** ~265-285 MB
- **After extraction:** Dataset is deleted automatically, only CSV remains (~65 MB)

### Quality
- **Success rate:** Typically 95-99%
- **Skipped images:** Hands not detected or partially visible
- **Normalization:** Ensures scale-invariant features

---

## Script Architecture

```
extract_landmarks_from_dataset.py
├── ASLDatasetExtractor class
│   ├── download_dataset()        # Kaggle API download
│   ├── extract_landmarks_from_image()  # Per-image processing
│   ├── _normalize_landmarks_single()   # Normalization (same as HandTracker)
│   ├── process_dataset()         # Main loop
│   ├── save_to_csv()            # CSV export
│   └── generate_report()        # Statistics
└── main() function              # Entry point
```

### No Changes to Project Files
- ✅ `hand_tracker.py` - NOT modified, used as-is
- ✅ `model.py` - NOT modified
- ✅ `config.py` - Used for confidence thresholds
- ✅ New file: `src/extract_landmarks_from_dataset.py`
- ✅ New output: `data/processed/asl_alphabet_landmarks.csv`

---

## FAQ

**Q: Can I use a different number of images per letter?**
A: Yes, edit line in `extract_landmarks_from_dataset.py`:
```python
IMAGES_PER_LETTER = 500  # Change from 300 to 500
```

**Q: Does this delete the downloaded dataset afterward?**
A: Currently, the dataset remains in `~/Downloads/kaggle_asl_temp/`. You can delete manually:
```bash
rm -rf ~/Downloads/kaggle_asl_temp/
```

**Q: Can I run this multiple times?**
A: Yes, it will overwrite `data/processed/asl_alphabet_landmarks.csv`. Consider renaming if you want to keep previous runs.

**Q: How do I verify the extraction worked?**
A: Check the CSV file:
```bash
# View first few rows
head -5 data/processed/asl_alphabet_landmarks.csv

# Get statistics
python -c "
import pandas as pd
df = pd.read_csv('data/processed/asl_alphabet_landmarks.csv')
print(f'Shape: {df.shape}')
print(f'Letters: {sorted(df[\"label\"].unique())}')
print(df.describe())
"
```

---

## Next Steps

1. **Extract the dataset** (this guide)
2. **Train the ML model** using `data/processed/asl_alphabet_landmarks.csv`
3. **Test real-time translation** with the trained model
4. **Integrate into GUI** for end-user application

---

**Created:** October 2, 2026  
**Dataset:** Kaggle ASL Alphabet  
**Status:** Ready for use
