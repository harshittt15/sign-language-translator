# 🚀 Quick Start: ASL Dataset Extraction

## TL;DR - Just the essentials

### 1️⃣ Setup Kaggle API (Once)
```bash
# Visit: https://www.kaggle.com/account → API → Create Token
# Save to: ~/.kaggle/kaggle.json
# Run: chmod 600 ~/.kaggle/kaggle.json && pip install kaggle
```

### 2️⃣ Install Dependencies
```bash
source venv/bin/activate
pip install tqdm kaggle
```

### 3️⃣ Run Extraction
```bash
python src/extract_landmarks_from_dataset.py
# Wait 10-15 minutes ☕
```

### 4️⃣ Verify Output
```bash
# CSV created with 7,650 samples × 64 columns (1 label + 63 features)
ls -lh data/processed/asl_alphabet_landmarks.csv
```

---

## 📊 Key Numbers

| What | Value |
|------|-------|
| **Images processed** | 7,800 (300 × 26 letters) |
| **Expected success** | ~98% (7,650 samples) |
| **Processing time** | 10-15 minutes |
| **Disk space needed** | 265-285 MB |
| **CSV output size** | 65 MB |
| **Features per sample** | 63 (21 landmarks × 3 coords) |
| **Output file** | `data/processed/asl_alphabet_landmarks.csv` |

---

## 🎯 What You'll Get

```
data/processed/asl_alphabet_landmarks.csv
├── 7,650 rows (samples)
├── 64 columns (1 label + 63 features)
├── Ready for scikit-learn training
├── Scale-invariant normalized features
└── No modifications to existing code
```

---

## 📋 Files Created

| File | Purpose |
|------|---------|
| `src/extract_landmarks_from_dataset.py` | Main extraction script |
| `DATASET_EXTRACTION_GUIDE.md` | Full documentation |
| `EXTRACTION_PLAN_SUMMARY.md` | Technical details |
| `QUICK_START_EXTRACTION.md` | This file |
| `requirements.txt` | Updated with tqdm, kaggle |

---

## ⚙️ How It Works

```
Kaggle Dataset
    ↓ (Download)
A-Z Image Folders
    ↓ (Load + Detect)
MediaPipe HandLandmarker
    ↓ (Extract 21 landmarks)
Raw Features (63 values)
    ↓ (Normalize)
Scale-Invariant Features
    ↓ (Save)
CSV File
```

---

## ✅ Checklist Before Running

- [ ] Kaggle API credentials in `~/.kaggle/kaggle.json`
- [ ] tqdm and kaggle installed
- [ ] MediaPipe 0.10.35 confirmed
- [ ] 300+ MB free disk space
- [ ] 15 minutes available

---

## 🔍 What to Expect

**Console output:**
```
✅ HandTracker initialized
Processing A: 300 images... ✅ A: 290/300 processed (10 skipped)
Processing B: 300 images... ✅ B: 298/300 processed (2 skipped)
...
📊 Total images processed: 7650
💾 Saved 7650 samples to .../asl_alphabet_landmarks.csv
🎉 Extraction complete! Ready for model training.
```

---

## 📊 Expected Statistics

```
Processing Summary:
  Total images processed: 7,650
  Total images skipped: 150
  Success rate: 98.1%

Samples Per Letter:
  A: 290 samples
  B: 298 samples
  C: 295 samples
  ...
  Z: 296 samples
```

---

## 🚨 Troubleshooting

| Problem | Solution |
|---------|----------|
| Kaggle credentials error | Check `ls -la ~/.kaggle/kaggle.json` exists |
| MediaPipe error | `pip install mediapipe==0.10.35` |
| High skip rate | Normal if 1-5%, expected dataset variability |
| Disk space error | Free up 300+ MB, dataset auto-deletes |
| Slow processing | Normal, ~5-10 images/second is expected |

---

## 📚 Learn More

- **Detailed guide:** `DATASET_EXTRACTION_GUIDE.md`
- **Technical details:** `EXTRACTION_PLAN_SUMMARY.md`
- **Script source:** `src/extract_landmarks_from_dataset.py`

---

## 🎓 What Happens

The script automatically:

1. ✅ Downloads Kaggle ASL Alphabet dataset (215 MB)
2. ✅ Extracts 300 images per letter (A-Z)
3. ✅ Detects hands using MediaPipe HandLandmarker
4. ✅ Extracts 21-point hand landmarks
5. ✅ Normalizes features (scale-invariant)
6. ✅ Saves to CSV with proper format
7. ✅ Reports statistics
8. ✅ Cleans up temp files

**Zero changes to your existing code!**

---

## ⏱️ Timeline

| Step | Time |
|------|------|
| Kaggle API setup | 5 minutes (one-time) |
| Dependency install | 2 minutes |
| Dataset download | 2-5 minutes |
| Image processing | 8-12 minutes |
| CSV save & report | 1 minute |
| **Total** | **~10-15 minutes** |

---

## 🎉 You're Ready!

```bash
cd /Users/harshit/Downloads/Mohak\ Project/sign_language_translator
source venv/bin/activate
python src/extract_landmarks_from_dataset.py
```

Then proceed to Phase 4: ML Model Training! 🚀

---

**Questions?** See `DATASET_EXTRACTION_GUIDE.md` for complete documentation.
