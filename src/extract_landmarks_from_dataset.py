"""
Extract hand landmarks from Kaggle ASL Alphabet dataset.

This script:
1. Downloads the Kaggle ASL Alphabet dataset
2. Loads images from A-Z folders
3. Extracts hand landmarks using MediaPipe HandLandmarker
4. Normalizes landmarks using the same method as HandTracker
5. Saves landmarks + labels to CSV

Requirements:
- Kaggle CLI authentication (~/.kaggle/kaggle.json or ~/.kaggle/access_token)
- MediaPipe 0.10.35+
- OpenCV, pandas, numpy
"""

import os
import sys
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import subprocess

sys.path.insert(0, os.path.dirname(__file__))
from hand_tracker import (
    HandTracker,
    NUM_LANDMARKS,
    LANDMARK_DIMENSIONS,
    WRIST_IDX,
    MIDDLE_MCP_IDX,
)
import config

# Kaggle dataset info
KAGGLE_DATASET = "grassknoted/asl-alphabet"
IMAGES_PER_LETTER = 300  # Manageable subset


class ASLDatasetExtractor:
    """Extract landmarks from ASL Alphabet dataset"""

    def __init__(self, images_per_letter=IMAGES_PER_LETTER):
        self.images_per_letter = images_per_letter
        self.tracker = None
        self.dataset_path = None
        self.processed_data = []
        self.skip_stats = {}

    def download_dataset(self):
        """Download Kaggle ASL Alphabet dataset"""
        print(f"\n📥 Downloading Kaggle dataset: {KAGGLE_DATASET}")

        try:
            # Download to temp location
            temp_dir = Path.home() / "Downloads" / "kaggle_asl_temp"
            temp_dir.mkdir(parents=True, exist_ok=True)

            print("🔑 Using Kaggle CLI authentication...")
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(temp_dir)],
                check=True,
                capture_output=True
            )

            # Unzip
            zip_file = temp_dir / f"{KAGGLE_DATASET.split('/')[-1]}.zip"
            if zip_file.exists():
                shutil.unpack_archive(zip_file, temp_dir)
                zip_file.unlink()

            self.dataset_path = temp_dir / "asl_alphabet_train" / "asl_alphabet_train"
            if not self.dataset_path.exists():
                self.dataset_path = temp_dir / "asl_alphabet_train"

            if not self.dataset_path.exists():
                print(f"❌ Extracted dataset not found at {self.dataset_path}")
                return False

            print(f"✅ Dataset downloaded to: {self.dataset_path}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Kaggle download failed: {e}")
            print("\n📝 Ensure Kaggle authentication is configured:")
            print("   • ~/.kaggle/kaggle.json (legacy), OR")
            print("   • ~/.kaggle/access_token (new method)")
            return False
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            return False

    def extract_landmarks_from_image(self, image_path):
        """Extract landmarks from a single image"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return None

            # Extract landmarks
            landmarks_list = self.tracker.extract_landmarks(image)

            if landmarks_list is None or len(landmarks_list) == 0:
                return None

            # Use first hand detected
            landmarks = landmarks_list[0]

            # Normalize landmarks (same method as HandTracker)
            normalized = self._normalize_landmarks_single(landmarks)

            return normalized

        except Exception as e:
            return None

    def _normalize_landmarks_single(self, landmarks):
        """
        Normalize a single hand's landmarks.
        Uses same method as HandTracker.normalize_landmarks()
        """
        landmarks = np.array(landmarks)

        # Reshape flat 63 values into 21 (x, y, z) points
        points = landmarks.reshape(NUM_LANDMARKS, LANDMARK_DIMENSIONS)

        # Reference point (wrist): landmark 0
        normalized = points - points[WRIST_IDX]

        # Scale by hand size (wrist to middle finger MCP distance)
        scale = np.linalg.norm(points[MIDDLE_MCP_IDX] - points[WRIST_IDX])
        if scale > 0:
            normalized = normalized / scale

        return normalized.reshape(-1)

    def process_dataset(self):
        """Process all images from dataset"""
        print("\n🔍 Processing ASL Alphabet dataset...\n")

        # Initialize HandTracker
        try:
            self.tracker = HandTracker(
                min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
            )
            print("✅ HandTracker initialized\n")
        except Exception as e:
            print(f"❌ Failed to initialize HandTracker: {e}")
            return False

        if self.dataset_path is None or not self.dataset_path.exists():
            print(f"❌ Dataset path not found: {self.dataset_path}")
            return False

        # Get all letter folders
        letter_folders = sorted([
            d for d in self.dataset_path.iterdir()
            if d.is_dir() and len(d.name) == 1 and d.name.isalpha()
        ])

        if not letter_folders:
            print(f"❌ No letter folders found in {self.dataset_path}")
            return False

        print(f"Found {len(letter_folders)} letter folders\n")

        # Process each letter
        total_processed = 0
        total_skipped = 0

        for letter_folder in letter_folders:
            letter = letter_folder.name.upper()
            self.skip_stats[letter] = 0

            # Get all images in this letter folder
            image_files = sorted([
                f for f in letter_folder.glob("*.jpg")
                if f.is_file()
            ][:self.images_per_letter])

            if not image_files:
                print(f"⚠️  No images found for letter {letter}")
                continue

            print(f"Processing {letter}: {len(image_files)} images...", end=" ")

            skipped = 0
            for image_path in tqdm(image_files, desc=letter, leave=False):
                normalized_landmarks = self.extract_landmarks_from_image(image_path)

                if normalized_landmarks is None:
                    skipped += 1
                    continue

                # Flatten landmarks (21 landmarks × 3 coords = 63 features)
                features = normalized_landmarks.flatten().tolist()

                # Store in dataset
                self.processed_data.append({
                    'label': letter,
                    **{f'feature_{i}': features[i] for i in range(len(features))}
                })
                total_processed += 1

            skipped_count = len(image_files) - (len(self.processed_data) - total_processed + skipped)
            self.skip_stats[letter] = skipped
            total_skipped += skipped

            status = f"✅ {letter}: {len(image_files) - skipped}/{len(image_files)} processed"
            if skipped > 0:
                status += f" ({skipped} skipped)"
            print(status)

        self.tracker.close()

        print(f"\n📊 Processing Summary:")
        print(f"  Total images processed: {total_processed}")
        print(f"  Total images skipped: {total_skipped}")
        print(f"  Success rate: {100*total_processed/(total_processed+total_skipped):.1f}%")

        return len(self.processed_data) > 0

    def save_to_csv(self):
        """Save processed landmarks to CSV"""
        if not self.processed_data:
            print("❌ No data to save")
            return False

        output_dir = Path(__file__).parent.parent / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "asl_alphabet_landmarks.csv"

        print(f"\n💾 Saving to CSV: {output_file}")

        # Create DataFrame
        df = pd.DataFrame(self.processed_data)

        # Reorder columns: label first, then features
        cols = ['label'] + [c for c in df.columns if c != 'label']
        df = df[cols]

        # Save
        df.to_csv(output_file, index=False)

        print(f"✅ Saved {len(df)} samples to {output_file}")

        return output_file

    def generate_report(self, output_file):
        """Generate extraction report"""
        if not self.processed_data:
            print("❌ No data for report")
            return

        df = pd.read_csv(output_file)

        print("\n" + "="*70)
        print("📋 EXTRACTION REPORT")
        print("="*70)

        print(f"\n📁 Dataset Source: {KAGGLE_DATASET}")
        print(f"📁 Output File: {output_file}")
        print(f"📊 Total Samples: {len(df)}")
        print(f"🏷️  Number of Features: {len(df.columns) - 1}")  # Exclude label
        print(f"📈 Feature Range: feature_0 to feature_{len(df.columns)-2}")

        print(f"\n📊 Samples Per Letter:")
        print("-" * 70)

        for letter in sorted(self.skip_stats.keys()):
            skipped = self.skip_stats[letter]
            samples = len(df[df['label'] == letter])

            if samples > 0 or skipped > 0:
                status = f"{letter}: {samples:4d} samples"
                if skipped > 0:
                    status += f" ({skipped} skipped)"
                print(f"  {status}")

        print("\n" + "-" * 70)
        total_samples_per_letter = [len(df[df['label'] == letter]) for letter in sorted(df['label'].unique())]
        print(f"Average samples per letter: {np.mean(total_samples_per_letter):.0f}")
        print(f"Min samples: {min(total_samples_per_letter) if total_samples_per_letter else 0}")
        print(f"Max samples: {max(total_samples_per_letter) if total_samples_per_letter else 0}")

        print(f"\n💾 CSV Details:")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Label column: 'label' (A-Z)")
        print(f"  Feature columns: 'feature_0' to 'feature_62'")
        print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

        print("\n" + "="*70 + "\n")


def main():
    """Main extraction pipeline"""
    print("\n" + "="*70)
    print("🤟 ASL ALPHABET DATASET LANDMARK EXTRACTION")
    print("="*70)

    extractor = ASLDatasetExtractor(images_per_letter=IMAGES_PER_LETTER)

    # Download dataset
    if not extractor.download_dataset():
        print("\n❌ Failed to download dataset. Exiting.")
        sys.exit(1)

    # Process images
    if not extractor.process_dataset():
        print("\n❌ Failed to process dataset. Exiting.")
        sys.exit(1)

    # Save to CSV
    output_file = extractor.save_to_csv()
    if not output_file:
        print("\n❌ Failed to save CSV. Exiting.")
        sys.exit(1)

    # Generate report
    extractor.generate_report(output_file)

    print("🎉 Extraction complete! Ready for model training.")


if __name__ == "__main__":
    main()
