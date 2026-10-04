"""
Machine learning model for sign language classification
Uses Random Forest classifier trained on hand landmark data
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import sys
sys.path.insert(0, '..')
import config

DATASET_FILENAME = "asl_alphabet_landmarks.csv"
LABEL_COLUMN = "label"


class SignLanguageModel:
    """Random Forest classifier for ASL alphabet signs (26 classes)"""

    def __init__(self):
        """Initialize model"""
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self._label_encoder = LabelEncoder()
        self.label_encoder = {}
        self.label_decoder = {}
        self.is_trained = False

    def build_model(self, num_classes):
        """
        Build Random Forest model

        Args:
            num_classes: Number of sign classes
        """
        print("\nModel Architecture:")
        print(f"Random Forest Classifier")
        print(f"- Estimators: 200")
        print(f"- Max Depth: 20")
        print(f"- Min Samples Split: 5")
        print(f"- Input Features: {config.MODEL_INPUT_SIZE}")

    def load_data(self, data_dir=config.PROCESSED_DATA_DIR):
        """
        Load landmark data from CSV files

        Args:
            data_dir: Directory containing CSV files

        Returns:
            Tuple of (X, y) where X is features and y is labels
        """
        csv_path = os.path.join(data_dir, DATASET_FILENAME)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        print(f"\nLoading data from {csv_path}...")
        df = pd.read_csv(csv_path)

        feature_cols = [f"feature_{i}" for i in range(config.TOTAL_FEATURES)]
        missing = [c for c in [LABEL_COLUMN] + feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing expected columns: {missing[:5]}")

        X = df[feature_cols].values
        y = df[LABEL_COLUMN].values

        print(f"\nTotal samples: {len(X)}")
        print(f"Feature dimensions: {X.shape}")
        print(f"Classes: {len(np.unique(y))}")

        return X, y

    def encode_labels(self, y):
        """
        Encode string labels to integers and create encoder/decoder

        Args:
            y: Array of string labels

        Returns:
            Encoded labels
        """
        y_encoded = self._label_encoder.fit_transform(y)

        # Mirror into plain dicts so metadata.json stays loadable by inference
        self.label_encoder = {
            str(label): int(idx)
            for idx, label in enumerate(self._label_encoder.classes_)
        }
        self.label_decoder = {
            int(idx): str(label)
            for idx, label in enumerate(self._label_encoder.classes_)
        }

        return y_encoded

    def prepare_data(self, X, y):
        """
        Prepare data for training

        Args:
            X: Feature array
            y: Label array

        Returns:
            Tuple of (X_train_scaled, X_test_scaled, y_train, y_test)
        """
        # Encode labels
        y_encoded = self.encode_labels(y)

        # Split data (stratified so every letter keeps its proportion)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=1 - config.TRAIN_TEST_SPLIT,
            random_state=config.RANDOM_STATE,
            stratify=y_encoded
        )

        # Scale features (inference predict() depends on this scaler)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        num_classes = len(self.label_encoder)

        print(f"\nData split:")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Number of classes: {num_classes}")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train(self, data_dir=config.PROCESSED_DATA_DIR):
        """
        Train the model on collected data

        Args:
            data_dir: Directory containing CSV training data
        """
        print("\n" + "="*60)
        print("TRAINING SIGN LANGUAGE MODEL")
        print("="*60)

        try:
            # Load data
            X, y = self.load_data(data_dir)

            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data(X, y)

            # Build model
            num_classes = len(self.label_encoder)
            self.build_model(num_classes)

            # Train model (scikit-learn takes integer class indices directly)
            print("\nTraining model...")
            self.model.fit(X_train, y_train)
            print("✓ Training complete")

            # Evaluate model
            print("\n" + "="*60)
            print("MODEL EVALUATION")
            print("="*60)

            y_pred = self.model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            print(f"\nTest Accuracy: {accuracy:.4f}")

            # Per-class metrics: both sides are integer indices, named A-Z
            class_indices = np.arange(num_classes)
            target_names = [self.label_decoder[i] for i in class_indices]

            print("\nClassification Report:")
            print(classification_report(
                y_test, y_pred,
                labels=class_indices,
                target_names=target_names,
                zero_division=0
            ))

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=class_indices)
            self._report_confusion_matrix(cm, target_names)

            # Save model and scaler
            os.makedirs(config.MODELS_DIR, exist_ok=True)
            model_path = config.MODEL_PATH.replace('.h5', '.pkl')
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, config.SCALER_PATH)

            print(f"\n✓ Model saved to {model_path}")
            print(f"✓ Scaler saved to {config.SCALER_PATH}")

            self.is_trained = True
            self.save_metadata()

            return accuracy

        except Exception as e:
            print(f"✗ Training error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _report_confusion_matrix(self, cm, target_names):
        """
        Display the 26-class confusion matrix and save it as CSV

        Args:
            cm: Confusion matrix array (num_classes x num_classes)
            target_names: Class names ordered by class index
        """
        print("\nConfusion Matrix (rows = true, cols = predicted):")
        header = "    " + " ".join(f"{n:>3}" for n in target_names)
        print(header)
        for name, row in zip(target_names, cm):
            print(f"{name:>3} " + " ".join(f"{v:>3}" for v in row))

        os.makedirs(config.MODELS_DIR, exist_ok=True)
        cm_path = os.path.join(config.MODELS_DIR, "confusion_matrix.csv")
        pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(cm_path)
        print(f"\n✓ Confusion matrix saved to {cm_path}")

    def load_model(self, model_path=config.MODEL_PATH, scaler_path=config.SCALER_PATH):
        """
        Load pre-trained model and scaler

        Args:
            model_path: Path to saved model (.pkl format)
            scaler_path: Path to saved scaler
        """
        try:
            # Update path to use .pkl extension for scikit-learn
            if model_path.endswith('.h5'):
                model_path = model_path.replace('.h5', '.pkl')

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.is_trained = True
            print(f"✓ Model loaded from {model_path}")
            print(f"✓ Scaler loaded from {scaler_path}")
        except FileNotFoundError as e:
            print(f"✗ Model file not found: {e}")
            raise

    def predict(self, landmarks):
        """
        Predict sign from landmarks

        Args:
            landmarks: Hand landmarks (numpy array or list)

        Returns:
            Tuple of (predicted_label, confidence)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load_model() first")

        try:
            # Convert to numpy array
            if not isinstance(landmarks, np.ndarray):
                landmarks = np.array(landmarks)

            # Handle both single and multiple hands
            if len(landmarks.shape) > 1 and landmarks.shape[0] > 1:
                landmarks = landmarks[0]  # Use first hand

            # Ensure 1D array
            landmarks = landmarks.flatten()

            # Scale
            landmarks_scaled = self.scaler.transform([landmarks])

            # Predict with scikit-learn
            predicted_class = self.model.predict(landmarks_scaled)[0]
            probabilities = self.model.predict_proba(landmarks_scaled)[0]
            confidence = np.max(probabilities)
            predicted_label = self.label_decoder[int(predicted_class)]

            return predicted_label, float(confidence)

        except Exception as e:
            print(f"✗ Prediction error: {e}")
            return None, 0.0

    def save_metadata(self):
        """Save model metadata (labels)"""
        metadata = {
            'label_encoder': self.label_encoder,
            'label_decoder': self.label_decoder
        }
        import json
        metadata_path = os.path.join(config.MODELS_DIR, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        print(f"✓ Metadata saved to {metadata_path}")

    def load_metadata(self):
        """Load model metadata (labels)"""
        import json
        metadata_path = os.path.join(config.MODELS_DIR, 'metadata.json')
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.label_encoder = metadata['label_encoder']
            self.label_decoder = {int(k): v for k, v in metadata['label_decoder'].items()}
            print(f"✓ Metadata loaded from {metadata_path}")
        except FileNotFoundError:
            print(f"⚠ Metadata file not found: {metadata_path}")


def main():
    """Main training interface"""
    try:
        model = SignLanguageModel()

        print("\n" + "="*60)
        print("SIGN LANGUAGE MODEL TRAINER")
        print("="*60)
        print("\nOptions:")
        print("1. Train new model")
        print("2. Load existing model")
        print("3. Exit")
        print("="*60)

        choice = input("\nSelect option (1-3): ").strip()

        if choice == '1':
            model.train()
            model.save_metadata()
        elif choice == '2':
            model.load_model()
            model.load_metadata()
        elif choice == '3':
            print("Exiting...")
        else:
            print("Invalid option")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
