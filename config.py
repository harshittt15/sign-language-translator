"""
Configuration settings for the Sign Language Translator application
"""

# Camera Settings
CAMERA_INDEX = 0  # Default webcam
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# MediaPipe Hand Detection
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 2  # Support single or dual-hand signs

# Hand Landmark Normalization
NUM_LANDMARKS = 21
LANDMARK_DIMENSIONS = 3  # x, y, z coordinates
TOTAL_FEATURES = NUM_LANDMARKS * LANDMARK_DIMENSIONS  # 63 features

# Model Settings
MODEL_INPUT_SIZE = TOTAL_FEATURES
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42

# Random Forest Classifier
N_ESTIMATORS = 200
MAX_DEPTH = 20
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 2

# Inference Settings
CONFIDENCE_THRESHOLD = 0.6  # Only display predictions above this confidence
PREDICTION_SMOOTHING = 5  # Rolling window of frames that must agree before accepting
NO_HAND_RESET_FRAMES = 5  # Consecutive no-hand frames before smoothing state resets

# Text-to-Speech Settings
TTS_RATE = 150  # Words per minute
TTS_VOLUME = 1.0  # 0.0 to 1.0
DEBOUNCE_TIME = 1.0  # Seconds before speaking same sign again

# Data Collection
MIN_SAMPLES_PER_SIGN = 30
FRAMES_PER_SAMPLE = 20  # Frames to capture per sign gesture

# Paths
DATA_DIR = "data/"
RAW_DATA_DIR = f"{DATA_DIR}raw/"
PROCESSED_DATA_DIR = f"{DATA_DIR}processed/"
MODELS_DIR = "models/"
MODEL_PATH = f"{MODELS_DIR}sign_language_model.h5"
SCALER_PATH = f"{MODELS_DIR}feature_scaler.pkl"

# GUI Settings
WINDOW_TITLE = "Real-Time Sign Language Translator"
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
UPDATE_INTERVAL = 30  # milliseconds
