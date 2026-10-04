"""
Hand tracking module using MediaPipe
Extracts 21 landmark points from hand gestures
Compatible with MediaPipe 1.0.1+
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import numpy as np
import sys
import os
sys.path.insert(0, '..')
import config

NUM_LANDMARKS = config.NUM_LANDMARKS
LANDMARK_DIMENSIONS = config.LANDMARK_DIMENSIONS
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9


def center_square_crop(frame):
    """
    Crop a frame to its centred square.

    ASPECT-RATIO COMPATIBILITY FIX. The model was trained on the 200x200
    (square) ASL Alphabet images, but the webcam delivers 1280x720. MediaPipe
    normalizes x by frame width and y by frame height, so a 16:9 frame
    compresses x relative to y by 720/1280 = 0.5625. The wrist-to-MCP
    normalization divides by a single scalar and therefore cannot undo an
    anisotropic distortion; measured drift was 0.155 on x versus 0.021 on y.
    Cropping to a square restores the geometry the model was trained on.

    Args:
        frame: BGR frame of any aspect ratio

    Returns:
        (cropped_frame, y_offset, x_offset, side) -- offsets locate the crop
        within the original frame so overlays can be drawn in register.
    """
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return frame[y0:y0 + side, x0:x0 + side], y0, x0, side


def _get_model_path():
    """Get path to hand_landmarker model"""
    model_path = os.path.join(
        os.path.dirname(__file__), '..', 'models', 'hand_landmarker.task'
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            f"Please ensure models/hand_landmarker.task exists"
        )
    return model_path


class HandTracker:
    """Handles hand detection and landmark extraction using MediaPipe"""

    def __init__(self, min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                 min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE):
        """
        Initialize hand tracker using MediaPipe task-based API

        Args:
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        # Suppress TensorFlow/MediaPipe warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Get model path
        model_path = _get_model_path()

        # Initialize HandLandmarker with CPU-only delegate
        base_options = BaseOptions(
            model_asset_path=model_path,
            delegate=BaseOptions.Delegate.CPU
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=config.MAX_NUM_HANDS,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Create landmarker
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.min_detection_confidence = min_detection_confidence

    def extract_landmarks(self, frame):
        """
        Extract hand landmarks from frame

        Args:
            frame: Input image (BGR format from OpenCV)

        Returns:
            List of hand landmarks or None if no hand detected
            Each hand has 21 points with (x, y, z) coordinates (numpy array of 63 floats)
        """
        # mp.Image requires a C-contiguous buffer; a reversed slice is a
        # non-contiguous view and silently yields zero detections.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = self.landmarker.detect(mp_image)

            if detection_result.hand_landmarks:
                hand_landmarks_list = []
                for hand_landmarks in detection_result.hand_landmarks:
                    landmarks = []
                    for landmark in hand_landmarks:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    hand_landmarks_list.append(np.array(landmarks))
                return hand_landmarks_list
        except Exception as e:
            # Silently handle initialization errors on some platforms
            pass

        return None

    def draw_landmarks(self, frame, landmarks_list):
        """
        Draw hand landmarks on frame for visualization

        Args:
            frame: Input image
            landmarks_list: List of landmarks from extract_landmarks()

        Returns:
            Frame with drawn landmarks
        """
        import cv2

        frame_display = frame.copy()
        h, w = frame.shape[:2]

        if landmarks_list:
            # Define hand connections (MediaPipe hand landmark connections)
            HAND_CONNECTIONS = [
                (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),      # Index
                (0, 9), (9, 10), (10, 11), (11, 12), # Middle
                (0, 13), (13, 14), (14, 15), (15, 16), # Ring
                (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
                (5, 9), (9, 13), (13, 17)            # Palm lines
            ]

            for landmarks in landmarks_list:
                # Convert landmarks to pixel coordinates
                landmark_points = []
                for i in range(0, len(landmarks), 3):
                    x = int(landmarks[i] * w)
                    y = int(landmarks[i + 1] * h)
                    landmark_points.append((x, y))

                # Draw connections (lines between landmarks)
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                        start_point = landmark_points[start_idx]
                        end_point = landmark_points[end_idx]
                        cv2.line(frame_display, start_point, end_point, (255, 0, 0), 2)

                # Draw landmarks (circles at each point)
                for point in landmark_points:
                    cv2.circle(frame_display, point, 4, (0, 255, 0), -1)

        return frame_display

    def normalize_landmarks(self, landmarks):
        """
        Normalize landmarks for model input

        Args:
            landmarks: Raw landmarks from extract_landmarks()

        Returns:
            Normalized landmarks array
        """
        if landmarks is None:
            return None

        normalized = []
        for hand_landmarks in landmarks:
            # Convert to numpy array if needed
            if not isinstance(hand_landmarks, np.ndarray):
                hand_landmarks = np.array(hand_landmarks)

            # Reshape flat 63 values into 21 (x, y, z) points so the wrist
            # offset and scale apply per-landmark rather than broadcasting.
            points = hand_landmarks.reshape(NUM_LANDMARKS, LANDMARK_DIMENSIONS)

            # Normalize: use the wrist (landmark 0) as reference point
            normalized_points = points - points[WRIST_IDX]

            # Scale by hand size (distance from wrist to middle finger MCP)
            scale = np.linalg.norm(points[MIDDLE_MCP_IDX] - points[WRIST_IDX])
            if scale > 0:
                normalized_points = normalized_points / scale

            normalized.append(normalized_points.reshape(-1))

        return normalized if len(normalized) > 0 else None

    def close(self):
        """Release MediaPipe resources"""
        if hasattr(self, 'landmarker'):
            try:
                self.landmarker.close()
            except:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def test_hand_tracker():
    """Test hand tracking with live camera feed"""
    import cv2
    from camera import CameraHandler

    print("Testing hand tracker...")

    try:
        camera = CameraHandler()
        tracker = HandTracker()

        print("✓ Hand tracker initialized")
        print("Press 'q' to exit hand tracking test\n")

        frame_count = 0
        while True:
            frame = camera.read()
            landmarks_list = tracker.extract_landmarks(frame)

            # Draw landmarks on frame
            frame_with_landmarks = tracker.draw_landmarks(frame, landmarks_list)

            # Display info
            if landmarks_list:
                print(f"Frame {frame_count}: {len(landmarks_list)} hand(s) detected")
            else:
                print(f"Frame {frame_count}: No hands detected")

            cv2.imshow("Hand Tracking Test", frame_with_landmarks)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        camera.release()
        tracker.close()
        cv2.destroyAllWindows()
        print("\n✓ Hand tracking test completed successfully")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_hand_tracker()
