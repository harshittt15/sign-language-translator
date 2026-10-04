"""
Camera module for webcam access and frame capture
"""

import cv2
import sys
sys.path.insert(0, '..')
import config


class CameraHandler:
    """Handles webcam access and frame capture"""

    def __init__(self, camera_index=config.CAMERA_INDEX):
        """
        Initialize camera handler

        Args:
            camera_index: Index of camera to use (default 0 for built-in)
        """
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.FPS)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera at index {camera_index}")

    def read(self):
        """
        Read frame from camera

        Returns:
            Tuple of (success, frame) where frame is BGR image
        """
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        return frame

    def get_frame_dimensions(self):
        """Get frame dimensions"""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def release(self):
        """Release camera resources"""
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def test_camera():
    """Test webcam connection and display feed with landmarks"""
    print("Testing camera connection...")

    try:
        camera = CameraHandler()
        print("✓ Camera connected successfully")
        print(f"✓ Frame dimensions: {camera.get_frame_dimensions()}")
        print("\nPress 'q' to exit camera test")

        while True:
            frame = camera.read()
            cv2.imshow("Camera Test - Press Q to Exit", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()
        print("✓ Camera test completed successfully")

    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_camera()
