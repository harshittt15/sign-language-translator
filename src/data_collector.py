"""
Data collection module for gathering sign language training data
Records hand landmarks for each sign gesture
"""

import cv2
import csv
import os
import numpy as np
import sys
sys.path.insert(0, '..')
import config
from camera import CameraHandler
from hand_tracker import HandTracker


class DataCollector:
    """Collects training data by recording hand landmarks for sign gestures"""

    # Define signs to collect (A-Z for alphabet)
    SIGNS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

    def __init__(self):
        """Initialize data collector"""
        self.camera = CameraHandler()
        self.tracker = HandTracker()
        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    def collect_sign_data(self, sign_name, num_samples=config.MIN_SAMPLES_PER_SIGN):
        """
        Collect training data for a specific sign

        Args:
            sign_name: Name of the sign (e.g., 'A', 'B', 'hello')
            num_samples: Number of samples to collect (default 30)
        """
        print(f"\n{'='*60}")
        print(f"Collecting data for sign: {sign_name}")
        print(f"{'='*60}")
        print(f"Total samples to collect: {num_samples}")
        print("Press 'SPACE' to start recording")
        print("Press 'q' to skip this sign")

        csv_path = os.path.join(config.PROCESSED_DATA_DIR, f"{sign_name}.csv")
        samples_collected = 0

        while samples_collected < num_samples:
            frame = self.camera.read()
            frame_display = frame.copy()

            # Show instructions
            cv2.putText(frame_display, f"Sign: {sign_name}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(frame_display, f"Samples: {samples_collected}/{num_samples}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame_display, "Press SPACE to record, Q to skip", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)

            cv2.imshow("Data Collection", frame_display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # Space to start recording
                print(f"\nRecording sample {samples_collected + 1}/{num_samples}...")
                landmarks_sequence = self.record_gesture(frame, sign_name)

                if landmarks_sequence:
                    self.save_landmarks(csv_path, landmarks_sequence, sign_name)
                    samples_collected += 1
                    print(f"✓ Sample {samples_collected} saved")

            elif key == ord('q'):  # Q to skip
                print(f"Skipping {sign_name}")
                break

        cv2.destroyAllWindows()
        print(f"\n✓ Completed collecting {samples_collected} samples for '{sign_name}'")
        return samples_collected > 0

    def record_gesture(self, initial_frame, sign_name, duration=config.FRAMES_PER_SAMPLE):
        """
        Record a hand gesture for specified duration

        Args:
            initial_frame: Starting frame
            sign_name: Name of sign being recorded
            duration: Number of frames to record

        Returns:
            List of landmark arrays or None
        """
        landmarks_sequence = []
        frame_count = 0

        print(f"Recording... {duration} frames")

        while frame_count < duration:
            frame = self.camera.read()
            frame_display = frame.copy()

            # Extract landmarks
            hand_landmarks = self.tracker.extract_landmarks(frame)

            if hand_landmarks:
                # Use first hand if multiple hands detected
                landmarks = hand_landmarks[0]
                landmarks_sequence.append(landmarks.copy())

                # Draw landmarks
                frame_display = self.tracker.draw_landmarks(frame, hand_landmarks)

                # Show progress
                cv2.putText(frame_display, f"Recording: {frame_count}/{duration}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame_display, "Hold gesture steady...", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
            else:
                cv2.putText(frame_display, "No hand detected!", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame_display, "Position hand in frame", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)

            cv2.imshow("Data Collection - Recording", frame_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                return None

            frame_count += 1

        cv2.destroyAllWindows()

        if landmarks_sequence:
            print(f"✓ Recorded {len(landmarks_sequence)} frames with hand detection")
            return landmarks_sequence
        else:
            print("✗ No hand landmarks detected during recording")
            return None

    def save_landmarks(self, csv_path, landmarks_sequence, sign_name):
        """
        Save landmarks to CSV file

        Args:
            csv_path: Path to save CSV
            landmarks_sequence: List of landmark arrays
            sign_name: Name of the sign
        """
        # Average landmarks across the sequence
        averaged_landmarks = np.mean(landmarks_sequence, axis=0)

        # Write to CSV
        file_exists = os.path.exists(csv_path)

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)

            # Write header if file is new
            if not file_exists:
                header = [f"landmark_{i//3}_{['x', 'y', 'z'][i%3]}"
                          for i in range(len(averaged_landmarks))]
                header.append('label')
                writer.writerow(header)

            # Write landmark data
            row = list(averaged_landmarks) + [sign_name]
            writer.writerow(row)

    def collect_all_signs(self):
        """Collect data for all defined signs"""
        print("\n" + "="*60)
        print("SIGN LANGUAGE DATA COLLECTION")
        print("="*60)
        print(f"Signs to collect: {', '.join(self.SIGNS)}")
        print("="*60)

        successful_signs = 0
        for sign in self.SIGNS:
            if self.collect_sign_data(sign):
                successful_signs += 1

        print(f"\n{'='*60}")
        print(f"COLLECTION COMPLETE: {successful_signs}/{len(self.SIGNS)} signs collected")
        print(f"{'='*60}")

    def close(self):
        """Release resources"""
        self.camera.release()
        self.tracker.close()
        cv2.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    """Main data collection interface"""
    try:
        collector = DataCollector()

        print("\n" + "="*60)
        print("SIGN LANGUAGE DATA COLLECTOR")
        print("="*60)
        print("\nOptions:")
        print("1. Collect all signs (A-Z)")
        print("2. Collect specific sign")
        print("3. Exit")
        print("="*60)

        choice = input("\nSelect option (1-3): ").strip()

        if choice == '1':
            collector.collect_all_signs()
        elif choice == '2':
            sign = input("Enter sign name (e.g., 'A', 'hello'): ").strip().upper()
            if sign:
                collector.collect_sign_data(sign)
        elif choice == '3':
            print("Exiting...")
        else:
            print("Invalid option")

        collector.close()

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
