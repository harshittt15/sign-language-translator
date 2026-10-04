"""
Main application entry point for Sign Language Translator
Integrates all components: camera, hand tracking, ML model, TTS, GUI
"""

import sys
import cv2
import numpy as np
sys.path.insert(0, '..')
import config
from camera import CameraHandler
from inference import SignLanguageInference
from prediction_smoother import PredictionSmoother
from tts import TextToSpeech


class SignLanguageTranslator:
    """Main application orchestrator"""

    def __init__(self):
        """Initialize translator with all components"""
        print("Initializing Sign Language Translator...")

        self.camera = None
        self.engine = None
        self.tts = None

        try:
            # Initialize camera
            print("- Initializing camera...")
            self.camera = CameraHandler()

            # Initialize the shared inference pipeline (hand tracker +
            # model + scaler). The web API uses this same class, so the
            # desktop and web paths cannot drift apart.
            print("- Initializing inference pipeline...")
            self.engine = SignLanguageInference()

            # Initialize TTS
            print("- Initializing text-to-speech...")
            self.tts = TextToSpeech()

            print("✓ All components initialized successfully\n")

        except Exception as e:
            print(f"✗ Initialization error: {e}")
            self.cleanup()
            raise

    def run(self):
        """Run the application in real-time translation mode"""
        print("="*60)
        print("SIGN LANGUAGE TRANSLATOR - RUNNING")
        print("="*60)
        print("Press 'q' to quit")
        print("Press 'a' to toggle audio")
        print("Press 'c' to clear history")
        print("Press 'd' to toggle per-frame diagnostics")
        print("="*60 + "\n")

        recognition_history = []
        audio_enabled = True
        frame_count = 0
        smoother = PredictionSmoother()
        debug_frames = False

        try:
            while True:
                frame = self.camera.read()
                frame_display = frame.copy()

                # One call runs the whole production pipeline: square crop
                # (the aspect-ratio fix) -> landmarks -> normalization ->
                # model -> temporal smoothing.
                result = self.engine.process_frame(frame, smoother=smoother)

                predicted_sign = result.raw_sign
                confidence = result.confidence
                accepted, emitted = result.sign, result.emitted
                normalized = result.hand_detected

                # Draw landmarks into the cropped region so the overlay
                # stays in register with the full display frame.
                crop_y, crop_x, crop_side = result.crop_offset
                frame_square = frame[crop_y:crop_y + crop_side,
                                     crop_x:crop_x + crop_side]
                frame_display[crop_y:crop_y + crop_side,
                              crop_x:crop_x + crop_side] = \
                    self.engine.tracker.draw_landmarks(frame_square, result.landmarks)

                if emitted:
                    recognition_history.append(accepted)

                    # Speak the sign
                    if audio_enabled:
                        self.tts.speak(accepted)

                    print(f"Frame {frame_count}: Recognized '{accepted}' "
                          f"(confidence: {confidence:.2f})")

                if debug_frames:
                    print(f"f {frame_count:>6} | hand:{'Y' if normalized else 'n'} "
                          f"| raw:{str(predicted_sign):>4} {confidence:5.2f} "
                          f"| smoothed:{str(accepted):>4} "
                          f"| miss:{smoother.missing_frames}"
                          f"{'  <-- EMIT ' + str(accepted) if emitted else ''}")

                if normalized:
                    # Display on frame
                    cv2.putText(frame_display, f"Sign: {accepted or 'None'}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    cv2.putText(frame_display, f"Confidence: {confidence:.2f}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame_display, f"raw: {predicted_sign or '-'}", (10, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
                else:
                    cv2.putText(frame_display, "No hand detected", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Display history at bottom
                if recognition_history:
                    history_text = " -> ".join(recognition_history[-10:])  # Last 10 signs
                    cv2.putText(frame_display, f"History: {history_text}", (10, frame_display.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

                # Display FPS
                cv2.putText(frame_display, f"Frame: {frame_count}", (frame_display.shape[1] - 200, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 1)

                # Display status
                audio_status = "Audio: ON" if audio_enabled else "Audio: OFF"
                cv2.putText(frame_display, audio_status, (frame_display.shape[1] - 200, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 100) if audio_enabled else (100, 100, 200), 1)

                # Show frame
                cv2.imshow("Sign Language Translator", frame_display)

                # Handle key input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):  # Quit
                    print("\nExiting...")
                    break
                elif key == ord('a'):  # Toggle audio
                    audio_enabled = self.tts.toggle()
                    print(f"\nAudio: {'ON' if audio_enabled else 'OFF'}")
                elif key == ord('c'):  # Clear history
                    recognition_history = []
                    smoother.reset()
                    print("\nHistory cleared")
                elif key == ord('d'):  # Toggle per-frame diagnostics
                    debug_frames = not debug_frames
                    print(f"\nFrame diagnostics: {'ON' if debug_frames else 'OFF'}")

                frame_count += 1

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"\n✗ Runtime error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """Release all resources"""
        print("\nCleaning up...")

        if self.camera:
            self.camera.release()
        if self.engine:
            self.engine.close()
        if self.tts:
            self.tts.close()

        cv2.destroyAllWindows()
        print("✓ Cleanup complete")


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("REAL-TIME SIGN LANGUAGE TRANSLATOR")
    print("="*60 + "\n")

    try:
        translator = SignLanguageTranslator()
        translator.run()

    except Exception as e:
        print(f"✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
