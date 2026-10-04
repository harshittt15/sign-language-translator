"""
Text-to-Speech module using pyttsx3
Provides audio output for recognized signs
"""

import pyttsx3
import threading
import time
import sys
sys.path.insert(0, '..')
import config


class TextToSpeech:
    """Handles text-to-speech audio output"""

    def __init__(self, rate=config.TTS_RATE, volume=config.TTS_VOLUME):
        """
        Initialize text-to-speech engine

        Args:
            rate: Speaking rate in words per minute
            volume: Volume level (0.0 to 1.0)
        """
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)

        self.enabled = True
        self.last_spoken = None
        self.debounce_time = config.DEBOUNCE_TIME
        self.speech_thread = None

    def speak(self, text):
        """
        Speak text asynchronously

        Args:
            text: Text to speak
        """
        if not self.enabled:
            return

        # Debounce: don't speak same text too frequently
        current_time = time.time()
        if self.last_spoken == text and \
           (current_time - self.last_speak_time) < self.debounce_time:
            return

        self.last_spoken = text
        self.last_speak_time = current_time

        # Speak in separate thread to avoid blocking
        self.speech_thread = threading.Thread(target=self._speak_sync, args=(text,))
        self.speech_thread.daemon = True
        self.speech_thread.start()

    def _speak_sync(self, text):
        """
        Synchronously speak text (runs in separate thread)

        Args:
            text: Text to speak
        """
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"✗ TTS Error: {e}")

    def set_rate(self, rate):
        """
        Set speaking rate

        Args:
            rate: Words per minute (typically 50-300)
        """
        self.engine.setProperty('rate', rate)

    def set_volume(self, volume):
        """
        Set volume level

        Args:
            volume: Volume (0.0 to 1.0)
        """
        volume = max(0.0, min(1.0, volume))
        self.engine.setProperty('volume', volume)

    def toggle(self):
        """Toggle audio on/off"""
        self.enabled = not self.enabled
        return self.enabled

    def stop(self):
        """Stop current speech"""
        self.engine.stop()

    def close(self):
        """Release TTS resources"""
        self.stop()
        if self.speech_thread and self.speech_thread.is_alive():
            self.speech_thread.join(timeout=1.0)


def test_tts():
    """Test text-to-speech functionality"""
    print("Testing Text-to-Speech...")

    try:
        tts = TextToSpeech()
        print("✓ TTS initialized successfully\n")

        # Test basic speech
        test_phrases = ["Hello", "This is a test", "Sign Language Translator"]

        print("Speaking test phrases:")
        for phrase in test_phrases:
            print(f"Speaking: '{phrase}'")
            tts.speak(phrase)
            time.sleep(2)

        print("\n✓ TTS test completed successfully")
        tts.close()

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_tts()
