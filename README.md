# Real-Time Sign Language Translator

A computer vision application that translates sign language gestures to text and audio in real-time. This application uses MediaPipe for hand tracking, machine learning for sign recognition, and text-to-speech for audio output.

## Features

- 🎥 **Real-Time Hand Detection**: Uses MediaPipe to track hand landmarks in real-time
- 🧠 **Machine Learning Classification**: Neural network trained on hand landmark data
- 🔊 **Text-to-Speech**: Speaks recognized signs aloud using pyttsx3
- 🖥️ **User-Friendly GUI**: PyQt5-based interface with live camera feed
- 📊 **Confidence Display**: Shows prediction confidence scores
- ⚙️ **Customizable Settings**: Adjust confidence threshold, audio settings, camera selection

## System Requirements

### Hardware
- Computer with working webcam
- Processor: Intel i5/i7 or equivalent (for real-time inference)
- RAM: 4GB minimum, 8GB recommended
- Disk Space: ~2GB for models and dependencies

### Software
- Python 3.7 or higher
- macOS, Windows, or Linux
- Administrator/sudo access for package installation

## Installation

### 1. Clone or Download the Project
```bash
cd sign_language_translator
```

### 2. Create Python Virtual Environment
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import cv2, mediapipe, tensorflow; print('All dependencies installed successfully!')"
```

## Project Structure

```
sign_language_translator/
├── data/
│   ├── raw/              # Raw video recordings
│   └── processed/        # Extracted landmark CSV files
├── models/               # Trained model files
├── src/
│   ├── __init__.py
│   ├── camera.py         # Webcam access & frame capture
│   ├── hand_tracker.py   # MediaPipe integration
│   ├── model.py          # ML model training & inference
│   ├── tts.py            # Text-to-speech functionality
│   ├── gui.py            # GUI implementation
│   ├── data_collector.py # Training data collection
│   └── main.py           # Application entry point
├── tests/                # Unit and integration tests
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Quick Start

### 1. Test Webcam & Hand Tracking
```bash
python src/camera.py
```
This will open your webcam and display live hand landmarks.

### 2. Collect Training Data
```bash
python src/data_collector.py
```
Follow on-screen prompts to record sign language gestures (A-Z and common words).

### 3. Train the ML Model
```bash
python src/model.py --train
```
Trains a neural network on collected landmark data.

### 4. Run the Application
```bash
python src/main.py
```
Launches the full GUI application with real-time sign translation.

## Configuration

Edit `config.py` to customize:
- **Camera Settings**: Resolution, FPS, camera index
- **Hand Detection**: Confidence thresholds, max number of hands
- **Model Architecture**: Hidden layer sizes, learning rate, epochs
- **Inference**: Confidence threshold, prediction smoothing
- **Text-to-Speech**: Voice rate, volume, debounce time
- **GUI**: Window size, update interval

## Usage Guide

### Basic Operation
1. Launch the application: `python src/main.py`
2. Position your hand in front of the webcam
3. Perform a trained sign gesture
4. The application will:
   - Display the recognized sign as text
   - Show confidence score
   - Speak the sign aloud (if audio enabled)
5. Use GUI buttons to:
   - Start/Stop recognition
   - Toggle audio output
   - Clear history
   - Access settings

### Adding Custom Signs
1. Open `src/data_collector.py`
2. Add new sign names to the signs list
3. Run the collector and record samples for each new sign
4. Retrain the model with `python src/model.py --train`

### Adjusting Recognition Sensitivity
1. Open the Settings in the GUI
2. Adjust the "Confidence Threshold" slider
3. Lower values = more sensitive (more false positives)
4. Higher values = more strict (fewer false positives)

## Troubleshooting

### Webcam Not Detected
- Check if camera is connected and enabled
- Verify camera permissions (especially on macOS/Linux)
- Try: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

### Poor Hand Recognition
- Ensure good lighting conditions
- Position hand clearly in frame (not too close/far)
- Perform signs slowly and clearly
- Retrain model with more diverse data

### Low Model Accuracy
- Collect more training samples (30-50 per sign minimum)
- Record data with varied hand sizes, distances, angles
- Ensure proper hand positioning during training
- Check model training logs for convergence issues

### Audio Not Playing
- Verify system audio is enabled
- Check pyttsx3 installation: `python -c "import pyttsx3; print('OK')"`
- Adjust TTS settings in config.py

### Performance Issues (Lag, Low FPS)
- Reduce frame resolution in config.py
- Lower detection confidence thresholds
- Close other applications
- Consider GPU acceleration (CUDA for TensorFlow)

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
Follow PEP 8 Python style guidelines:
```bash
pip install flake8
flake8 src/
```

### Adding New Features
1. Create feature branch: `git checkout -b feature/your-feature`
2. Implement changes
3. Add tests in `tests/` directory
4. Submit pull request with description

## API Reference

### camera.py
```python
from src.camera import CameraHandler

camera = CameraHandler(camera_index=0)
frame = camera.read()
camera.release()
```

### hand_tracker.py
```python
from src.hand_tracker import HandTracker

tracker = HandTracker()
landmarks = tracker.extract_landmarks(frame)
```

### model.py
```python
from src.model import SignLanguageModel

model = SignLanguageModel()
model.train(data_path='data/processed/')
prediction = model.predict(landmarks)
```

## Performance Benchmarks

Target specifications:
- **Inference Latency**: <500ms per prediction
- **FPS**: 24+ frames per second
- **Memory Usage**: <500MB during operation
- **Model Accuracy**: >85% on test set

## Future Enhancements

- [ ] Support for continuous gesture sequences
- [ ] Multi-language audio output
- [ ] Custom sign vocabulary management
- [ ] Video file processing (not just webcam)
- [ ] Cloud deployment option
- [ ] Mobile app version
- [ ] Gesture confidence calibration
- [ ] User-specific model adaptation

## Known Limitations

- Single hand recognition (can be extended to dual-hand)
- Requires clear hand visibility in frame
- Performance depends on lighting conditions
- Model training requires representative data
- Currently supports pre-trained sign vocabulary

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request with detailed description

## License

This project is open source and available under the MIT License.

## Acknowledgments

- **MediaPipe**: Hand landmark detection framework by Google
- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Computer vision library
- **pyttsx3**: Cross-platform text-to-speech

## Support

For issues, questions, or suggestions:
1. Check the Troubleshooting section
2. Review existing documentation
3. Create an issue with detailed description
4. Include system info (OS, Python version, dependencies)

## Contact

For inquiries about this project, please reach out to the development team.

---

**Version**: 1.0.0  
**Last Updated**: September 2026  
**Status**: Active Development
