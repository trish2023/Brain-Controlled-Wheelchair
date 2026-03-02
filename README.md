# Brain-Controlled Wheelchair

A real-time Brain-Computer Interface (BCI) system for wheelchair navigation using EEG signals and eye blink detection.

## Overview

This project enables individuals with severe motor impairments to control a wheelchair using only brain signals and eye blinks. The system combines:

- **EEG-based mental state classification** using a CNN-LSTM deep learning model
- **Eye blink pattern detection** using OpenCV and dlib facial landmarks
- **WebSocket integration** for real-time communication with Unity-based wheelchair simulation

## Features

- Real-time EEG signal processing from Neiry BCI headset
- Mental state classification: Focused, Relaxed, Drowsy, Neutral
- Eye blink detection with pattern recognition (single, double, triple, long blinks)
- Multi-stage command validation with authentication for safety
- WebSocket connectivity for Unity integration
- Comprehensive logging for debugging and analysis

## Hardware Requirements

- **Neiry BCI Headset** (LE Headband) with 4 EEG channels (T3, T4, O1, O2)
- Webcam for eye blink detection
- Computer with Python 3.10+

## Project Structure

```
Brain-Controlled-Wheelchair/
├── dataset/
│   └── brain_data_log.csv              # EEG training dataset (~14,000+ samples)
│
├── real_time_system/
│   ├── blink_detector.py               # Eye blink detection module
│   ├── cnn_lstm_model.py               # CNN-LSTM neural network architecture
│   ├── control_logic.py                # Main controller with WebSocket integration
│   ├── eeg_inference.py                # Real-time EEG inference pipeline
│   ├── eeg_logger.py                   # EEG data acquisition from Neiry headset
│   ├── model_weights.pth               # Pre-trained model weights
│   ├── predicted_state_log.txt         # Inference output log
│   ├── brain_data_log.csv              # Real-time EEG data log
│   ├── data_logger/                    # Additional logging utilities
│   └── logs/                           # Runtime logs
│
└── README.md
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Brain-Controlled-Wheelchair.git
   cd Brain-Controlled-Wheelchair
   ```

2. **Install dependencies**
   ```bash
   pip install torch numpy pandas opencv-python dlib scipy websockets
   pip install pyneurosdk2  # For Neiry headset support
   ```

3. **Download the facial landmark model**
   - Download `shape_predictor_68_face_landmarks.dat` from [dlib's model repository](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
   - Extract and place it in the `real_time_system/` directory

## Usage

### Running the Eye Blink Controller (Standalone)
```bash
cd real_time_system
python control_logic.py
```
This starts the webcam-based blink detection with WebSocket connectivity.

### Running EEG Inference
```bash
cd real_time_system
python eeg_inference.py
```
This requires the Neiry headset to be connected and will log predicted mental states.

### Running EEG Data Collection
```bash
cd real_time_system
python eeg_logger.py
```
Connects to the Neiry headset, performs calibration, and logs EEG data to CSV.

## Control Commands

### Blink-Based Commands
| Blink Pattern | Duration/Count | Action |
|---------------|----------------|--------|
| Long Blink | >2 seconds | Toggle System ON/OFF |
| Double Blink | 2 blinks | Turn Right |
| Triple Blink | 3 blinks | Turn Left |

### Mental State + Blink Fusion
| Mental State | Blink Type | Action |
|--------------|------------|--------|
| Focused | Single Blink | Move Forward |
| Focused | Double Blink | Turn Right |
| Focused | Triple Blink | Turn Left |
| Relaxed/Other | Any | Stop (Safe Mode) |

## Model Architecture

### CNN-LSTM Network
- **Input**: 30-frame sliding window of 8 EEG features
- **Conv1D Layers**: 8→32→64 channels with ReLU activation and max pooling
- **LSTM Layer**: 64→128 hidden units for temporal pattern learning
- **Output**: 4-class classification (Focused, Relaxed, Drowsy, Neutral)

### EEG Features
The model processes spectral band powers:
- Delta, Theta, Alpha, Beta, Gamma
- Mental parameters from emotional math processing

### Classification Logic
```python
if alpha >= 0.52 and theta < 0.21 and beta >= 0.26:
    return "Focused"
elif alpha >= 0.52 and beta < 0.26:
    return "Relaxed"
elif theta >= 0.21 and beta < 0.26:
    return "Drowsy"
else:
    return "Neutral"
```

## Eye Aspect Ratio (EAR) Calculation

Blink detection uses the Eye Aspect Ratio formula:
```
EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
```
- Threshold: EAR < 0.2 indicates eye closure
- Long blink: Eyes closed for >2 seconds

## WebSocket Integration

The system connects to a WebSocket server (e.g., Unity via ngrok) to send movement commands:
```json
{"command": "MOVE", "direction": "RIGHT"}
{"command": "MOVE", "direction": "LEFT"}
```

Configure the WebSocket URI in `control_logic.py`:
```python
self.websocket_uri = "wss://your-ngrok-url.ngrok-free.app"
```

## Safety Features

- **Multi-stage command validation**: Commands require detection + authentication phases
- **EAR stability check**: Commands rejected if eye tracking is unstable
- **Safe mode**: System stops when user is not in "Focused" mental state
- **Manual override**: Long blink toggles between automatic and manual modes

## Dataset Format

The EEG dataset (`brain_data_log.csv`) contains:
```
timestamp, calibration_percent, is_artifacted_sequence,
mental_param_1, mental_param_2, mental_param_3,
spectral_delta, spectral_theta, spectral_alpha, spectral_beta, spectral_gamma,
attention_state
```

## Troubleshooting

- **Headset not found**: Ensure Bluetooth is enabled and headset is in pairing mode
- **Poor signal quality**: Check electrode placement and resistance values (<2MΩ)
- **Blinks not detected**: Ensure adequate lighting and face is visible to camera
- **WebSocket connection failed**: Verify ngrok URL and network connectivity

## License

This project is developed for assistive technology research and educational purposes.

