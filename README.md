# Brain-Controlled-Wheelchair


🧠 Brain-Controlled Wheelchair – EEG + Blink-Based Navigation
This repository contains the implementation of a real-time Brain-Computer Interface (BCI) system designed to help individuals with severe motor impairments control a wheelchair using only their brain signals and eye blinks.

Using EEG data from the Neiry BCI headset and blink patterns detected via OpenCV, the system interprets user intent through a combination of cognitive state classification and blink-based commands. A lightweight Decision Tree or CNN-LSTM model processes the EEG inputs, while a rule-based logic layer fuses it with blink events to issue commands such as move forward, turn left/right, or stop.

This repo includes everything from data collection and model training to real-time inference, logging, and control logic — all geared toward accessible, low-cost, and explainable assistive technology.

📁 Project Structure

├── dataset/
│   └── brain_data_log.csv                # Raw EEG dataset for training
│
├── real_time_system/
│   ├── __pycache__/                      # Python bytecode cache
│   ├── data_logger/                      # Module to log EEG signals and predictions
│   ├── logs/                             # Runtime log files
│   ├── blink_detector.py                 # Blink detection using OpenCV and dlib
│   ├── brain_data_log.csv                # EEG log for live testing
│   ├── cnn_lstm_model.py                 # CNN-LSTM architecture (optional)
│   ├── cnn_lstm_model.cpython-313.pyc    # Compiled version
│   ├── control_logic.py                  # Fusion of EEG and blink inputs into actions
│   ├── eeg_inference.py                  # Real-time EEG classification logic
│   ├── eeg_logger.py                     # Logger for live EEG stream
│   ├── model_weights.pth                 # Saved model weights
│   ├── predicted_state_log.txt           # Output log of classified mental states
│   ├── shape_predictor_68_face_landmark.dat  # Landmark model for face/eye detection
│   ├── shape_predictor_68_face_landmark.zip  # Compressed version (backup)
│   ├── train_model.py                    # Model training script
│   ├── emo.py                            # (Optional) Emotion-related utilities
│   ├── pyneurosdk2-1.0.15.tar.gz         # Neiry BCI SDK package
│   └── requirements.txt                  # Required Python libraries



🧠 Control Logic
| Mental State  | Blink Type   | Action           |
| ------------- | ------------ | ---------------- |
| Focused       | Single Blink | Move Forward     |
| Focused       | Double Blink | Turn Right       |
| Focused       | Triple Blink | Turn Left        |
| Relaxed/Other | Any Input    | Stop (Safe Mode) |

