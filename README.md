# Brain-Controlled-Wheelchair


🧠 Brain-Controlled Wheelchair – EEG + Blink-Based Navigation
This repository contains the implementation of a real-time Brain-Computer Interface (BCI) system designed to help individuals with severe motor impairments control a wheelchair using only their brain signals and eye blinks.

Using EEG data from the Neiry BCI headset and blink patterns detected via OpenCV, the system interprets user intent through a combination of cognitive state classification and blink-based commands. A lightweight Decision Tree or CNN-LSTM model processes the EEG inputs, while a rule-based logic layer fuses it with blink events to issue commands such as move forward, turn left/right, or stop.

This repo includes everything from data collection and model training to real-time inference, logging, and control logic — all geared toward accessible, low-cost, and explainable assistive technology.

📁 Project Structure
BCI/
│
├── dataset/
│   └── brain_data_log.csv               # Raw EEG dataset used for training
│
├── real_time_system/
│   ├── __pycache__/                     # Cached Python files
│   ├── data_logger/                     # Module for logging EEG and system states
│   ├── logs/                            # Runtime logs for debugging
│   ├── blink_detector.py                # Detects blinks using OpenCV + dlib
│   ├── brain_data_log.csv               # EEG data used in real-time simulation
│   ├── cnn_lstm_model.py                # CNN-LSTM model definition (for comparison)
│   ├── cnn_lstm_model.cpython-313.pyc   # Compiled version of the model
│   ├── control_logic.py                 # Rule-based logic combining EEG & blink inputs
│   ├── eeg_inference.py                 # Live EEG classification using trained model
│   ├── eeg_logger.py                    # Collects and logs EEG data in real-time
│   ├── model_weights.pth                # Saved weights for the trained model
│   ├── predicted_state_log.txt          # Output log of predicted mental states
│   ├── shape_predictor_68_face_landmark.dat # Landmark model for blink detection
│   ├── shape_predictor_68_face_landmark.zip # (Compressed version)
│   ├── train_model.py                   # Script to train Decision Tree or CNN+LSTM
│   ├── emo.py                           # (Optional) Emo state-related processing
│   ├── pyneurosdk2-1.0.15.tar.gz        # SDK for Neiry EEG headset integration
│   └── requirements.txt                 # All required Python packages

🧠 Control Logic
| Mental State  | Blink Type   | Action           |
| ------------- | ------------ | ---------------- |
| Focused       | Single Blink | Move Forward     |
| Focused       | Double Blink | Turn Right       |
| Focused       | Triple Blink | Turn Left        |
| Relaxed/Other | Any Input    | Stop (Safe Mode) |

