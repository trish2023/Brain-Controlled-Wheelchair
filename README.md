# Brain-Controlled-Wheelchair


Brain-Controlled Wheelchair – EEG + Blink-Based Navigation
This repository contains the implementation of a real-time Brain-Computer Interface (BCI) system designed to help individuals with severe motor impairments control a wheelchair using only their brain signals and eye blinks.

Using EEG data from the Neiry BCI headset and blink patterns detected via OpenCV, the system interprets user intent through a combination of cognitive state classification and blink-based commands. A lightweight Decision Tree or CNN-LSTM model processes the EEG inputs, while a rule-based logic layer fuses it with blink events to issue commands such as move forward, turn left/right, or stop.

This repo includes everything from data collection and model training to real-time inference, logging, and control logic — all geared toward accessible, low-cost, and explainable assistive technology.

## Project Structure

```plaintext
BCI/
├── dataset/
│   └── brain_data_log.csv                # Raw EEG dataset for model training
│
├── real_time_system/
│   ├── __pycache__/                      # Python bytecode cache
│   ├── data_logger/                      # EEG and system state logger module
│   ├── logs/                             # Stores real-time logs
│   ├── blink_detector.py                 # Eye blink detection using OpenCV + dlib
│   ├── brain_data_log.csv                # EEG log for real-time inference
│   ├── cnn_lstm_model.py                 # CNN-LSTM model for EEG classification
│   ├── cnn_lstm_model.cpython-313.pyc    # Compiled Python model file
│   ├── control_logic.py                  # Combines EEG + blink to generate commands
│   ├── eeg_inference.py                  # Loads model & performs real-time EEG inference
│   ├── eeg_logger.py                     # Reads EEG data and logs it
│   ├── model_weights.pth                 # Pre-trained model weights
│   ├── predicted_state_log.txt           # Log of model's predicted mental states
│   ├── shape_predictor_68_face_landmark.dat # Facial landmark model for blink detection
│   ├── shape_predictor_68_face_landmark.zip # Zipped version of facial landmark model
│   ├── train_model.py                    # Training script for the EEG classifier
│   ├── emo.py                            # (Optional) Emotion detection-related logic
│   ├── pyneurosdk2-1.0.15.tar.gz         # Neiry EEG headset SDK archive
│   └── requirements.txt                  # List of Python dependencies
```



## Control Logic
| Mental State  | Blink Type   | Action           |
| ------------- | ------------ | ---------------- |
| Focused       | Single Blink | Move Forward     |
| Focused       | Double Blink | Turn Right       |
| Focused       | Triple Blink | Turn Left        |
| Relaxed/Other | Any Input    | Stop (Safe Mode) |

