# Real-time EEG inference using CNN+LSTM
import subprocess
import threading
import queue
import torch
import pandas as pd
import time
from cnn_lstm_model import CNNLSTM

# Configurations
WINDOW_SIZE = 30
LABELS = ['Focused', 'Relaxed', 'Drowsy', 'Neutral']
OUTPUT_PATH = 'predicted_state_log.txt'

# Queue for EEG data
eeg_queue = queue.Queue()

# Load model
input_size = 8
hidden_size = 128
num_classes = 4
model = CNNLSTM(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
model.load_state_dict(torch.load('model_weights.pth', map_location='cpu'))
model.eval()

# Producer: Reads data from the logger subprocess
def eeg_logger_producer():
    logger = subprocess.Popen(['python', 'eeg_logger.py'], stdout=subprocess.PIPE, text=True)

    for line in logger.stdout:
        if line.strip():
            eeg_data = line.strip().split(',')
            eeg_queue.put(eeg_data)

# Consumer: Reads data from the queue, processes, and predicts
def eeg_inference_consumer():
    buffer = []

    while True:
        data = eeg_queue.get()
        if len(data) >= 8:  # Make sure it has 8 features
            buffer.append(list(map(float, data[3:11])))  # Extract relevant features

        if len(buffer) >= WINDOW_SIZE:
            x = torch.tensor(buffer[-WINDOW_SIZE:], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                output = model(x)
                predicted_class = torch.argmax(output, dim=1).item()
                predicted_state = LABELS[predicted_class]

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
              f.write(f"{timestamp} → {predicted_state}\n")


            print(f"[{timestamp}] Predicted: {predicted_state}")

# Run threads
logger_thread = threading.Thread(target=eeg_logger_producer, daemon=True)
inference_thread = threading.Thread(target=eeg_inference_consumer, daemon=True)

logger_thread.start()
inference_thread.start()

logger_thread.join()
inference_thread.join()
