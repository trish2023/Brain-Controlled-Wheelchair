import cv2
import numpy as np
import time
import threading
import dlib
import queue
import os
import sys
from scipy.spatial import distance as dist
from collections import deque

class EyeBlinkDetector:
    def __init__(self, blink_queue):
        """
        Initialize the Eye Blink Detector.
        
        Args:
            blink_queue: Queue to send detected blink patterns to
        """
        self.blink_queue = blink_queue
        
        # Initialize the face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        
        # Try to find the shape predictor file
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(predictor_path):
            # Try common paths
            possible_paths = [
                "models/shape_predictor_68_face_landmarks.dat",
                "data/shape_predictor_68_face_landmarks.dat",
                os.path.join(os.getcwd(), "shape_predictor_68_face_landmarks.dat")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    predictor_path = path
                    break
            else:
                print("ERROR: shape_predictor_68_face_landmarks.dat not found!")
                print("Please download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
                print("Extract it and place it in the same directory as this script.")
                raise FileNotFoundError("Shape predictor file not found")
        
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Define indexes for facial landmarks for the eyes
        self.LEFT_EYE_START = 36
        self.LEFT_EYE_END = 41
        self.RIGHT_EYE_START = 42
        self.RIGHT_EYE_END = 47
        
        # Blink detection variables
        self.eyes_open = True
        self.blink_start_time = 0
        self.blinks_count = 0
        self.last_blink_time = 0
        
        # EAR thresholds and timing
        self.EAR_THRESHOLD = 0.2  # Eye Aspect Ratio threshold for eye closure
        self.long_blink_threshold = 2.0  # Long blink threshold (2 seconds)
        self.blink_sequence_timeout = 1.0  # seconds between blinks to count as a sequence
        self.blink_timeout = 0.5  # seconds to reset blink count
        
        # EAR tracking
        self.ear_history = []
        self.current_ear = 1.0
        
        # Control variables
        self.running = False
        self.thread = None
        
        # Debug
        self.debug = True
        self.command_history = []
        
        print("[BLINK] EyeBlinkDetector initialized successfully")
    
    def eye_aspect_ratio(self, eye_points):
        """Calculate the Eye Aspect Ratio (EAR) for a set of eye landmarks."""
        # Compute the euclidean distances between the vertical eye landmarks
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        
        # Compute the euclidean distance between the horizontal eye landmarks
        C = dist.euclidean(eye_points[0], eye_points[3])
        
        # Calculate the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_eyes(self, frame):
        """Detect eyes in the frame and determine if they are open or closed using EAR."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        # Default values if no face is detected
        current_ear = 1.0
        eyes_open = True
        
        for face in faces:
            # Get facial landmarks
            landmarks = self.predictor(gray, face)
            
            # Extract eye regions
            left_eye_points = []
            right_eye_points = []
            
            for i in range(self.LEFT_EYE_START, self.LEFT_EYE_END + 1):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                left_eye_points.append((x, y))
                if self.debug:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            for i in range(self.RIGHT_EYE_START, self.RIGHT_EYE_END + 1):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                right_eye_points.append((x, y))
                if self.debug:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            if self.debug:
                # Draw convex hull around each eye
                left_eye_hull = cv2.convexHull(np.array(left_eye_points))
                right_eye_hull = cv2.convexHull(np.array(right_eye_points))
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
                
                # Draw bounding box for the face
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Calculate EAR for both eyes
            left_ear = self.eye_aspect_ratio(left_eye_points)
            right_ear = self.eye_aspect_ratio(right_eye_points)
            
            # Average EAR for both eyes
            current_ear = (left_ear + right_ear) / 2.0
            
            # Determine if eyes are open or closed based on EAR
            eyes_open = current_ear > self.EAR_THRESHOLD
            
            if self.debug:
                # Display EAR on frame
                cv2.putText(frame, f"EAR: {current_ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Store the current EAR
        self.current_ear = current_ear
        self.ear_history.append(current_ear)
        if len(self.ear_history) > 30:  # Keep only the last 30 values
            self.ear_history.pop(0)
        
        return eyes_open, frame
    
    def process_blink(self, eyes_open):
        """Process the current eye state to detect blinks and sequences."""
        current_time = time.time()
        
        # Detect eye state changes
        if self.eyes_open and not eyes_open:  # Eyes just closed
            self.blink_start_time = current_time
            self.eyes_open = False
            
        elif not self.eyes_open and eyes_open:  # Eyes just opened
            self.eyes_open = True
            blink_duration = current_time - self.blink_start_time
            
            # Check if this is a long blink
            if blink_duration > self.long_blink_threshold:
                self.send_blink_command("long")
                self.add_command(f"Long blink detected ({blink_duration:.1f}s)")
            else:
                # This is a normal short blink, part of a potential sequence
                if self.blinks_count == 0 or (current_time - self.last_blink_time) < self.blink_sequence_timeout:
                    self.blinks_count += 1
                else:
                    # Too much time has passed, start a new sequence
                    self.blinks_count = 1
                
                self.last_blink_time = current_time
                self.add_command(f"Blink {self.blinks_count} detected")
                
        # Check if we need to process a completed blink sequence
        if self.blinks_count > 0 and (current_time - self.last_blink_time) > self.blink_timeout:
            if self.blinks_count == 2:  # Double blink
                self.send_blink_command("double")
                self.add_command("Double blink sequence completed")
            elif self.blinks_count == 3:  # Triple blink
                self.send_blink_command("triple")
                self.add_command("Triple blink sequence completed")
            elif self.blinks_count > 3:
                self.add_command(f"Complex sequence ({self.blinks_count} blinks) - ignored")
            
            self.blinks_count = 0
    
    def send_blink_command(self, blink_type):
        """Send blink command to the queue."""
        try:
            self.blink_queue.put(blink_type, block=False)
            print(f"[BLINK] Sent: {blink_type}")
        except queue.Full:
            print(f"[BLINK] Warning: Queue full, dropping {blink_type} command")
    
    def add_command(self, command):
        """Add a command to the history with a timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        self.command_history.append(f"[{timestamp}] {command}")
        # Keep only the last 5 commands
        if len(self.command_history) > 5:
            self.command_history.pop(0)
    
    def draw_status(self, frame):
        """Draw status information on the frame."""
        if not self.debug:
            return frame
            
        # Current blink count
        cv2.putText(frame, f"Blinks: {self.blinks_count}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Eye state
        state = "OPEN" if self.eyes_open else "CLOSED"
        color = (0, 255, 0) if self.eyes_open else (0, 0, 255)
        cv2.putText(frame, f"Eyes: {state}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Commands info
        cv2.putText(frame, "Commands:", (10, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Long blink (2s): Toggle system", (10, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "Double blink: Turn right", (10, 180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "Triple blink: Turn left", (10, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Command history
        y_pos = 250
        for cmd in self.command_history[-3:]:  # Show last 3 commands
            cv2.putText(frame, cmd, (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            y_pos += 20
        
        # EAR visualization
        if len(self.ear_history) > 1:
            self.draw_ear_graph(frame)
        
        return frame
    
    def draw_ear_graph(self, frame):
        """Draw EAR history graph."""
        # Graph parameters
        graph_width = 200
        graph_height = 50
        graph_x = 400
        graph_y = 30
        
        # Normalize EAR history for visualization
        if self.ear_history:
            hist_max = max(1.0, max(self.ear_history))
            hist_min = min(0.0, min(self.ear_history))
            hist_range = max(0.1, hist_max - hist_min)
            
            # Draw background
            cv2.rectangle(frame, (graph_x, graph_y), 
                          (graph_x + graph_width, graph_y + graph_height), 
                          (0, 0, 0), -1)
            
            # Draw threshold line
            threshold_y = int(graph_y + graph_height - (self.EAR_THRESHOLD - hist_min) / hist_range * graph_height)
            cv2.line(frame, (graph_x, threshold_y), (graph_x + graph_width, threshold_y), 
                     (0, 255, 255), 1)
            
            # Draw EAR history
            for i in range(1, len(self.ear_history)):
                pt1_x = graph_x + int((i-1) * graph_width / len(self.ear_history))
                pt1_y = int(graph_y + graph_height - (self.ear_history[i-1] - hist_min) / hist_range * graph_height)
                
                pt2_x = graph_x + int(i * graph_width / len(self.ear_history))
                pt2_y = int(graph_y + graph_height - (self.ear_history[i] - hist_min) / hist_range * graph_height)
                
                cv2.line(frame, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 255, 0), 1)
            
            # Label the graph
            cv2.putText(frame, "EAR History", (graph_x, graph_y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _run(self):
        """Main detection loop (runs in separate thread)."""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[BLINK] ERROR: Could not open camera!")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("[BLINK] Camera initialized, starting detection...")
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("[BLINK] WARNING: Failed to read from camera!")
                    continue
                    
                # Mirror the frame horizontally for a more natural view
                frame = cv2.flip(frame, 1)
                
                # Detect eyes and process the frame
                eyes_open, processed_frame = self.detect_eyes(frame)
                
                # Process blinks
                self.process_blink(eyes_open)
                
                if self.debug:
                    # Draw status information
                    processed_frame = self.draw_status(processed_frame)
                    
                    # Display the processed frame
                    cv2.imshow('Eye Blink Detector', processed_frame)
                    
                    # Check for quit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or cv2.getWindowProperty('Eye Blink Detector', cv2.WND_PROP_VISIBLE) < 1:
                        break
        
        except Exception as e:
            print(f"[BLINK] Error in detection loop: {e}")
        
        finally:
            cap.release()
            if self.debug:
                cv2.destroyAllWindows()
            print("[BLINK] Detection stopped")
    
    def start(self):
        """Start the blink detection in a separate thread."""
        if self.running:
            print("[BLINK] Already running!")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("[BLINK] Detection started")
    
    def stop(self):
        """Stop the blink detection."""
        if not self.running:
            return
        
        self.running = False
        print("[BLINK] Stopping detection...")
    
    def join(self):
        """Wait for the detection thread to finish."""
        if self.thread and self.thread.is_alive():
            self.thread.join()
    
    def is_running(self):
        """Check if the detector is currently running."""
        return self.running and (self.thread is not None) and self.thread.is_alive()


# For standalone testing
if __name__ == "__main__":
    print("Testing Eye Blink Detector...")
    
    # Create a test queue
    test_queue = queue.Queue()
    
    # Create detector
    detector = EyeBlinkDetector(test_queue)
    
    try:
        # Start detection
        detector.start()
        
        print("Press 'q' in the camera window to quit")
        print("Listening for blink commands...")
        
        # Monitor the queue for commands
        start_time = time.time()
        while detector.is_running():
            try:
                # Check for blink commands
                blink_type = test_queue.get(timeout=1.0)
                print(f"Received blink command: {blink_type}")
            except queue.Empty:
                # Check if we should quit (after 60 seconds for testing)
                if time.time() - start_time > 60:
                    print("Test timeout reached")
                    break
                continue
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    finally:
        detector.stop()
        detector.join()
        print("Test completed")
