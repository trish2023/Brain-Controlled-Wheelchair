import cv2
import numpy as np
import time
import dlib
from scipy.spatial import distance as dist
import threading
import json
import asyncio

# Try to import websockets with fallback
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    print("Warning: websockets not available. Install with: pip install websockets")
    WEBSOCKETS_AVAILABLE = False

class EyeBlinkController:
    def __init__(self):
        # Initialize the face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # Define indexes for facial landmarks for the eyes
        self.LEFT_EYE_START = 36
        self.LEFT_EYE_END = 41
        self.RIGHT_EYE_START = 42
        self.RIGHT_EYE_END = 47

        # State variables
        self.is_system_active = False
        self.mode = "manual"  # "manual" or "automatic" 

        # Blink detection variables
        self.eyes_open = True
        self.blink_start_time = 0
        self.blinks_count = 0
        self.last_blink_time = 0
        self.blink_sequence_start_time = 0

        # EAR thresholds
        self.EAR_THRESHOLD = 0.25
        self.long_blink_threshold = 2.0
        self.blink_sequence_timeout = 1.0
        self.blink_timeout = 0.5

        # Command timing parameters
        self.command_detection_time = 3.0
        self.command_auth_time = 2.0

        # EAR tracking
        self.ear_history = []
        self.current_ear = 1.0
        self.is_blink = False
        self.blink_counter = 0
        self.last_blink_check = time.time()

        # Movement command variables
        self.current_direction = "stop"
        self.pending_command = None
        self.is_detecting_command = False
        self.command_detection_start = 0
        self.is_authenticating_command = False
        self.command_auth_start = 0
        self.command_sequence = []

        # Debug and UI
        self.debug = True
        self.command_history = []

        # WebSocket variables
        self.websocket_uri = "wss://db55-103-213-211-203.ngrok-free.app"
        self.websocket_thread = None
        self.websocket_running = False

    def eye_aspect_ratio(self, eye_points):
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        C = dist.euclidean(eye_points[0], eye_points[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def detect_eyes(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        current_ear = 1.0
        eyes_open = True

        for face in faces:
            landmarks = self.predictor(gray, face)
            left_eye_points, right_eye_points = [], []

            for i in range(self.LEFT_EYE_START, self.LEFT_EYE_END + 1):
                x, y = landmarks.part(i).x, landmarks.part(i).y
                left_eye_points.append((x, y))
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            for i in range(self.RIGHT_EYE_START, self.RIGHT_EYE_END + 1):
                x, y = landmarks.part(i).x, landmarks.part(i).y
                right_eye_points.append((x, y))
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            left_eye_hull = cv2.convexHull(np.array(left_eye_points))
            right_eye_hull = cv2.convexHull(np.array(right_eye_points))
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

            left_ear = self.eye_aspect_ratio(left_eye_points)
            right_ear = self.eye_aspect_ratio(right_eye_points)
            current_ear = (left_ear + right_ear) / 2.0
            eyes_open = current_ear > self.EAR_THRESHOLD

            cv2.putText(frame, f"EAR: {current_ear:.2f}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        self.current_ear = current_ear
        self.ear_history.append(current_ear)
        if len(self.ear_history) > 30:
            self.ear_history.pop(0)

        return eyes_open, frame

    def process_blink(self, eyes_open):
        current_time = time.time()

        if self.is_authenticating_command:
            if current_time - self.command_auth_start >= self.command_auth_time:
                self.validate_and_execute_command()
                return

        elif self.is_detecting_command:
            self.command_sequence.append((current_time, eyes_open, self.current_ear))
            if current_time - self.command_detection_start >= self.command_detection_time:
                self.analyze_command_sequence()
                return

        if self.eyes_open and not eyes_open:
            self.blink_start_time = current_time
            self.eyes_open = False

        elif not self.eyes_open and eyes_open:
            self.eyes_open = True
            blink_duration = current_time - self.blink_start_time

            if blink_duration > self.long_blink_threshold:
                self.handle_long_blink()
            else:
                if self.blinks_count == 0 or (current_time - self.last_blink_time) < self.blink_sequence_timeout:
                    if self.blinks_count == 0:
                        self.blink_sequence_start_time = current_time
                    self.blinks_count += 1
                else:
                    self.blinks_count = 1
                    self.blink_sequence_start_time = current_time
                self.last_blink_time = current_time

        if self.blinks_count > 0 and (current_time - self.last_blink_time) > self.blink_timeout:
            if self.blinks_count == 2:
                self.start_command_detection("DOUBLE_BLINK")
            elif self.blinks_count == 3:
                self.start_command_detection("TRIPLE_BLINK")
            self.blinks_count = 0

    def start_command_detection(self, command_type):
        self.is_detecting_command = True
        self.command_detection_start = time.time()
        self.pending_command = command_type
        self.command_sequence = []

        message = f"Potential {command_type} detected - collecting data for {self.command_detection_time}s..."
        self.add_command(message)
        print(message)

    def analyze_command_sequence(self):
        self.is_detecting_command = False
        if len(self.command_sequence) < 10:
            self.add_command("Command detection failed: Insufficient data")
            print("Command detection failed: Insufficient data")
            return

        ear_values = [ear for _, _, ear in self.command_sequence]
        ear_stability = np.std(ear_values)
        command_valid = False

        if self.pending_command in ["DOUBLE_BLINK", "TRIPLE_BLINK"]:
            command_valid = ear_stability < 0.15

        if command_valid:
            self.start_command_authentication()
        else:
            self.add_command(f"{self.pending_command} command rejected: Pattern not recognized")
            print(f"{self.pending_command} command rejected: Pattern not recognized")

    def start_command_authentication(self):
        self.is_authenticating_command = True
        self.command_auth_start = time.time()

        message = f"{self.pending_command} detected - authenticating for {self.command_auth_time}s..."
        self.add_command(message)
        print(message)

    def validate_and_execute_command(self):
        self.is_authenticating_command = False
        recent_ear_values = self.ear_history[-15:]
        if len(recent_ear_values) < 15:
            recent_ear_values = self.ear_history
        ear_stability = np.std(recent_ear_values)

        if ear_stability < 0.1:
            if self.pending_command == "DOUBLE_BLINK":
                self.handle_double_blink()
            elif self.pending_command == "TRIPLE_BLINK":
                self.handle_triple_blink()
        else:
            self.add_command(f"{self.pending_command} command failed: Unstable during authentication")
            print(f"{self.pending_command} command failed: Unstable during authentication")

    def handle_long_blink(self):
        self.is_system_active = not self.is_system_active
        if self.is_system_active:
            status = "STARTED"
            self.mode = "automatic"
            self.add_command(f"System {status}")
            self.add_command("Switched to automatic mode")
            print(f"System {status}")
            print("Switched to automatic mode")
        else:
            status = "STOPPED"
            self.mode = "manual"
            self.add_command(f"System {status}")
            self.add_command("Switched to manual mode")
            print(f"System {status}")
            print("Switched to manual mode")

    def handle_double_blink(self):
        if not self.is_system_active:
            return
        self.current_direction = "right"
        self.add_command("Turn RIGHT")
        print("Turn RIGHT")
        self.mode = "automatic"
        # Send WebSocket command
        self.send_websocket_command({"command": "MOVE", "direction": "RIGHT"})

    def handle_triple_blink(self):
        if not self.is_system_active:
            return
        self.current_direction = "left"
        self.add_command("Turn LEFT")
        print("Turn LEFT")
        self.mode = "automatic"
        # Send WebSocket command
        self.send_websocket_command({"command": "MOVE", "direction": "LEFT"})

    def add_command(self, command):
        timestamp = time.strftime("%H:%M:%S")
        self.command_history.append(f"[{timestamp}] {command}")
        if len(self.command_history) > 5:
            self.command_history.pop(0)

    def draw_status(self, frame):
        status = "ACTIVE" if self.is_system_active else "INACTIVE"
        cv2.putText(frame, f"System: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Mode: {self.mode.upper()}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Direction: {self.current_direction.upper()}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # WebSocket status
        ws_status = "CONNECTED" if WEBSOCKETS_AVAILABLE and self.websocket_running else "DISCONNECTED"
        ws_color = (0, 255, 0) if WEBSOCKETS_AVAILABLE and self.websocket_running else (0, 0, 255)
        cv2.putText(frame, f"WebSocket: {ws_status}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ws_color, 2)

        command_box_y = 300
        cv2.rectangle(frame, (10, command_box_y), (300, command_box_y + 120), (0, 0, 0), -1)
        cv2.putText(frame, "COMMANDS:", (15, command_box_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Long Blink (2s): Start/Stop", (15, command_box_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "Double Blink: Turn Right", (15, command_box_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "Triple Blink: Turn Left", (15, command_box_y + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if self.is_detecting_command:
            elapsed = time.time() - self.command_detection_start
            remaining = self.command_detection_time - elapsed
            cmd_text = "TURN RIGHT" if self.pending_command == "DOUBLE_BLINK" else "TURN LEFT"
            cv2.putText(frame, f"Detecting: {cmd_text}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            cv2.putText(frame, f"Time remaining: {remaining:.1f}s", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

        elif self.is_authenticating_command:
            elapsed = time.time() - self.command_auth_start
            remaining = self.command_auth_time - elapsed
            cmd_text = "TURN RIGHT" if self.pending_command == "DOUBLE_BLINK" else "TURN LEFT"
            cv2.putText(frame, f"Authenticating: {cmd_text}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            cv2.putText(frame, f"Time remaining: {remaining:.1f}s", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

        return frame

    # WebSocket methods
    def send_websocket_command(self, command):
        """Add command to queue for WebSocket sending"""
        if hasattr(self, 'command_queue'):
            self.command_queue.put(command)

    async def websocket_client(self):
        """WebSocket client coroutine"""
        if not WEBSOCKETS_AVAILABLE:
            print("[WebSocket] WebSockets not available - running in simulation mode")
            while self.websocket_running:
                try:
                    command = self.command_queue.get(timeout=1.0)
                    print(f"[SIMULATION] Would send to Unity: {command}")
                except:
                    continue
            return

        max_retries = 5
        retry_count = 0
        
        while self.websocket_running and retry_count < max_retries:
            try:
                print(f"[WebSocket] Attempting to connect to {self.websocket_uri}...")
                
                # Configure headers for ngrok
                extra_headers = {}
                if "ngrok" in self.websocket_uri:
                    extra_headers["ngrok-skip-browser-warning"] = "true"
                
                async with websockets.connect(self.websocket_uri, extra_headers=extra_headers) as ws:
                    print("[WebSocket] Connected successfully")
                    retry_count = 0  # Reset retry count on successful connection
                    
                    while self.websocket_running:
                        try:
                            # Check for commands to send
                            command = self.command_queue.get(timeout=1.0)
                            await ws.send(json.dumps(command))
                            print(f"[WebSocket] Sent to Unity: {command}")
                        except:
                            # Send ping to keep connection alive
                            try:
                                await ws.ping()
                                await asyncio.sleep(1)
                            except:
                                break  # Connection lost
                            
            except Exception as e:
                retry_count += 1
                print(f"[WebSocket] Connection failed (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    print(f"[WebSocket] Retrying in 5 seconds...")
                    await asyncio.sleep(5)
                else:
                    print("[WebSocket] Max retries reached. Switching to simulation mode.")
                    # Fall back to simulation mode
                    while self.websocket_running:
                        try:
                            command = self.command_queue.get(timeout=1.0)
                            print(f"[SIMULATION] Would send to Unity: {command}")
                        except:
                            continue

    def websocket_thread_worker(self):
        """WebSocket thread worker function"""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.websocket_client())
        except Exception as e:
            print(f"[WebSocket] Thread error: {e}")
        finally:
            try:
                loop.close()
            except:
                pass

    def start_websocket(self):
        """Start WebSocket connection in separate thread"""
        import queue
        self.command_queue = queue.Queue()
        self.websocket_running = True
        self.websocket_thread = threading.Thread(target=self.websocket_thread_worker, daemon=True)
        self.websocket_thread.start()
        print("[WebSocket] WebSocket thread started")

    def stop_websocket(self):
        """Stop WebSocket connection"""
        self.websocket_running = False
        if self.websocket_thread and self.websocket_thread.is_alive():
            self.websocket_thread.join(timeout=5)
        print("[WebSocket] WebSocket stopped")

controller = EyeBlinkController()

# Start WebSocket connection
controller.start_websocket()

cap = cv2.VideoCapture(0)

print("Eye Blink Controller with WebSocket started")
print("WebSocket URI:", controller.websocket_uri)
print("Press 'q' to quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        eyes_open, frame = controller.detect_eyes(frame)
        controller.process_blink(eyes_open)
        frame = controller.draw_status(frame)

        cv2.imshow("Eye Blink Controller", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    controller.stop_websocket()
    cap.release()
    cv2.destroyAllWindows()
