import cv2
import threading
import time
import queue
from typing import Optional
import numpy as np
from .face_detector import FaceDetector
from .eye_tracker import EyeTracker
from .voice_detector import VoiceDetector


class CameraMonitor:
    def __init__(self, camera_index: int = 0, violation_queue: Optional[queue.Queue] = None):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Detectors
        self.face_detector = FaceDetector()
        self.eye_tracker = EyeTracker()
        self.voice_detector = VoiceDetector()
        
        # Violation queue for async communication
        self.violation_queue = violation_queue or queue.Queue()
        
        # State tracking
        self.last_face_detection_time: Optional[float] = None
        self.face_absence_threshold = 3.0  # seconds
        self.frame_rate = 30
        self.frame_interval = 1.0 / self.frame_rate
        self.last_voice_check_time = 0.0
        self.voice_check_interval = 1.0  # Check voice every second
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.monitor_start_time: float = 0.0
        self.startup_grace_period = 5.0  # seconds to ignore violations after start

    def start_monitoring(self):
        """Start monitoring camera and microphone."""
        if self.is_monitoring:
            return

        try:
            # Open camera
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                # Try default camera
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise Exception(f"Could not open camera. Please ensure camera is connected and accessible.")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, self.frame_rate)
            
            # Start voice monitoring (may fail if microphone is not available)
            try:
                self.voice_detector.start_monitoring()
            except Exception as mic_error:
                print(f"Warning: Could not start voice monitoring: {mic_error}")
                # Continue without voice detection
            
            # Start camera monitoring thread
            self.is_monitoring = True
            self.monitor_start_time = time.time()
            self.last_voice_check_time = self.monitor_start_time
            self.last_face_detection_time = self.monitor_start_time
            self.monitoring_thread = threading.Thread(target=self._monitor_camera, daemon=True)
            self.monitoring_thread.start()
            
        except Exception as e:
            print(f"Error starting camera monitoring: {e}")
            # Don't raise exception - allow service to continue
            # Camera monitoring may not work, but WebSocket connections will
            self.is_monitoring = False
            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None

    def _monitor_camera(self):
        """Monitor camera in background thread."""
        while self.is_monitoring:
            try:
                if self.cap is None or not self.cap.isOpened():
                    break
                
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                
                # Process frame for violations
                self._process_frame(frame)
                
                # Control frame rate
                time.sleep(self.frame_interval)
                
            except Exception as e:
                print(f"Error in camera monitoring: {e}")
                break

    def _process_frame(self, frame: np.ndarray):
        """Process frame to detect violations."""
        current_time = time.time()

        # Skip violation checks during initial grace period to allow user setup.
        if current_time - self.monitor_start_time < self.startup_grace_period:
            self.last_face_detection_time = current_time
            self.last_voice_check_time = current_time
            return
        
        # 1. Face presence detection
        is_face_present, face_data = self.face_detector.detect_face(frame)
        
        if not is_face_present:
            if self.last_face_detection_time is None:
                self.last_face_detection_time = current_time
            else:
                # Check if face has been absent for threshold duration
                absence_duration = current_time - self.last_face_detection_time
                if absence_duration >= self.face_absence_threshold:
                    # Put violation in queue
                    try:
                        self.violation_queue.put_nowait((
                            "face_presence",
                            f"Face not detected for {absence_duration:.1f} seconds"
                        ))
                    except queue.Full:
                        print("Violation queue is full")
                    return
        else:
            self.last_face_detection_time = current_time
            
            # 2. Eye gaze detection
            is_looking_away, message = self.eye_tracker.is_looking_away(frame)
            if is_looking_away:
                # Put violation in queue
                try:
                    self.violation_queue.put_nowait(("eye_gaze", message))
                except queue.Full:
                    print("Violation queue is full")
                return
        
        # 3. Voice detection (check periodically, not every frame)
        if current_time - self.last_voice_check_time >= self.voice_check_interval:
            self.last_voice_check_time = current_time
            if self.voice_detector.is_human_speech_detected():
                # Put violation in queue
                try:
                    self.violation_queue.put_nowait(("voice", "Human speech detected"))
                except queue.Full:
                    print("Violation queue is full")
                return

    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame from camera."""
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return frame

    def stop_monitoring(self):
        """Stop monitoring camera and microphone."""
        self.is_monitoring = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        
        self.voice_detector.stop_monitoring()
        self.last_face_detection_time = None
        with self.frame_lock:
            self.latest_frame = None
        self.monitor_start_time = 0.0

    def release(self):
        """Release all resources."""
        self.stop_monitoring()
        self.face_detector.release()
        self.eye_tracker.release()
        self.voice_detector.release()

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the latest captured frame."""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def is_camera_open(self) -> bool:
        """Check if the underlying camera device is currently opened."""
        return self.cap is not None and self.cap.isOpened()

