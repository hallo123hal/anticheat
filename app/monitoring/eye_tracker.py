import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional
import time


class EyeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Eye landmarks indices (left and right eye)
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

        # Looking away threshold (in degrees)
        self.LOOK_AWAY_THRESHOLD = 30  # degrees
        self.look_away_start_time: Optional[float] = None
        self.LOOK_AWAY_DURATION = 5.0  # seconds

    def calculate_eye_aspect_ratio(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio (EAR) to detect if eyes are open."""
        eye_points = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
        
        # Calculate distances
        vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])
        vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])
        horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Calculate EAR
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    def calculate_head_pose(self, landmarks, frame_shape):
        """Calculate head pose to detect if user is looking away."""
        h, w = frame_shape[:2]
        
        # Get key facial landmarks for head pose estimation
        # Landmarks indices reference (MediaPipe FaceMesh):
        # 1: Nose tip, 33: Left eye corner, 263: Right eye corner,
        # 175: Chin, 10: Forehead (glabella area),
        # 61: Mouth left corner, 291: Mouth right corner
        try:
            nose_tip = landmarks[1]
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            chin = landmarks[175]
            forehead = landmarks[10]
            mouth_left = landmarks[61]
            mouth_right = landmarks[291]
        except Exception:
            return None, None, None

        # Convert to image coordinates
        image_points = np.array([
            (nose_tip.x * w, nose_tip.y * h),
            (left_eye.x * w, left_eye.y * h),
            (right_eye.x * w, right_eye.y * h),
            (chin.x * w, chin.y * h),
            (forehead.x * w, forehead.y * h),
            (mouth_left.x * w, mouth_left.y * h),
            (mouth_right.x * w, mouth_right.y * h),
        ], dtype=np.float32)

        # 3D model points (approximate).
        # Units here are arbitrary but should keep relative geometry.
        model_points = np.array([
            (0.0,    0.0,    0.0),     # Nose tip
            (-225.0, 170.0, -135.0),   # Left eye
            (225.0,  170.0, -135.0),   # Right eye
            (0.0,    330.0,  -65.0),   # Chin
            (0.0,   -200.0, -100.0),   # Forehead
            (-150.0, -50.0, -125.0),   # Mouth left
            (150.0,  -50.0, -125.0),   # Mouth right
        ], dtype=np.float32)

        # Camera internals (approximate)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float32,
        )

        dist_coeffs = np.zeros((4, 1))

        # Solve PnP
        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
        except Exception:
            return None, None, None

        if success:
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Calculate Euler angles
            sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
            singular = sy < 1e-6

            if not singular:
                x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                y = np.arctan2(-rotation_matrix[2, 0], sy)
                z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                y = np.arctan2(-rotation_matrix[2, 0], sy)
                z = 0

            # Convert to degrees
            pitch = np.degrees(x)
            yaw = np.degrees(y)
            roll = np.degrees(z)

            return pitch, yaw, roll

        return None, None, None

    def is_looking_away(self, frame: np.ndarray) -> Tuple[bool, Optional[str]]:
        """
        Check if user is looking away.
        Returns: (is_looking_away, message)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return False, None

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        # Calculate head pose
        pitch, yaw, roll = self.calculate_head_pose(landmarks, frame.shape)

        if pitch is None or yaw is None:
            return False, None

        # Check if looking away (yaw angle indicates left/right looking)
        is_looking_away = abs(yaw) > self.LOOK_AWAY_THRESHOLD or abs(pitch) > 25

        current_time = time.time()

        if is_looking_away:
            if self.look_away_start_time is None:
                self.look_away_start_time = current_time
            else:
                # Check if looking away for 5+ seconds
                elapsed = current_time - self.look_away_start_time
                if elapsed >= self.LOOK_AWAY_DURATION:
                    self.look_away_start_time = None
                    return True, f"Looking away detected for {elapsed:.1f} seconds"
        else:
            # Reset timer if looking at screen
            self.look_away_start_time = None

        return False, None

    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """Draw face mesh landmarks on frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    None,
                    self.mp_drawing_styles.get_default_face_mesh_contours_style(),
                )

        return frame

    def release(self):
        """Release resources."""
        self.face_mesh.close()

