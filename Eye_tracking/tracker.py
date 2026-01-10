"""
Eye tracking module using MediaPipe FaceLandmarker (Tasks API).
Extracts iris positions and predicts gaze coordinates using a calibrated model.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict
import logging
import os
import urllib.request
import time

# MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MediaPipe iris landmark indices
# Left iris: 468-472 (5 points, center is average)
# Right iris: 473-477 (5 points, center is average)
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

# Eye corner landmarks for optional relative positioning
LEFT_EYE_INNER_CORNER = 133
LEFT_EYE_OUTER_CORNER = 33
RIGHT_EYE_INNER_CORNER = 362
RIGHT_EYE_OUTER_CORNER = 263

logger = logging.getLogger(__name__)

# Model URL for FaceLandmarker (includes iris model)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"


def download_model():
    """Download MediaPipe FaceLandmarker model if not present."""
    if not os.path.exists(MODEL_PATH):
        logger.info(f"Downloading FaceLandmarker model from {MODEL_URL}...")
        print("Downloading additional AI models... please wait...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            logger.info(f"Model downloaded to {MODEL_PATH}")
            print("Download complete!")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise


class EyeTracker:
    """Handles eye tracking using MediaPipe FaceLandmarker."""
    
    def __init__(self):
        """Initialize MediaPipe FaceLandmarker."""
        download_model()
        
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # Calibration model coefficients (set after calibration)
        self.model_x: Optional[np.ndarray] = None
        self.model_y: Optional[np.ndarray] = None
        self.is_calibrated = False
        self.start_time = time.time()
        
        logger.info("EyeTracker initialized with MediaPipe FaceLandmarker")
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, any]:
        """
        Process a single frame and extract eye tracking data.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            Dictionary containing:
                - face_detected: bool
                - left_iris_x_norm: float (0-1) or None
                - left_iris_y_norm: float (0-1) or None
                - right_iris_x_norm: float (0-1) or None
                - right_iris_y_norm: float (0-1) or None
                - features: np.ndarray or None (feature vector for calibration/prediction)
        """
        result = {
            'face_detected': False,
            'left_iris_x_norm': None,
            'left_iris_y_norm': None,
            'right_iris_x_norm': None,
            'right_iris_y_norm': None,
            'features': None
        }
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Calculate timestamp (ms)
        timestamp_ms = int((time.time() - self.start_time) * 1000)
        
        detection_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if not detection_result.face_landmarks:
            return result
        
        # Get first face landmarks
        landmarks = detection_result.face_landmarks[0]
        
        # Extract iris centers
        left_iris_center = self._get_iris_center(landmarks, LEFT_IRIS_INDICES)
        right_iris_center = self._get_iris_center(landmarks, RIGHT_IRIS_INDICES)
        
        if left_iris_center is None or right_iris_center is None:
            return result
        
        # Update result
        result['face_detected'] = True
        result['left_iris_x_norm'] = left_iris_center[0]
        result['left_iris_y_norm'] = left_iris_center[1]
        result['right_iris_x_norm'] = right_iris_center[0]
        result['right_iris_y_norm'] = right_iris_center[1]
        
        # Build feature vector
        result['features'] = self._build_features(landmarks, left_iris_center, right_iris_center)
        
        return result
    
    def _get_iris_center(self, landmarks, iris_indices: list) -> Optional[Tuple[float, float]]:
        """
        Calculate iris center from iris landmarks.
        """
        try:
            # Note: tasks API landmarks have .x, .y, .z attributes just like solutions
            x_coords = [landmarks[i].x for i in iris_indices]
            y_coords = [landmarks[i].y for i in iris_indices]
            
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            
            return (float(center_x), float(center_y))
        except (IndexError, AttributeError) as e:
            logger.warning(f"Failed to extract iris center: {e}")
            return None
    
    def _build_features(self, landmarks, left_iris: Tuple[float, float], 
                       right_iris: Tuple[float, float]) -> np.ndarray:
        """
        Build feature vector for gaze prediction.
        """
        features = [
            left_iris[0],   # left_iris_x_norm
            left_iris[1],   # left_iris_y_norm
            right_iris[0],  # right_iris_x_norm
            right_iris[1],  # right_iris_y_norm
        ]
        
        # Optional: Add relative features for robustness against head movement
        try:
            # Left eye relative position
            left_inner = (landmarks[LEFT_EYE_INNER_CORNER].x, landmarks[LEFT_EYE_INNER_CORNER].y)
            left_outer = (landmarks[LEFT_EYE_OUTER_CORNER].x, landmarks[LEFT_EYE_OUTER_CORNER].y)
            left_rel_x = (left_iris[0] - left_inner[0]) / (left_outer[0] - left_inner[0] + 1e-6)
            left_rel_y = left_iris[1] - (left_inner[1] + left_outer[1]) / 2
            
            # Right eye relative position
            right_inner = (landmarks[RIGHT_EYE_INNER_CORNER].x, landmarks[RIGHT_EYE_INNER_CORNER].y)
            right_outer = (landmarks[RIGHT_EYE_OUTER_CORNER].x, landmarks[RIGHT_EYE_OUTER_CORNER].y)
            right_rel_x = (right_iris[0] - right_inner[0]) / (right_outer[0] - right_inner[0] + 1e-6)
            right_rel_y = right_iris[1] - (right_inner[1] + right_outer[1]) / 2
            
            features.extend([left_rel_x, left_rel_y, right_rel_x, right_rel_y])
        except (IndexError, AttributeError, ZeroDivisionError):
            pass
        
        return np.array(features, dtype=np.float32)
    
    def calibrate(self, calibration_data: list):
        """Calibrate gaze prediction model."""
        if len(calibration_data) < 9:
            raise ValueError(f"Insufficient calibration data: {len(calibration_data)} points (need 9)")
        
        X = []
        y_x = []
        y_y = []
        
        for features, screen_x, screen_y in calibration_data:
            X.append(features)
            y_x.append(screen_x)
            y_y.append(screen_y)
        
        X = np.array(X)
        y_x = np.array(y_x)
        y_y = np.array(y_y)
        
        X_bias = np.column_stack([X, np.ones(len(X))])
        
        self.model_x, _, _, _ = np.linalg.lstsq(X_bias, y_x, rcond=None)
        self.model_y, _, _, _ = np.linalg.lstsq(X_bias, y_y, rcond=None)
        
        self.is_calibrated = True
        logger.info(f"Calibration complete with {len(calibration_data)} points")
    
    def predict_gaze(self, features: np.ndarray, screen_w: int, screen_h: int) -> Tuple[Optional[float], Optional[float], bool]:
        """Predict gaze coordinates."""
        if not self.is_calibrated:
            return (None, None, False)
        
        features_bias = np.append(features, 1.0)
        
        gaze_x = float(np.dot(features_bias, self.model_x))
        gaze_y = float(np.dot(features_bias, self.model_y))
        
        gaze_valid = (0 <= gaze_x < screen_w) and (0 <= gaze_y < screen_h)
        
        return (gaze_x, gaze_y, gaze_valid)
    
    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'face_landmarker') and self.face_landmarker:
            self.face_landmarker.close()
            logger.info("EyeTracker closed")
