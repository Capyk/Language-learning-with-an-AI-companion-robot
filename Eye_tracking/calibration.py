"""
Calibration module for eye tracking.
Displays fullscreen 9-point calibration grid and collects training samples.
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Tuple, Optional
from tracker import EyeTracker

logger = logging.getLogger(__name__)

# Calibration grid: 3x3 points
# Margins from screen edges (as fraction of screen size)
MARGIN_X = 0.1
MARGIN_Y = 0.1

# Number of valid frames to collect per calibration point
FRAMES_PER_POINT = 15

# Timeout for collecting frames (seconds)
COLLECTION_TIMEOUT = 5.0


class CalibrationUI:
    """Handles the 9-point calibration process."""
    
    def __init__(self, screen_w: int, screen_h: int, tracker: EyeTracker, camera):
        """
        Initialize calibration UI.
        
        Args:
            screen_w: Screen width in pixels
            screen_h: Screen height in pixels
            tracker: EyeTracker instance
            camera: OpenCV VideoCapture instance
        """
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.tracker = tracker
        self.camera = camera
        
        # Generate 9 calibration points (3x3 grid)
        self.calibration_points = self._generate_calibration_points()
        self.current_point_idx = 0
        
        # Collected calibration data: (features, screen_x, screen_y)
        self.calibration_data: List[Tuple[np.ndarray, int, int]] = []
        
        logger.info(f"CalibrationUI initialized with {len(self.calibration_points)} points")
    
    def _generate_calibration_points(self) -> List[Tuple[int, int]]:
        """
        Generate 9 calibration points in a 3x3 grid.
        
        Returns:
            List of (x, y) pixel coordinates
        """
        points = []
        
        # Calculate positions
        x_positions = [
            int(self.screen_w * MARGIN_X),
            int(self.screen_w * 0.5),
            int(self.screen_w * (1 - MARGIN_X))
        ]
        y_positions = [
            int(self.screen_h * MARGIN_Y),
            int(self.screen_h * 0.5),
            int(self.screen_h * (1 - MARGIN_Y))
        ]
        
        # Create 3x3 grid (row by row)
        for y in y_positions:
            for x in x_positions:
                points.append((x, y))
        
        logger.info(f"Generated calibration points: {points}")
        return points
    
    def run_calibration(self) -> bool:
        """
        Run the full calibration process.
        
        Returns:
            True if calibration successful, False if aborted
        """
        # Create fullscreen window
        window_name = "Calibration"
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        logger.info("Starting calibration process")
        
        try:
            for idx, (point_x, point_y) in enumerate(self.calibration_points):
                self.current_point_idx = idx
                logger.info(f"Calibration point {idx + 1}/9: ({point_x}, {point_y})")
                
                # Collect samples for this point
                success = self._collect_point_samples(window_name, point_x, point_y)
                
                if not success:
                    logger.warning("Calibration aborted by user")
                    cv2.destroyWindow(window_name)
                    return False
            
            # Calibration complete, fit model
            logger.info(f"Calibration data collected: {len(self.calibration_data)} samples")
            self.tracker.calibrate(self.calibration_data)
            
            # Keep window open for next step
            # cv2.destroyWindow(window_name) 
            return True
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            cv2.destroyWindow(window_name)
            return False
    
    def _collect_point_samples(self, window_name: str, point_x: int, point_y: int) -> bool:
        """
        Collect samples for a single calibration point.
        
        Args:
            window_name: OpenCV window name
            point_x: Point x coordinate
            point_y: Point y coordinate
            
        Returns:
            True if samples collected, False if aborted
        """
        # Wait for user to press SPACE
        waiting_for_space = True
        
        while waiting_for_space:
            # Read camera frame
            ret, frame = self.camera.read()
            if not ret:
                logger.error("Failed to read camera frame")
                return False
            
            # Process frame to check face detection
            tracking_data = self.tracker.process_frame(frame)
            face_detected = tracking_data['face_detected']
            
            # Draw calibration screen
            calib_screen = self._draw_calibration_screen(point_x, point_y, face_detected, 
                                                         waiting=True)
            cv2.imshow(window_name, calib_screen)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                return False
            elif key == 32:  # SPACE
                waiting_for_space = False
        
        # Collect FRAMES_PER_POINT valid frames
        collected_frames = []
        start_time = time.time()
        
        while len(collected_frames) < FRAMES_PER_POINT:
            # Check timeout
            if time.time() - start_time > COLLECTION_TIMEOUT:
                logger.warning(f"Timeout collecting samples for point ({point_x}, {point_y})")
                # Show error message and retry
                self._show_retry_message(window_name, point_x, point_y)
                return self._collect_point_samples(window_name, point_x, point_y)
            
            # Read camera frame
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            # Process frame
            tracking_data = self.tracker.process_frame(frame)
            
            if tracking_data['face_detected'] and tracking_data['features'] is not None:
                collected_frames.append(tracking_data['features'])
            
            # Draw calibration screen with progress
            progress = len(collected_frames)
            calib_screen = self._draw_calibration_screen(point_x, point_y, 
                                                         tracking_data['face_detected'],
                                                         waiting=False, progress=progress)
            cv2.imshow(window_name, calib_screen)
            
            # Check for ESC
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                return False
        
        # Average collected features
        avg_features = np.mean(collected_frames, axis=0)
        self.calibration_data.append((avg_features, point_x, point_y))
        
        logger.info(f"Collected {len(collected_frames)} frames for point ({point_x}, {point_y})")
        
        # Brief pause before next point
        time.sleep(0.3)
        
        return True
    
    def _draw_calibration_screen(self, point_x: int, point_y: int, face_detected: bool,
                                 waiting: bool = True, progress: int = 0) -> np.ndarray:
        """
        Draw calibration screen with point and instructions.
        
        Args:
            point_x: Calibration point x coordinate
            point_y: Calibration point y coordinate
            face_detected: Whether face is currently detected
            waiting: Whether waiting for SPACE press
            progress: Number of frames collected (if not waiting)
            
        Returns:
            BGR image
        """
        # Black background
        screen = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        
        # Draw calibration point (white circle)
        cv2.circle(screen, (point_x, point_y), 20, (255, 255, 255), -1)
        cv2.circle(screen, (point_x, point_y), 22, (200, 200, 200), 2)
        
        # Draw instructions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (255, 255, 255)
        
        # Point number
        point_text = f"Punkt {self.current_point_idx + 1}/9"
        cv2.putText(screen, point_text, (50, 50), font, font_scale, color, thickness)
        
        # Face detection status
        face_status = "Face detected: YES" if face_detected else "Face detected: NO"
        face_color = (0, 255, 0) if face_detected else (0, 0, 255)
        cv2.putText(screen, face_status, (50, 100), font, font_scale, face_color, thickness)
        
        # Instructions
        if waiting:
            instruction = "Schau auf den Punkt und druecke SPACE"
            cv2.putText(screen, instruction, (50, 150), font, font_scale, color, thickness)
        else:
            instruction = f"Sammle Daten... {progress}/{FRAMES_PER_POINT}"
            cv2.putText(screen, instruction, (50, 150), font, font_scale, (0, 255, 255), thickness)
        
        # ESC to abort
        cv2.putText(screen, "ESC = Abbrechen", (50, self.screen_h - 50), 
                   font, 0.6, (150, 150, 150), 1)
        
        return screen
    
    def _show_retry_message(self, window_name: str, point_x: int, point_y: int):
        """Show retry message when face detection fails."""
        screen = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Error message
        cv2.putText(screen, "Face not detected!", 
                   (self.screen_w // 2 - 200, self.screen_h // 2 - 50),
                   font, 1.2, (0, 0, 255), 2)
        cv2.putText(screen, "Bitte positioniere dein Gesicht vor der Kamera", 
                   (self.screen_w // 2 - 400, self.screen_h // 2 + 20),
                   font, 0.8, (255, 255, 255), 2)
        cv2.putText(screen, "Versuche es erneut...", 
                   (self.screen_w // 2 - 200, self.screen_h // 2 + 80),
                   font, 0.8, (255, 255, 0), 2)
        
        cv2.imshow(window_name, screen)
        cv2.waitKey(2000)  # Show for 2 seconds
    def wait_for_start_trigger(self, window_name: str = "Calibration") -> bool:
        """
        Show 'Start Tracking' button and wait for user input.
        
        Args:
            window_name: OpenCV window name
        
        Returns:
            True if start requested, False if aborted (ESC)
        """
        # Ensure window exists and is fullscreen
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Button dimensions
        btn_w, btn_h = 400, 100
        btn_x = (self.screen_w - btn_w) // 2
        btn_y = (self.screen_h - btn_h) // 2
        
        # Mouse callback state
        self.mouse_clicked = False
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Check if click is inside button
                if btn_x <= x <= btn_x + btn_w and btn_y <= y <= btn_y + btn_h:
                    self.mouse_clicked = True
        
        cv2.setMouseCallback(window_name, mouse_callback)
        
        while True:
            screen = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
            
            # Draw instructions
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Success symbol
            cv2.circle(screen, (self.screen_w // 2, self.screen_h // 2 - 200), 50, (0, 255, 0), 3)
            cv2.polylines(screen, [np.array([
                [self.screen_w // 2 - 20, self.screen_h // 2 - 200], 
                [self.screen_w // 2 - 5, self.screen_h // 2 - 185], 
                [self.screen_w // 2 + 25, self.screen_h // 2 - 225]], dtype=np.int32)], 
                False, (0, 255, 0), 4)

            cv2.putText(screen, "Kalibrierung erfolgreich!", 
                       (self.screen_w // 2 - 250, self.screen_h // 2 - 100),
                       font, 1.2, (255, 255, 255), 2)
            
            # Draw Button
            # Hover effect
            # Note: We can't easily get mouse pos without callback movement events
            # Just draw static button
            cv2.rectangle(screen, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), (0, 180, 0), -1) # Green fill
            cv2.rectangle(screen, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), (255, 255, 255), 3) # White border
            
            text = "START TRACKING"
            text_size = cv2.getTextSize(text, font, 1.0, 2)[0]
            text_x = btn_x + (btn_w - text_size[0]) // 2
            text_y = btn_y + (btn_h + text_size[1]) // 2
            cv2.putText(screen, text, (text_x, text_y), font, 1.0, (255, 255, 255), 2)
            
            cv2.putText(screen, "Oder druecke SPACE", (self.screen_w // 2 - 120, btn_y + btn_h + 40), 
                       font, 0.7, (200, 200, 200), 1)
            
            cv2.imshow(window_name, screen)
            
            key = cv2.waitKey(10) & 0xFF
            if key == 27:  # ESC
                return False
            elif key == 32:  # SPACE
                return True
            
            if self.mouse_clicked:
                return True
                
    def show_completion_screen(self, output_file: str):
        """Show completion message."""
        window_name = "Completion"
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        start_time = time.time()
        
        while True:
            screen = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Success Icon
            cv2.circle(screen, (self.screen_w // 2, self.screen_h // 2 - 100), 60, (0, 255, 255), 4) # Yellow circle
            
            cv2.putText(screen, "Tracking beendet!", 
                       (self.screen_w // 2 - 200, self.screen_h // 2),
                       font, 1.5, (0, 255, 255), 2)
            
            cv2.putText(screen, "Daten wurden gespeichert.", 
                       (self.screen_w // 2 - 180, self.screen_h // 2 + 60),
                       font, 0.8, (255, 255, 255), 1)
                       
            # Timer to auto-close
            elapsed = time.time() - start_time
            remaining = int(5 - elapsed)
            if remaining <= 0:
                break
                
            cv2.putText(screen, f"Schliesse in {remaining}s...", 
                       (self.screen_w // 2 - 100, self.screen_h - 50),
                       font, 0.6, (150, 150, 150), 1)
            
            cv2.imshow(window_name, screen)
            
            key = cv2.waitKey(10) & 0xFF
            if key == 27 or key == 32: # ESC or SPACE to close early
                break
        
        cv2.destroyWindow(window_name)
