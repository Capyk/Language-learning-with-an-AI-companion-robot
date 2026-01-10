"""
Eye Tracking Application - Main Entry Point
Webcam-based eye tracking with 9-point calibration and Excel export.
"""

import cv2
import argparse
import logging
import sys
import time
import ctypes
from typing import List, Dict

from tracker import EyeTracker
from calibration import CalibrationUI
from export_excel import export_to_excel, create_frame_record

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_dpi_awareness():
    """
    Enable Windows DPI awareness for accurate screen coordinates.
    """
    try:
        # Try per-monitor DPI awareness (Windows 8.1+)
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        logger.info("DPI awareness set: Per-Monitor DPI Aware")
    except Exception:
        try:
            # Fallback to system DPI awareness (Windows Vista+)
            ctypes.windll.user32.SetProcessDPIAware()
            logger.info("DPI awareness set: System DPI Aware")
        except Exception as e:
            logger.warning(f"Failed to set DPI awareness: {e}")


def get_screen_resolution() -> tuple:
    """
    Get actual screen resolution using Windows API.
    
    Returns:
        (width, height) in pixels
    """
    try:
        user32 = ctypes.windll.user32
        screen_w = user32.GetSystemMetrics(0)  # SM_CXSCREEN
        screen_h = user32.GetSystemMetrics(1)  # SM_CYSCREEN
        logger.info(f"Screen resolution: {screen_w}x{screen_h}")
        return screen_w, screen_h
    except Exception as e:
        logger.error(f"Failed to get screen resolution: {e}")
        # Fallback to common resolution
        return 1920, 1080


def initialize_camera(camera_index: int = 0) -> cv2.VideoCapture:
    """
    Initialize camera with error handling.
    
    Args:
        camera_index: Camera device index
        
    Returns:
        VideoCapture object or None if failed
    """
    logger.info(f"Initializing camera {camera_index}...")
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # DirectShow for Windows
    
    if not camera.isOpened():
        logger.error(f"Failed to open camera {camera_index}")
        return None
    
    # Set camera properties for better performance
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    # Read test frame
    ret, frame = camera.read()
    if not ret:
        logger.error("Failed to read from camera")
        camera.release()
        return None
    
    logger.info(f"Camera initialized: {frame.shape[1]}x{frame.shape[0]}")
    return camera


def run_tracking_phase(tracker: EyeTracker, camera: cv2.VideoCapture,
                      screen_w: int, screen_h: int,
                      duration_sec: int = 30) -> List[Dict]:
    """
    Run the tracking phase and collect frame data.
    
    Args:
        tracker: Calibrated EyeTracker instance
        camera: VideoCapture object
        screen_w: Screen width in pixels
        screen_h: Screen height in pixels
        duration_sec: Tracking duration in seconds
        
    Returns:
        List of frame records
    """
    logger.info(f"Starting tracking phase ({duration_sec} seconds)...")
    
    frame_data = []
    frame_idx = 0
    start_time = time.time()
    
    # Create preview window (Fullscreen for App-like feel)
    window_name = "Eye Tracking - Preview"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            remaining = max(0, duration_sec - elapsed)
            
            # Check if time is up
            if elapsed >= duration_sec:
                logger.info("Tracking phase complete")
                break
            
            # Read frame
            ret, frame = camera.read()
            if not ret:
                logger.warning("Failed to read camera frame")
                continue
            
            # Process frame
            timestamp_ms = int(elapsed * 1000)
            tracking_data = tracker.process_frame(frame)
            
            # Predict gaze if face detected
            gaze_x, gaze_y, gaze_valid = None, None, False
            if tracking_data['face_detected'] and tracking_data['features'] is not None:
                gaze_x, gaze_y, gaze_valid = tracker.predict_gaze(
                    tracking_data['features'], screen_w, screen_h
                )
            
            # Create frame record
            record = create_frame_record(
                timestamp_ms=timestamp_ms,
                frame_idx=frame_idx,
                phase='tracking',
                screen_w=screen_w,
                screen_h=screen_h,
                tracking_data=tracking_data,
                gaze_x=gaze_x,
                gaze_y=gaze_y,
                gaze_valid=gaze_valid
            )
            frame_data.append(record)
            frame_idx += 1
            
            # Draw preview
            # Resize frame to screen size for better visibility if needed, 
            # but standard imshow scaling works well in fullscreen.
            preview = frame.copy()
            
            # Draw info overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Elapsed Time (User requested "how long it has been tracking")
            time_text = f"Zeit: {int(elapsed)}s / {duration_sec}s"
            cv2.putText(preview, time_text, (20, 50), font, 1.0, (0, 255, 0), 2)
            
            face_status = "Face: YES" if tracking_data['face_detected'] else "Face: NO"
            face_color = (0, 255, 0) if tracking_data['face_detected'] else (0, 0, 255)
            cv2.putText(preview, face_status, (20, 90), font, 0.8, face_color, 2)
            
            if gaze_valid:
                gaze_text = f"Gaze: ({int(gaze_x)}, {int(gaze_y)})"
                cv2.putText(preview, gaze_text, (20, 130), font, 0.8, (255, 255, 0), 2)
            
            cv2.putText(preview, "ESC = Abbrechen", (20, preview.shape[0] - 20),
                       font, 0.6, (150, 150, 150), 1)
            
            cv2.imshow(window_name, preview)
            
            # Check for ESC key
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                logger.info("Tracking aborted by user")
                break
    
    finally:
        cv2.destroyWindow(window_name)
    
    logger.info(f"Collected {len(frame_data)} frames")
    return frame_data


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description='Eye Tracking Application')
    parser.add_argument('--duration', type=int, default=30,
                       help='Tracking duration in seconds (default: 30)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index (default: 0)')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Output directory for Excel files (default: ./output)')
    args = parser.parse_args()
    
    logger.info("=== Eye Tracking Application ===")
    logger.info(f"Tracking duration: {args.duration} seconds")
    logger.info(f"Camera index: {args.camera}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Setup Windows DPI awareness
    setup_dpi_awareness()
    screen_w, screen_h = get_screen_resolution()
    
    # Initialize camera
    camera = initialize_camera(args.camera)
    if camera is None:
        logger.error("Camera initialization failed. Exiting.")
        sys.exit(1)
    
    # Initialize tracker
    try:
        tracker = EyeTracker()
    except Exception as e:
        logger.error(f"Failed to initialize tracker: {e}")
        camera.release()
        sys.exit(1)
    
    try:
        # Run calibration
        logger.info("Starting calibration phase...")
        calibration_ui = CalibrationUI(screen_w, screen_h, tracker, camera)
        calibration_success = calibration_ui.run_calibration()
        
        if not calibration_success:
            logger.warning("Calibration aborted or failed")
            return
        
        logger.info("Calibration successful!")
        
        # Wait for user to start tracking
        logger.info("Waiting for user start trigger...")
        start_tracking = calibration_ui.wait_for_start_trigger()
        cv2.destroyWindow("Calibration")
        
        if not start_tracking:
            logger.info("Tracking aborted before start")
            return
        
        # Brief pause before tracking
        time.sleep(0.5)
        
        # Run tracking phase
        frame_data = run_tracking_phase(tracker, camera, screen_w, screen_h, args.duration)
        
        # Export to Excel
        if frame_data:
            logger.info("Exporting data to Excel...")
            excel_path = export_to_excel(frame_data, args.output_dir)
            if excel_path:
                logger.info(f"Data exported successfully: {excel_path}")
                print(f"\n[OK] Eye tracking complete!")
                print(f"[OK] Data saved to: {excel_path}")
                print(f"[OK] Total frames: {len(frame_data)}")
                
                # Show completion screen
                calibration_ui.show_completion_screen(excel_path)
            else:
                logger.error("Failed to export Excel file")
        else:
            logger.warning("No data collected")
    
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    
    finally:
        # Cleanup
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()
        logger.info("Application closed")


if __name__ == '__main__':
    main()
