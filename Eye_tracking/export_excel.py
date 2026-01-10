"""
Excel export module for eye tracking data.
Exports raw frame data to .xlsx format.
"""

import pandas as pd
from datetime import datetime
import logging
from typing import List, Dict
import os

logger = logging.getLogger(__name__)


def export_to_excel(frame_data: List[Dict], output_dir: str = ".") -> str:
    """
    Export frame data to Excel file.
    
    Args:
        frame_data: List of dictionaries containing frame data
        output_dir: Directory to save Excel file
        
    Returns:
        Path to created Excel file
    """
    if not frame_data:
        logger.warning("No frame data to export")
        return None
    
    # Create DataFrame with exact column order
    df = pd.DataFrame(frame_data, columns=[
        'timestamp_ms',
        'frame_idx',
        'phase',
        'screen_w_px',
        'screen_h_px',
        'face_detected',
        'left_iris_x_norm',
        'left_iris_y_norm',
        'right_iris_x_norm',
        'right_iris_y_norm',
        'gaze_x_px',
        'gaze_y_px',
        'gaze_valid'
    ])
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eye_tracking_{timestamp}.xlsx"
    filepath = os.path.join(output_dir, filename)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Export to Excel
    try:
        df.to_excel(filepath, sheet_name='raw', index=False, engine='openpyxl')
        logger.info(f"Exported {len(df)} frames to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to export Excel: {e}")
        raise


def create_frame_record(timestamp_ms: int, frame_idx: int, phase: str,
                       screen_w: int, screen_h: int,
                       tracking_data: Dict,
                       gaze_x: float = None, gaze_y: float = None,
                       gaze_valid: bool = False) -> Dict:
    """
    Create a frame record dictionary with all required fields.
    
    Args:
        timestamp_ms: Timestamp in milliseconds
        frame_idx: Frame index
        phase: "calibration" or "tracking"
        screen_w: Screen width in pixels
        screen_h: Screen height in pixels
        tracking_data: Dictionary from tracker.process_frame()
        gaze_x: Predicted gaze x coordinate (or None)
        gaze_y: Predicted gaze y coordinate (or None)
        gaze_valid: Whether gaze prediction is valid
        
    Returns:
        Dictionary with all frame data fields
    """
    return {
        'timestamp_ms': timestamp_ms,
        'frame_idx': frame_idx,
        'phase': phase,
        'screen_w_px': screen_w,
        'screen_h_px': screen_h,
        'face_detected': 1 if tracking_data['face_detected'] else 0,
        'left_iris_x_norm': tracking_data['left_iris_x_norm'],
        'left_iris_y_norm': tracking_data['left_iris_y_norm'],
        'right_iris_x_norm': tracking_data['right_iris_x_norm'],
        'right_iris_y_norm': tracking_data['right_iris_y_norm'],
        'gaze_x_px': gaze_x,
        'gaze_y_px': gaze_y,
        'gaze_valid': 1 if gaze_valid else 0
    }
