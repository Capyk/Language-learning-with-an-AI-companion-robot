"""
Pydantic models for eye-tracking data.
Matches the exact Excel schema required for export.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class EyeTrackingFrame(BaseModel):
    """Single frame of eye-tracking data matching Excel export schema."""
    timestamp_ms: int = Field(..., description="Milliseconds since tracking start")
    frame_idx: int = Field(..., description="Frame number (0-indexed)")
    phase: str = Field(default="tracking", description="Always 'tracking' for export")
    screen_w_px: int = Field(..., description="Screen width in physical pixels")
    screen_h_px: int = Field(..., description="Screen height in physical pixels")
    face_detected: int = Field(..., description="1 if face detected, 0 otherwise")
    left_iris_x_norm: Optional[float] = Field(None, description="Left iris X (0-1) or NaN")
    left_iris_y_norm: Optional[float] = Field(None, description="Left iris Y (0-1) or NaN")
    right_iris_x_norm: Optional[float] = Field(None, description="Right iris X (0-1) or NaN")
    right_iris_y_norm: Optional[float] = Field(None, description="Right iris Y (0-1) or NaN")
    gaze_x_px: Optional[float] = Field(None, description="Predicted gaze X in pixels or NaN")
    gaze_y_px: Optional[float] = Field(None, description="Predicted gaze Y in pixels or NaN")
    gaze_valid: int = Field(..., description="1 if gaze within screen bounds, 0 otherwise")
    on_screen: int = Field(..., description="1 if user looking at screen (smoothed), 0 otherwise")


class BatchUploadRequest(BaseModel):
    """Request body for batch upload endpoint."""
    frames: List[EyeTrackingFrame] = Field(..., description="List of tracking frames to append")


class FinalizeRequest(BaseModel):
    """Request body for finalize endpoint."""
    # Could add metadata here if needed (e.g., tracking duration, notes)
    pass
