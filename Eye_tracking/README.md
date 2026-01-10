# Eye Tracking Application

Webcam-based eye tracking application for Windows with 9-point calibration and Excel export.

## Features

- **9-Point Calibration**: Fullscreen calibration UI with 3×3 grid
- **MediaPipe Iris Tracking**: Robust iris detection using Google MediaPipe
- **Real-time Tracking**: 30-second tracking phase with live preview
- **Excel Export**: Raw frame data exported to `.xlsx` format
- **Windows DPI Awareness**: Accurate screen coordinates on high-DPI displays
- **Face Detection Feedback**: Visual feedback during calibration and tracking

## Requirements

- **OS**: Windows 10/11
- **Python**: 3.10 or higher
- **Hardware**: Laptop webcam (minimum 30 FPS recommended)

## Installation

### 1. Clone or Download Project

```bash
cd "c:\Users\abreh\Documents\Studium\7. Semester\Human AI Interaction\Eye_tracking"
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `opencv-python` - Camera capture and UI
- `mediapipe` - Face mesh and iris landmarks
- `pandas` - Data manipulation
- `openpyxl` - Excel export
- `numpy` - Linear algebra for calibration

## Usage

### Basic Usage

```bash
python main.py
```

This will:
1. Start 9-point calibration (fullscreen)
2. Run 30-second tracking phase
3. Export data to `./output/eye_tracking_YYYYMMDD_HHMMSS.xlsx`

### Custom Duration

```bash
python main.py --duration 60
```

Track for 60 seconds instead of default 30.

### Custom Camera

```bash
python main.py --camera 1
```

Use camera index 1 (if you have multiple cameras).

### Custom Output Directory

```bash
python main.py --output-dir ./my_data
```

Save Excel files to `./my_data` instead of `./output`.

### All Options

```bash
python main.py --duration 45 --camera 0 --output-dir ./results
```

## Calibration Process

1. **Fullscreen Window Opens**: Black background with white calibration point
2. **For Each of 9 Points**:
   - Look at the white point
   - Wait for "Face detected: YES" status
   - Press **SPACE** to start collecting samples
   - Keep looking at the point while data is collected (15 frames)
3. **After All Points**: Calibration model is fitted automatically
4. **Tracking Starts**: 30-second tracking phase begins immediately

### Calibration Tips

- Sit comfortably at a consistent distance from the screen
- Ensure good lighting (avoid backlighting)
- Keep your head relatively still during calibration
- If face detection fails, adjust your position and try again
- Press **ESC** to abort at any time

## Tracking Phase

During tracking:
- Small preview window shows webcam feed
- Timer displays remaining seconds
- Face detection status shown in real-time
- Current gaze coordinates displayed (if valid)
- Press **ESC** to abort early (data still exported)

## Output Format

Excel file with one sheet (`raw`) containing these columns:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp_ms` | int | Milliseconds since tracking start |
| `frame_idx` | int | Frame number (0-indexed) |
| `phase` | string | "tracking" (or "calibration" if logged) |
| `screen_w_px` | int | Screen width in pixels |
| `screen_h_px` | int | Screen height in pixels |
| `face_detected` | int | 1 if face detected, 0 otherwise |
| `left_iris_x_norm` | float | Left iris X (0-1), NaN if invalid |
| `left_iris_y_norm` | float | Left iris Y (0-1), NaN if invalid |
| `right_iris_x_norm` | float | Right iris X (0-1), NaN if invalid |
| `right_iris_y_norm` | float | Right iris Y (0-1), NaN if invalid |
| `gaze_x_px` | float | Predicted gaze X in pixels, NaN if invalid |
| `gaze_y_px` | float | Predicted gaze Y in pixels, NaN if invalid |
| `gaze_valid` | int | 1 if gaze prediction valid, 0 otherwise |

**Note**: Normalized iris coordinates (0-1) are relative to camera frame. Gaze coordinates are in screen pixels (origin: top-left).

## Troubleshooting

### Camera Not Found

**Error**: `Failed to open camera 0`

**Solutions**:
- Check if camera is being used by another application (Zoom, Teams, etc.)
- Try different camera index: `python main.py --camera 1`
- Restart your computer
- Check Windows Privacy Settings → Camera → Allow apps to access camera

### MediaPipe Initialization Failed

**Error**: `Failed to initialize tracker`

**Solutions**:
- Reinstall MediaPipe: `pip uninstall mediapipe && pip install mediapipe`
- Check Python version: `python --version` (must be 3.10+)
- Try installing specific version: `pip install mediapipe==0.10.9`

### Face Not Detected During Calibration

**Symptoms**: "Face detected: NO" stays red

**Solutions**:
- Improve lighting (face should be well-lit, avoid backlighting)
- Move closer to camera (but not too close)
- Remove glasses if they cause reflections
- Ensure camera lens is clean
- Look directly at camera briefly to help detection

### DPI Scaling Issues

**Symptoms**: Gaze coordinates seem offset or incorrect

**Solutions**:
- Application automatically handles DPI scaling
- If issues persist, check Windows Display Settings → Scale (should be detected correctly)
- Try running as administrator: `Right-click → Run as administrator`

### Excel Export Failed

**Error**: `Failed to export Excel`

**Solutions**:
- Ensure output directory is writable
- Close any open Excel files with similar names
- Check disk space
- Reinstall openpyxl: `pip install --upgrade openpyxl`

### Low Frame Rate / Laggy

**Solutions**:
- Close other applications using camera
- Reduce camera resolution (edit `main.py` line 67-68)
- Ensure good lighting (helps MediaPipe performance)
- Check CPU usage (MediaPipe is CPU-intensive)

### Gaze Prediction Inaccurate

**Expected Behavior**: This is a simple linear model, not scientific-grade eye tracking.

**To Improve Accuracy**:
- Perform calibration carefully (look exactly at each point)
- Keep head position consistent between calibration and tracking
- Ensure good lighting throughout
- Avoid large head movements during tracking
- Re-calibrate if you change position

## Project Structure

```
Eye_tracking/
├── main.py              # Entry point, orchestrates flow
├── tracker.py           # MediaPipe integration, feature extraction
├── calibration.py       # 9-point calibration UI
├── export_excel.py      # Excel export functionality
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── output/             # Excel files saved here (created automatically)
```

## Technical Details

### Iris Landmarks

- **Left Iris**: MediaPipe landmarks 468-472
- **Right Iris**: MediaPipe landmarks 473-477
- **Eye Corners**: Used for relative positioning (optional robustness)

### Feature Vector

Minimal features:
- Left iris X, Y (normalized)
- Right iris X, Y (normalized)

Optional robustness features:
- Iris position relative to eye corners (reduces head movement impact)

### Calibration Model

- **Algorithm**: Linear regression (least squares)
- **Implementation**: `numpy.linalg.lstsq`
- **Separate models** for X and Y coordinates
- **Input**: Feature vector + bias term
- **Output**: Screen pixel coordinates

### Validation

- `gaze_valid=0` if prediction outside screen bounds
- `face_detected=0` if MediaPipe fails to detect face
- NaN values for invalid measurements

## Limitations

⚠️ **This is NOT scientific-grade eye tracking**

- Accuracy limited by webcam quality (~30 FPS, low resolution)
- Linear model cannot account for complex eye movements
- Head movement significantly affects accuracy
- Lighting conditions impact performance
- Individual eye characteristics vary

**Use Case**: Rough gaze patterns, UI testing, research prototypes

**Not Suitable For**: Medical applications, precise gaze analysis, accessibility tools requiring high accuracy

## License

This project is provided as-is for educational and research purposes.

## Support

For issues or questions:
1. Check Troubleshooting section above
2. Verify all dependencies installed correctly
3. Test with default parameters first
4. Check console logs for detailed error messages

---

**Version**: 1.0  
**Last Updated**: 2025-12-26
