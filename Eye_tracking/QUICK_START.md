# Quick Start Guide

## ⚠️ Important: Python Version Issue

You're using **Python 3.14**, which is very new. NumPy doesn't have pre-built packages for it yet, which causes installation failures.

## Solution: Use Python 3.11 or 3.12

### Option 1: Install Python 3.12 (Recommended)

1. Download Python 3.12 from: https://www.python.org/downloads/
2. During installation, check "Add Python to PATH"
3. After installation, double-click: **`install_dependencies.bat`**
4. Then double-click: **`run_eye_tracking.bat`**

### Option 2: Manual Installation (If you want to use Python 3.14)

You'll need to install Visual Studio Build Tools to compile numpy from source:

1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install "Desktop development with C++"
3. Run: `install_dependencies.bat`

---

## Files Created for You

### 🚀 `run_eye_tracking.bat`
**Double-click this to start the app!**
- Shows instructions
- Runs the calibration and tracking
- Handles errors gracefully

### 📦 `install_dependencies.bat`
**Run this FIRST to install required packages**
- Installs opencv, mediapipe, pandas, etc.
- Shows helpful error messages if it fails

---

## Manual Installation (Alternative)

If the batch files don't work, open PowerShell or Command Prompt and run:

```bash
# Navigate to project folder
cd "c:\Users\abreh\Documents\Studium\7. Semester\Human AI Interaction\Eye_tracking"

# Install dependencies
python -m pip install opencv-python mediapipe pandas openpyxl numpy

# Run the app
python main.py
```

---

## What Happens When You Run

1. **Fullscreen calibration** appears (black background, white dots)
2. Look at each dot and press **SPACE** (9 times total)
3. **Tracking starts** automatically for 30 seconds
4. **Excel file** is saved to `./output/` folder
5. Done!

---

## Troubleshooting

### "ModuleNotFoundError"
→ Run `install_dependencies.bat` first

### "Camera not found"
→ Close Zoom, Teams, or other apps using your webcam

### "Face not detected"
→ Improve lighting, move closer to camera

---

**Need help?** Check the full [README.md](README.md) for detailed troubleshooting.
