# ⚠️ CRITICAL: Python 3.14 Compatibility Issue

## Problem

**MediaPipe does NOT work with Python 3.14** due to low-level C library incompatibilities.

Error: `function 'free' not found`

This is a known issue with MediaPipe on Python 3.14 (which is still in development/alpha).

---

## ✅ SOLUTION: Install Python 3.12

### Step 1: Download Python 3.12

Go to: **https://www.python.org/downloads/**

Download: **Python 3.12.8** (latest stable 3.12 version)

### Step 2: Install Python 3.12

- ✅ Check "Add Python 3.12 to PATH"
- ✅ Choose "Install Now"

### Step 3: Verify Installation

Open a NEW terminal and run:
```bash
python --version
```

Should show: `Python 3.12.x`

### Step 4: Install Dependencies

Double-click: **`install_dependencies.bat`**

Or manually run:
```bash
cd "c:\Users\abreh\Documents\Studium\7. Semester\Human AI Interaction\Eye_tracking"
python -m pip install -r requirements.txt
```

### Step 5: Run the App

Double-click: **`run_eye_tracking.bat`**

---

## Why Python 3.14 Doesn't Work

- Python 3.14 is still in **alpha/development** (not stable yet)
- MediaPipe's C extensions are not compiled for Python 3.14
- Missing binary dependencies cause runtime errors
- **Recommended versions**: Python 3.11 or 3.12 (stable)

---

## Alternative: Keep Python 3.14 and Install Python 3.12 Side-by-Side

You can have both versions installed:

1. Install Python 3.12 (see above)
2. Keep Python 3.14 for other projects
3. Use `py -3.12` to run with Python 3.12
4. Use `py -3.14` to run with Python 3.14

Update `run_eye_tracking.bat` to use:
```batch
py -3.12 main.py
```

---

## Quick Test After Installing Python 3.12

```bash
python -c "import mediapipe; print('MediaPipe works!')"
```

If this prints "MediaPipe works!" without errors, you're good to go!

---

**Bottom line**: Python 3.14 is too new. Use Python 3.12 for this project.
