import sys
import os
import time

print("=== System Check ===")
print(f"Python: {sys.version.split()[0]}")

try:
    import cv2
    print(f"OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"[FAIL] OpenCV Error: {e}")
    sys.exit(1)

try:
    import mediapipe as mp
    print(f"MediaPipe: {mp.__version__}")
except ImportError as e:
    print(f"[FAIL] MediaPipe Error: {e}")
    sys.exit(1)

try:
    import numpy
    print(f"NumPy: {numpy.__version__}")
except ImportError as e:
    print(f"[FAIL] NumPy Error: {e}")
    sys.exit(1)

print("\nChecking Camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[FAIL] Camera NOT found! (Index 0)")
    # Try index 1
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[FAIL] Camera NOT found! (Index 1)")
        print("   -> Please check if camera is connected or used by another app.")
    else:
        print("[OK] Camera found (Index 1)")
        cap.release()
else:
    print("[OK] Camera found (Index 0)")
    cap.release()

print("\nChecking MediaPipe Model...")
try:
    if os.path.exists("face_landmarker.task"):
        file_size = os.path.getsize("face_landmarker.task")
        print(f"[OK] Model file exists ({file_size / 1024 / 1024:.2f} MB)")
    else:
        print("[WARN] Model file missing (will be downloaded automatically)")
except Exception as e:
    print(f"[WARN] Warning checking model: {e}")

print("\n=== Check Complete ===\n")
