"""
landmark_dlib.py — 68-point facial landmark detector (legacy fallback)
"""
from __future__ import annotations
import dlib
import numpy as np
import cv2
import os

# Load models once at module level
_detector = dlib.get_frontal_face_detector()
_predictor = None  # Loaded lazily via _load_predictor()

# Resolve model path relative to this file's directory so it works regardless
# of the working directory the caller uses.  Falls back to bare filename so
# any existing scripts that place the .dat next to demo.py still work.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_PATHS = [
    os.path.join(_HERE, "..", "models", "shape_predictor_68_face_landmarks.dat"),
    os.path.join(_HERE, "shape_predictor_68_face_landmarks.dat"),
    "shape_predictor_68_face_landmarks.dat",
]
MODEL_PATH = next((p for p in _CANDIDATE_PATHS if os.path.exists(p)), _CANDIDATE_PATHS[0])


def _load_predictor():
    global _predictor
    if _predictor is None:
        try:
            _predictor = dlib.shape_predictor(MODEL_PATH)
        except RuntimeError:
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'. "
                "Run: bash scripts/download_model.sh"
            )
    return _predictor


def detect_landmarks(image: np.ndarray) -> np.ndarray:
    """
    Detect 68 facial landmarks in an image using dlib.

    Returns (68, 2) float32, or None if not found.
    """
    predictor = _load_predictor()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _detector(gray, 1)

    if len(faces) == 0:
        print("[landmark_dlib] Warning: no face detected.")
        return None

    # Use the first (largest) detected face
    shape = predictor(gray, faces[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
    return landmarks


def get_region_indices(n_points: int | None = 68) -> dict:
    """
    Return region index sets for dlib 68-point layout.
    Keys: left_eye, right_eye, lips, brows_left, brows_right, face_oval
    """
    if n_points != 68:
        raise ValueError("dlib fallback expects 68 landmarks")
    return {
        "left_eye": list(range(36, 42)),
        "right_eye": list(range(42, 48)),
        "lips": list(range(48, 68)),
        "brows_left": list(range(17, 22)),
        "brows_right": list(range(22, 27)),
        # Rough oval: jawline 0..16
        "face_oval": list(range(0, 17)),
    }