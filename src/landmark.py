"""
Phase 1: Facial Landmark Detection
Owner: Member A

Detects 68 facial landmarks using dlib's shape predictor.
Output shape: (68, 2) float32 array of (x, y) pixel coordinates.
"""

import dlib
import numpy as np
import cv2

# Load models once at module level
_detector = dlib.get_frontal_face_detector()
_predictor = None  # Loaded lazily via _load_predictor()

MODEL_PATH = "shape_predictor_68_face_landmarks.dat"


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
    Detect 68 facial landmarks in an image.

    Args:
        image: BGR image as numpy array (H, W, 3)

    Returns:
        landmarks: float32 array of shape (68, 2) — (x, y) pixel coords
                   Returns None if no face is detected.
    """
    predictor = _load_predictor()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _detector(gray, 1)

    if len(faces) == 0:
        print("[landmark] Warning: no face detected.")
        return None

    # Use the first (largest) detected face
    shape = predictor(gray, faces[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
    return landmarks  # shape: (68, 2)


def visualize_landmarks(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Draw landmarks on a copy of the image for debugging."""
    vis = image.copy()
    for i, (x, y) in enumerate(landmarks.astype(int)):
        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
        cv2.putText(vis, str(i), (x + 2, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 200, 255), 1)
    return vis


if __name__ == "__main__":
    # Quick test
    import sys
    img = cv2.imread(sys.argv[1]) if len(sys.argv) > 1 else None
    if img is None:
        print("Usage: python landmark.py <image_path>")
        sys.exit(1)
    lm = detect_landmarks(img)
    if lm is not None:
        print(f"Detected landmarks: {lm.shape}")
        vis = visualize_landmarks(img, lm)
        cv2.imwrite("output/landmarks_debug.jpg", vis)
        print("Saved: output/landmarks_debug.jpg")
