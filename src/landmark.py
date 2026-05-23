"""
Facial landmark detection via MediaPipe Face Mesh.
Returns the raw 478-point array (468 mesh + 10 iris refinement points).
"""
from __future__ import annotations
import numpy as np

from .landmark_mp import detect_landmarks_mp


def detect_landmarks(image) -> np.ndarray | None:
    """Return (478, 2) float32 pixel-coordinate landmark array, or None."""
    return detect_landmarks_mp(image, refine=True)
