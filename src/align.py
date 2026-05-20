"""
Face alignment utilities.

Provides a simple similarity alignment that rotates the face so that the
inter-ocular line is horizontal. This reduces geometric variance and
improves warping stability.

We intentionally keep the canvas size the same as the input image and only
apply a rotation (no scaling) to preserve FOV and avoid cropping.
"""
from __future__ import annotations
import numpy as np
import cv2

# Landmark indices (dlib 68)
LEFT_EYE_OUTER = 36
RIGHT_EYE_OUTER = 45


def _eye_angle(landmarks: np.ndarray) -> float:
    p1 = landmarks[LEFT_EYE_OUTER]
    p2 = landmarks[RIGHT_EYE_OUTER]
    dx, dy = (p2 - p1).astype(float)
    return float(np.arctan2(dy, dx))


def align_face(image: np.ndarray, landmarks: np.ndarray):
    """
    Rotate image so that the line between outer eye corners is horizontal.

    Args:
        image: BGR image (H, W, 3)
        landmarks: (68, 2) array

    Returns:
        aligned_img: rotated image (same H, W)
        aligned_landmarks: rotated landmarks
        M: 2x3 affine transform applied to image (cv2.warpAffine)
        M_inv: inverse 2x3 transform
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    theta = _eye_angle(landmarks)
    # Rotate by -theta to make eye line horizontal
    angle_deg = -theta * 180.0 / np.pi
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    aligned_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)

    # Transform landmarks (augment with ones)
    ones = np.ones((landmarks.shape[0], 1), dtype=np.float32)
    pts = np.hstack([landmarks.astype(np.float32), ones])  # (68, 3)
    aligned_lm = (pts @ M.T).astype(np.float32)

    M_inv = cv2.invertAffineTransform(M)
    return aligned_img, aligned_lm, M, M_inv
