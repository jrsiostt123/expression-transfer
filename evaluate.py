"""
evaluate.py — Quantitative evaluation metrics
Usage: python evaluate.py --result <path> --reference <path> --result-lm-img <path>
"""

import argparse
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from skimage.metrics import structural_similarity as ssim
from landmark import detect_landmarks


def compute_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Compute SSIM between two BGR images (resized to same shape if needed)."""
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray_a, gray_b, full=True)
    return score


def compute_landmark_deviation(result_img: np.ndarray, target_lm: np.ndarray) -> float:
    """
    Detect landmarks on result image and compute mean deviation
    from the expected target landmark positions.
    """
    detected_lm = detect_landmarks(result_img)
    if detected_lm is None:
        print("[eval] Could not detect landmarks on result image.")
        return float("nan")
    deviation = np.linalg.norm(detected_lm - target_lm, axis=1).mean()
    return float(deviation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate expression transfer quality")
    parser.add_argument("--result", required=True, help="Result image path")
    parser.add_argument("--reference", required=True, help="Reference/ground truth image path")
    args = parser.parse_args()

    result = cv2.imread(args.result)
    reference = cv2.imread(args.reference)

    if result is None or reference is None:
        print("Error: could not load images.")
        sys.exit(1)

    score = compute_ssim(result, reference)
    print(f"SSIM: {score:.4f}  (1.0 = identical, higher is better)")
