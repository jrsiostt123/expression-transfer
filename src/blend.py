"""
Phase 4: Seamless Poisson Blending
Owner: Member C

Composites the warped face region onto the source image
using OpenCV's seamlessClone (Poisson blending).
"""

import numpy as np
import cv2


def blend(
    source_img: np.ndarray,
    warped_img: np.ndarray,
    face_mask: np.ndarray
) -> np.ndarray:
    """
    Seamlessly blend warped face region into source image.

    Args:
        source_img: BGR image (H, W, 3) — the original source face
        warped_img: BGR image (H, W, 3) — warped face from Phase 3
        face_mask:  (H, W) uint8 mask — white = face region to blend

    Returns:
        result: (H, W, 3) final composited image
    """
    # Find the center of the face mask for seamlessClone
    M = cv2.moments(face_mask)
    if M["m00"] == 0:
        print("[blend] Warning: empty mask — returning source image unchanged.")
        return source_img.copy()

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    center = (cx, cy)

    # Expand mask to 3-channel for seamlessClone
    mask_3ch = cv2.merge([face_mask, face_mask, face_mask])

    try:
        result = cv2.seamlessClone(warped_img, source_img, face_mask, center, cv2.NORMAL_CLONE)
    except cv2.error as e:
        print(f"[blend] seamlessClone failed: {e}")
        print("[blend] Falling back to direct alpha composite.")
        alpha = face_mask.astype(float) / 255.0
        alpha_3ch = np.stack([alpha, alpha, alpha], axis=-1)
        result = (warped_img * alpha_3ch + source_img * (1 - alpha_3ch)).astype(np.uint8)

    return result


def save_comparison(source_img, driver_img, result_img, path="output/comparison.jpg"):
    """Save a side-by-side comparison of source | driver | result."""
    h = max(source_img.shape[0], driver_img.shape[0], result_img.shape[0])

    def pad(img):
        top = (h - img.shape[0]) // 2
        return cv2.copyMakeBorder(img, top, h - img.shape[0] - top, 0, 0,
                                   cv2.BORDER_CONSTANT, value=(240, 240, 240))

    divider = np.ones((h, 4, 3), dtype=np.uint8) * 180
    comparison = np.hstack([pad(source_img), divider, pad(driver_img), divider, pad(result_img)])

    # Labels
    for i, label in enumerate(["Source", "Driver", "Result"]):
        x = i * (source_img.shape[1] + 4) + 10
        cv2.putText(comparison, label, (x, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 200), 2)

    cv2.imwrite(path, comparison)
    print(f"Saved comparison: {path}")


if __name__ == "__main__":
    # Quick sanity check
    src = np.ones((480, 640, 3), dtype=np.uint8) * 100
    wrp = np.ones((480, 640, 3), dtype=np.uint8) * 150
    msk = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(msk, (320, 240), 150, 255, -1)
    result = blend(src, wrp, msk)
    print(f"Result shape: {result.shape}")
    print("blend.py OK")
