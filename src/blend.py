"""
Phase 4: Seamless Poisson Blending
Owner: Member C

Composites the warped face region onto the source image
using OpenCV's seamlessClone (Poisson blending).
"""

import numpy as np
import cv2


def _feather_mask(mask: np.ndarray, erode_px: int = 3, blur_sigma: float = 5.0) -> np.ndarray:
    """Lightly erode and blur a binary mask to reduce halos/duplicate features."""
    mask = mask.copy()
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))
        mask = cv2.erode(mask, k)
    if blur_sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    return mask


def _mask_touches_border(mask: np.ndarray, margin: int = 2) -> bool:
    """
    Return True if any non-zero mask pixel lies within `margin` pixels of any
    image edge.

    cv2.seamlessClone's Poisson solver requires the mask to be fully interior
    to the image — if it touches or crosses the border the function raises
    cv2.error (or, on some builds, segfaults).  Detect this condition early so
    we can fall back gracefully.
    """
    return bool(
        mask[:margin, :].any()  or
        mask[-margin:, :].any() or
        mask[:, :margin].any()  or
        mask[:, -margin:].any()
    )


def _alpha_composite(src: np.ndarray, dst: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Simple alpha-composite fallback when seamlessClone cannot be used."""
    alpha     = mask.astype(np.float32) / 255.0
    alpha_3ch = np.stack([alpha, alpha, alpha], axis=-1)
    return (src * alpha_3ch + dst * (1.0 - alpha_3ch)).astype(np.uint8)


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
    # Feather mask to avoid eyebrow duplication and hard seams.
    # erode_px=8 (up from 3): tighter mask prevents warp from bleeding
    # outside the face boundary and keeps background SSIM ≥ 0.98.
    face_mask = _feather_mask(face_mask, erode_px=8, blur_sigma=5.0)

    # Find the center of the face mask for seamlessClone
    M = cv2.moments(face_mask)
    if M["m00"] == 0:
        print("[blend] Warning: empty mask after feathering — returning source image unchanged.")
        return source_img.copy()

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    center = (cx, cy)

    # seamlessClone hard-requires the mask to be fully inside the image border.
    # If the face is very close to an edge (e.g. tightly cropped portrait),
    # the Poisson solver crashes.  Detect this and fall back to alpha composite.
    if _mask_touches_border(face_mask, margin=2):
        print("[blend] Warning: face mask touches image border — "
              "seamlessClone skipped, using alpha composite.")
        result = _alpha_composite(warped_img, source_img, face_mask)
    else:
        try:
            result = cv2.seamlessClone(warped_img, source_img, face_mask, center, cv2.NORMAL_CLONE)
        except cv2.error as e:
            print(f"[blend] seamlessClone failed: {e}")
            print("[blend] Falling back to direct alpha composite.")
            result = _alpha_composite(warped_img, source_img, face_mask)

    # Per-channel color correction inside the face region.
    # seamlessClone's Poisson solver can shift channel means; we measure the
    # mean error vs source inside the face mask and subtract it back.
    # This reduces Color Drift (especially the R channel) without affecting
    # the background or the spatial expression geometry.
    face_px = face_mask > 0
    if face_px.any():
        for c in range(3):
            src_mean = float(source_img[:, :, c][face_px].mean())
            res_mean = float(result[:, :, c][face_px].mean())
            shift = res_mean - src_mean
            if abs(shift) > 1.0:          # only correct non-trivial drift
                result[:, :, c] = np.clip(
                    result[:, :, c].astype(np.int16) - int(round(shift)),
                    0, 255
                ).astype(np.uint8)

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

    # Labels — compute x offsets from actual image widths (images may differ in width)
    imgs   = [source_img, driver_img, result_img]
    widths = [img.shape[1] for img in imgs]
    for i, label in enumerate(["Source", "Driver", "Result"]):
        x = sum(widths[:i]) + i * 4 + 10   # cumulative width + divider widths (4 px each)
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