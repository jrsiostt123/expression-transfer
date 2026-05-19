"""
Phase 2: Expression Parameterization
Owner: Member A

Computes displacement vectors representing a facial expression,
normalized by face scale for robustness across different image sizes.
"""

import numpy as np


def _interocular_distance(landmarks: np.ndarray) -> float:
    """
    Compute inter-ocular distance for scale normalization.
    Left eye outer corner: landmark 36
    Right eye outer corner: landmark 45
    """
    return float(np.linalg.norm(landmarks[45] - landmarks[36]))


def compute_displacement(
    source_lm: np.ndarray,
    driver_lm: np.ndarray,
    driver_neutral_lm: np.ndarray,
    scale: float = 1.0
) -> np.ndarray:
    """
    Compute expression displacement vectors from driver's neutral to expressive pose,
    then rescale to match the source face's proportions.

    Args:
        source_lm:         (68, 2) landmarks of the source (target) face
        driver_lm:         (68, 2) landmarks of the driver face (expressive)
        driver_neutral_lm: (68, 2) landmarks of the driver face (neutral)
        scale:             Optional manual scale multiplier (0.7–1.0 to reduce exaggeration)

    Returns:
        displacement: (68, 2) float32 array of (dx, dy) vectors
    """
    # Raw expression delta on driver face
    raw_delta = driver_lm - driver_neutral_lm  # (68, 2)

    # Normalize by driver's face scale, re-apply at source's face scale
    driver_scale = _interocular_distance(driver_neutral_lm)
    source_scale = _interocular_distance(source_lm)

    if driver_scale < 1e-6:
        raise ValueError("Driver neutral landmarks degenerate — interocular distance near zero.")

    displacement = raw_delta * (source_scale / driver_scale) * scale
    return displacement.astype(np.float32)


def apply_displacement(landmarks: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    """
    Apply displacement vectors to a set of landmarks.

    Args:
        landmarks:    (68, 2) source landmark positions
        displacement: (68, 2) displacement vectors

    Returns:
        new_landmarks: (68, 2) displaced landmark positions
    """
    return (landmarks + displacement).astype(np.float32)


if __name__ == "__main__":
    # Mock test with random data
    src = np.random.rand(68, 2).astype(np.float32) * 200 + 100
    drv = np.random.rand(68, 2).astype(np.float32) * 200 + 100
    drv_n = drv + np.random.rand(68, 2).astype(np.float32) * 5

    disp = compute_displacement(src, drv, drv_n)
    print(f"Displacement shape: {disp.shape}")
    print(f"Max displacement: {np.abs(disp).max():.2f}px")
    new_lm = apply_displacement(src, disp)
    print(f"New landmarks shape: {new_lm.shape}")
