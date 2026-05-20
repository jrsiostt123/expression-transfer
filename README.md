# Expression Transfer

A classical computer vision pipeline for transferring facial expressions between images — no deep learning required.

**Course:** Image Processing, Spring 2026 — NTHU CS  
**Demo:** May 25, 2026

---

## Pipeline Overview

```
Input: source face (neutral) + driver face (expressive) [driver neutral OPTIONAL]
  │
  ├── Phase 0: Face Alignment (stabilization)   [src/align.py]
  ├── Phase 1: Landmark Detection               [src/landmark.py]
  ├── Phase 2: Expression Parameterization      [src/expression.py]
  ├── Phase 3: Face Warping                     [src/warp.py]
  └── Phase 4: Poisson Blending                 [src/blend.py]
  │
Output: source face with driver's expression
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dlib landmark model
bash scripts/download_model.sh

# 3. Run demo (no driver neutral)
python demo.py --source data/sample_images/source.jpg --driver data/sample_images/driver.jpg

# Optionally, if you have a neutral image of the driver (better):
python demo.py --source data/sample_images/source.jpg --driver data/sample_images/driver_expr.jpg \
               --driver-neutral data/sample_images/driver_neutral.jpg
```

---

## Project Structure

```
expression-transfer/
├── src/
│   ├── landmark.py       # Phase 1: dlib 68-point detection
│   ├── expression.py     # Phase 2: displacement vector computation
│   ├── warp.py           # Phase 3: Delaunay triangulation + affine warping
│   └── blend.py          # Phase 4: Poisson blending + compositing
├── data/
│   └── sample_images/    # Test face images (not committed — see below)
├── output/               # Generated results (gitignored)
├── docs/                 # Proposal and spec documents
├── notebooks/            # Jupyter notebooks for exploration
├── demo.py               # End-to-end demo script
├── evaluate.py           # SSIM and landmark deviation metrics
├── requirements.txt
└── scripts/
    └── download_model.sh
```

---

## Module Interface Contract

> ⚠️ **All members must respect these interfaces.** Do not change argument/return types without team discussion.

### `landmark.py`
```python
detect_landmarks(image: np.ndarray) -> np.ndarray  # shape: (68, 2), dtype: float32
# Returns (x, y) pixel coordinates for each of 68 landmarks
```

### `expression.py`
```python
compute_displacement(
    source_lm: np.ndarray,
    driver_lm: np.ndarray,
    driver_neutral_lm: np.ndarray | None = None,
    scale: float = 0.7,
) -> np.ndarray
# shape: (68, 2), dtype: float32
# With driver_neutral_lm: uses (driver_expression - driver_neutral) rescaled to source face size.
# Without driver_neutral_lm: directly aligns driver to source and warps toward those positions.
```

### `warp.py`
```python
warp_face(source_img: np.ndarray, source_lm: np.ndarray, displacement: np.ndarray) -> tuple[np.ndarray, np.ndarray]
# Returns: (warped_image, face_mask) — both same shape as source_img
# Notes: Per-triangle affine warping with corrected patch coordinate frames.
```

### `blend.py`
```python
blend(source_img: np.ndarray, warped_img: np.ndarray, mask: np.ndarray) -> np.ndarray
# Returns: final composited image, same shape as source_img
```

---

## Team & Branch Structure

| Member | Phase | Branch |
|--------|-------|--------|
| Member A | Landmark detection + Expression parameterization + Alignment | `feature/landmark` |
| Member B | Face warping | `feature/warp` |
| Member C | Poisson blending + Evaluation | `feature/blend` |

**Integration branch:** `develop` — merge here first, then to `main` after testing.

---

## Dependencies

- Python 3.9+
- OpenCV
- dlib
- NumPy
- SciPy
- scikit-image (for SSIM evaluation)

---

## Dataset

We use the [Radboud Faces Database (RaFD)](https://rafd.socsci.ru.nl/) and [CK+](https://www.pitt.edu/~emotion/ck-spread.htm) for evaluation. These require registration — do not commit images to the repo.
