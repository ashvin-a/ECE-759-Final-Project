"""
Manual HOG implementation in Python.
Parameters must stay in sync with include/types.h.

Used for:
  - Training feature extraction in train_svm.py
  - Ground-truth reference vectors for C++ correctness validation
"""

import numpy as np

# Must match include/types.h
WIN_W        = 64
WIN_H        = 64
CELL_SIZE    = 8
BLOCK_SIZE   = 2    # in cells
NBINS        = 9
BLOCK_STRIDE = 1    # in cells
L2HYS_CLIP   = 0.2
L2HYS_EPS    = 1e-6

CELLS_X  = WIN_W // CELL_SIZE        # 8
CELLS_Y  = WIN_H // CELL_SIZE        # 8
BLOCKS_X = CELLS_X - BLOCK_SIZE + 1  # 7
BLOCKS_Y = CELLS_Y - BLOCK_SIZE + 1  # 7
FEAT_DIM = BLOCKS_X * BLOCKS_Y * BLOCK_SIZE * BLOCK_SIZE * NBINS  # 1764


def compute_gradients(gray: np.ndarray):
    """
    Compute per-pixel gradient magnitude and unsigned orientation [0, 180).
    Uses [-1, 0, 1] kernel (no Gaussian smoothing, matching Dalal & Triggs).
    gray: (H, W) float32 in [0, 255]
    Returns: mag (H, W), ori (H, W) in degrees
    """
    # Horizontal and vertical gradients with zero-padding at borders
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gx[:, 0]    = gray[:, 1]  - gray[:, 0]
    gx[:, -1]   = gray[:, -1] - gray[:, -2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    gy[0, :]    = gray[1, :]  - gray[0, :]
    gy[-1, :]   = gray[-1, :] - gray[-2, :]

    mag = np.sqrt(gx * gx + gy * gy)
    ori = np.degrees(np.arctan2(np.abs(gy), np.abs(gx)))  # unsigned: fold to [0, 90]
    # arctan2(|gy|, |gx|) gives [0,90]; remap to [0,180) using the sign of gy vs gx
    # Standard unsigned HOG: use arctan2 of the full vectors then mod 180
    ori = np.degrees(np.arctan2(gy, gx)) % 180.0

    return mag.astype(np.float32), ori.astype(np.float32)


def build_cell_histograms(mag: np.ndarray, ori: np.ndarray):
    """
    Build 9-bin unsigned orientation histograms for each 8x8 cell.
    Uses soft (bilinear) bin interpolation.
    Returns: hists (CELLS_Y, CELLS_X, NBINS) float32
    """
    H, W = mag.shape
    hists = np.zeros((CELLS_Y, CELLS_X, NBINS), dtype=np.float32)

    bin_width = 180.0 / NBINS  # 20 degrees per bin

    for cy in range(CELLS_Y):
        for cx in range(CELLS_X):
            y0, x0 = cy * CELL_SIZE, cx * CELL_SIZE
            cell_mag = mag[y0:y0+CELL_SIZE, x0:x0+CELL_SIZE]
            cell_ori = ori[y0:y0+CELL_SIZE, x0:x0+CELL_SIZE]

            # Bilinear interpolation across bins
            bin_f  = cell_ori / bin_width          # fractional bin index
            bin_lo = np.floor(bin_f).astype(int) % NBINS
            bin_hi = (bin_lo + 1) % NBINS
            alpha  = bin_f - np.floor(bin_f)       # weight for hi bin

            for b_lo, b_hi, a, m in zip(
                bin_lo.ravel(), bin_hi.ravel(), alpha.ravel(), cell_mag.ravel()
            ):
                hists[cy, cx, b_lo] += m * (1.0 - a)
                hists[cy, cx, b_hi] += m * a

    return hists


def l2hys_normalize(block: np.ndarray) -> np.ndarray:
    """
    L2-Hys normalization:
      1. L2 normalize
      2. Clip at L2HYS_CLIP
      3. L2 normalize again
    block: 1D float32 array of length BLOCK_SIZE*BLOCK_SIZE*NBINS = 36
    """
    norm = np.sqrt(np.dot(block, block) + L2HYS_EPS ** 2)
    v = block / norm
    v = np.clip(v, 0.0, L2HYS_CLIP)
    norm2 = np.sqrt(np.dot(v, v) + L2HYS_EPS ** 2)
    return (v / norm2).astype(np.float32)


def extract_hog(patch: np.ndarray) -> np.ndarray:
    """
    Compute the full 1764-dim HOG descriptor for a 64x64 BGR or gray patch.
    patch: (64, 64) or (64, 64, 3) uint8 or float32
    Returns: (1764,) float32
    """
    if patch.ndim == 3:
        # Convert BGR to grayscale using OpenCV coefficients
        gray = (0.114 * patch[:, :, 0] +
                0.587 * patch[:, :, 1] +
                0.299 * patch[:, :, 2]).astype(np.float32)
    else:
        gray = patch.astype(np.float32)

    assert gray.shape == (WIN_H, WIN_W), \
        f"Patch must be {WIN_H}x{WIN_W}, got {gray.shape}"

    mag, ori = compute_gradients(gray)
    hists    = build_cell_histograms(mag, ori)

    descriptor = []
    for by in range(BLOCKS_Y):
        for bx in range(BLOCKS_X):
            block_cells = hists[by:by+BLOCK_SIZE, bx:bx+BLOCK_SIZE, :]
            block_vec   = block_cells.ravel()
            descriptor.append(l2hys_normalize(block_vec))

    feat = np.concatenate(descriptor)
    assert feat.shape[0] == FEAT_DIM, f"Expected {FEAT_DIM} dims, got {feat.shape[0]}"
    return feat
