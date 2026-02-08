# src/degeneracy.py
"""
Degeneracy and sanity checks for homography estimation.

Rejects degenerate configurations (collinear points, ill-conditioned
matrices, orientation-flipping warps) before they pollute the NFA scoring.
"""

import numpy as np
from itertools import combinations


def check_collinearity(pts: np.ndarray, threshold: float = 1e-4) -> bool:
    """Check if any 3 of the given points are nearly collinear.

    Parameters
    ----------
    pts : (m, 2) array of 2D points (typically m=4 for a minimal sample)
    threshold : minimum triangle area to consider non-degenerate

    Returns
    -------
    True if the configuration is degenerate (collinear), False if OK.
    """
    n = len(pts)
    for i, j, k in combinations(range(n), 3):
        # Signed area of triangle = 0.5 * |det([pj-pi, pk-pi])|
        v1 = pts[j] - pts[i]
        v2 = pts[k] - pts[i]
        area = abs(v1[0] * v2[1] - v1[1] * v2[0])
        if area < threshold:
            return True  # degenerate
    return False


def check_conditioning(H: np.ndarray, threshold: float = 1e-6) -> bool:
    """Check that H is well-conditioned.

    Parameters
    ----------
    H : (3, 3) homography matrix
    threshold : minimum acceptable inverse condition number.
        Set low (1e-6) because denormalized H can have large condition
        numbers even for valid homographies with significant translations.

    Returns
    -------
    True if H is well-conditioned, False if degenerate.
    """
    sigma = np.linalg.svd(H, compute_uv=False)
    if sigma[0] < 1e-15:
        return False
    inv_cond = sigma[-1] / sigma[0]
    return bool(inv_cond >= threshold)


def check_orientation_preserving(
    H: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    indices: np.ndarray | None = None,
) -> bool:
    """Check that H preserves orientation at the given correspondences.

    The homography maps x1 -> H @ [x1, y1, 1]^T = [x', y', w']^T.
    We need w' > 0 for the mapping to be orientation-preserving.
    We check on the points from image 1.

    Parameters
    ----------
    H : (3, 3) homography matrix
    pts1 : (n, 2) points in image 1
    pts2 : (n, 2) points in image 2 (unused, kept for API consistency)
    indices : optional subset of indices to check; if None, check all

    Returns
    -------
    True if orientation is preserved at all checked points, False otherwise.
    """
    if indices is not None:
        pts = pts1[indices]
    else:
        pts = pts1

    # w' = H[2,0]*x + H[2,1]*y + H[2,2]
    w_prime = H[2, 0] * pts[:, 0] + H[2, 1] * pts[:, 1] + H[2, 2]
    return bool(np.all(w_prime > 0))


def check_valid_warp(H: np.ndarray, img_shape: tuple, max_area_ratio: float = 100.0) -> bool:
    """Check that H does not warp image corners to absurd locations.

    Parameters
    ----------
    H : (3, 3) homography matrix
    img_shape : (h, w) of the source image
    max_area_ratio : maximum ratio of warped area to original area

    Returns
    -------
    True if the warp is reasonable, False otherwise.
    """
    h, w = img_shape[:2]
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1],
    ], dtype=float).T  # (3, 4)

    mapped = H @ corners  # (3, 4)

    # Check all w' > 0
    if np.any(mapped[2, :] <= 0):
        return False

    mapped_xy = mapped[:2, :] / mapped[2:, :]  # (2, 4)

    # Check coordinates are finite and not too large
    if not np.all(np.isfinite(mapped_xy)):
        return False
    if np.any(np.abs(mapped_xy) > 1e6):
        return False

    # Check area ratio using shoelace formula
    x = mapped_xy[0, :]
    y = mapped_xy[1, :]
    # Shoelace formula for quadrilateral area
    area_mapped = 0.5 * abs(
        x[0] * y[1] - x[1] * y[0]
        + x[1] * y[2] - x[2] * y[1]
        + x[2] * y[3] - x[3] * y[2]
        + x[3] * y[0] - x[0] * y[3]
    )
    area_original = h * w
    if area_original < 1:
        return False
    ratio = area_mapped / area_original
    return bool(ratio < max_area_ratio and ratio > 1.0 / max_area_ratio)
