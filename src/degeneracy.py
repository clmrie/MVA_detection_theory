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
    # if 3 points are collinear the DLT system becomes rank-deficient
    # and we get a garbage homography, so we reject these samples early
    n = len(pts)
    for i, j, k in combinations(range(n), 3):
        # cross product gives twice the triangle area
        v1 = pts[j] - pts[i]
        v2 = pts[k] - pts[i]
        area = abs(v1[0] * v2[1] - v1[1] * v2[0])
        if area < threshold:
            return True  # degenerate
    return False


def check_conditioning(H: np.ndarray, threshold: float = 0.1) -> bool:
    """Check that H is well-conditioned.

    Parameters
    ----------
    H : (3, 3) homography matrix
    threshold : minimum acceptable inverse condition number.
        The IPOL reference rejects homographies with condition number
        above ~10 (inv_cond < 0.1).

    Returns
    -------
    True if H is well-conditioned, False if degenerate.
    """
    # we check on the normalized H before denormalization as recommended by IPOL
    # a badly conditioned H means the mapping is nearly singular and unreliable
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

    We need w' > 0 when we project a point through H otherwise it means
    the point got mapped behind the camera which is physically nonsensical
    We only check on the sample points not all points because invalid
    correspondences will just get infinite error later in symmetric_transfer_error
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

    We added this because some homographies pass all the other checks but
    still collapse the image to a tiny sliver or blow it up to crazy sizes
    This catches those degenerate cases that would pollute the NFA scoring
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

    # we use the shoelace formula to compute how much the area changed
    # if it grew or shrank by more than 100x the homography is degenerate
    x = mapped_xy[0, :]
    y = mapped_xy[1, :]
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
