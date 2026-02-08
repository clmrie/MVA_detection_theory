"""
Homography estimation, error computation, and refinement.

Implements the Direct Linear Transform (DLT) with Hartley normalization,
symmetric transfer error, and Levenberg-Marquardt refinement.
"""

import numpy as np
from scipy.optimize import least_squares

from .degeneracy import check_conditioning


def normalize_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalization: translate centroid to origin, scale so
    mean distance from origin equals sqrt(2).

    Parameters
    ----------
    pts : (n, 2) array of 2D points

    Returns
    -------
    pts_norm : (n, 2) normalized points
    T : (3, 3) normalization matrix such that
        pts_norm_h = T @ pts_h  (homogeneous coordinates)
    """
    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid
    mean_dist = np.mean(np.sqrt(np.sum(pts_centered ** 2, axis=1)))
    if mean_dist < 1e-10:
        mean_dist = 1e-10
    scale = np.sqrt(2.0) / mean_dist

    T = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0],
    ])

    pts_h = np.column_stack([pts, np.ones(len(pts))])
    pts_norm_h = (T @ pts_h.T).T
    pts_norm = pts_norm_h[:, :2]

    return pts_norm, T


def fit_homography_dlt(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray | None:
    """Compute homography via DLT from n >= 4 correspondences.

    Parameters
    ----------
    pts1 : (n, 2) source points
    pts2 : (n, 2) destination points (pts2 ~ H @ pts1 in homogeneous coords)

    Returns
    -------
    H : (3, 3) homography matrix, or None if estimation fails.
    """
    n = len(pts1)
    if n < 4:
        return None

    # Normalize
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    # Build the 2n x 9 matrix A
    A = np.zeros((2 * n, 9))
    for i in range(n):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]
        A[2 * i] = [x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2]
        A[2 * i + 1] = [0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2]

    # SVD
    try:
        _, S, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None

    # Last row of Vt (= last column of V)
    h = Vt[-1, :]
    H_norm = h.reshape(3, 3)

    # Denormalize
    H = np.linalg.inv(T2) @ H_norm @ T1

    # Normalize so H[2,2] = 1 (if possible)
    if abs(H[2, 2]) < 1e-15:
        return None
    H = H / H[2, 2]

    # Check conditioning
    if not check_conditioning(H):
        return None

    return H


def symmetric_transfer_error(
    H: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute symmetric transfer error for each correspondence.

    error_i = max(d(x2, H @ x1)^2, d(x1, H^{-1} @ x2)^2)

    Parameters
    ----------
    H : (3, 3) homography matrix
    pts1, pts2 : (n, 2) point arrays

    Returns
    -------
    errors : (n,) max of forward and backward squared distances
    sides : (n,) int, 0 = backward (left) error dominates,
            1 = forward (right) error dominates
    """
    n = len(pts1)
    large_error = 1e18

    # Forward: project pts1 through H -> compare with pts2
    pts1_h = np.column_stack([pts1, np.ones(n)])  # (n, 3)
    proj_fwd = (H @ pts1_h.T).T  # (n, 3)
    w_fwd = proj_fwd[:, 2]

    # Backward: project pts2 through H^{-1} -> compare with pts1
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(n, large_error), np.zeros(n, dtype=int)

    pts2_h = np.column_stack([pts2, np.ones(n)])
    proj_bwd = (H_inv @ pts2_h.T).T
    w_bwd = proj_bwd[:, 2]

    # Forward error (right image)
    err_right = np.full(n, large_error)
    valid_fwd = np.abs(w_fwd) > 1e-10
    if np.any(valid_fwd):
        proj_fwd_xy = proj_fwd[valid_fwd, :2] / proj_fwd[valid_fwd, 2:3]
        err_right[valid_fwd] = np.sum((pts2[valid_fwd] - proj_fwd_xy) ** 2, axis=1)

    # Backward error (left image)
    err_left = np.full(n, large_error)
    valid_bwd = np.abs(w_bwd) > 1e-10
    if np.any(valid_bwd):
        proj_bwd_xy = proj_bwd[valid_bwd, :2] / proj_bwd[valid_bwd, 2:3]
        err_left[valid_bwd] = np.sum((pts1[valid_bwd] - proj_bwd_xy) ** 2, axis=1)

    # Symmetric: take max
    errors = np.maximum(err_left, err_right)
    # side = 1 if right error dominates (forward), 0 if left (backward)
    sides = (err_right >= err_left).astype(int)

    return errors, sides


def compute_inliers(
    H: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Return boolean mask of inliers (error <= epsilon^2).

    Parameters
    ----------
    H : (3, 3) homography
    pts1, pts2 : (n, 2) points
    epsilon : distance threshold in pixels (not squared)

    Returns
    -------
    mask : (n,) boolean, True for inliers
    """
    errors, _ = symmetric_transfer_error(H, pts1, pts2)
    return errors <= epsilon ** 2


def refine_homography(
    H_init: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    inlier_mask: np.ndarray,
) -> np.ndarray | None:
    """Refine H using Levenberg-Marquardt minimization on inlier
    correspondences, minimizing symmetric transfer error.

    Parameterization: 8 free parameters (H[2,2] = 1 fixed).

    Parameters
    ----------
    H_init : (3, 3) initial homography
    pts1, pts2 : (n, 2) all points
    inlier_mask : (n,) boolean mask

    Returns
    -------
    H_refined : (3, 3) refined homography, or None if refinement fails.
    """
    p1 = pts1[inlier_mask]
    p2 = pts2[inlier_mask]
    n_inliers = len(p1)

    if n_inliers < 4:
        return None

    # Normalize H so H[2,2] = 1
    H0 = H_init.copy()
    if abs(H0[2, 2]) < 1e-15:
        return None
    H0 = H0 / H0[2, 2]

    # Initial parameters: 8 values (exclude H[2,2])
    h0 = H0.flatten()[:8]  # first 8 elements

    p1_h = np.column_stack([p1, np.ones(n_inliers)])

    def residuals(h):
        H = np.array([
            [h[0], h[1], h[2]],
            [h[3], h[4], h[5]],
            [h[6], h[7], 1.0],
        ])
        # Forward projection
        proj = (H @ p1_h.T).T
        w = proj[:, 2]
        # Avoid division by zero
        w = np.where(np.abs(w) < 1e-10, 1e-10, w)
        proj_xy = proj[:, :2] / w[:, np.newaxis]
        return (p2 - proj_xy).flatten()  # 2*n_inliers residuals

    try:
        result = least_squares(residuals, h0, method='lm', max_nfev=200)
        h_opt = result.x
        H_refined = np.array([
            [h_opt[0], h_opt[1], h_opt[2]],
            [h_opt[3], h_opt[4], h_opt[5]],
            [h_opt[6], h_opt[7], 1.0],
        ])
        if not check_conditioning(H_refined):
            return None
        return H_refined
    except Exception:
        return None
