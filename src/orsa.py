# src/orsa.py
"""
ORSA (Optimized Random SAmpling) for homography estimation.

Implements the a-contrario RANSAC variant from:
Moisan, Moulon, Monasse — "Automatic Homographic Registration of a Pair
of Images, with A Contrario Elimination of Outliers", IPOL 2012.

The key difference from RANSAC: instead of a fixed inlier threshold,
ORSA adaptively selects the threshold that minimizes the NFA (Number of
False Alarms). A detection is meaningful when NFA < 1.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from .nfa import (
    compute_best_nfa,
    compute_nfa_for_all_k,
    precompute_log_combi_k,
    precompute_log_combi_n,
)
from .homography import (
    fit_homography_dlt,
    refine_homography,
    symmetric_transfer_error,
)
from .degeneracy import (
    check_collinearity,
    check_orientation_preserving,
    check_valid_warp,
)


@dataclass
class OrsaResult:
    """Result of ORSA homography estimation."""
    H: np.ndarray | None           # 3x3 best homography (None if no detection)
    inlier_mask: np.ndarray        # boolean mask (n,)
    nfa: float                     # NFA value (10^log_nfa)
    log_nfa: float                 # log10(NFA)
    n_inliers: int                 # number of inliers
    epsilon: float                 # optimal error threshold (pixel distance)
    n_iterations: int              # actual iterations performed
    n_models_tested: int           # valid models evaluated (not rejected by degeneracy)
    runtime: float                 # wall-clock seconds
    n_matches: int                 # total input matches
    reprojection_errors: np.ndarray | None = None  # errors on inliers
    log_nfa_history: list = field(default_factory=list)  # best log_nfa over iterations


def orsa_homography(
    pts1: np.ndarray,
    pts2: np.ndarray,
    img1_shape: tuple,
    img2_shape: tuple,
    max_iter: int = 1000,
    confidence: float = 0.99,
    max_threshold: float | None = None,
    seed: int | None = None,
    verbose: bool = False,
) -> OrsaResult:
    """Main ORSA loop for homography estimation.

    Parameters
    ----------
    pts1 : (n, 2) matches in image 1
    pts2 : (n, 2) matches in image 2
    img1_shape : (h1, w1) of image 1
    img2_shape : (h2, w2) of image 2
    max_iter : maximum number of random samples
    confidence : probability of finding the correct model (for adaptive iter)
    max_threshold : maximum allowed squared error threshold. If None,
        defaults to the squared diagonal of the larger image. This prevents
        counting matches with errors larger than the image as "inliers",
        which would violate the NFA's independence assumption.
    seed : random seed for reproducibility
    verbose : print progress

    Returns
    -------
    OrsaResult with best homography, inliers, NFA, and diagnostics.
    """
    t_start = time.time()
    n = len(pts1)
    sample_size = 4
    n_outcomes = 1  # one model per 4-point sample

    # Default max_threshold: squared diagonal of the larger image
    if max_threshold is None:
        h1, w1 = img1_shape[:2]
        h2, w2 = img2_shape[:2]
        diag = max(np.sqrt(w1**2 + h1**2), np.sqrt(w2**2 + h2**2))
        max_threshold = diag ** 2

    # Edge case: not enough matches
    if n < sample_size + 1:
        return OrsaResult(
            H=None, inlier_mask=np.zeros(n, dtype=bool),
            nfa=1.0, log_nfa=0.0, n_inliers=0, epsilon=0.0,
            n_iterations=0, n_models_tested=0,
            runtime=time.time() - t_start, n_matches=n,
        )

    rng = np.random.default_rng(seed)

    # Precompute logalpha0 for each image
    h1, w1 = img1_shape[:2]
    h2, w2 = img2_shape[:2]
    logalpha0 = np.array([
        np.log10(np.pi / (w1 * h1)),  # left image (side=0)
        np.log10(np.pi / (w2 * h2)),  # right image (side=1)
    ])

    # Precompute binomial coefficient tables
    n_minus_p = n - sample_size
    log_combi_n = precompute_log_combi_n(n_minus_p)
    log_combi_k = precompute_log_combi_k(sample_size, n)

    # Initialize best model tracking
    best_log_nfa = 0.0  # log10(1) = 0; NFA >= 1 means not meaningful
    best_H = None
    best_inlier_mask = np.zeros(n, dtype=bool)
    best_epsilon = 0.0
    best_k = 0

    n_models_tested = 0
    n_iter = max_iter
    log_nfa_history = []

    # Reserve last 10% iterations for focused sampling
    n_iter_reserve = max(1, max_iter // 10)
    n_iter_main = max_iter - n_iter_reserve

    actual_iter = 0
    for it in range(max_iter):
        actual_iter = it + 1

        # Adaptive stopping
        if it >= n_iter:
            break

        # Sample 4 random matches
        if it < n_iter_main or best_H is None or best_k < sample_size:
            indices = rng.choice(n, size=sample_size, replace=False)
        else:
            # Focused sampling from current inlier set
            inlier_indices = np.where(best_inlier_mask)[0]
            if len(inlier_indices) >= sample_size:
                indices = rng.choice(inlier_indices, size=sample_size, replace=False)
            else:
                indices = rng.choice(n, size=sample_size, replace=False)

        # Degeneracy checks on sample
        if check_collinearity(pts1[indices]) or check_collinearity(pts2[indices]):
            continue

        # Estimate H via DLT
        H = fit_homography_dlt(pts1[indices], pts2[indices])
        if H is None:
            continue

        # Orientation check
        if not check_orientation_preserving(H, pts1, pts2):
            continue

        # Valid warp check: reject degenerate H that collapses space
        if not check_valid_warp(H, img1_shape):
            continue

        n_models_tested += 1

        # Compute all residuals
        errors, sides = symmetric_transfer_error(H, pts1, pts2)

        # Sort by error
        order = np.argsort(errors)
        sorted_errors = errors[order]
        sorted_sides = sides[order]

        # Compute NFA with adaptive epsilon
        log_nfa, k, err_at_k = compute_best_nfa(
            sorted_errors, sorted_sides, logalpha0,
            n, sample_size, n_outcomes,
            log_combi_n, log_combi_k,
            mult_error=1.0,
            max_threshold=max_threshold,
        )

        log_nfa_history.append(min(log_nfa, best_log_nfa if best_H is not None else log_nfa))

        # Update best model if improved
        if log_nfa < best_log_nfa:
            best_log_nfa = log_nfa
            best_H = H.copy()
            best_k = k
            best_epsilon = np.sqrt(max(err_at_k, 0.0))

            # Build inlier mask (first k matches by error)
            inlier_indices_sorted = order[:k]
            best_inlier_mask = np.zeros(n, dtype=bool)
            best_inlier_mask[inlier_indices_sorted] = True

            # Adaptive iteration count
            inlier_ratio = k / n
            if 0 < inlier_ratio < 1:
                p_all_inliers = inlier_ratio ** sample_size
                if p_all_inliers > 1e-15:
                    new_n_iter = int(np.ceil(
                        np.log(1 - confidence) / np.log(1 - p_all_inliers)
                    ))
                    n_iter = min(max_iter, max(it + 1, new_n_iter))

            if verbose:
                print(
                    f"  Iter {it}: log10(NFA) = {log_nfa:.2f}, "
                    f"inliers = {k}/{n}, eps = {best_epsilon:.2f} px"
                )

    # Refinement phase: refit on inliers until convergence
    if best_H is not None and best_log_nfa < 0:
        best_H, best_inlier_mask, best_log_nfa, best_k, best_epsilon = _refine_until_convergence(
            best_H, best_inlier_mask, pts1, pts2,
            logalpha0, n, sample_size, n_outcomes,
            log_combi_n, log_combi_k,
            max_threshold, verbose,
        )

        # LM polish
        H_refined = refine_homography(best_H, pts1, pts2, best_inlier_mask)
        if H_refined is not None:
            # Verify refinement didn't make things worse
            errors_ref, sides_ref = symmetric_transfer_error(H_refined, pts1, pts2)
            order_ref = np.argsort(errors_ref)
            log_nfa_ref, k_ref, err_ref = compute_best_nfa(
                errors_ref[order_ref], sides_ref[order_ref], logalpha0,
                n, sample_size, n_outcomes,
                log_combi_n, log_combi_k,
                mult_error=1.0, max_threshold=max_threshold,
            )
            if log_nfa_ref <= best_log_nfa:
                best_H = H_refined
                best_log_nfa = log_nfa_ref
                best_k = k_ref
                best_epsilon = np.sqrt(max(err_ref, 0.0))
                inlier_indices_ref = order_ref[:k_ref]
                best_inlier_mask = np.zeros(n, dtype=bool)
                best_inlier_mask[inlier_indices_ref] = True

    # Compute final reprojection errors on inliers
    reproj_errors = None
    if best_H is not None and np.any(best_inlier_mask):
        all_errors, _ = symmetric_transfer_error(best_H, pts1, pts2)
        reproj_errors = np.sqrt(all_errors[best_inlier_mask])

    runtime = time.time() - t_start

    return OrsaResult(
        H=best_H,
        inlier_mask=best_inlier_mask,
        nfa=10 ** best_log_nfa if best_log_nfa < 300 else float('inf'),
        log_nfa=best_log_nfa,
        n_inliers=int(np.sum(best_inlier_mask)),
        epsilon=best_epsilon,
        n_iterations=actual_iter,
        n_models_tested=n_models_tested,
        runtime=runtime,
        n_matches=n,
        reprojection_errors=reproj_errors,
        log_nfa_history=log_nfa_history,
    )


def _refine_until_convergence(
    H, inlier_mask, pts1, pts2,
    logalpha0, n, sample_size, n_outcomes,
    log_combi_n, log_combi_k,
    max_threshold, verbose,
    max_refine_iter=10,
):
    """Iteratively refit H on inliers and recompute NFA until convergence."""
    best_log_nfa_ref = np.inf
    # Compute initial NFA
    errors, sides = symmetric_transfer_error(H, pts1, pts2)
    order = np.argsort(errors)
    current_log_nfa, current_k, current_err = compute_best_nfa(
        errors[order], sides[order], logalpha0,
        n, sample_size, n_outcomes,
        log_combi_n, log_combi_k,
        mult_error=1.0, max_threshold=max_threshold,
    )
    best_log_nfa_ref = current_log_nfa
    best_H = H.copy()
    best_mask = inlier_mask.copy()
    best_k = current_k
    best_eps = current_err

    for _ in range(max_refine_iter):
        # Refit DLT on current inliers
        inlier_pts1 = pts1[best_mask]
        inlier_pts2 = pts2[best_mask]

        if len(inlier_pts1) < 4:
            break

        H_refit = fit_homography_dlt(inlier_pts1, inlier_pts2)
        if H_refit is None:
            break

        # Recompute errors and NFA
        errors_new, sides_new = symmetric_transfer_error(H_refit, pts1, pts2)
        order_new = np.argsort(errors_new)
        log_nfa_new, k_new, err_new = compute_best_nfa(
            errors_new[order_new], sides_new[order_new], logalpha0,
            n, sample_size, n_outcomes,
            log_combi_n, log_combi_k,
            mult_error=1.0, max_threshold=max_threshold,
        )

        if log_nfa_new < best_log_nfa_ref:
            best_log_nfa_ref = log_nfa_new
            best_H = H_refit.copy()
            best_k = k_new
            best_eps = err_new
            inlier_indices_new = order_new[:k_new]
            best_mask = np.zeros(n, dtype=bool)
            best_mask[inlier_indices_new] = True

            if verbose:
                print(
                    f"  Refine: log10(NFA) = {log_nfa_new:.2f}, "
                    f"inliers = {k_new}/{n}"
                )
        else:
            break  # No improvement, stop

    return best_H, best_mask, best_log_nfa_ref, best_k, np.sqrt(max(best_eps, 0.0))
