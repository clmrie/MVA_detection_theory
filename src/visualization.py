"""
Visualization utilities for ORSA homography registration.

Provides functions to draw matches, display warped images,
plot NFA curves, and generate experiment summaries.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2

matplotlib.use('Agg')  # non-interactive backend for saving figures


def draw_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    inlier_mask: np.ndarray | None = None,
    max_display: int = 200,
    title: str = "",
    figsize: tuple = (16, 8),
) -> plt.Figure:
    """Draw matches between two images side by side.

    Green lines for inliers, red for outliers. If inlier_mask is None,
    all matches are drawn in blue.

    Parameters
    ----------
    img1, img2 : input images
    pts1, pts2 : (n, 2) matched points
    inlier_mask : (n,) boolean mask; None means no classification
    max_display : maximum matches to draw (subsampled if more)
    title : figure title
    figsize : figure size

    Returns
    -------
    fig : matplotlib Figure
    """
    # Convert to RGB for display
    if len(img1.shape) == 2:
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    else:
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    if len(img2.shape) == 2:
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    else:
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    h1, w1 = img1_rgb.shape[:2]
    h2, w2 = img2_rgb.shape[:2]
    h_max = max(h1, h2)

    # Composite image
    canvas = np.zeros((h_max, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1_rgb
    canvas[:h2, w1:w1 + w2] = img2_rgb

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(canvas)
    ax.set_axis_off()

    n = len(pts1)
    if n == 0:
        if title:
            ax.set_title(title)
        return fig

    # Subsample if too many
    if n > max_display:
        indices = np.random.choice(n, max_display, replace=False)
    else:
        indices = np.arange(n)

    for idx in indices:
        x1, y1 = pts1[idx]
        x2, y2 = pts2[idx]
        x2_shifted = x2 + w1

        if inlier_mask is not None:
            if inlier_mask[idx]:
                color = (0, 0.8, 0)  # green for inliers
                alpha = 0.7
                lw = 0.8
            else:
                color = (0.9, 0, 0)  # red for outliers
                alpha = 0.3
                lw = 0.4
        else:
            color = (0.2, 0.5, 1.0)  # blue
            alpha = 0.5
            lw = 0.6

        ax.plot([x1, x2_shifted], [y1, y2], '-', color=color, alpha=alpha, linewidth=lw)
        ax.plot(x1, y1, '.', color=color, markersize=3, alpha=alpha)
        ax.plot(x2_shifted, y2, '.', color=color, markersize=3, alpha=alpha)

    if title:
        ax.set_title(title, fontsize=14)

    fig.tight_layout()
    return fig


def warp_and_blend(
    img1: np.ndarray,
    img2: np.ndarray,
    H: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Warp img1 onto img2's frame using H and alpha-blend.

    Parameters
    ----------
    img1 : source image
    img2 : destination image
    H : (3, 3) homography mapping img1 -> img2
    alpha : blend weight (0 = only img2, 1 = only warped img1)

    Returns
    -------
    blended : RGB image (numpy array)
    """
    h2, w2 = img2.shape[:2]
    warped = cv2.warpPerspective(img1, H, (w2, h2))

    # Convert to float for blending
    warped_f = warped.astype(np.float64)
    img2_f = img2.astype(np.float64)

    # Only blend where warped image has content
    mask = (warped.sum(axis=2) > 0) if len(warped.shape) == 3 else (warped > 0)
    blended = img2_f.copy()
    if len(blended.shape) == 3:
        blended[mask] = alpha * warped_f[mask] + (1 - alpha) * img2_f[mask]
    else:
        blended[mask] = alpha * warped_f[mask] + (1 - alpha) * img2_f[mask]

    return blended.astype(np.uint8)


def plot_nfa_curve(
    sorted_errors: np.ndarray,
    log_nfas: np.ndarray,
    best_k: int,
    title: str = "NFA as a function of inlier count",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plot log10(NFA) as a function of k (number of inliers).

    Parameters
    ----------
    sorted_errors : (n,) sorted ascending residuals
    log_nfas : (n,) log10(NFA) for each k (inf for invalid k)
    best_k : optimal inlier count
    title : figure title
    figsize : figure size

    Returns
    -------
    fig : matplotlib Figure
    """
    fig, ax1 = plt.subplots(1, 1, figsize=figsize)

    n = len(sorted_errors)
    k_values = np.arange(1, n + 1)

    # Filter valid NFA values
    valid = np.isfinite(log_nfas)

    # Plot log_NFA
    ax1.plot(k_values[valid], log_nfas[valid], 'b-', linewidth=1.5, label='log$_{10}$(NFA)')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='NFA = 1 threshold')
    if best_k > 0:
        best_idx = best_k - 1
        if best_idx < n and valid[best_idx]:
            ax1.axvline(x=best_k, color='r', linestyle='--', alpha=0.7,
                        label=f'Best k = {best_k}')
            ax1.plot(best_k, log_nfas[best_idx], 'ro', markersize=8,
                     label=f'min log$_{{10}}$(NFA) = {log_nfas[best_idx]:.1f}')

    ax1.set_xlabel('Number of inliers (k)', fontsize=12)
    ax1.set_ylabel('log$_{10}$(NFA)', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper right')

    # Secondary axis: sorted errors
    ax2 = ax1.twinx()
    ax2.plot(k_values, np.sqrt(np.maximum(sorted_errors, 0)), 'g-', alpha=0.4, linewidth=1,
             label='Error (px)')
    ax2.set_ylabel('Error threshold (px)', fontsize=12, color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    ax1.set_title(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_error_histogram(
    errors: np.ndarray,
    inlier_mask: np.ndarray,
    epsilon: float,
    title: str = "Reprojection Error Distribution",
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Plot histogram of reprojection errors, with inlier/outlier coloring.

    Parameters
    ----------
    errors : (n,) squared errors for all matches
    inlier_mask : (n,) boolean
    epsilon : threshold in pixels
    title : figure title
    figsize : figure size

    Returns
    -------
    fig : matplotlib Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    distances = np.sqrt(np.maximum(errors, 0))
    max_dist = min(np.percentile(distances, 95), epsilon * 5)

    inlier_dists = distances[inlier_mask]
    outlier_dists = distances[~inlier_mask]

    bins = np.linspace(0, max_dist, 50)
    ax.hist(inlier_dists[inlier_dists <= max_dist], bins=bins, alpha=0.7,
            color='green', label=f'Inliers ({len(inlier_dists)})')
    ax.hist(outlier_dists[outlier_dists <= max_dist], bins=bins, alpha=0.5,
            color='red', label=f'Outliers ({len(outlier_dists)})')
    ax.axvline(x=epsilon, color='blue', linestyle='--', linewidth=2,
               label=f'Threshold = {epsilon:.1f} px')

    ax.set_xlabel('Reprojection error (px)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_experiment_summary(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    result,
    sorted_errors: np.ndarray | None = None,
    log_nfas: np.ndarray | None = None,
    title: str = "",
    figsize: tuple = (20, 12),
) -> plt.Figure:
    """Multi-panel summary of an ORSA experiment.

    Panel 1: Matches with inlier/outlier coloring
    Panel 2: Warped blend (if H found)
    Panel 3: NFA curve (if data provided)
    Panel 4: Error histogram (if H found)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel 1: Matches
    _draw_matches_on_ax(axes[0, 0], img1, img2, pts1, pts2, result.inlier_mask)
    axes[0, 0].set_title(
        f'Matches: {result.n_inliers} inliers / {result.n_matches} total\n'
        f'log$_{{10}}$(NFA) = {result.log_nfa:.1f}, '
        f'eps = {result.epsilon:.1f} px',
        fontsize=11,
    )

    # Panel 2: Warped blend
    if result.H is not None:
        blended = warp_and_blend(img1, img2, result.H)
        if len(blended.shape) == 3:
            blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        else:
            blended_rgb = blended
        axes[0, 1].imshow(blended_rgb)
        axes[0, 1].set_title('Warped blend', fontsize=11)
    else:
        axes[0, 1].text(0.5, 0.5, 'No homography found', ha='center', va='center',
                        fontsize=14, transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Warped blend (N/A)', fontsize=11)
    axes[0, 1].set_axis_off()

    # Panel 3: NFA curve
    if sorted_errors is not None and log_nfas is not None:
        n = len(sorted_errors)
        k_values = np.arange(1, n + 1)
        valid = np.isfinite(log_nfas)
        axes[1, 0].plot(k_values[valid], log_nfas[valid], 'b-', linewidth=1.5)
        axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        if result.n_inliers > 0:
            axes[1, 0].axvline(x=result.n_inliers, color='r', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('k')
        axes[1, 0].set_ylabel('log$_{10}$(NFA)')
        axes[1, 0].set_title('NFA curve', fontsize=11)
    else:
        axes[1, 0].text(0.5, 0.5, 'NFA curve not available', ha='center', va='center',
                        fontsize=14, transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('NFA curve (N/A)', fontsize=11)

    # Panel 4: Error histogram
    if result.H is not None:
        from .homography import symmetric_transfer_error
        all_errors, _ = symmetric_transfer_error(result.H, pts1, pts2)
        distances = np.sqrt(np.maximum(all_errors, 0))
        max_dist = min(np.percentile(distances, 95), result.epsilon * 5) if result.epsilon > 0 else np.percentile(distances, 95)
        bins = np.linspace(0, max(max_dist, 0.1), 50)
        inlier_d = distances[result.inlier_mask]
        outlier_d = distances[~result.inlier_mask]
        axes[1, 1].hist(inlier_d[inlier_d <= max_dist], bins=bins, alpha=0.7,
                        color='green', label='Inliers')
        axes[1, 1].hist(outlier_d[outlier_d <= max_dist], bins=bins, alpha=0.5,
                        color='red', label='Outliers')
        axes[1, 1].axvline(x=result.epsilon, color='blue', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Error (px)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Error distribution', fontsize=11)
        axes[1, 1].legend(fontsize=9)
    else:
        axes[1, 1].text(0.5, 0.5, 'No model', ha='center', va='center',
                        fontsize=14, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Error distribution (N/A)', fontsize=11)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def _draw_matches_on_ax(ax, img1, img2, pts1, pts2, inlier_mask, max_display=200):
    """Helper to draw matches on a given matplotlib axis."""
    if len(img1.shape) == 2:
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    else:
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    if len(img2.shape) == 2:
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    else:
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    h1, w1 = img1_rgb.shape[:2]
    h2, w2 = img2_rgb.shape[:2]
    h_max = max(h1, h2)
    canvas = np.zeros((h_max, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1_rgb
    canvas[:h2, w1:w1 + w2] = img2_rgb
    ax.imshow(canvas)
    ax.set_axis_off()

    n = len(pts1)
    if n == 0:
        return
    indices = np.random.choice(n, min(n, max_display), replace=False) if n > max_display else np.arange(n)
    for idx in indices:
        x1, y1 = pts1[idx]
        x2, y2 = pts2[idx]
        color = (0, 0.8, 0) if inlier_mask[idx] else (0.9, 0, 0)
        alpha = 0.7 if inlier_mask[idx] else 0.3
        lw = 0.8 if inlier_mask[idx] else 0.4
        ax.plot([x1, x2 + w1], [y1, y2], '-', color=color, alpha=alpha, linewidth=lw)
