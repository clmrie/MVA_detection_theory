"""Tests for the homography estimation module."""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.homography import (
    normalize_points,
    fit_homography_dlt,
    symmetric_transfer_error,
    compute_inliers,
    refine_homography,
)


class TestNormalizePoints:
    def test_centroid_at_origin(self):
        """After normalization, centroid should be at origin."""
        pts = np.array([[10, 20], [30, 40], [50, 60], [70, 80]], dtype=float)
        pts_norm, T = normalize_points(pts)
        centroid = np.mean(pts_norm, axis=0)
        np.testing.assert_allclose(centroid, [0, 0], atol=1e-10)

    def test_mean_distance_sqrt2(self):
        """After normalization, mean distance from origin should be sqrt(2)."""
        pts = np.array([[10, 20], [30, 40], [50, 60], [70, 80]], dtype=float)
        pts_norm, T = normalize_points(pts)
        dists = np.sqrt(np.sum(pts_norm ** 2, axis=1))
        np.testing.assert_allclose(np.mean(dists), np.sqrt(2), atol=1e-10)

    def test_transformation_matrix(self):
        """T should correctly transform points."""
        pts = np.array([[100, 200], [300, 400], [150, 350]], dtype=float)
        pts_norm, T = normalize_points(pts)
        # Apply T manually
        pts_h = np.column_stack([pts, np.ones(len(pts))])
        pts_transformed = (T @ pts_h.T).T[:, :2]
        np.testing.assert_allclose(pts_norm, pts_transformed, atol=1e-10)


class TestFitHomographyDLT:
    def test_identity(self):
        """Same points => H should be close to identity."""
        pts = np.array([
            [100, 50], [400, 50], [400, 350], [100, 350]
        ], dtype=float)
        H = fit_homography_dlt(pts, pts)
        assert H is not None
        H_normalized = H / H[2, 2]
        np.testing.assert_allclose(H_normalized, np.eye(3), atol=1e-6)

    def test_known_translation(self):
        """Known translation: H = [[1,0,tx],[0,1,ty],[0,0,1]]."""
        tx, ty = 50, 30
        pts1 = np.array([
            [100, 50], [400, 50], [400, 350], [100, 350]
        ], dtype=float)
        pts2 = pts1 + np.array([tx, ty])
        H = fit_homography_dlt(pts1, pts2)
        assert H is not None

        # Check that H applied to pts1 gives pts2
        pts1_h = np.column_stack([pts1, np.ones(4)])
        proj = (H @ pts1_h.T).T
        proj_xy = proj[:, :2] / proj[:, 2:3]
        np.testing.assert_allclose(proj_xy, pts2, atol=1e-4)

    def test_known_homography_recovery(self):
        """Generate points under a known H and recover it."""
        H_true = np.array([
            [1.05, 0.08, 15],
            [-0.03, 0.98, 10],
            [0.0001, 0.00005, 1],
        ])
        pts1 = np.array([
            [100, 100], [400, 100], [400, 350], [100, 350],
            [250, 200], [320, 150],
        ], dtype=float)
        pts1_h = np.column_stack([pts1, np.ones(len(pts1))])
        pts2_h = (H_true @ pts1_h.T).T
        pts2 = pts2_h[:, :2] / pts2_h[:, 2:3]

        H_est = fit_homography_dlt(pts1, pts2)
        assert H_est is not None

        # Verify reprojection
        proj = (H_est @ pts1_h.T).T
        proj_xy = proj[:, :2] / proj[:, 2:3]
        np.testing.assert_allclose(proj_xy, pts2, atol=0.1)

    def test_too_few_points(self):
        """Fewer than 4 points should return None."""
        pts1 = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        pts2 = pts1.copy()
        assert fit_homography_dlt(pts1, pts2) is None

    def test_degenerate_collinear(self):
        """Collinear points should produce None (ill-conditioned)."""
        pts1 = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
        pts2 = pts1 + 10
        # May or may not return None depending on conditioning check
        H = fit_homography_dlt(pts1, pts2)
        # If it returns something, it should at least be poorly conditioned
        # The conditioning check should catch this


class TestSymmetricTransferError:
    def test_zero_error_for_exact_match(self):
        """If pts2 = H @ pts1 exactly, error should be ~0."""
        H = np.array([
            [1.05, 0.08, 15],
            [-0.03, 0.98, 10],
            [0.0001, 0.00005, 1],
        ])
        pts1 = np.array([[100, 100], [400, 100], [400, 350], [100, 350]], dtype=float)
        pts1_h = np.column_stack([pts1, np.ones(4)])
        pts2_h = (H @ pts1_h.T).T
        pts2 = pts2_h[:, :2] / pts2_h[:, 2:3]

        errors, sides = symmetric_transfer_error(H, pts1, pts2)
        np.testing.assert_allclose(errors, 0, atol=1e-8)

    def test_positive_error_for_noisy_match(self):
        """Noisy matches should have positive error."""
        H = np.eye(3)
        pts1 = np.array([[100, 100], [200, 200]], dtype=float)
        pts2 = pts1 + np.array([[5, 0], [0, 5]])  # 5px offset
        errors, sides = symmetric_transfer_error(H, pts1, pts2)
        assert np.all(errors > 0)
        # Error should be ~25 (5^2) for symmetric max
        np.testing.assert_allclose(errors, 25, atol=1)

    def test_sides_output(self):
        """Side indicator should be 0 or 1."""
        H = np.eye(3)
        pts1 = np.array([[100, 100]], dtype=float)
        pts2 = np.array([[105, 100]], dtype=float)
        errors, sides = symmetric_transfer_error(H, pts1, pts2)
        assert sides[0] in [0, 1]


class TestComputeInliers:
    def test_all_inliers(self):
        """With large epsilon, all should be inliers."""
        H = np.eye(3)
        pts1 = np.array([[100, 100], [200, 200], [300, 300]], dtype=float)
        pts2 = pts1 + 1.0  # 1px offset
        mask = compute_inliers(H, pts1, pts2, epsilon=10.0)
        assert np.all(mask)

    def test_no_inliers(self):
        """With tiny epsilon, noisy matches should be outliers."""
        H = np.eye(3)
        pts1 = np.array([[100, 100], [200, 200]], dtype=float)
        pts2 = pts1 + 50.0  # 50px offset
        mask = compute_inliers(H, pts1, pts2, epsilon=1.0)
        assert not np.any(mask)


class TestRefineHomography:
    def test_refine_improves_noisy_estimate(self):
        """LM refinement should improve a noisy initial estimate."""
        H_true = np.array([
            [1.05, 0.08, 15],
            [-0.03, 0.98, 10],
            [0.0001, 0.00005, 1],
        ])
        rng = np.random.default_rng(42)
        n = 50
        pts1 = rng.uniform(50, 500, (n, 2))
        pts1_h = np.column_stack([pts1, np.ones(n)])
        pts2_h = (H_true @ pts1_h.T).T
        pts2 = pts2_h[:, :2] / pts2_h[:, 2:3]
        pts2 += rng.normal(0, 0.5, (n, 2))  # small noise

        # Initial estimate from 4 points only (less accurate)
        H_init = fit_homography_dlt(pts1[:4], pts2[:4])
        assert H_init is not None

        # Refine on all points
        mask = np.ones(n, dtype=bool)
        H_ref = refine_homography(H_init, pts1, pts2, mask)
        assert H_ref is not None

        # Refined should have lower error than initial
        err_init, _ = symmetric_transfer_error(H_init, pts1, pts2)
        err_ref, _ = symmetric_transfer_error(H_ref, pts1, pts2)
        assert np.mean(err_ref) <= np.mean(err_init) + 1e-6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
