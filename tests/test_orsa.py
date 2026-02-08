"""Tests for the ORSA homography estimation module."""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orsa import orsa_homography
from experiments.synthetic import generate_synthetic_matches, evaluate_homography


IMG_SHAPE = (480, 640)


class TestOrsaBasic:
    def test_identity(self):
        """Points near identity: pts2 = pts1 + small noise."""
        rng = np.random.default_rng(42)
        n = 100
        pts1 = rng.uniform(50, 500, (n, 2))
        pts2 = pts1 + rng.normal(0, 0.5, (n, 2))

        result = orsa_homography(
            pts1, pts2, IMG_SHAPE, IMG_SHAPE,
            max_iter=500, seed=42,
        )
        assert result.H is not None
        assert result.log_nfa < 0, f"Expected detection, got log_NFA = {result.log_nfa}"
        assert result.n_inliers > 50

        # H should be close to identity (small noise causes slight offset)
        H = result.H / result.H[2, 2]
        np.testing.assert_allclose(H, np.eye(3), atol=0.5)

    def test_known_homography(self):
        """Synthetic data with known H: should recover it."""
        H_true = np.array([
            [1.05, 0.08, 15],
            [-0.03, 0.98, 10],
            [0.0001, 0.00005, 1],
        ])
        pts1, pts2, gt_mask = generate_synthetic_matches(
            n_inliers=100, n_outliers=50,
            H_true=H_true, noise_sigma=1.0,
            img_shape=IMG_SHAPE, seed=42,
        )
        result = orsa_homography(
            pts1, pts2, IMG_SHAPE, IMG_SHAPE,
            max_iter=1000, seed=42,
        )
        assert result.H is not None
        assert result.log_nfa < 0

        # Check accuracy (tolerance accounts for noise + estimation variance)
        h_error = evaluate_homography(result.H, H_true, IMG_SHAPE)
        assert h_error['corner_error_mean'] < 20.0, \
            f"Corner error too high: {h_error['corner_error_mean']:.2f} px"

    def test_pure_outliers_no_detection(self):
        """All random matches: ORSA should NOT detect a homography.

        Statistical test: run multiple seeds and verify that the majority
        do not produce a false detection (NFA >= 1). The a-contrario
        framework guarantees a low false alarm rate on average, but
        individual runs may still produce occasional false alarms.
        """
        n_trials = 10
        false_alarms = 0
        for seed in range(n_trials):
            pts1, pts2, _ = generate_synthetic_matches(
                n_inliers=0, n_outliers=100,
                H_true=np.eye(3),
                img_shape=IMG_SHAPE, seed=seed * 100,
            )
            result = orsa_homography(
                pts1, pts2, IMG_SHAPE, IMG_SHAPE,
                max_iter=300, seed=seed,
            )
            if result.log_nfa < 0:
                false_alarms += 1
        # Allow at most 2 false alarms out of 10 trials
        assert false_alarms <= 2, \
            f"Too many false alarms: {false_alarms}/{n_trials}"

    def test_high_outlier_ratio(self):
        """80% outliers: should still detect the homography."""
        H_true = np.array([
            [1.05, 0.08, 15],
            [-0.03, 0.98, 10],
            [0.0001, 0.00005, 1],
        ])
        pts1, pts2, gt_mask = generate_synthetic_matches(
            n_inliers=40, n_outliers=160,
            H_true=H_true, noise_sigma=1.0,
            img_shape=IMG_SHAPE, seed=42,
        )
        result = orsa_homography(
            pts1, pts2, IMG_SHAPE, IMG_SHAPE,
            max_iter=2000, seed=42,
        )
        assert result.H is not None
        assert result.log_nfa < 0, \
            f"Failed to detect with 80% outliers: log_NFA = {result.log_nfa}"

    def test_deterministic_with_seed(self):
        """Same seed should give identical results."""
        pts1, pts2, _ = generate_synthetic_matches(
            n_inliers=80, n_outliers=80,
            H_true=np.array([[1.05, 0.08, 15], [-0.03, 0.98, 10], [0.0001, 0.00005, 1]]),
            noise_sigma=1.0, img_shape=IMG_SHAPE, seed=42,
        )
        result1 = orsa_homography(pts1, pts2, IMG_SHAPE, IMG_SHAPE, seed=123)
        result2 = orsa_homography(pts1, pts2, IMG_SHAPE, IMG_SHAPE, seed=123)

        assert result1.log_nfa == result2.log_nfa
        assert result1.n_inliers == result2.n_inliers
        if result1.H is not None and result2.H is not None:
            np.testing.assert_allclose(result1.H, result2.H, atol=1e-10)


class TestOrsaEdgeCases:
    def test_too_few_matches(self):
        """Fewer than 5 matches: cannot detect."""
        pts1 = np.array([[100, 100], [200, 200], [300, 300], [400, 400]], dtype=float)
        pts2 = pts1 + 1
        result = orsa_homography(pts1, pts2, IMG_SHAPE, IMG_SHAPE, seed=42)
        assert result.n_inliers == 0

    def test_exactly_4_good_matches(self):
        """Exactly 4 matches that form a valid homography, plus 1 outlier."""
        H_true = np.array([[1.0, 0, 10], [0, 1.0, 5], [0, 0, 1]])
        pts1 = np.array([
            [100, 100], [400, 100], [400, 350], [100, 350], [250, 250],
        ], dtype=float)
        pts1_h = np.column_stack([pts1[:4], np.ones(4)])
        pts2_top = (H_true @ pts1_h.T).T[:, :2]
        pts2 = np.vstack([pts2_top, [500, 10]])  # last is outlier
        result = orsa_homography(pts1, pts2, IMG_SHAPE, IMG_SHAPE, max_iter=500, seed=42)
        # With only 5 points total, detection may or may not happen
        # but it shouldn't crash
        assert result.n_iterations > 0

    def test_result_fields(self):
        """Verify all result fields are populated."""
        pts1, pts2, _ = generate_synthetic_matches(
            n_inliers=50, n_outliers=50,
            H_true=np.eye(3), noise_sigma=1.0,
            img_shape=IMG_SHAPE, seed=42,
        )
        result = orsa_homography(pts1, pts2, IMG_SHAPE, IMG_SHAPE, seed=42)
        assert isinstance(result.log_nfa, float)
        assert isinstance(result.n_inliers, int)
        assert isinstance(result.n_iterations, int)
        assert isinstance(result.n_models_tested, int)
        assert isinstance(result.runtime, float)
        assert result.runtime >= 0
        assert len(result.inlier_mask) == len(pts1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
