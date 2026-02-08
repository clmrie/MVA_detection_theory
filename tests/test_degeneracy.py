"""Tests for the degeneracy check module."""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.degeneracy import (
    check_collinearity,
    check_conditioning,
    check_orientation_preserving,
    check_valid_warp,
)


class TestCheckCollinearity:
    def test_collinear_points(self):
        """4 points on a line should be flagged as degenerate."""
        pts = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
        assert check_collinearity(pts)

    def test_three_collinear(self):
        """3 out of 4 collinear should also be flagged."""
        pts = np.array([[0, 0], [1, 0], [2, 0], [1, 5]], dtype=float)
        assert check_collinearity(pts)

    def test_general_position(self):
        """4 points in general position: no 3 collinear."""
        pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
        assert not check_collinearity(pts)

    def test_near_collinear(self):
        """Points that are almost but not quite collinear."""
        pts = np.array([[0, 0], [100, 0], [200, 0.5], [50, 80]], dtype=float)
        assert not check_collinearity(pts)


class TestCheckConditioning:
    def test_identity(self):
        """Identity matrix is well-conditioned."""
        assert check_conditioning(np.eye(3))

    def test_singular(self):
        """Singular matrix should fail."""
        H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
        assert not check_conditioning(H)

    def test_near_singular(self):
        """Near-singular matrix should fail."""
        H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1e-12]], dtype=float)
        assert not check_conditioning(H)

    def test_good_homography(self):
        """Typical homography should pass."""
        H = np.array([
            [1.05, 0.08, 15],
            [-0.03, 0.98, 10],
            [0.0001, 0.00005, 1],
        ])
        assert check_conditioning(H)


class TestCheckOrientationPreserving:
    def test_identity(self):
        """Identity preserves orientation."""
        pts1 = np.array([[100, 100], [200, 200], [300, 150]], dtype=float)
        pts2 = pts1.copy()
        assert check_orientation_preserving(np.eye(3), pts1, pts2)

    def test_flip(self):
        """Homography that maps w' < 0 should fail."""
        H = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0.01, 0, -1],  # w' = 0.01*x - 1, negative for x < 100
        ], dtype=float)
        pts1 = np.array([[50, 50], [30, 100]], dtype=float)
        pts2 = pts1.copy()
        assert not check_orientation_preserving(H, pts1, pts2)


class TestCheckValidWarp:
    def test_identity(self):
        """Identity warp should be valid."""
        assert check_valid_warp(np.eye(3), (480, 640))

    def test_mild_perspective(self):
        """Mild perspective should be valid."""
        H = np.array([
            [1.05, 0.08, 15],
            [-0.03, 0.98, 10],
            [0.0001, 0.00005, 1],
        ])
        assert check_valid_warp(H, (480, 640))

    def test_degenerate_warp(self):
        """Warp that sends corners to infinity should fail."""
        H = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0.01, 0.01, 0],  # w' = 0 for many points
        ], dtype=float)
        assert not check_valid_warp(H, (480, 640))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
