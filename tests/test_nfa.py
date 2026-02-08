# tests/test_nfa.py
"""Tests for the NFA (Number of False Alarms) computation module."""

import numpy as np
import pytest
from scipy.special import comb

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nfa import (
    log10_combi,
    precompute_log_combi_n,
    precompute_log_combi_k,
    compute_best_nfa,
)


class TestLog10Combi:
    def test_small_values(self):
        """Verify against scipy.special.comb for small n, k."""
        for n in range(1, 20):
            for k in range(0, n + 1):
                expected = np.log10(comb(n, k, exact=True))
                result = log10_combi(n, k)
                assert abs(result - expected) < 1e-10, \
                    f"log10(C({n},{k})): expected {expected}, got {result}"

    def test_symmetry(self):
        """C(n, k) == C(n, n-k)."""
        for n in [10, 20, 50]:
            for k in range(n + 1):
                assert abs(log10_combi(n, k) - log10_combi(n, n - k)) < 1e-10

    def test_edge_cases(self):
        """C(n, 0) = C(n, n) = 1 => log10 = 0."""
        for n in [0, 1, 5, 100]:
            assert log10_combi(n, 0) == 0.0
            assert log10_combi(n, n) == 0.0

    def test_invalid_k(self):
        """k < 0 or k > n should return -inf."""
        assert log10_combi(5, -1) == -np.inf
        assert log10_combi(5, 6) == -np.inf

    def test_large_values(self):
        """Test with large n to ensure no overflow."""
        result = log10_combi(1000, 500)
        # C(1000, 500) is astronomically large but log10 should be finite
        assert np.isfinite(result)
        assert result > 0


class TestPrecomputeLogCombiN:
    def test_consistency(self):
        """Precomputed table should match direct computation."""
        n = 50
        table = precompute_log_combi_n(n)
        assert len(table) == n + 1
        for k in range(n + 1):
            assert abs(table[k] - log10_combi(n, k)) < 1e-10


class TestPrecomputeLogCombiK:
    def test_consistency(self):
        """Precomputed table should match direct computation."""
        k = 4
        n_max = 100
        table = precompute_log_combi_k(k, n_max)
        assert len(table) == n_max + 1
        for m in range(k, n_max + 1):
            assert abs(table[m] - log10_combi(m, k)) < 1e-10, \
                f"C({m},{k}): table={table[m]}, direct={log10_combi(m, k)}"


class TestComputeBestNFA:
    def _setup_tables(self, n, sample_size=4):
        """Helper to create precomputed tables."""
        log_combi_n = precompute_log_combi_n(n)
        log_combi_k = precompute_log_combi_k(sample_size, n)
        return log_combi_n, log_combi_k

    def test_perfect_inliers(self):
        """All matches are perfect inliers => NFA should be very small."""
        n = 100
        sample_size = 4
        # Very small errors (near-perfect matches)
        sorted_errors = np.linspace(0.01, 1.0, n)
        sorted_sides = np.zeros(n, dtype=int)
        logalpha0 = np.array([np.log10(np.pi / (640 * 480))] * 2)
        lcn, lck = self._setup_tables(n)

        log_nfa, best_k, best_err = compute_best_nfa(
            sorted_errors, sorted_sides, logalpha0,
            n, sample_size, 1, lcn, lck,
        )
        assert log_nfa < 0, f"Expected log_NFA < 0 for perfect inliers, got {log_nfa}"
        assert best_k > sample_size

    def test_pure_outliers(self):
        """When all errors are huge (alpha clips to 1), NFA should be > 1.

        With alpha=1, having k inliers out of n is always expected, so
        the NFA reduces to combinatorial terms which are always >= 1.
        """
        n = 100
        sample_size = 4
        # Errors so large that alpha clips to 1: need logalpha0 + 0.5*log10(err) >= 0
        # logalpha0 ≈ -5, so need 0.5*log10(err) >= 5, i.e., err >= 1e10
        rng = np.random.default_rng(42)
        errors = rng.uniform(1e10, 1e12, n)
        sorted_errors = np.sort(errors)
        sorted_sides = np.zeros(n, dtype=int)
        logalpha0 = np.array([np.log10(np.pi / (640 * 480))] * 2)
        lcn, lck = self._setup_tables(n)

        log_nfa, best_k, best_err = compute_best_nfa(
            sorted_errors, sorted_sides, logalpha0,
            n, sample_size, 1, lcn, lck,
        )
        assert log_nfa >= 0, f"Expected log_NFA >= 0 for clipped alpha, got {log_nfa}"

    def test_mixed_inliers_outliers(self):
        """Mix of inliers (small error) and outliers (huge error).

        50 inliers with sub-pixel error, 50 outliers with error so large
        that alpha clips to 1. The NFA should detect exactly the inlier group.
        """
        n = 100
        sample_size = 4
        inlier_errors = np.linspace(0.1, 2.0, 50)
        outlier_errors = np.linspace(1e10, 1e12, 50)  # alpha clips to 1
        all_errors = np.concatenate([inlier_errors, outlier_errors])
        sorted_errors = np.sort(all_errors)
        sorted_sides = np.zeros(n, dtype=int)
        logalpha0 = np.array([np.log10(np.pi / (640 * 480))] * 2)
        lcn, lck = self._setup_tables(n)

        log_nfa, best_k, best_err = compute_best_nfa(
            sorted_errors, sorted_sides, logalpha0,
            n, sample_size, 1, lcn, lck,
        )
        # Should detect the inlier group
        assert log_nfa < 0, f"Expected detection, got log_NFA = {log_nfa}"
        # best_k should be close to 50
        assert 30 <= best_k <= 60, f"Expected ~50 inliers, got {best_k}"

    def test_too_few_points(self):
        """Fewer than sample_size+1 points => no detection."""
        n = 4
        sample_size = 4
        sorted_errors = np.array([0.1, 0.2, 0.3, 0.4])
        sorted_sides = np.zeros(n, dtype=int)
        logalpha0 = np.array([np.log10(np.pi / (640 * 480))] * 2)
        lcn = precompute_log_combi_n(4)
        lck = precompute_log_combi_k(4, 4)

        log_nfa, best_k, best_err = compute_best_nfa(
            sorted_errors, sorted_sides, logalpha0,
            n, sample_size, 1, lcn, lck,
        )
        assert log_nfa >= 0
        assert best_k == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
