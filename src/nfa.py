# src/nfa.py
"""
A-contrario NFA (Number of False Alarms) computation for ORSA.

All computations in log10 space to avoid underflow/overflow with
large binomial coefficients and small probabilities raised to large powers.

Reference: Moisan, Moulon, Monasse — "Automatic Homographic Registration
of a Pair of Images, with A Contrario Elimination of Outliers", IPOL 2012.
"""

import numpy as np


def log10_combi(n: int, k: int) -> float:
    """Compute log10(C(n, k)) iteratively for numerical stability.

    Uses the identity: C(n, k) = prod_{i=1}^{k} (n - i + 1) / i
    so log10(C(n, k)) = sum_{i=1}^{k} [log10(n - i + 1) - log10(i)].
    """
    if k < 0 or k > n:
        return -np.inf
    if k == 0 or k == n:
        return 0.0
    # Use smaller k for efficiency: C(n, k) = C(n, n-k)
    if k > n - k:
        k = n - k
    result = 0.0
    for i in range(1, k + 1):
        result += np.log10(n - i + 1) - np.log10(i)
    return result


def precompute_log_combi_n(n: int) -> np.ndarray:
    """Tabulate log10(C(n, k)) for k = 0, 1, ..., n.

    Returns array of length n + 1 where out[k] = log10(C(n, k)).
    """
    table = np.zeros(n + 1)
    for k in range(1, n + 1):
        table[k] = table[k - 1] + np.log10(n - k + 1) - np.log10(k)
    return table


def precompute_log_combi_k(k: int, n_max: int) -> np.ndarray:
    """Tabulate log10(C(m, k)) for m = 0, 1, ..., n_max.

    Returns array of length n_max + 1 where out[m] = log10(C(m, k)).
    Entries for m < k are set to 0 (log10(0) would be -inf, but these
    are never used in the NFA loop).
    """
    table = np.zeros(n_max + 1)
    if k > n_max:
        return table
    # C(k, k) = 1
    table[k] = 0.0
    for m in range(k + 1, n_max + 1):
        # C(m, k) = C(m-1, k) * m / (m - k)
        table[m] = table[m - 1] + np.log10(m) - np.log10(m - k)
    return table


def compute_best_nfa(
    sorted_errors: np.ndarray,
    sorted_sides: np.ndarray,
    logalpha0: np.ndarray,
    n_data: int,
    sample_size: int,
    n_outcomes: int,
    log_combi_n: np.ndarray,
    log_combi_k: np.ndarray,
    mult_error: float = 1.0,
    max_threshold: float = np.inf,
) -> tuple[float, int, float]:
    """Core NFA computation with adaptive epsilon.

    For each candidate inlier count k (from sample_size+1 to n_data),
    use the k-th sorted residual as the precision threshold and compute
    the NFA. Return the minimum.

    Parameters
    ----------
    sorted_errors : (n_data,) sorted ascending residuals (squared distances)
    sorted_sides : (n_data,) int, 0=left image dominates, 1=right
    logalpha0 : (2,) log10(pi / (w*h)) for [left, right] images
    n_data : total number of matches
    sample_size : minimal sample size (4 for homography)
    n_outcomes : number of models per sample (1 for homography)
    log_combi_n : precomputed log10(C(n_data - sample_size, j)) for j=0..n-p
    log_combi_k : precomputed log10(C(m, sample_size)) for m=0..n_data
    mult_error : multiplier for log10(error); 0.5 for squared distances
    max_threshold : maximum allowed error threshold

    Returns
    -------
    best_log_nfa : log10(NFA) of the best (minimum) NFA found
    best_k : number of inliers at the optimum
    best_error : the squared error threshold at the optimum
    """
    eps_machine = np.finfo(float).eps

    # Number-of-tests factor: log10(n_outcomes * (n - p))
    n_minus_p = n_data - sample_size
    if n_minus_p <= 0:
        return 0.0, 0, 0.0
    loge0 = np.log10(n_outcomes * n_minus_p)

    best_log_nfa = np.inf
    best_k = 0
    best_error = 0.0

    for i in range(sample_size, n_data):
        # i is 0-indexed; this corresponds to the (i+1)-th match
        # k = i + 1 = number of inliers if we use error[i] as threshold
        error_i = sorted_errors[i]

        if error_i > max_threshold:
            break

        side_i = int(sorted_sides[i])
        logalpha = logalpha0[side_i] + mult_error * np.log10(error_i + eps_machine)

        # Clip: probability cannot exceed 1 => logalpha <= 0
        if logalpha > 0.0:
            logalpha = 0.0

        # j = number of inliers beyond the sample_size points
        # j = (i + 1) - sample_size = i - sample_size + 1
        j = i - sample_size + 1  # j >= 1

        # NFA(k) = n_outcomes * (n-p) * C(n-p, j) * C(i+1, p) * alpha^j
        # But log_combi_n is indexed from 0..n-p, log_combi_k from 0..n
        log_nfa = (
            loge0
            + logalpha * j
            + log_combi_n[j]       # log10(C(n-p, j))
            + log_combi_k[i + 1]   # log10(C(k, p)) where k = i+1
        )

        if log_nfa < best_log_nfa:
            best_log_nfa = log_nfa
            best_k = i + 1
            best_error = error_i

    # If no candidate was tested, return log_nfa = 0 (NFA = 1, not meaningful)
    if best_log_nfa == np.inf:
        return 0.0, 0, 0.0

    return best_log_nfa, best_k, best_error


def compute_nfa_for_all_k(
    sorted_errors: np.ndarray,
    sorted_sides: np.ndarray,
    logalpha0: np.ndarray,
    n_data: int,
    sample_size: int,
    n_outcomes: int,
    log_combi_n: np.ndarray,
    log_combi_k: np.ndarray,
    mult_error: float = 1.0,
) -> np.ndarray:
    """Compute log10(NFA) for every k from sample_size+1 to n_data.

    Useful for plotting the NFA curve as a function of k.

    Returns array of shape (n_data,) where entry i contains the log_NFA
    when using i+1 inliers (entries for i < sample_size are set to inf).
    """
    eps_machine = np.finfo(float).eps
    n_minus_p = n_data - sample_size
    if n_minus_p <= 0:
        return np.full(n_data, np.inf)

    loge0 = np.log10(n_outcomes * n_minus_p)
    log_nfas = np.full(n_data, np.inf)

    for i in range(sample_size, n_data):
        error_i = sorted_errors[i]
        side_i = int(sorted_sides[i])
        logalpha = logalpha0[side_i] + mult_error * np.log10(error_i + eps_machine)
        if logalpha > 0.0:
            logalpha = 0.0
        j = i - sample_size + 1
        log_nfas[i] = (
            loge0
            + logalpha * j
            + log_combi_n[j]
            + log_combi_k[i + 1]
        )

    return log_nfas
