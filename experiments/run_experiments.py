# experiments/run_experiments.py
"""
Experiment runner for ORSA homography registration.

Runs all experiment types (null model, synthetic, real images, failure cases,
sensitivity analysis) and saves results as JSON + figures.

Usage:
    python -m experiments.run_experiments [--experiment NAME] [--output-dir DIR]

Run from the project root (MVA_detection_theory/).
"""

import argparse
import json
import os
import sys
import time

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orsa import orsa_homography
from src.homography import symmetric_transfer_error
from src.matching import detect_and_match
from src.nfa import (
    compute_nfa_for_all_k,
    precompute_log_combi_k,
    precompute_log_combi_n,
)
from src.visualization import (
    draw_matches,
    plot_experiment_summary,
    plot_nfa_curve,
    plot_error_histogram,
)
from experiments.synthetic import (
    evaluate_homography,
    generate_synthetic_matches,
    make_test_homographies,
)


def save_result(result_dict: dict, path: str):
    """Save result dictionary as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convert numpy types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    with open(path, 'w') as f:
        json.dump(result_dict, f, indent=2, default=convert)
    print(f"  Saved: {path}")


# ─── Helper: load imgB matches and reference ORSA result ──────────────────

_imgB_cache = {}


def _load_imgB_matches(data_dir: str):
    """Load imgB pair, run SIFT matching, and compute reference ORSA result.

    Returns a dict with pts1, pts2, img_shape1, img_shape2, and ref_result.
    Cached so repeated calls don't recompute.
    """
    if data_dir in _imgB_cache:
        return _imgB_cache[data_dir]

    pairs = _find_image_pairs(data_dir, prefix='imgB')
    if not pairs:
        raise FileNotFoundError(
            f"No imgB image pair found in {data_dir}. "
            "Place imgB_1.jpg and imgB_2.jpg in the data directory."
        )
    _, path1, path2 = pairs[0]
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Could not read images: {path1}, {path2}")

    match_result = detect_and_match(img1, img2, method='sift')
    ref_result = orsa_homography(
        match_result.pts1, match_result.pts2,
        img1.shape[:2], img2.shape[:2],
        max_iter=1000, seed=42,
    )
    print(f"  imgB reference: {match_result.n_matches} matches, "
          f"{ref_result.n_inliers} inliers, "
          f"log10(NFA) = {ref_result.log_nfa:.1f}")

    data = {
        'pts1': match_result.pts1,
        'pts2': match_result.pts2,
        'img_shape1': img1.shape[:2],
        'img_shape2': img2.shape[:2],
        'img1': img1,
        'img2': img2,
        'ref_result': ref_result,
    }
    _imgB_cache[data_dir] = data
    return data


# Experiment 1: Null Model Validation

def run_null_model(output_dir: str, data_dir: str):
    """Verify that ORSA does NOT detect homographies when correspondences
    are randomly shuffled, using real SIFT keypoints from imgB.

    We shuffle pts2 to destroy all true correspondences while preserving
    the realistic spatial distribution of keypoints.
    """
    print("\n=== Experiment 1: Null Model Validation (shuffled real keypoints) ===")
    imgB = _load_imgB_matches(data_dir)
    pts1_all, pts2_all = imgB['pts1'], imgB['pts2']
    shape1, shape2 = imgB['img_shape1'], imgB['img_shape2']
    n_total = len(pts1_all)

    subsample_sizes = [50, 100, 200, 500, 1000]
    n_trials = 50
    results = []

    for n in subsample_sizes:
        if n > n_total:
            print(f"  Skipping n={n} (only {n_total} matches available)")
            continue
        raw_log_nfas = []
        for trial in range(n_trials):
            rng = np.random.default_rng(trial * 1000 + n)
            idx = rng.choice(n_total, size=n, replace=False)
            pts1_sub = pts1_all[idx]
            pts2_sub = pts2_all[idx]
            # Shuffle pts2 to break all true correspondences
            pts2_shuffled = pts2_sub[rng.permutation(n)]

            result = orsa_homography(
                pts1_sub, pts2_shuffled, shape1, shape2,
                max_iter=500, seed=trial * 1000 + n,
            )
            raw_log_nfas.append(result.raw_log_nfa)

        raw_log_nfas = np.array(raw_log_nfas)
        finite_mask = np.isfinite(raw_log_nfas)
        finite_vals = raw_log_nfas[finite_mask]
        n_false_alarms = int(np.sum(raw_log_nfas < 0))
        n_no_model = int(np.sum(~finite_mask))
        mean_val = float(np.mean(finite_vals)) if len(finite_vals) > 0 else float('inf')
        min_val = float(np.min(finite_vals)) if len(finite_vals) > 0 else float('inf')
        print(
            f"  n={n}: "
            f"false alarms = {n_false_alarms}/{n_trials}, "
            f"mean raw log_NFA = {mean_val:.2f}, "
            f"min raw log_NFA = {min_val:.2f}"
            f"{f', no model tested = {n_no_model}' if n_no_model > 0 else ''}"
        )
        results.append({
            'n': n,
            'n_trials': n_trials,
            'false_alarms': n_false_alarms,
            'n_no_model': n_no_model,
            'raw_log_nfas': finite_vals.tolist(),
            'mean_raw_log_nfa': mean_val,
            'min_raw_log_nfa': min_val,
        })

    save_result({'experiment': 'null_model', 'source': 'imgB_shuffled',
                 'results': results},
                os.path.join(output_dir, 'null_model.json'))

    # Plot distribution of raw log_NFA
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for r in results:
        ax.hist(r['raw_log_nfas'], bins=30, alpha=0.6,
                label=f"n={r['n']}")
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2,
               label='Detection threshold (NFA = 1)')
    ax.set_xlabel('log$_{10}$(NFA$_{\\mathrm{min}}$)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Null Model: best log$_{10}$(NFA) per trial '
                 '(shuffled real keypoints)', fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'null_model.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: {os.path.join(output_dir, 'null_model.png')}")


# Experiment 2: Outlier Injection on Real Data

def run_outlier_injection(output_dir: str, data_dir: str):
    """Test ORSA robustness by injecting random outliers into real imgB matches.

    We start with the real SIFT matches from imgB and progressively add
    random correspondences. The ORSA result on clean data serves as
    pseudo ground-truth for measuring inlier recovery and H consistency.
    """
    print("\n=== Experiment 2: Outlier Injection (imgB) ===")
    imgB = _load_imgB_matches(data_dir)
    pts1_real, pts2_real = imgB['pts1'], imgB['pts2']
    shape1, shape2 = imgB['img_shape1'], imgB['img_shape2']
    ref_result = imgB['ref_result']
    n_real = len(pts1_real)
    ref_inlier_mask = ref_result.inlier_mask  # which of the originals are inliers
    n_ref_inliers = int(np.sum(ref_inlier_mask))

    # Injection counts: add N random correspondences on top of the real ones
    injection_counts = [0, 250, 550, 1100, 2750, 5500]
    n_trials = 10
    results = []

    for n_inject in injection_counts:
        n_total = n_real + n_inject
        effective_noise = n_inject / n_total if n_total > 0 else 0
        trial_results = []

        for trial in range(n_trials):
            rng = np.random.default_rng(trial * 1000 + n_inject)

            if n_inject > 0:
                # Generate random correspondences within image bounds
                h1, w1 = shape1
                h2, w2 = shape2
                noise_pts1 = np.column_stack([
                    rng.uniform(0, w1, n_inject),
                    rng.uniform(0, h1, n_inject),
                ])
                noise_pts2 = np.column_stack([
                    rng.uniform(0, w2, n_inject),
                    rng.uniform(0, h2, n_inject),
                ])
                pts1 = np.vstack([pts1_real, noise_pts1])
                pts2 = np.vstack([pts2_real, noise_pts2])
            else:
                pts1, pts2 = pts1_real.copy(), pts2_real.copy()

            result = orsa_homography(
                pts1, pts2, shape1, shape2,
                max_iter=1000, seed=trial,
            )

            # Inlier recovery: how many of the original ref inliers are
            # still found as inliers?
            if result.n_inliers > 0:
                # The first n_real entries correspond to original matches
                orsa_mask_on_real = result.inlier_mask[:n_real]
                recovered = int(np.sum(orsa_mask_on_real & ref_inlier_mask))
                noise_as_inlier = int(np.sum(result.inlier_mask[n_real:]))
                precision = recovered / (recovered + noise_as_inlier) \
                    if (recovered + noise_as_inlier) > 0 else 0
                recall = recovered / n_ref_inliers if n_ref_inliers > 0 else 0
            else:
                recovered = 0
                noise_as_inlier = 0
                precision = 0
                recall = 0

            # H consistency
            corner_error = np.nan
            if result.H is not None and ref_result.H is not None:
                h_err = evaluate_homography(result.H, ref_result.H, shape1)
                corner_error = h_err['corner_error_mean']

            trial_results.append({
                'detected': result.log_nfa < 0,
                'log_nfa': float(result.log_nfa),
                'n_inliers': result.n_inliers,
                'recovered': recovered,
                'noise_as_inlier': noise_as_inlier,
                'precision': float(precision),
                'recall': float(recall),
                'corner_error': float(corner_error),
                'epsilon': float(result.epsilon),
                'runtime': float(result.runtime),
            })

        detected = sum(1 for t in trial_results if t['detected'])
        mean_recall = np.mean([t['recall'] for t in trial_results])
        mean_prec = np.mean([t['precision'] for t in trial_results])
        print(
            f"  +{n_inject} noise ({effective_noise:.0%} extra): "
            f"detected={detected}/{n_trials}, "
            f"precision={mean_prec:.3f}, "
            f"recall={mean_recall:.3f}"
        )
        results.append({
            'n_injected': n_inject,
            'n_total': n_total,
            'effective_noise_ratio': float(effective_noise),
            'trials': trial_results,
        })

    save_result({
        'experiment': 'outlier_injection', 'source': 'imgB',
        'n_real_matches': n_real, 'n_ref_inliers': n_ref_inliers,
        'results': results,
    }, os.path.join(output_dir, 'outlier_injection.json'))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ratios = [r['effective_noise_ratio'] for r in results]
    mean_prec = [np.mean([t['precision'] for t in r['trials']]) for r in results]
    mean_rec = [np.mean([t['recall'] for t in r['trials']]) for r in results]
    std_prec = [np.std([t['precision'] for t in r['trials']]) for r in results]
    std_rec = [np.std([t['recall'] for t in r['trials']]) for r in results]

    ax1.errorbar(ratios, mean_prec, yerr=std_prec, marker='o', capsize=5,
                 label='Precision')
    ax1.errorbar(ratios, mean_rec, yerr=std_rec, marker='s', capsize=5,
                 label='Recall')
    ax1.set_xlabel('Fraction of injected noise', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Inlier Recovery vs. Noise Injection', fontsize=13)
    ax1.legend()
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    mean_corner = []
    for r in results:
        errs = [t['corner_error'] for t in r['trials']
                if t['detected'] and not np.isnan(t['corner_error'])]
        mean_corner.append(np.mean(errs) if errs else np.nan)
    ax2.plot(ratios, mean_corner, 'o-', color='purple')
    ax2.set_xlabel('Fraction of injected noise', fontsize=12)
    ax2.set_ylabel('Corner error vs. H$_{\\mathrm{ref}}$ (px)', fontsize=12)
    ax2.set_title('Homography Consistency vs. Noise Injection', fontsize=13)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'outlier_injection.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: {os.path.join(output_dir, 'outlier_injection.png')}")


# Experiment 3: Real Images — Easy Case

def run_real_easy(output_dir: str, data_dir: str):
    """Real image pair where a homography is clearly valid."""
    print("\n=== Experiment 3: Real Images — Easy Case ===")

    # Look for image pairs in data directory
    pairs = _find_image_pairs(data_dir, prefix='imgA')
    if not pairs:
        print("  No imgA image pairs found. Generating synthetic images instead.")
        pairs = [_generate_synthetic_image_pair(
            data_dir, 'imgA', difficulty='easy')]

    for name, img1_path, img2_path in pairs:
        print(f"  Processing: {name}")
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            print(f"    Failed to load images, skipping.")
            continue

        # Feature matching
        match_result = detect_and_match(img1, img2, method='sift', ratio_thresh=0.75)
        print(f"    Matches: {match_result.n_matches}")
        if match_result.n_matches < 10:
            print(f"    Too few matches, skipping.")
            continue

        # ORSA
        result = orsa_homography(
            match_result.pts1, match_result.pts2,
            img1.shape[:2], img2.shape[:2],
            max_iter=1000, seed=42, verbose=True,
        )

        print(
            f"    log10(NFA) = {result.log_nfa:.2f}, "
            f"inliers = {result.n_inliers}/{result.n_matches}, "
            f"eps = {result.epsilon:.2f} px, "
            f"runtime = {result.runtime:.3f} s"
        )

        # Compute NFA curve for visualization
        sorted_errors, log_nfas = _compute_nfa_curve(result, match_result, img1, img2)

        # Save summary figure
        fig = plot_experiment_summary(
            img1, img2, match_result.pts1, match_result.pts2, result,
            sorted_errors=sorted_errors, log_nfas=log_nfas,
            title=f'Easy Case: {name}',
        )
        fig_path = os.path.join(output_dir, f'real_easy_{name}.png')
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"    Saved: {fig_path}")

        # Save JSON
        save_result({
            'experiment': 'real_easy',
            'name': name,
            'n_matches': result.n_matches,
            'n_inliers': result.n_inliers,
            'log_nfa': float(result.log_nfa),
            'epsilon': float(result.epsilon),
            'runtime': float(result.runtime),
            'n_iterations': result.n_iterations,
            'n_models_tested': result.n_models_tested,
            'reproj_error_mean': float(np.mean(result.reprojection_errors))
                if result.reprojection_errors is not None else None,
            'reproj_error_median': float(np.median(result.reprojection_errors))
                if result.reprojection_errors is not None else None,
        }, os.path.join(output_dir, f'real_easy_{name}.json'))


# Experiment 4: Real Images — Hard Case

def run_real_hard(output_dir: str, data_dir: str):
    """Real image pair with significant viewpoint change or many outliers."""
    print("\n=== Experiment 4: Real Images — Hard Case ===")

    pairs = _find_image_pairs(data_dir, prefix='imgB')
    if not pairs:
        print("  No imgB image pairs found. Generating synthetic images instead.")
        pairs = [_generate_synthetic_image_pair(
            data_dir, 'imgB', difficulty='hard')]

    for name, img1_path, img2_path in pairs:
        print(f"  Processing: {name}")
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            print(f"    Failed to load images, skipping.")
            continue

        match_result = detect_and_match(img1, img2, method='sift', ratio_thresh=0.8)
        print(f"    Matches: {match_result.n_matches}")
        if match_result.n_matches < 10:
            print(f"    Too few matches, skipping.")
            continue

        result = orsa_homography(
            match_result.pts1, match_result.pts2,
            img1.shape[:2], img2.shape[:2],
            max_iter=2000, seed=42, verbose=True,
        )

        print(
            f"    log10(NFA) = {result.log_nfa:.2f}, "
            f"inliers = {result.n_inliers}/{result.n_matches}, "
            f"eps = {result.epsilon:.2f} px, "
            f"runtime = {result.runtime:.3f} s"
        )

        sorted_errors, log_nfas = _compute_nfa_curve(result, match_result, img1, img2)
        fig = plot_experiment_summary(
            img1, img2, match_result.pts1, match_result.pts2, result,
            sorted_errors=sorted_errors, log_nfas=log_nfas,
            title=f'Hard Case: {name}',
        )
        fig_path = os.path.join(output_dir, f'real_hard_{name}.png')
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"    Saved: {fig_path}")

        save_result({
            'experiment': 'real_hard',
            'name': name,
            'n_matches': result.n_matches,
            'n_inliers': result.n_inliers,
            'log_nfa': float(result.log_nfa),
            'epsilon': float(result.epsilon),
            'runtime': float(result.runtime),
            'n_iterations': result.n_iterations,
            'n_models_tested': result.n_models_tested,
            'reproj_error_mean': float(np.mean(result.reprojection_errors))
                if result.reprojection_errors is not None else None,
        }, os.path.join(output_dir, f'real_hard_{name}.json'))


# Experiment 5: Failure Case

def run_failure_case(output_dir: str, data_dir: str):
    """Cases where ORSA should NOT find a meaningful homography.

    We test shuffled real matches, multi-structure scenes, and extreme
    outlier ratios to make sure ORSA correctly says "no" when there is
    nothing to find.
    """
    print("\n=== Experiment 5: Failure Cases ===")
    imgB = _load_imgB_matches(data_dir)
    pts1_all, pts2_all = imgB['pts1'], imgB['pts2']
    shape1, shape2 = imgB['img_shape1'], imgB['img_shape2']
    ref_result = imgB['ref_result']
    ref_inlier_mask = ref_result.inlier_mask

    # 5a: Shuffled real matches (1000 points) — no geometry
    print("  5a: Shuffled real matches (1000 pts from imgB)")
    rng = np.random.default_rng(12345)
    idx = rng.choice(len(pts1_all), size=1000, replace=False)
    pts1_sub = pts1_all[idx]
    pts2_shuffled = pts2_all[idx][rng.permutation(1000)]
    result = orsa_homography(
        pts1_sub, pts2_shuffled, shape1, shape2,
        max_iter=1000, seed=42,
    )
    print(
        f"    log10(NFA) = {result.log_nfa:.2f}, "
        f"inliers = {result.n_inliers}/1000, "
        f"detection = {'YES' if result.log_nfa < 0 else 'NO (correct)'}"
    )
    save_result({
        'experiment': 'failure_shuffled', 'n_matches': 1000,
        'log_nfa': float(result.log_nfa), 'detected': result.log_nfa < 0,
    }, os.path.join(output_dir, 'failure_shuffled.json'))

    # 5b: Multi-structure — two conflicting homographies (synthetic)
    print("  5b: Multi-structure — two conflicting homographies")
    img_shape = (480, 640)
    H1 = np.array([[1.05, 0.08, 15], [-0.03, 0.98, 10], [0.0001, 0.00005, 1]])
    H2 = np.array([[0.95, -0.1, -20], [0.06, 1.05, 15], [-0.0002, 0.0001, 1]])
    pts1_a, pts2_a, _ = generate_synthetic_matches(
        n_inliers=80, n_outliers=0, H_true=H1, noise_sigma=2.0,
        img_shape=img_shape, seed=100,
    )
    pts1_b, pts2_b, _ = generate_synthetic_matches(
        n_inliers=80, n_outliers=50, H_true=H2, noise_sigma=2.0,
        img_shape=img_shape, seed=200,
    )
    pts1_mixed = np.vstack([pts1_a, pts1_b])
    pts2_mixed = np.vstack([pts2_a, pts2_b])
    result = orsa_homography(
        pts1_mixed, pts2_mixed, img_shape, img_shape,
        max_iter=1000, seed=42,
    )
    print(
        f"    log10(NFA) = {result.log_nfa:.2f}, "
        f"inliers = {result.n_inliers}/{len(pts1_mixed)}, "
        f"(ORSA detects dominant plane)"
    )
    save_result({
        'experiment': 'multi_structure',
        'n_matches': len(pts1_mixed),
        'log_nfa': float(result.log_nfa),
        'n_inliers': result.n_inliers,
    }, os.path.join(output_dir, 'multi_structure.json'))

    # 5c: Extreme outlier — 5 real inlier matches + 300 random
    print("  5c: Extreme outlier (5 real inliers + 300 random)")
    inlier_indices = np.where(ref_inlier_mask)[0]
    n_trials_failure = 20
    n_detected = 0
    log_nfas_failure = []
    for trial in range(n_trials_failure):
        rng_f = np.random.default_rng(trial * 31 + 7777)
        # Pick 5 real inlier correspondences
        chosen = rng_f.choice(inlier_indices, size=5, replace=False)
        pts1_real5 = pts1_all[chosen]
        pts2_real5 = pts2_all[chosen]
        # Generate 300 random correspondences
        h1, w1 = shape1
        h2, w2 = shape2
        noise_pts1 = np.column_stack([
            rng_f.uniform(0, w1, 300), rng_f.uniform(0, h1, 300)])
        noise_pts2 = np.column_stack([
            rng_f.uniform(0, w2, 300), rng_f.uniform(0, h2, 300)])
        pts1_f = np.vstack([pts1_real5, noise_pts1])
        pts2_f = np.vstack([pts2_real5, noise_pts2])

        result_f = orsa_homography(
            pts1_f, pts2_f, shape1, shape2,
            max_iter=1000, seed=trial,
        )
        log_nfas_failure.append(result_f.log_nfa)
        if result_f.log_nfa < 0:
            n_detected += 1
    print(
        f"    Detected: {n_detected}/{n_trials_failure}, "
        f"mean log_NFA = {np.mean(log_nfas_failure):.2f}, "
        f"min log_NFA = {np.min(log_nfas_failure):.2f}"
    )
    save_result({
        'experiment': 'failure_extreme_outliers',
        'n_inliers_true': 5,
        'n_random': 300,
        'n_trials': n_trials_failure,
        'n_detected': n_detected,
        'log_nfas': [float(x) for x in log_nfas_failure],
    }, os.path.join(output_dir, 'failure_extreme_outliers.json'))


# Experiment 6: Sensitivity Analysis

def run_sensitivity(output_dir: str, data_dir: str):
    """Check that ORSA gives consistent results across different seeds
    and that increasing max_iter improves accuracy, using real imgB matches."""
    print("\n=== Experiment 6: Sensitivity Analysis (imgB) ===")
    imgB = _load_imgB_matches(data_dir)
    pts1, pts2 = imgB['pts1'], imgB['pts2']
    shape1, shape2 = imgB['img_shape1'], imgB['img_shape2']
    # 6a: Vary max_iter on real imgB matches
    print("  6a: Varying max_iter")
    max_iters = [50, 100, 500, 1000, 5000]
    iter_results = []
    for max_it in max_iters:
        result = orsa_homography(
            pts1, pts2, shape1, shape2,
            max_iter=max_it, seed=42,
        )
        print(
            f"    max_iter={max_it}: "
            f"log_NFA={result.log_nfa:.1f}, "
            f"n_inliers={result.n_inliers}, "
            f"epsilon={result.epsilon:.1f} px, "
            f"runtime={result.runtime*1000:.0f} ms"
        )
        iter_results.append({
            'max_iter': max_it,
            'log_nfa': float(result.log_nfa),
            'n_inliers': result.n_inliers,
            'epsilon': float(result.epsilon),
            'runtime': float(result.runtime),
        })

    # 6b: same data, different seeds
    print("  6b: Seed stability (max_iter=1000)")
    seed_log_nfas = []
    seed_n_inliers = []
    for seed in range(20):
        result = orsa_homography(
            pts1, pts2, shape1, shape2,
            max_iter=1000, seed=seed,
        )
        seed_log_nfas.append(result.log_nfa)
        seed_n_inliers.append(result.n_inliers)

    print(
        f"    log_NFA: mean={np.mean(seed_log_nfas):.1f}, "
        f"std={np.std(seed_log_nfas):.1f}, "
        f"range=[{np.min(seed_log_nfas):.1f}, {np.max(seed_log_nfas):.1f}]"
    )
    print(
        f"    n_inliers: mean={np.mean(seed_n_inliers):.1f}, "
        f"std={np.std(seed_n_inliers):.1f}"
    )

    save_result({
        'experiment': 'sensitivity', 'source': 'imgB',
        'iter_results': iter_results,
        'seed_log_nfas': [float(x) for x in seed_log_nfas],
        'seed_n_inliers': [int(x) for x in seed_n_inliers],
    }, os.path.join(output_dir, 'sensitivity.json'))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    iters = [r['max_iter'] for r in iter_results]
    nfas = [r['log_nfa'] for r in iter_results]
    n_inl = [r['n_inliers'] for r in iter_results]
    ax1.plot(iters, n_inl, 'o-', color='steelblue')
    ax1.set_xlabel('max_iter', fontsize=12)
    ax1.set_ylabel('Number of inliers', fontsize=12)
    ax1.set_title('Inlier count vs. max_iter (imgB)', fontsize=13)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)

    ax2.hist(seed_log_nfas, bins=15, alpha=0.7, color='steelblue')
    ax2.set_xlabel('log$_{10}$(NFA)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('NFA Stability Across 20 Seeds (imgB)', fontsize=13)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'sensitivity.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: {os.path.join(output_dir, 'sensitivity.png')}")


# Helpers

def _find_image_pairs(data_dir: str, prefix: str) -> list[tuple[str, str, str]]:
    """Find image pairs in data_dir with naming convention: {prefix}_1.jpg, {prefix}_2.jpg
    or {prefix}_a.jpg, {prefix}_b.jpg."""
    pairs = []
    if not os.path.isdir(data_dir):
        return pairs

    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        for suffix_pair in [('_1', '_2'), ('_a', '_b')]:
            path1 = os.path.join(data_dir, f'{prefix}{suffix_pair[0]}{ext}')
            path2 = os.path.join(data_dir, f'{prefix}{suffix_pair[1]}{ext}')
            if os.path.isfile(path1) and os.path.isfile(path2):
                pairs.append((prefix, path1, path2))
    return pairs


def _generate_synthetic_image_pair(
    data_dir: str, name: str, difficulty: str = 'easy',
) -> tuple[str, str, str]:
    """Generate a pair of synthetic images with a textured pattern and a known H."""
    rng = np.random.default_rng(42)
    h, w = 480, 640

    # Create a textured image with random rectangles and circles
    img1 = np.ones((h, w, 3), dtype=np.uint8) * 200
    for _ in range(30):
        color = tuple(rng.integers(0, 255, 3).tolist())
        x1, y1 = rng.integers(0, w), rng.integers(0, h)
        x2, y2 = x1 + rng.integers(20, 100), y1 + rng.integers(20, 80)
        cv2.rectangle(img1, (x1, y1), (x2, y2), color, -1)
    for _ in range(20):
        color = tuple(rng.integers(0, 255, 3).tolist())
        cx, cy = rng.integers(30, w - 30), rng.integers(30, h - 30)
        r = rng.integers(10, 50)
        cv2.circle(img1, (cx, cy), r, color, -1)

    # Apply homography to create img2
    if difficulty == 'easy':
        H = np.array([[1.02, 0.03, 10], [-0.01, 0.99, 5], [0.00005, 0.00002, 1]])
    else:  # hard
        H = np.array([[0.85, 0.2, 40], [-0.15, 1.1, -25], [0.0005, 0.0003, 1]])

    img2 = cv2.warpPerspective(img1, H, (w, h))

    # Add noise
    noise = rng.normal(0, 5, img2.shape).astype(np.int16)
    img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    os.makedirs(data_dir, exist_ok=True)
    path1 = os.path.join(data_dir, f'{name}_1.png')
    path2 = os.path.join(data_dir, f'{name}_2.png')
    cv2.imwrite(path1, img1)
    cv2.imwrite(path2, img2)
    print(f"    Generated synthetic pair: {path1}, {path2}")
    return (name, path1, path2)


def _compute_nfa_curve(result, match_result, img1, img2):
    """Recompute NFA for all k values using the best H so we can plot the curve."""
    if result.H is None:
        return None, None

    errors, sides = symmetric_transfer_error(result.H, match_result.pts1, match_result.pts2)
    order = np.argsort(errors)
    sorted_errors = errors[order]
    sorted_sides = sides[order]

    n = len(sorted_errors)
    sample_size = 4
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    logalpha0 = np.array([
        np.log10(np.pi / (w1 * h1)),
        np.log10(np.pi / (w2 * h2)),
    ])
    log_combi_n = precompute_log_combi_n(n)
    log_combi_k = precompute_log_combi_k(sample_size, n)

    log_nfas = compute_nfa_for_all_k(
        sorted_errors, sorted_sides, logalpha0,
        n, sample_size, 1,
        log_combi_n, log_combi_k,
        mult_error=1.0,
    )
    return sorted_errors, log_nfas


# Main

def main():
    parser = argparse.ArgumentParser(description='ORSA Homography Experiments')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'null', 'outlier_injection',
                                 'real_easy', 'real_hard', 'failure',
                                 'sensitivity'],
                        help='Which experiment to run')
    parser.add_argument('--output-dir', type=str, default='experiments/results',
                        help='Output directory for results')
    parser.add_argument('--data-dir', type=str, default='experiments/data',
                        help='Directory containing test images')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    experiments = {
        'null': lambda: run_null_model(args.output_dir, args.data_dir),
        'outlier_injection': lambda: run_outlier_injection(args.output_dir, args.data_dir),
        'real_easy': lambda: run_real_easy(args.output_dir, args.data_dir),
        'real_hard': lambda: run_real_hard(args.output_dir, args.data_dir),
        'failure': lambda: run_failure_case(args.output_dir, args.data_dir),
        'sensitivity': lambda: run_sensitivity(args.output_dir, args.data_dir),
    }

    if args.experiment == 'all':
        for name, func in experiments.items():
            try:
                func()
            except Exception as e:
                print(f"\n  ERROR in {name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        experiments[args.experiment]()

    print("\nDone! Results saved to:", args.output_dir)


if __name__ == '__main__':
    main()
