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


# Experiment 1: Null Model Validation

def run_null_model(output_dir: str):
    """Verify that ORSA does NOT detect homographies in purely random matches.

    Under H0, all matches are random. The NFA should always be > 1
    (log_NFA > 0), confirming the a-contrario false alarm control.
    """
    print("\n=== Experiment 1: Null Model Validation ===")
    img_shape = (480, 640)
    H_dummy = np.eye(3)  # not used for outliers
    n_outlier_counts = [50, 100, 200, 500]
    n_trials = 50
    results = []

    for n_outliers in n_outlier_counts:
        log_nfas = []
        for trial in range(n_trials):
            pts1, pts2, _ = generate_synthetic_matches(
                n_inliers=0, n_outliers=n_outliers,
                H_true=H_dummy, seed=trial * 1000 + n_outliers,
                img_shape=img_shape,
            )
            result = orsa_homography(
                pts1, pts2, img_shape, img_shape,
                max_iter=500, seed=trial,
            )
            log_nfas.append(result.log_nfa)

        log_nfas = np.array(log_nfas)
        n_false_alarms = np.sum(log_nfas < 0)
        print(
            f"  n_outliers={n_outliers}: "
            f"false alarms = {n_false_alarms}/{n_trials}, "
            f"mean log_NFA = {np.mean(log_nfas):.2f}, "
            f"min log_NFA = {np.min(log_nfas):.2f}"
        )
        results.append({
            'n_outliers': n_outliers,
            'n_trials': n_trials,
            'false_alarms': int(n_false_alarms),
            'log_nfas': log_nfas.tolist(),
            'mean_log_nfa': float(np.mean(log_nfas)),
            'min_log_nfa': float(np.min(log_nfas)),
        })

    save_result({'experiment': 'null_model', 'results': results},
                os.path.join(output_dir, 'null_model.json'))

    # Plot distribution of log_NFA
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for r in results:
        ax.hist(r['log_nfas'], bins=30, alpha=0.6,
                label=f"n={r['n_outliers']}")
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='NFA = 1')
    ax.set_xlabel('log$_{10}$(NFA)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Null Model: NFA Distribution (should be > 0)', fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'null_model.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: {os.path.join(output_dir, 'null_model.png')}")


# Experiment 2: Simple Synthetic Case

def run_simple_synthetic(output_dir: str):
    """Synthetic homography with varying outlier ratios.

    Known H, moderate noise. Measure precision, recall, and H error.
    """
    print("\n=== Experiment 2: Simple Synthetic Case ===")
    img_shape = (480, 640)
    homographies = make_test_homographies(img_shape)
    H_true = homographies['perspective_mild']
    n_total = 200
    noise_sigma = 1.0
    outlier_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_trials = 10

    results = []
    for ratio in outlier_ratios:
        n_outliers = int(n_total * ratio)
        n_inliers = n_total - n_outliers
        trial_results = []

        for trial in range(n_trials):
            pts1, pts2, gt_mask = generate_synthetic_matches(
                n_inliers=n_inliers, n_outliers=n_outliers,
                H_true=H_true, noise_sigma=noise_sigma,
                img_shape=img_shape, seed=trial * 100 + int(ratio * 100),
            )
            result = orsa_homography(
                pts1, pts2, img_shape, img_shape,
                max_iter=1000, seed=trial,
            )

            # Compute precision / recall
            if result.n_inliers > 0:
                tp = np.sum(result.inlier_mask & gt_mask)
                fp = np.sum(result.inlier_mask & ~gt_mask)
                fn = np.sum(~result.inlier_mask & gt_mask)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            else:
                precision = 0
                recall = 0

            # Homography error
            h_error = {}
            if result.H is not None:
                h_error = evaluate_homography(result.H, H_true, img_shape)

            trial_results.append({
                'precision': float(precision),
                'recall': float(recall),
                'log_nfa': float(result.log_nfa),
                'n_inliers': result.n_inliers,
                'epsilon': float(result.epsilon),
                'runtime': float(result.runtime),
                'h_error': h_error,
                'detected': result.log_nfa < 0,
            })

        precisions = [t['precision'] for t in trial_results]
        recalls = [t['recall'] for t in trial_results]
        detected = sum(1 for t in trial_results if t['detected'])
        print(
            f"  outlier_ratio={ratio:.0%}: "
            f"detected={detected}/{n_trials}, "
            f"precision={np.mean(precisions):.3f} +/- {np.std(precisions):.3f}, "
            f"recall={np.mean(recalls):.3f} +/- {np.std(recalls):.3f}"
        )
        results.append({
            'outlier_ratio': ratio,
            'n_inliers_true': n_inliers,
            'n_outliers': n_outliers,
            'trials': trial_results,
        })

    save_result({'experiment': 'simple_synthetic', 'H_true': H_true.tolist(),
                 'noise_sigma': noise_sigma, 'results': results},
                os.path.join(output_dir, 'simple_synthetic.json'))

    # Plot precision/recall vs outlier ratio
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ratios = [r['outlier_ratio'] for r in results]
    mean_prec = [np.mean([t['precision'] for t in r['trials']]) for r in results]
    mean_rec = [np.mean([t['recall'] for t in r['trials']]) for r in results]
    std_prec = [np.std([t['precision'] for t in r['trials']]) for r in results]
    std_rec = [np.std([t['recall'] for t in r['trials']]) for r in results]

    ax1.errorbar(ratios, mean_prec, yerr=std_prec, marker='o', capsize=5, label='Precision')
    ax1.errorbar(ratios, mean_rec, yerr=std_rec, marker='s', capsize=5, label='Recall')
    ax1.set_xlabel('Outlier ratio', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Precision / Recall vs. Outlier Ratio', fontsize=13)
    ax1.legend()
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # Corner error
    mean_corner_err = []
    for r in results:
        errs = [t['h_error'].get('corner_error_mean', np.nan) for t in r['trials'] if t['detected']]
        mean_corner_err.append(np.mean(errs) if errs else np.nan)
    ax2.plot(ratios, mean_corner_err, 'o-', color='purple')
    ax2.set_xlabel('Outlier ratio', fontsize=12)
    ax2.set_ylabel('Mean corner error (px)', fontsize=12)
    ax2.set_title('Homography Accuracy vs. Outlier Ratio', fontsize=13)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'simple_synthetic.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: {os.path.join(output_dir, 'simple_synthetic.png')}")


# Experiment 3: Real Images — Easy Case

def run_real_easy(output_dir: str, data_dir: str):
    """Real image pair where a homography is clearly valid."""
    print("\n=== Experiment 3: Real Images — Easy Case ===")

    # Look for image pairs in data directory
    pairs = _find_image_pairs(data_dir, prefix='easy')
    if not pairs:
        print("  No easy image pairs found. Generating synthetic images instead.")
        pairs = [_generate_synthetic_image_pair(
            data_dir, 'easy', difficulty='easy')]

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

    pairs = _find_image_pairs(data_dir, prefix='hard')
    if not pairs:
        print("  No hard image pairs found. Generating synthetic images instead.")
        pairs = [_generate_synthetic_image_pair(
            data_dir, 'hard', difficulty='hard')]

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
    """Cases where homography is NOT valid: non-planar, repeated textures, etc.

    ORSA should report NFA > 1 or find only a partial model.
    """
    print("\n=== Experiment 5: Failure Case ===")

    # Sub-experiment 5a: purely random matches (no real geometric relation)
    print("  5a: Random matches (no geometry)")
    img_shape = (480, 640)
    pts1, pts2, _ = generate_synthetic_matches(
        n_inliers=0, n_outliers=300, H_true=np.eye(3),
        img_shape=img_shape, seed=12345,
    )
    result = orsa_homography(
        pts1, pts2, img_shape, img_shape,
        max_iter=1000, seed=42,
    )
    print(
        f"    log10(NFA) = {result.log_nfa:.2f}, "
        f"inliers = {result.n_inliers}/300, "
        f"detection = {'YES' if result.log_nfa < 0 else 'NO (correct)'}"
    )
    save_result({
        'experiment': 'failure_random', 'n_matches': 300,
        'log_nfa': float(result.log_nfa), 'detected': result.log_nfa < 0,
    }, os.path.join(output_dir, 'failure_random.json'))

    # Sub-experiment 5b: two conflicting homographies (multi-structure)
    # Note: ORSA correctly detects the dominant plane — this is a *success*,
    # not a failure. We keep it to demonstrate robustness to mixed models.
    print("  5b: Multi-structure — two conflicting homographies")
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

    # Sub-experiment 5c: extreme outlier ratio (true failure case)
    # Very few inliers swamped by outliers — detection should fail.
    print("  5c: Extreme outlier ratio (5 inliers / 300 outliers)")
    n_trials_failure = 20
    n_detected = 0
    log_nfas_failure = []
    H_fail = make_test_homographies(img_shape)['perspective_mild']
    for trial in range(n_trials_failure):
        pts1_f, pts2_f, gt_f = generate_synthetic_matches(
            n_inliers=5, n_outliers=300,
            H_true=H_fail, noise_sigma=2.0,
            img_shape=img_shape, seed=trial * 31 + 7777,
        )
        result_f = orsa_homography(
            pts1_f, pts2_f, img_shape, img_shape,
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
        'n_outliers': 300,
        'n_trials': n_trials_failure,
        'n_detected': n_detected,
        'log_nfas': [float(x) for x in log_nfas_failure],
    }, os.path.join(output_dir, 'failure_extreme_outliers.json'))

    # Real failure images if available
    pairs = _find_image_pairs(data_dir, prefix='failure')
    if not pairs:
        print("  No failure image pairs found in data directory.")
    for name, img1_path, img2_path in pairs:
        print(f"  Processing: {name}")
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            continue
        match_result = detect_and_match(img1, img2, method='sift')
        if match_result.n_matches < 10:
            continue
        result = orsa_homography(
            match_result.pts1, match_result.pts2,
            img1.shape[:2], img2.shape[:2],
            max_iter=1000, seed=42,
        )
        print(
            f"    log10(NFA) = {result.log_nfa:.2f}, "
            f"inliers = {result.n_inliers}/{result.n_matches}"
        )


# Experiment 6: Sensitivity Analysis

def run_sensitivity(output_dir: str):
    """Vary ORSA parameters and random seeds to assess stability."""
    print("\n=== Experiment 6: Sensitivity Analysis ===")
    img_shape = (480, 640)
    H_true = make_test_homographies(img_shape)['perspective_mild']
    n_total = 200
    outlier_ratio = 0.5
    n_inliers = int(n_total * (1 - outlier_ratio))
    n_outliers = n_total - n_inliers

    # 6a: Vary max_iter
    print("  6a: Varying max_iter")
    max_iters = [50, 100, 500, 1000, 5000]
    iter_results = []
    for max_it in max_iters:
        log_nfas = []
        corner_errors = []
        runtimes = []
        for trial in range(10):
            pts1, pts2, gt_mask = generate_synthetic_matches(
                n_inliers=n_inliers, n_outliers=n_outliers,
                H_true=H_true, noise_sigma=1.0,
                img_shape=img_shape, seed=trial * 77,
            )
            result = orsa_homography(
                pts1, pts2, img_shape, img_shape,
                max_iter=max_it, seed=trial,
            )
            log_nfas.append(result.log_nfa)
            runtimes.append(result.runtime)
            if result.H is not None:
                h_err = evaluate_homography(result.H, H_true, img_shape)
                corner_errors.append(h_err['corner_error_mean'])
            else:
                corner_errors.append(np.nan)

        print(
            f"    max_iter={max_it}: "
            f"mean log_NFA={np.mean(log_nfas):.1f}, "
            f"mean corner_err={np.nanmean(corner_errors):.2f} px, "
            f"mean runtime={np.mean(runtimes):.3f} s"
        )
        iter_results.append({
            'max_iter': max_it,
            'log_nfas': [float(x) for x in log_nfas],
            'corner_errors': [float(x) for x in corner_errors],
            'runtimes': [float(x) for x in runtimes],
        })

    # 6b: Vary random seed (stability)
    print("  6b: Seed stability (max_iter=1000)")
    seed_log_nfas = []
    seed_n_inliers = []
    for seed in range(20):
        pts1, pts2, gt_mask = generate_synthetic_matches(
            n_inliers=n_inliers, n_outliers=n_outliers,
            H_true=H_true, noise_sigma=1.0,
            img_shape=img_shape, seed=42,  
        )
        result = orsa_homography(
            pts1, pts2, img_shape, img_shape,
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
        'experiment': 'sensitivity',
        'iter_results': iter_results,
        'seed_log_nfas': [float(x) for x in seed_log_nfas],
        'seed_n_inliers': [int(x) for x in seed_n_inliers],
    }, os.path.join(output_dir, 'sensitivity.json'))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    bp_data = [r['corner_errors'] for r in iter_results]
    bp_labels = [str(r['max_iter']) for r in iter_results]
    ax1.boxplot(bp_data, labels=bp_labels)
    ax1.set_xlabel('max_iter', fontsize=12)
    ax1.set_ylabel('Corner error (px)', fontsize=12)
    ax1.set_title('Accuracy vs. max_iter', fontsize=13)

    ax2.hist(seed_log_nfas, bins=15, alpha=0.7, color='steelblue')
    ax2.set_xlabel('log$_{10}$(NFA)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('NFA Stability Across Seeds (same data)', fontsize=13)

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
    """Compute the full NFA curve for plotting."""
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
                        choices=['all', 'null', 'synthetic', 'real_easy',
                                 'real_hard', 'failure', 'sensitivity'],
                        help='Which experiment to run')
    parser.add_argument('--output-dir', type=str, default='experiments/results',
                        help='Output directory for results')
    parser.add_argument('--data-dir', type=str, default='experiments/data',
                        help='Directory containing test images')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    experiments = {
        'null': lambda: run_null_model(args.output_dir),
        'synthetic': lambda: run_simple_synthetic(args.output_dir),
        'real_easy': lambda: run_real_easy(args.output_dir, args.data_dir),
        'real_hard': lambda: run_real_hard(args.output_dir, args.data_dir),
        'failure': lambda: run_failure_case(args.output_dir, args.data_dir),
        'sensitivity': lambda: run_sensitivity(args.output_dir),
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
