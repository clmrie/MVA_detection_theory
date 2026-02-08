# ORSA Homography Registration

A Python implementation of the **ORSA** (Optimized Random SAmpling) algorithm for robust homography estimation between image pairs, based on the a-contrario framework from:

> L. Moisan, P. Moulon, and P. Monasse, "Automatic Homographic Registration of a Pair of Images, with A Contrario Elimination of Outliers," *Image Processing On Line (IPOL)*, 2012.

Unlike traditional RANSAC which requires a manually-set inlier threshold, ORSA adaptively selects the threshold that minimizes the **Number of False Alarms (NFA)**. A detection is considered meaningful when NFA < 1, providing automatic false alarm control.

## Project Structure

```
├── src/
│   ├── orsa.py            # Main ORSA algorithm (adaptive RANSAC loop)
│   ├── nfa.py             # A-contrario NFA computation in log-space
│   ├── homography.py      # DLT estimation, symmetric transfer error, LM refinement
│   ├── degeneracy.py      # Geometric validation (collinearity, conditioning, warp checks)
│   ├── matching.py        # Feature detection and matching (SIFT, ORB)
│   └── visualization.py   # Plotting utilities (matches, NFA curves, error histograms)
│
├── experiments/
│   ├── run_experiments.py # Experiment runner (6 experiment types)
│   ├── synthetic.py       # Synthetic data generation with ground truth
│   ├── data/              # Test images
│   └── results/           # Output JSON and figures
│
├── tests/                 # 46 unit tests covering all modules
│   ├── test_orsa.py
│   ├── test_nfa.py
│   ├── test_homography.py
│   └── test_degeneracy.py
│
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python >= 3.10.

## Usage

```python
import cv2
from src.orsa import orsa_homography
from src.matching import detect_and_match

img1 = cv2.imread("image1.jpg")
img2 = cv2.imread("image2.jpg")

# Detect and match features
match_result = detect_and_match(img1, img2, method="sift", ratio_thresh=0.75)

# Run ORSA
result = orsa_homography(
    match_result.pts1,
    match_result.pts2,
    img1.shape[:2],
    img2.shape[:2],
    max_iter=1000,
    seed=42,
)

if result.log_nfa < 0:
    print(f"Homography detected (NFA = 10^{result.log_nfa:.2f})")
    print(f"Inliers: {result.n_inliers}/{result.n_matches}")
    print(f"Homography:\n{result.H}")
else:
    print("No meaningful homography found")
```

## Experiments

Six experiments are available:

| Experiment | Description |
|---|---|
| `null_model` | Validates rejection of random matches (NFA > 1) |
| `synthetic` | Known homographies with varying outlier ratios |
| `real_easy` | Easy image pairs with clear homographies |
| `real_hard` | Challenging cases with large viewpoint changes |
| `failure` | Cases where no valid homography exists |
| `sensitivity` | Parameter sensitivity analysis |

```bash
# Run all experiments
python -m experiments.run_experiments

# Run a specific experiment
python -m experiments.run_experiments --experiment synthetic

# Custom output directory
python -m experiments.run_experiments --output-dir my_results/
```

Results are saved as JSON files and PNG figures in `experiments/results/`.

## Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific module
python -m pytest tests/test_orsa.py -v
```

## Algorithm Overview

**ORSA** combines RANSAC sampling with a-contrario validation:

1. **Random sampling**: Draw minimal 4-point samples and estimate homographies via DLT with Hartley normalization.
2. **Adaptive threshold**: For each candidate, test all possible inlier counts *k* and select the threshold that minimizes NFA(*k*, *epsilon*).
3. **NFA criterion**: NFA = (*n*-4) x C(*n*, *k*) x C(*k*, 4) x *alpha*^(*k*-4), where *alpha* = *pi* x *epsilon*^2 / (*w* x *h*) is the probability of a random inlier. Computed in log-space for numerical stability.
4. **Adaptive iterations**: Iteration count adjusts based on the current estimated inlier ratio.
5. **Refinement**: Best model is iteratively refit on its inlier set, then polished with Levenberg-Marquardt optimization.
6. **Degeneracy checks**: Collinearity detection, orientation preservation, conditioning, and valid warp verification prevent degenerate solutions.
