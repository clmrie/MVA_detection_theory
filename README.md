# ORSA Homography Registration

A Python implementation of the **ORSA** (Optimized Random Sampling) algorithm for automatic homographic registration of image pairs, based on the **a-contrario** framework of Moisan, Moulon, and Monasse (IPOL, 2012).

## The Problem

Registering two images of a planar scene requires estimating a projective homography from point correspondences. Feature matchers like SIFT produce many outliers that corrupt direct estimation. RANSAC handles outliers but requires a manually tuned inlier threshold. The **a-contrario approach** removes this parameter: it selects the threshold that minimizes the **Number of False Alarms (NFA)**, declaring a detection meaningful only when NFA < 1 -- meaning the observed geometric agreement would occur less than once in purely random data.

## The Algorithm

1. **Random sampling** -- Draw 4-point samples, estimate homographies via DLT with Hartley normalization
2. **Adaptive threshold** -- For each candidate, test all inlier counts *k* and minimize NFA(*k*, *epsilon*)
3. **NFA criterion** -- `log10 NFA = log10 C(n,k) + log10 C(k,4) + (k-4) * log10(pi * epsilon^2 / (w*h))`, computed in log-space with precomputed binomial tables
4. **Degeneracy checks** -- Collinearity, conditioning, orientation preservation, valid warp verification
5. **Adaptive stopping** -- Iteration count adjusts based on estimated inlier ratio
6. **Refinement** -- Iterative DLT refit on inliers, then Levenberg-Marquardt polishing

## Results

All experiments use real SIFT matches from real image pairs.

### Real image pairs

| Case | Matches | Inliers | log10 NFA | Threshold |
|---|---|---|---|---|
| Easy (imgA) | 1254 | 564 (45%) | -1645 | 30.3 px |
| Hard (imgB) | 1212 | 1023 (84%) | -3991 | 4.1 px |

![Easy case](experiments/results/real_easy_imgA.png)
![Hard case](experiments/results/real_hard_imgB.png)

### Null model validation

ORSA on shuffled real keypoints (all correspondences broken): **zero false alarms across 250 trials**. Smallest observed log10 NFA is +3.7, three orders of magnitude above the detection threshold.

![Null model](experiments/results/null_model.png)

### Robustness to outlier injection

Random false correspondences injected into 1106 real matches:

| Injected noise | 0% | 18% | 33% | 50% | 71% | 83% |
|---|---|---|---|---|---|---|
| Detection rate | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 9/10 |
| Precision | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 |
| Recall | 0.99 | 1.00 | 0.98 | 0.98 | 0.98 | 0.86 |

![Outlier injection](experiments/results/outlier_injection.png)

### Sensitivity

Stable across iteration budgets (50-5000 yield identical results) and random seeds (inlier count std = 11 over 20 trials).

![Sensitivity](experiments/results/sensitivity.png)

## Project Structure

```
src/
  orsa.py            # Main ORSA algorithm
  nfa.py             # A-contrario NFA computation
  homography.py      # DLT, symmetric transfer error, LM refinement
  degeneracy.py      # Geometric validation checks
  matching.py        # SIFT/ORB feature matching
  visualization.py   # Plotting utilities
experiments/
  run_experiments.py  # Experiment runner
  synthetic.py        # Synthetic data generation
  data/               # Test images
  results/            # Output JSON and figures
tests/                # Unit tests
report/               # LaTeX report
```

## Experiments

```bash
pip install -r requirements.txt

# Run all experiments
python -m experiments.run_experiments

# Run a specific experiment
python -m experiments.run_experiments --experiment outlier_injection
```

## Tests

```bash
python -m pytest tests/ -v
```

## References

- Moisan, Moulon, Monasse. *Automatic Homographic Registration of a Pair of Images, with A Contrario Elimination of Outliers.* IPOL, 2012.
- Desolneux, Moisan, Morel. *From Gestalt Theory to Image Analysis: A Probabilistic Approach.* Springer, 2008.
- Fischler, Bolles. *Random Sample Consensus.* Communications of the ACM, 1981.
