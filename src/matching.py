# src/matching.py
"""
Feature detection and matching using OpenCV.

Supports SIFT and ORB detectors with Lowe's ratio test and
optional mutual consistency check.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MatchResult:
    """Structured result from feature matching."""
    pts1: np.ndarray       # (n, 2) keypoint coordinates in image 1
    pts2: np.ndarray       # (n, 2) keypoint coordinates in image 2
    n_matches: int         # number of matches
    n_keypoints1: int      # total keypoints detected in image 1
    n_keypoints2: int      # total keypoints detected in image 2
    method: str            # 'sift' or 'orb'


def detect_and_match(
    img1: np.ndarray,
    img2: np.ndarray,
    method: str = 'sift',
    ratio_thresh: float = 0.75,
    mutual: bool = False,
    max_features: int = 5000,
) -> MatchResult:
    """Detect features and match between two images.

    Parameters
    ----------
    img1, img2 : input images (BGR or grayscale)
    method : 'sift' or 'orb'
    ratio_thresh : Lowe's ratio test threshold
    mutual : if True, also apply cross-check (reduces matches but increases quality)
    max_features : maximum features to detect (used by ORB; SIFT uses nfeatures=0)

    Returns
    -------
    MatchResult with matched point coordinates.
    """
    # Convert to grayscale if needed
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

    # Create detector
    if method.lower() == 'sift':
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2
    elif method.lower() == 'orb':
        detector = cv2.ORB_create(nfeatures=max_features)
        norm_type = cv2.NORM_HAMMING
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sift' or 'orb'.")

    # Detect and compute descriptors
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)

    if desc1 is None or desc2 is None or len(kp1) < 4 or len(kp2) < 4:
        return MatchResult(
            pts1=np.zeros((0, 2)),
            pts2=np.zeros((0, 2)),
            n_matches=0,
            n_keypoints1=len(kp1) if kp1 else 0,
            n_keypoints2=len(kp2) if kp2 else 0,
            method=method,
        )

    # Match descriptors with ratio test
    if mutual:
        # Cross-check matching (no ratio test needed)
        bf = cv2.BFMatcher(norm_type, crossCheck=True)
        raw_matches = bf.match(desc1, desc2)
        good_matches = raw_matches
    else:
        # KNN matching with Lowe's ratio test
        bf = cv2.BFMatcher(norm_type)
        raw_matches = bf.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for m_pair in raw_matches:
            if len(m_pair) == 2:
                m, nn = m_pair
                if m.distance < ratio_thresh * nn.distance:
                    good_matches.append(m)

    if len(good_matches) == 0:
        return MatchResult(
            pts1=np.zeros((0, 2)),
            pts2=np.zeros((0, 2)),
            n_matches=0,
            n_keypoints1=len(kp1),
            n_keypoints2=len(kp2),
            method=method,
        )

    # Remove exact duplicate correspondences (same pair of endpoints).
    # The IPOL paper warns only about exact duplicates, not shared single
    # endpoints, which occur legitimately in repetitive textures.
    seen_pairs = set()
    unique_matches = []
    # Sort by distance (best first)
    good_matches = sorted(good_matches, key=lambda m: m.distance)
    for m in good_matches:
        pair = (m.queryIdx, m.trainIdx)
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            unique_matches.append(m)

    # Extract point coordinates
    pts1 = np.array([kp1[m.queryIdx].pt for m in unique_matches], dtype=np.float64)
    pts2 = np.array([kp2[m.trainIdx].pt for m in unique_matches], dtype=np.float64)

    return MatchResult(
        pts1=pts1,
        pts2=pts2,
        n_matches=len(unique_matches),
        n_keypoints1=len(kp1),
        n_keypoints2=len(kp2),
        method=method,
    )
