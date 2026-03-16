"""
Patch-Codebook Utilities for TimeMAE-inspired E-step Refinement.

Provides codebook learning from observed sub-series patches and
post-MMPS refinement that quantizes imputed patches toward learned prototypes.

Reference:
    TimeMAE (Cheng et al., WSDM 2026) — Masked Codeword Classification (Sec 3.4.1):
    learns a codebook C = {c_1, ..., c_K} and assigns each sub-series to its
    nearest codeword via argmax similarity.
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def build_patch_codebook(corrupted_data, obs_masks, patch_size=4, n_codes=256, seed=None):
    """
    Learn a codebook of representative sub-series patches from observed data.

    Extracts all fully-observed, non-overlapping patches of length `patch_size`,
    flattens each to a vector, and fits MiniBatchKMeans to produce K centroids.

    Args:
        corrupted_data: (N, T, C) numpy array with NaN for missing values
        obs_masks:      (N, T) boolean array, True = observed
        patch_size:     length of each sub-series patch (TimeMAE's window sigma)
        n_codes:        number of codebook entries K (TimeMAE's vocabulary size)
        seed:           random seed for reproducibility

    Returns:
        codebook: fitted MiniBatchKMeans object
                  (codebook.cluster_centers_ has shape (n_codes, patch_size * C))
    """
    N, T, C = corrupted_data.shape
    observed_patches = []

    for i in range(N):
        for t_start in range(0, T - patch_size + 1, patch_size):
            t_end = t_start + patch_size
            if obs_masks[i, t_start:t_end].all():
                patch = corrupted_data[i, t_start:t_end, :]
                if not np.isnan(patch).any():
                    observed_patches.append(patch.flatten())

    if len(observed_patches) == 0:
        print("WARNING: No fully-observed patches found. Codebook refinement will be skipped.")
        return None

    observed_patches = np.array(observed_patches)
    effective_n_codes = min(n_codes, len(observed_patches))

    if effective_n_codes < n_codes:
        print(f"  Reducing codebook size from {n_codes} to {effective_n_codes} "
              f"(only {len(observed_patches)} observed patches available)")

    kmeans = MiniBatchKMeans(
        n_clusters=effective_n_codes,
        batch_size=min(1024, len(observed_patches)),
        random_state=seed,
        n_init=3,
    )
    kmeans.fit(observed_patches)

    print(f"Codebook built: {effective_n_codes} codes from {len(observed_patches)} "
          f"observed patches (patch_size={patch_size}, dim={patch_size * C})")

    return kmeans


def codebook_refine_reconstructions(reconstructions, obs_masks, codebook,
                                     patch_size=4, blend_strength=0.3):
    """
    Refine MMPS-imputed sequences by blending imputed patches toward their
    nearest codebook entry.

    For each non-overlapping patch that contains at least one originally-missing
    position, the patch is soft-blended with the nearest codebook centroid.
    The blend weight is adaptive: fully-missing patches receive maximum codebook
    influence, while mostly-observed patches receive minimal correction.

    Args:
        reconstructions: (N, T, C) numpy array of MMPS-imputed sequences (no NaN)
        obs_masks:       (N, T) boolean array, True = originally observed
        codebook:        fitted MiniBatchKMeans from build_patch_codebook, or None
        patch_size:      sub-series patch length (must match codebook's patch_size)
        blend_strength:  base blending coefficient; actual alpha per patch is
                         blend_strength * (1 - obs_ratio)

    Returns:
        refined: (N, T, C) numpy array, clipped to [0, 1]
    """
    if codebook is None:
        return reconstructions

    N, T, C = reconstructions.shape
    refined = reconstructions.copy()
    n_refined = 0

    for i in range(N):
        for t_start in range(0, T - patch_size + 1, patch_size):
            t_end = t_start + patch_size
            patch_mask = obs_masks[i, t_start:t_end]

            if patch_mask.all():
                continue

            obs_ratio = patch_mask.mean()
            alpha = blend_strength * (1.0 - obs_ratio)

            if alpha < 1e-6:
                continue

            patch_flat = reconstructions[i, t_start:t_end, :].flatten().reshape(1, -1)
            nearest_idx = codebook.predict(patch_flat)[0]
            nearest_code = codebook.cluster_centers_[nearest_idx].reshape(patch_size, C)

            refined[i, t_start:t_end, :] = (
                (1.0 - alpha) * reconstructions[i, t_start:t_end, :] + alpha * nearest_code
            )
            n_refined += 1

    # Handle tail patch if T is not divisible by patch_size
    tail_len = T % patch_size
    if tail_len > 0:
        for i in range(N):
            t_start = T - tail_len
            patch_mask = obs_masks[i, t_start:T]
            if patch_mask.all():
                continue
            obs_ratio = patch_mask.mean()
            alpha = blend_strength * (1.0 - obs_ratio)
            if alpha < 1e-6:
                continue
            # Pad tail to patch_size, find nearest, then take only tail_len
            tail_data = reconstructions[i, t_start:T, :]
            padded = np.zeros((patch_size, C))
            padded[:tail_len, :] = tail_data
            nearest_idx = codebook.predict(padded.flatten().reshape(1, -1))[0]
            nearest_code = codebook.cluster_centers_[nearest_idx].reshape(patch_size, C)
            refined[i, t_start:T, :] = (
                (1.0 - alpha) * tail_data + alpha * nearest_code[:tail_len, :]
            )
            n_refined += 1

    refined = np.clip(refined, 0, 1)
    print(f"  Codebook refinement: {n_refined} patches blended (blend_strength={blend_strength:.3f})")

    return refined
