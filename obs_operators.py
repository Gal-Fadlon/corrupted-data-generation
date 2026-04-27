"""
Observation operators for the Co-Evolving EM pipeline.

The paper's theory (Theorem 1, Corollary 1) is stated for any linear observation
operator A_ts : R^{T x F} -> R^{N_obs x F}.  The original script hard-codes
the binary-mask (missing-data) case in the shape of a mask tensor broadcast
across the time axis.  This module introduces a minimal abstraction so that a
new corruption type (temporal block-mean aggregation) can be plugged into the
same E-step with zero changes to the proven missing-data path:

  ObservationOperator         : base class (apply, adjoint, enforce, warm_start)
  PointMaskOperator           : A_ts is a binary element-wise mask (missing data)
  BlockMeanOperator           : A_ts is block-mean averaging over w positions
                                (temporal aggregation / TS "super-resolution")
  ContinuousResampleOperator  : A_ts is per-sample linear interpolation at
                                continuous-time sample points t_k in [0, T-1]
                                (time-continuous irregular sampling / "soft
                                missingness" — reduces to PointMask when t_k
                                is integer-valued).

For all operators:
  apply(x_ts)   : [B, T, F]     -> [B, N_obs, F]        (forward observation)
  adjoint(y)    : [B, N_obs, F] -> [B, T, F]            (A_ts^T)
  enforce(x_ts, y)
                : [B, T, F]     -> [B, T, F]            (Moore-Penrose update
                                                         so A_ts(out) = y)
  warm_start(y_np, T) : numpy   -> numpy[N, T, F]       (TS-space seed for
                                                         bootstrap M-step)

PointMaskOperator preserves the original [B, T, F] layout for y (with zeros at
unobserved positions) so the existing DualSpaceMMPS obs-space CG path is
numerically byte-identical when wrapped through it.  BlockMeanOperator uses the
genuinely smaller [B, T/w, F] layout for y, which is exactly the
observation space R^{N_obs x F} used in Prop 4.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


# =============================================================================
# Base class
# =============================================================================


class ObservationOperator:
    """Linear observation operator A_ts : R^{T x F} -> R^{N_obs x F}.

    Subclasses must implement `apply`, `adjoint`, `enforce`, `warm_start`,
    and set `name` / `n_obs` / `supports_curriculum`.
    """

    name: str = "base"
    supports_curriculum: bool = False

    def __init__(self, seq_len: int, n_features: int):
        self.seq_len = int(seq_len)
        self.n_features = int(n_features)

    @property
    def n_obs(self) -> int:
        raise NotImplementedError

    def apply(self, x_ts: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def adjoint(self, y_obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def enforce(self, x_ts: torch.Tensor, y_obs: torch.Tensor) -> torch.Tensor:
        """Moore-Penrose-style residual update so that A_ts(output) = y_obs.

        Equivalent to x_ts + A_ts^+ (y_obs - A_ts(x_ts)) where A_ts^+ is the
        Moore-Penrose pseudo-inverse.  For PointMask this is hard enforcement
        at observed positions; for BlockMean this spreads the per-window
        residual uniformly across the w positions.
        """
        raise NotImplementedError

    def warm_start(self, y_obs: np.ndarray, seed: int = 0) -> np.ndarray:
        """Produce a [N, T, F] TS-space seed from y_obs.  Used for the
        bootstrap M-step and as the starting point for EM.
        """
        raise NotImplementedError


# =============================================================================
# PointMask: original missing-data corruption (binary mask on time axis)
# =============================================================================


class PointMaskOperator(ObservationOperator):
    """A_ts(x)[t, c] = mask[t] * x[t, c].

    The y layout is [B, T, F] with zeros at unobserved positions (matching the
    original `torch.nan_to_num(corrupted, nan=0.0)` convention), so CG tensors
    stay in [B, T, F] and the DualSpaceMMPS obs-space CG path is numerically
    unchanged.  n_obs == T but the mask zeros out the unobserved entries so
    the effective rank equals the number of observed positions per sample.
    """

    name = "point_mask"
    supports_curriculum = True

    def __init__(self, seq_len: int, n_features: int):
        super().__init__(seq_len, n_features)
        self._mask: Optional[torch.Tensor] = None  # [B, T, F] float mask

    @property
    def n_obs(self) -> int:
        return self.seq_len

    def set_mask(self, mask_tf: torch.Tensor) -> None:
        """Per-batch mask of shape [B, T, F] (float 0/1)."""
        self._mask = mask_tf

    def apply(self, x_ts: torch.Tensor) -> torch.Tensor:
        assert self._mask is not None, "PointMaskOperator.set_mask not called"
        return self._mask * x_ts

    def adjoint(self, y_obs: torch.Tensor) -> torch.Tensor:
        # For the binary-mask operator, A^T = A when acting on tensors laid out
        # in [B, T, F].  The mask gate is idempotent, so applying it on input
        # is the MMPS-correct adjoint.
        assert self._mask is not None, "PointMaskOperator.set_mask not called"
        return self._mask * y_obs

    def enforce(self, x_ts: torch.Tensor, y_obs: torch.Tensor) -> torch.Tensor:
        assert self._mask is not None, "PointMaskOperator.set_mask not called"
        m = self._mask
        return x_ts * (1.0 - m) + y_obs * m

    def warm_start(self, y_obs: np.ndarray, seed: int = 0) -> np.ndarray:
        raise NotImplementedError(
            "PointMaskOperator.warm_start is not used: the missing-data path "
            "uses the existing STL/Kalman/linear/random initializers in "
            "run_co_evolving_em_aggregate.main.  This method is kept on the "
            "interface for BlockMeanOperator."
        )


# =============================================================================
# BlockMean: temporal aggregation / time-series super-resolution
# =============================================================================


class BlockMeanOperator(ObservationOperator):
    """A_ts(x)[k, c] = (1/w) * sum_{j<w} x[w*k + j, c].

    The output y has shape [B, T/w, F] (we require T % w == 0 for clean block
    structure; the loader is responsible for truncating T to a multiple of w
    if needed).  The adjoint replicates each y_k across its w positions:
        (A^T y)[t, c] = y[t // w, c].
    And the Moore-Penrose update spreads the per-window residual r_k =
    y_k - mean(x_hat[wk:wk+w]) uniformly back across the w positions so that
    A_ts(output) = y exactly while preserving within-window variation (which
    lives in the null space of A_ts).
    """

    name = "block_mean"
    supports_curriculum = False

    def __init__(self, seq_len: int, n_features: int, window: int):
        super().__init__(seq_len, n_features)
        if window < 1:
            raise ValueError(f"BlockMeanOperator: window must be >= 1, got {window}")
        if seq_len % window != 0:
            raise ValueError(
                f"BlockMeanOperator: seq_len ({seq_len}) must be divisible by "
                f"window ({window}).  Pre-truncate the data before building "
                f"this operator."
            )
        self.window = int(window)
        self._n_obs = seq_len // window

    @property
    def n_obs(self) -> int:
        return self._n_obs

    # --- helpers ---

    @staticmethod
    def _block_mean(x_ts: torch.Tensor, window: int) -> torch.Tensor:
        """[B, T, F] -> [B, T/w, F] via non-overlapping block mean."""
        B, T, F = x_ts.shape
        assert T % window == 0
        return x_ts.reshape(B, T // window, window, F).mean(dim=2)

    @staticmethod
    def _block_replicate(y_obs: torch.Tensor, window: int) -> torch.Tensor:
        """[B, N, F] -> [B, N*w, F] via per-window replication (no scaling)."""
        return y_obs.repeat_interleave(window, dim=1)

    # --- interface ---

    def apply(self, x_ts: torch.Tensor) -> torch.Tensor:
        return self._block_mean(x_ts, self.window)

    def adjoint(self, y_obs: torch.Tensor) -> torch.Tensor:
        # A_ts (block-mean) has matrix entries (1/w).  Its transpose has
        # entries (1/w) in the same support, i.e. adjoint is also "replicate
        # and scale by 1/w".  We absorb the 1/w scaling into the MMPS
        # sigma_y_sq via the apply/adjoint composition:
        #     (A A^T)_{kk'} = (1/w) * delta_{kk'}   for BlockMean.
        # Concretely we use "replicate" for A^T here and rely on sigma_y^2 I
        # dominating the diagonal; the small 1/w factor is absorbed by CG's
        # scale invariance.  Using pure replicate (no scaling) matches what
        # the DualSpaceMMPS code expects because it does not apply any extra
        # scaling between G and G^T -- it's the SAME G^T in the normal
        # equations on both sides.  To be mathematically precise we use
        # A^T (the true transpose) = (1/w) * replicate, i.e. scale by 1/w.
        return self._block_replicate(y_obs, self.window) / float(self.window)

    def enforce(self, x_ts: torch.Tensor, y_obs: torch.Tensor) -> torch.Tensor:
        # Moore-Penrose: for A_ts = (1/w) 1^T per window,
        # A^+ = (1/w) 1^T^+ = w * (1/w) * 1^T ... let's just derive.
        # A A^T = (1/w^2) * w * I = (1/w) I  -> (A A^T)^{-1} = w * I.
        # A^+ = A^T (A A^T)^{-1} = w * A^T = w * (1/w) * replicate = replicate.
        # So the pseudo-inverse of block-mean acting on a residual y-vector
        # is pure replication.  Residual r_k = y_k - A(x_hat)_k is spread
        # back to all w positions of the window (no scaling).
        residual = y_obs - self._block_mean(x_ts, self.window)  # [B, T/w, F]
        correction = self._block_replicate(residual, self.window)  # [B, T, F]
        return x_ts + correction

    def warm_start(self, y_obs: np.ndarray, seed: int = 0,
                   stl_period: Optional[int] = None,
                   use_stl: bool = True,
                   sigma_hf: float = 0.1) -> np.ndarray:
        """y_obs: [N, T/w, F] -> x_init: [N, T, F].

        Three-stage init chosen so that:
          (i)  the trend across windows is smooth (not piecewise-constant),
          (ii) there is realistic high-frequency content INSIDE each window
               (without which the M-step diffusion model collapses onto a
               low-frequency manifold — observed empirically on whole-totem-2288,
               disc_mean flat at 0.47 for 4 EM iterations),
          (iii) A_ts(x_init) == y_obs exactly (no constraint violation).

        Stages:
          (1) Cross-window trend = linear interpolation of y_k values treated
              as samples at window midpoints t_k = w*k + (w-1)/2.  Gives a
              piecewise-linear (not piecewise-constant) init with realistic
              low-frequency structure.
          (2) Null-space Gaussian noise (sigma = sigma_hf) added at every
              position, then centered per-window so block_mean(noise) == 0 —
              i.e. the noise lives entirely in the null space of A and
              therefore CANNOT change A_ts(x_init).  Default sigma_hf=0.1
              matches typical per-timestep stdev on MinMax-normalized Energy.
          (3) Window-mean correction: reset each window to exactly match y_k.
              Required because stage (1) only approximately matches y_k on
              the midpoint grid after interpolation; this ensures A x = y
              to float32 precision regardless of the interpolation scheme.

        STL is NOT applied here — on a linear-interp + null-space-noise init
        it mainly oversmooths and re-collapses onto the low-freq manifold
        (the exact failure mode we saw on whole-totem-2288).  The diffusion
        M-step's own denoiser acts as the spectrum regularizer instead.

        Args:
          y_obs:     [N, T/w, F] aggregated observations (MinMax-in-[0,1]).
          seed:      RNG seed for the null-space noise.
          stl_period: accepted for API compatibility with older callers,
                      ignored (kept to avoid breaking `op.warm_start(y, seed=..., stl_period=...)`).
          use_stl:    accepted for API compatibility, ignored.
          sigma_hf:  stdev of null-space noise.  0.05-0.15 is a good range
                     for MinMax-normalized data.  Larger values help when the
                     dataset has more per-step variation; too large will
                     push many positions into the clip boundary and lose
                     A x = y exactness on the clamp.
        """
        if y_obs.ndim != 3:
            raise ValueError(
                f"BlockMeanOperator.warm_start expects y_obs [N, T/w, F]; "
                f"got shape {y_obs.shape}"
            )
        N, N_obs, F = y_obs.shape
        w = self.window
        T = self.seq_len
        assert N_obs * w == T

        # Stage 1: linear interpolation across window midpoints.
        # y_k represents the mean over positions [w*k, w*k+w-1], which has
        # midpoint t_k = w*k + (w-1)/2 (a half-integer for even w).  We fit
        # a linear interpolant through these midpoints, sampled at the full
        # grid t=0..T-1.  Extrapolation at the edges uses the nearest y.
        t_mid = np.arange(N_obs, dtype=np.float64) * w + (w - 1) / 2.0  # [N_obs]
        t_full = np.arange(T, dtype=np.float64)                          # [T]
        x_init = np.empty((N, T, F), dtype=np.float32)
        for i in range(N):
            for c in range(F):
                x_init[i, :, c] = np.interp(
                    t_full, t_mid, y_obs[i, :, c].astype(np.float64),
                ).astype(np.float32)

        # Stage 2: null-space noise.  Sample iid Gaussian at every position,
        # then subtract the window-mean of the noise from every position in
        # that window.  The residual has per-window mean exactly zero, so
        # A_ts(noise_centered) = 0 and adding it preserves A_ts(x_init).
        rng = np.random.default_rng(seed)
        noise = (sigma_hf * rng.standard_normal(x_init.shape)).astype(np.float32)
        # [N, T, F] -> [N, N_obs, w, F] -> subtract per-window mean -> flat.
        noise_win = noise.reshape(N, N_obs, w, F)
        noise_win = noise_win - noise_win.mean(axis=2, keepdims=True)
        x_init = x_init + noise_win.reshape(N, T, F)

        # Stage 3: exact window-mean correction.  After stages (1) and (2)
        # we have A(x_init) ≈ y (stage 1 gets the mean right only when the
        # interp is exact, i.e. interior windows — edge windows can be off
        # by ~epsilon).  Spread the residual uniformly across the window
        # (Moore-Penrose of block-mean, same as `enforce`).
        current_mean = x_init.reshape(N, N_obs, w, F).mean(axis=2)       # [N, N_obs, F]
        residual = y_obs.astype(np.float32) - current_mean                # [N, N_obs, F]
        correction = np.repeat(residual, w, axis=1)                       # [N, T, F]
        x_init = x_init + correction

        # Deliberately NOT clipping to [0, 1] here: the stage-3 correction
        # (and the noise in stage 2) can push a few positions slightly out of
        # range (<5% of positions by <0.05 in practice), but clipping them
        # would break the A_ts(x_init) = y_obs invariant — which was the
        # exact failure mode that killed whole-totem-2288.  The diffusion
        # pipeline handles slightly-out-of-range inputs fine.
        return x_init


# =============================================================================
# ContinuousResample: time-continuous irregular sampling via linear interpolation
# =============================================================================


class ContinuousResampleOperator(ObservationOperator):
    """A_ts(x)[k, c] = (1 - alpha_k) * x[floor(t_k), c] + alpha_k * x[ceil(t_k), c].

    Per-sample observation operator S in R^{N_obs x T} with two non-zeros per
    row at the floor and ceil grid positions of t_k in [0, T-1].  This is the
    natural linear-interpolation resampling operator: it degenerates exactly to
    PointMaskOperator when t_k is integer-valued (alpha_k = 0) and has the same
    2-tap local support as nearest-neighbor up/down sampling in DSP.

    Properties used by MMPS / the obs-space CG:
      * Row sums are 1 (convex combination) so lambda_max(S S^T) <= 1, which
        keeps the CG condition number bounded by the same c^2 + 1 / c^2 factor
        that PointMask enjoys under the c·sigma_t adaptive noise schedule.
      * S has 2 non-zeros per row, so apply / adjoint are O(B · N_obs) and
        internally implemented via a dense [B, N_obs, T] matrix (T=24 in the
        paper's runs; dense at this size is faster than sparse kernels).
      * Moore-Penrose: x <- x + S^T (S S^T)^{-1} (y - S x).  (S S^T) is
        almost-surely full-rank for continuous t_k; we add a tiny ridge for
        numerical safety.

    Usage:
      op = ContinuousResampleOperator(seq_len, n_features, n_obs)
      op.set_grid(timestamps)   # timestamps: [B, N_obs] float in [0, T-1]
      y = op.apply(x)           # x: [B, T, F]  -> y: [B, N_obs, F]

    The timestamps tensor is expected to be sorted per sample (the sampler in
    run_co_evolving_em_continuous produces sorted times); sortedness is not
    required for correctness but gives a numerically well-behaved S S^T.
    """

    name = "continuous_resample"
    supports_curriculum = False  # v1: no timestamp-curriculum (see plan)

    def __init__(self, seq_len: int, n_features: int, n_obs: int):
        super().__init__(seq_len, n_features)
        if n_obs < 1:
            raise ValueError(f"ContinuousResampleOperator: n_obs must be >= 1, got {n_obs}")
        if n_obs > seq_len:
            # Technically allowed (over-determined), but flag it: with linear
            # interpolation and t_k in [0, T-1], rows are 2-sparse combinations
            # of T grid points -- you can have at most T linearly independent
            # rows, so n_obs > T makes S S^T singular and enforce() will hit
            # the ridge regularizer instead of doing a true Moore-Penrose.
            pass
        self._n_obs = int(n_obs)
        # Lazily set by set_grid(); shape [B, N_obs, T], float.
        self._S: Optional[torch.Tensor] = None
        self._timestamps: Optional[torch.Tensor] = None

    @property
    def n_obs(self) -> int:
        return self._n_obs

    # --- grid setup ---

    def set_grid(self, timestamps: torch.Tensor) -> None:
        """Build the per-batch observation matrix S from continuous timestamps.

        Args:
          timestamps: [B, N_obs] float tensor with values in [0, T-1].  Each
            row is the set of observation times for one sample.  Values are
            clamped to [0, T-1] internally so callers do not need to be
            paranoid about floating-point edge cases.
        """
        if timestamps.ndim != 2:
            raise ValueError(
                f"ContinuousResampleOperator.set_grid expects [B, N_obs]; "
                f"got shape {tuple(timestamps.shape)}"
            )
        B, N_obs = timestamps.shape
        if N_obs != self._n_obs:
            raise ValueError(
                f"ContinuousResampleOperator: timestamps has N_obs={N_obs} "
                f"but operator was built with n_obs={self._n_obs}. "
                f"Rebuild the operator or pad/truncate the timestamps."
            )
        T = self.seq_len
        device = timestamps.device

        t_clamped = timestamps.clamp(min=0.0, max=float(T - 1))
        floor_idx = t_clamped.floor().long().clamp(max=T - 1)          # [B, N_obs]
        ceil_idx = (floor_idx + 1).clamp(max=T - 1)                     # [B, N_obs]
        alpha = (t_clamped - floor_idx.to(t_clamped.dtype))             # [B, N_obs]
        # When t_k is exactly on a grid point, floor==ceil and alpha==0, so
        # the row degenerates cleanly to a one-hot (identical to PointMask).

        # Build dense S: [B, N_obs, T].  Only 2 non-zeros per row.
        S = torch.zeros(B, N_obs, T, device=device, dtype=t_clamped.dtype)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N_obs)
        obs_idx = torch.arange(N_obs, device=device).unsqueeze(0).expand(B, N_obs)
        # Contribution at floor: weight (1 - alpha)
        S[batch_idx, obs_idx, floor_idx] = (1.0 - alpha)
        # Contribution at ceil: weight alpha.  When floor==ceil (integer t_k),
        # this ADDS 0 on top of the (1-alpha)=1 we wrote above, so the final
        # weight on that position is exactly 1.  (Using assignment here would
        # overwrite to 0 for integer t_k; we use accumulation via index_put_
        # below to be safe across duplicate floor/ceil indices.)
        S.index_put_(
            (batch_idx, obs_idx, ceil_idx),
            alpha,
            accumulate=True,
        )

        self._S = S
        self._timestamps = t_clamped

    def _require_grid(self) -> torch.Tensor:
        if self._S is None:
            raise RuntimeError(
                "ContinuousResampleOperator: set_grid(timestamps) must be called "
                "before apply / adjoint / enforce."
            )
        return self._S

    # --- interface ---

    def apply(self, x_ts: torch.Tensor) -> torch.Tensor:
        S = self._require_grid().to(dtype=x_ts.dtype, device=x_ts.device)
        # y[b, k, f] = sum_t S[b, k, t] * x[b, t, f]
        return torch.einsum('bkt,btf->bkf', S, x_ts)

    def adjoint(self, y_obs: torch.Tensor) -> torch.Tensor:
        S = self._require_grid().to(dtype=y_obs.dtype, device=y_obs.device)
        # (S^T y)[b, t, f] = sum_k S[b, k, t] * y[b, k, f]
        return torch.einsum('bkt,bkf->btf', S, y_obs)

    def enforce(self, x_ts: torch.Tensor, y_obs: torch.Tensor) -> torch.Tensor:
        """Moore-Penrose residual update: x <- x + S^T (S S^T + eps I)^{-1} (y - S x).

        For 2-tap linear interpolation S S^T is sparse (at most pentadiagonal)
        and almost-surely full-rank; solving the N_obs x N_obs system with
        torch.linalg.solve is cheap at N_obs <= T (= 24 in our runs).  A small
        ridge (eps) absorbs the rare rank-deficient case (e.g. two obs at the
        same continuous time).  For pathological batches that remain singular
        in fp32 we fall back to lstsq (least-squares) which always succeeds.
        """
        S_ref = self._require_grid()
        dtype = x_ts.dtype
        device = x_ts.device
        S = S_ref.to(dtype=dtype, device=device)

        # Residual in y-space.
        residual = y_obs - torch.einsum('bkt,btf->bkf', S, x_ts)   # [B, N_obs, F]

        # M = S S^T, shape [B, N_obs, N_obs].  Symmetric PSD.
        M = torch.einsum('bkt,bjt->bkj', S, S)
        eye = torch.eye(M.shape[-1], device=device, dtype=dtype).unsqueeze(0)
        M = M + 1e-8 * eye

        # Solve M z = residual per batch; PyTorch broadcasts the RHS features.
        # Fast path: torch.linalg.solve. If any batch element is singular it
        # raises _LinAlgError, in which case we fall back to lstsq, which
        # returns the minimum-norm least-squares solution and never raises.
        # For the common (non-singular) batches lstsq and solve agree, so
        # this preserves numerics on the existing happy path.
        try:
            z = torch.linalg.solve(M, residual)                     # [B, N_obs, F]
        except (torch._C._LinAlgError, RuntimeError):
            z = torch.linalg.lstsq(M, residual).solution            # [B, N_obs, F]

        # Correction = S^T z.
        correction = torch.einsum('bkt,bkf->btf', S, z)             # [B, T, F]
        return x_ts + correction

    def warm_start(self, y_obs: np.ndarray, timestamps: np.ndarray,
                   seed: int = 0, use_stl: bool = True,
                   stl_period: Optional[int] = None) -> np.ndarray:
        """Piecewise-linear interpolation onto the regular grid, optional STL.

        Args:
          y_obs:       [N, N_obs, F] float array of observations.
          timestamps:  [N, N_obs]   float array of sample times in [0, T-1].
          seed:        reserved for future randomized variants.
          use_stl:     if True, apply STL trend+seasonal refinement on the
                       interpolated signal to add within-gap structure.
          stl_period:  STL period (auto-detected from seq_len if None).

        Returns:
          x_init: [N, seq_len, F] float32, clipped to [0, 1].
        """
        if y_obs.ndim != 3:
            raise ValueError(
                f"ContinuousResampleOperator.warm_start expects y_obs [N, N_obs, F]; "
                f"got shape {y_obs.shape}"
            )
        if timestamps.ndim != 2:
            raise ValueError(
                f"ContinuousResampleOperator.warm_start expects timestamps [N, N_obs]; "
                f"got shape {timestamps.shape}"
            )
        N, N_obs, F = y_obs.shape
        if timestamps.shape != (N, N_obs):
            raise ValueError(
                f"shape mismatch: y_obs {y_obs.shape} vs timestamps {timestamps.shape}"
            )
        T = self.seq_len
        x_init = np.zeros((N, T, F), dtype=np.float32)
        grid = np.arange(T, dtype=np.float64)

        for i in range(N):
            # Sort by time per sample (np.interp requires increasing xp).
            order = np.argsort(timestamps[i])
            t_sorted = timestamps[i, order].astype(np.float64)
            y_sorted = y_obs[i, order, :].astype(np.float64)
            # Deduplicate identical timestamps (rare but numerically harmful).
            if t_sorted.size > 1:
                keep = np.concatenate([[True], np.diff(t_sorted) > 0])
                t_sorted = t_sorted[keep]
                y_sorted = y_sorted[keep, :]
            if t_sorted.size == 0:
                x_init[i, :, :] = 0.5
                continue
            for c in range(F):
                x_init[i, :, c] = np.interp(grid, t_sorted, y_sorted[:, c]).astype(np.float32)

        x_init = np.clip(x_init, 0.0, 1.0)

        if not use_stl:
            return x_init

        if stl_period is None:
            from utils.utils_stl import auto_detect_period
            stl_period = auto_detect_period(T)
        if stl_period % 2 == 0:
            stl_period += 1
        if stl_period < 3:
            stl_period = 3
        if T < 2 * stl_period:
            return x_init

        try:
            from statsmodels.tsa.seasonal import STL
        except ImportError:
            return x_init

        out = x_init.copy()
        for i in range(N):
            for c in range(F):
                series = x_init[i, :, c].astype(np.float64)
                try:
                    stl = STL(series, period=stl_period, robust=True).fit()
                    out[i, :, c] = (stl.trend + stl.seasonal).astype(np.float32)
                except Exception:
                    pass
        return np.clip(out, 0.0, 1.0)


# =============================================================================
# Factory
# =============================================================================


def build_operator(corruption_type: str, seq_len: int, n_features: int,
                   agg_window: int = 1,
                   n_obs: Optional[int] = None,
                   interp_kernel: str = "linear") -> ObservationOperator:
    """Factory for CLI-driven construction.

    Callers pass the CLI args (corruption_type, agg_window, n_obs,
    interp_kernel) directly.  For continuous-time resampling n_obs is the
    number of sample points per sample (derived from n_obs_ratio * seq_len
    in the run script); interp_kernel is 'linear' in v1 (cubic/sinc reserved
    for v2).
    """
    ct = corruption_type.lower().strip()
    if ct in ("missing", "point_mask"):
        return PointMaskOperator(seq_len, n_features)
    if ct in ("aggregate", "block_mean", "agg"):
        return BlockMeanOperator(seq_len, n_features, agg_window)
    if ct in ("continuous", "continuous_resample", "resample"):
        if n_obs is None:
            raise ValueError(
                "build_operator: --corruption_type continuous requires n_obs "
                "(set via --n_obs_ratio in the run script)."
            )
        kernel = interp_kernel.lower().strip()
        if kernel != "linear":
            raise NotImplementedError(
                f"build_operator: --interp_kernel={interp_kernel!r} is a v1 stub. "
                f"Only 'linear' is implemented in v1; cubic/sinc are reserved "
                f"for v2 (see Continuous-Time Resampling plan, §Ablation Row F)."
            )
        return ContinuousResampleOperator(seq_len, n_features, n_obs)
    raise ValueError(
        f"build_operator: unknown corruption_type={corruption_type!r}. "
        f"Supported: 'missing', 'aggregate', 'continuous'."
    )
