"""
Unit tests for obs_operators.ObservationOperator implementations.

Checks the three mathematical properties that any operator bug would break:

  1. Adjoint identity:  <A x, y> == <x, A^T y>
  2. Moore-Penrose:     A (x + A^+ (y - A x)) == y
  3. Shape / n_obs contract

Run with:  pytest tests/test_obs_operators.py
"""

import numpy as np
import pytest
import torch

from obs_operators import (
    BlockMeanOperator,
    ContinuousResampleOperator,
    PointMaskOperator,
    build_operator,
)


# -----------------------------------------------------------------------------
# PointMaskOperator
# -----------------------------------------------------------------------------


def test_point_mask_adjoint_identity():
    """<A x, y> == <x, A^T y> for arbitrary x, y."""
    torch.manual_seed(0)
    B, T, F = 4, 24, 3
    x = torch.randn(B, T, F, dtype=torch.float64)
    y = torch.randn(B, T, F, dtype=torch.float64)

    op = PointMaskOperator(seq_len=T, n_features=F)
    mask = (torch.rand(B, T, F) > 0.5).to(torch.float64)
    op.set_mask(mask)

    lhs = (op.apply(x) * y).sum()
    rhs = (x * op.adjoint(y)).sum()

    assert torch.allclose(lhs, rhs, atol=1e-12), (
        f"PointMask adjoint failure: <Ax,y>={lhs.item()} vs <x,A^T y>={rhs.item()}"
    )


def test_point_mask_enforce_idempotent():
    """A_ts(enforce(x, y)) == A_ts(y) == mask*y, and enforce is idempotent."""
    torch.manual_seed(1)
    B, T, F = 4, 24, 3
    x = torch.randn(B, T, F, dtype=torch.float64)
    y = torch.randn(B, T, F, dtype=torch.float64)

    op = PointMaskOperator(T, F)
    mask = (torch.rand(B, T, F) > 0.5).to(torch.float64)
    op.set_mask(mask)

    # On observed support, enforced tensor == y.
    x_enf = op.enforce(x, mask * y)
    assert torch.allclose(mask * x_enf, mask * y, atol=1e-14)
    # Off support, enforced tensor == x (unchanged).
    assert torch.allclose((1 - mask) * x_enf, (1 - mask) * x, atol=1e-14)

    # Idempotent.
    assert torch.allclose(op.enforce(x_enf, mask * y), x_enf, atol=1e-14)


def test_point_mask_n_obs():
    op = PointMaskOperator(seq_len=48, n_features=5)
    assert op.n_obs == 48
    assert op.name == "point_mask"
    assert op.supports_curriculum is True


# -----------------------------------------------------------------------------
# BlockMeanOperator
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("window", [2, 3, 4, 6])
def test_block_mean_adjoint_identity(window):
    """<A x, y>_{Y} == <x, A^T y>_{X} with A^T = (1/w) replicate."""
    torch.manual_seed(2)
    B, T, F = 3, 24, 4
    assert T % window == 0

    op = BlockMeanOperator(seq_len=T, n_features=F, window=window)
    x = torch.randn(B, T, F, dtype=torch.float64)
    y = torch.randn(B, T // window, F, dtype=torch.float64)

    lhs = (op.apply(x) * y).sum()
    rhs = (x * op.adjoint(y)).sum()

    assert torch.allclose(lhs, rhs, atol=1e-12), (
        f"BlockMean(w={window}) adjoint failure: "
        f"<Ax,y>={lhs.item()} vs <x,A^T y>={rhs.item()}"
    )


@pytest.mark.parametrize("window", [2, 3, 4, 6])
def test_block_mean_pseudo_inverse_lands_on_y(window):
    """A_ts(enforce(x, y)) == y to machine precision (Moore-Penrose property)."""
    torch.manual_seed(3)
    B, T, F = 3, 24, 4

    op = BlockMeanOperator(T, F, window)
    x = torch.randn(B, T, F, dtype=torch.float64)
    y = torch.randn(B, T // window, F, dtype=torch.float64)

    x_enf = op.enforce(x, y)
    y_recovered = op.apply(x_enf)

    assert torch.allclose(y_recovered, y, atol=1e-12), (
        f"BlockMean(w={window}) pseudo-inverse failure: "
        f"A(enforce(x,y)) != y; max err = {(y_recovered - y).abs().max().item()}"
    )


@pytest.mark.parametrize("window", [2, 3, 4])
def test_block_mean_preserves_within_window_null_space(window):
    """enforce must preserve within-window variation (null space of A_ts).

    If x_hat already satisfies A(x_hat) = y, enforce is a no-op.  This tests
    that the pseudo-inverse spreads only the per-window *residual*, not the
    whole observation.
    """
    torch.manual_seed(4)
    B, T, F = 2, 24, 3
    op = BlockMeanOperator(T, F, window)
    x_base = torch.randn(B, T, F, dtype=torch.float64)
    y = op.apply(x_base)  # so A(x_base) = y exactly

    # Any x_hat that satisfies A(x_hat) = y should be returned unchanged.
    x_hat = x_base.clone()
    x_enf = op.enforce(x_hat, y)
    assert torch.allclose(x_hat, x_enf, atol=1e-12)


@pytest.mark.parametrize("window", [2, 3, 4])
def test_block_mean_forward_is_mean(window):
    """A_ts really computes the block mean (not sum, not first element)."""
    B, T, F = 2, 24, 2
    op = BlockMeanOperator(T, F, window)
    x = torch.arange(T, dtype=torch.float64).reshape(1, T, 1).expand(B, T, F).contiguous()
    y = op.apply(x)  # [B, T/w, F]

    # First window's values are 0..w-1, mean == (w-1)/2.
    expected_first = (window - 1) / 2.0
    assert torch.allclose(y[:, 0, :], torch.full_like(y[:, 0, :], expected_first)), (
        f"BlockMean(w={window}) apply != mean over window"
    )


def test_block_mean_rejects_non_divisible_seq_len():
    with pytest.raises(ValueError):
        BlockMeanOperator(seq_len=24, n_features=3, window=5)  # 24 % 5 != 0


def test_block_mean_rejects_zero_window():
    with pytest.raises(ValueError):
        BlockMeanOperator(seq_len=24, n_features=3, window=0)


# -----------------------------------------------------------------------------
# warm_start shape contract
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("window", [2, 4])
def test_block_mean_warm_start_shape(window):
    """warm_start returns [N, T, F] with correct shape and no NaNs.

    Note: we intentionally DO NOT clip to [0,1] (would break the A x = y
    invariant near the boundary), so we only assert finiteness and a loose
    value-range sanity check.
    """
    np.random.seed(0)
    N, T, F = 6, 24, 3
    op = BlockMeanOperator(T, F, window)
    y = np.random.rand(N, T // window, F).astype(np.float32)
    x_init = op.warm_start(y, seed=0)
    assert x_init.shape == (N, T, F)
    assert np.isfinite(x_init).all()
    # Loose sanity: sigma_hf=0.1 means values can go slightly outside [0,1].
    assert x_init.min() >= -0.5
    assert x_init.max() <= 1.5


@pytest.mark.parametrize("window", [2, 3, 4])
def test_block_mean_warm_start_preserves_mean_constraint(window):
    """A_ts(warm_start(y)) must equal y to float32 precision.

    This is the critical invariant that failed on whole-totem-2288: the old
    STL-on-replicate path produced degenerate piecewise-constant sequences
    that the M-step then memorized.  The new linear-interp + null-space-noise
    path must preserve A x = y exactly (modulo the [0,1] clip on edge cases).
    """
    np.random.seed(1)
    N, T, F = 4, 24, 3
    op = BlockMeanOperator(T, F, window)
    # y in [0.1, 0.9] to stay well away from the clip boundary, so clipping
    # cannot break the A x = y exactness at non-edge positions.
    y = 0.1 + 0.8 * np.random.rand(N, T // window, F).astype(np.float32)
    x_init = op.warm_start(y, seed=0, sigma_hf=0.1)

    # Recover block means from x_init and compare to y.
    x_t = torch.from_numpy(x_init.astype(np.float64))
    op_t = BlockMeanOperator(T, F, window)
    recovered = op_t.apply(x_t).numpy().astype(np.float32)

    np.testing.assert_allclose(recovered, y, atol=1e-5)


@pytest.mark.parametrize("window", [2, 3, 4])
def test_block_mean_warm_start_has_within_window_variation(window):
    """warm_start must inject non-trivial within-window variation.

    Empirically: stdev of within-window residuals should be roughly sigma_hf
    (the null-space noise scale).  If it's ~0, we've regressed to the old
    piecewise-constant init and the M-step will collapse again.
    """
    np.random.seed(2)
    N, T, F = 8, 24, 3
    sigma_hf = 0.1
    op = BlockMeanOperator(T, F, window)
    y = 0.3 + 0.4 * np.random.rand(N, T // window, F).astype(np.float32)
    x_init = op.warm_start(y, seed=0, sigma_hf=sigma_hf)

    # Residuals = x_init minus replicated window-mean.
    x_win = x_init.reshape(N, T // window, window, F)
    residual_std = (x_win - x_win.mean(axis=2, keepdims=True)).std()

    # With sigma_hf=0.1 and per-window centering of N(0, 0.01), the expected
    # residual std is sigma_hf * sqrt(1 - 1/w) (correction for removing the
    # window mean).  Check it's within a factor of 2 of that.
    expected = sigma_hf * np.sqrt(max(1.0 - 1.0 / window, 1e-6))
    assert residual_std > 0.5 * expected, (
        f"Within-window variation collapsed: std={residual_std:.4f}, "
        f"expected ~{expected:.4f} (w={window}, sigma_hf={sigma_hf})"
    )


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------


def test_build_operator_dispatch():
    op_m = build_operator("missing", seq_len=24, n_features=3)
    op_a = build_operator("aggregate", seq_len=24, n_features=3, agg_window=4)
    op_c = build_operator("continuous", seq_len=24, n_features=3, n_obs=12)
    assert isinstance(op_m, PointMaskOperator)
    assert isinstance(op_a, BlockMeanOperator)
    assert isinstance(op_c, ContinuousResampleOperator)
    assert op_a.window == 4
    assert op_a.n_obs == 6
    assert op_c.n_obs == 12


def test_build_operator_unknown_raises():
    with pytest.raises(ValueError):
        build_operator("unknown_type", seq_len=24, n_features=3)


def test_build_operator_continuous_requires_n_obs():
    with pytest.raises(ValueError):
        build_operator("continuous", seq_len=24, n_features=3)


def test_build_operator_continuous_rejects_non_linear_kernel():
    with pytest.raises(NotImplementedError):
        build_operator(
            "continuous", seq_len=24, n_features=3, n_obs=12, interp_kernel="cubic",
        )


# -----------------------------------------------------------------------------
# ContinuousResampleOperator
# -----------------------------------------------------------------------------


def test_continuous_resample_adjoint_identity():
    """<S x, y> == <x, S^T y> for arbitrary x, y with continuous timestamps."""
    torch.manual_seed(10)
    B, T, F, N_obs = 4, 24, 3, 12
    op = ContinuousResampleOperator(seq_len=T, n_features=F, n_obs=N_obs)
    # Random continuous timestamps in [0, T-1], sorted per row.
    t = torch.rand(B, N_obs, dtype=torch.float64) * (T - 1)
    t, _ = torch.sort(t, dim=-1)
    op.set_grid(t)

    x = torch.randn(B, T, F, dtype=torch.float64)
    y = torch.randn(B, N_obs, F, dtype=torch.float64)

    lhs = (op.apply(x) * y).sum()
    rhs = (x * op.adjoint(y)).sum()

    assert torch.allclose(lhs, rhs, atol=1e-10), (
        f"ContinuousResample adjoint failure: "
        f"<Sx,y>={lhs.item()} vs <x,S^T y>={rhs.item()}"
    )


def test_continuous_resample_enforce_round_trip():
    """After x' = enforce(x, y), ||S x' - y|| ~ 0."""
    torch.manual_seed(11)
    B, T, F, N_obs = 4, 24, 3, 12
    op = ContinuousResampleOperator(seq_len=T, n_features=F, n_obs=N_obs)
    t = torch.rand(B, N_obs, dtype=torch.float64) * (T - 1)
    t, _ = torch.sort(t, dim=-1)
    # Enforce spacing >= 1 to keep S S^T well-conditioned.
    spacing = torch.linspace(0, T - 1, N_obs, dtype=torch.float64).unsqueeze(0).expand(B, -1)
    jitter = (torch.rand(B, N_obs, dtype=torch.float64) - 0.5) * 0.4
    t = (spacing + jitter).clamp(0.0, float(T - 1))
    op.set_grid(t)

    x = torch.randn(B, T, F, dtype=torch.float64)
    y = torch.randn(B, N_obs, F, dtype=torch.float64)

    x_enf = op.enforce(x, y)
    y_recovered = op.apply(x_enf)

    assert torch.allclose(y_recovered, y, atol=1e-6), (
        f"ContinuousResample enforce round-trip failure: "
        f"max err = {(y_recovered - y).abs().max().item()}"
    )


def test_continuous_resample_integer_timestamps_match_point_mask():
    """With integer t_k, ContinuousResample == PointMask on apply & adjoint.

    This is the "clean numerical regression test" from the plan: integer-valued
    timestamps make alpha=0 for every row, so S degenerates to a one-hot
    matrix -- the exact same observation operator as PointMask restricted to
    the sampled positions.
    """
    torch.manual_seed(12)
    B, T, F, N_obs = 2, 24, 3, 8
    # Pick disjoint integer timestamps per sample.
    perms = torch.stack([torch.randperm(T)[:N_obs] for _ in range(B)])
    t_int, _ = torch.sort(perms, dim=-1)
    t_float = t_int.to(torch.float64)

    op_c = ContinuousResampleOperator(seq_len=T, n_features=F, n_obs=N_obs)
    op_c.set_grid(t_float)

    op_m = PointMaskOperator(seq_len=T, n_features=F)
    mask = torch.zeros(B, T, F, dtype=torch.float64)
    for b in range(B):
        mask[b, t_int[b], :] = 1.0
    op_m.set_mask(mask)

    x = torch.randn(B, T, F, dtype=torch.float64)

    # apply: gather ContinuousResample y and compare to masked x at those positions.
    y_c = op_c.apply(x)                    # [B, N_obs, F]
    # Equivalent PointMask operator also selects x at the same positions.
    y_m = torch.stack([x[b, t_int[b], :] for b in range(B)], dim=0)

    assert torch.allclose(y_c, y_m, atol=1e-12), (
        f"ContinuousResample[integer t] != PointMask gather: "
        f"max err = {(y_c - y_m).abs().max().item()}"
    )

    # adjoint: scattering y back onto the grid equals the PointMask masked y.
    y_obs = torch.randn(B, N_obs, F, dtype=torch.float64)
    adj_c = op_c.adjoint(y_obs)           # [B, T, F]
    adj_expected = torch.zeros(B, T, F, dtype=torch.float64)
    for b in range(B):
        adj_expected[b, t_int[b], :] = y_obs[b]
    assert torch.allclose(adj_c, adj_expected, atol=1e-12), (
        f"ContinuousResample[integer t] adjoint != PointMask scatter: "
        f"max err = {(adj_c - adj_expected).abs().max().item()}"
    )

    # enforce: (integer timestamps ==> Moore-Penrose == hard overwrite).
    x_enf = op_c.enforce(x, y_obs)
    for b in range(B):
        assert torch.allclose(x_enf[b, t_int[b], :], y_obs[b], atol=1e-5)


def test_continuous_resample_warm_start_shape():
    np.random.seed(13)
    N, T, F, N_obs = 6, 24, 3, 12
    op = ContinuousResampleOperator(seq_len=T, n_features=F, n_obs=N_obs)
    y = np.random.rand(N, N_obs, F).astype(np.float32)
    timestamps = np.sort(
        np.random.rand(N, N_obs).astype(np.float32) * (T - 1), axis=-1,
    )
    x_init = op.warm_start(y, timestamps, seed=0, use_stl=False)
    assert x_init.shape == (N, T, F)
    assert np.isfinite(x_init).all()
    assert x_init.min() >= 0.0 - 1e-6
    assert x_init.max() <= 1.0 + 1e-6


def test_continuous_resample_n_obs_contract():
    op = ContinuousResampleOperator(seq_len=24, n_features=3, n_obs=10)
    assert op.n_obs == 10
    assert op.name == "continuous_resample"
    assert op.supports_curriculum is False


def test_continuous_resample_rejects_wrong_timestamps_shape():
    op = ContinuousResampleOperator(seq_len=24, n_features=3, n_obs=10)
    with pytest.raises(ValueError):
        op.set_grid(torch.randn(4, 12))  # n_obs mismatch


def test_continuous_resample_requires_set_grid():
    op = ContinuousResampleOperator(seq_len=24, n_features=3, n_obs=10)
    with pytest.raises(RuntimeError):
        op.apply(torch.randn(2, 24, 3))


# -----------------------------------------------------------------------------
# L-TMPD row-sum diagonal sanity
# -----------------------------------------------------------------------------


def test_ltmpd_row_sum_diagonal_matches_matrix_form():
    """The row-sum diagonal used by L-TMPD (posterior_denoise_tmpd_obs_space)
    must equal the exact row-sum of G J^T G^T on a small toy system.

    TMPD reference (bb515/tmpdtorch, condition_methods.py::TweedieMomentProjection):
        C_yy_ts = G (J^T (G^T ones))  + reg
    which is the TS-space row-sum  (G J^T G^T) @ ones  (plus regularizer).

    Here we set J = I (identity denoiser Jacobian) so the expected C_yy is
    the row-sum of  G G^T  on observed positions, which for a point mask
    equals the mask itself (G G^T is diagonal, each observed row sums to 1).
    """
    torch.manual_seed(42)
    B, T, F = 2, 16, 3

    op = PointMaskOperator(seq_len=T, n_features=F)
    mask = (torch.rand(B, T, F) > 0.5).to(torch.float64)
    op.set_mask(mask)

    # L-TMPD's update for identity J:
    #   ones_ts  = mask
    #   ones_img = G^T ones_ts  = adjoint(ones_ts) = mask (point mask self-adjoint)
    #   Jones    = ones_img  (since J = I)
    #   C_yy     = mask * (G Jones) = mask * mask = mask
    ones_ts = mask.clone()
    ones_img = op.adjoint(ones_ts)
    Jones_img = ones_img                          # J = I
    C_yy_ts = mask * op.apply(Jones_img)

    # Exact matrix form: (G J^T G^T)_{ij} = mask_i * delta_{ij} * mask_j
    # row-sum = mask_i * mask_i = mask_i (binary mask)
    expected = mask.clone()

    assert torch.allclose(C_yy_ts, expected, atol=1e-12), (
        f"L-TMPD row-sum diagonal mismatch: "
        f"max err = {(C_yy_ts - expected).abs().max().item()}"
    )
    # Physical sanity: row-sum of a PSD matrix is non-negative.
    assert (C_yy_ts >= 0).all()


def test_ltmpd_row_sum_diagonal_general_linear_J():
    """Row-sum diagonal  G J^T G^T @ ones  must equal the dense matrix
    form built by full materialization, for a small random symmetric J.
    """
    torch.manual_seed(43)
    T, F = 8, 2
    N = T * F

    op = PointMaskOperator(seq_len=T, n_features=F)
    mask = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 1]], dtype=torch.float64).unsqueeze(-1).expand(1, T, F).contiguous()
    op.set_mask(mask)

    # Symmetric PSD J (stand-in for J^T = Tweedie covariance).
    A = torch.randn(N, N, dtype=torch.float64)
    J = A @ A.T + 0.1 * torch.eye(N, dtype=torch.float64)

    # Row-sum via the TMPD recipe:  mask * G ( J^T ( G^T mask ) )
    ones_img = op.adjoint(mask)
    Jones_img = (J @ ones_img.reshape(1, N).T).T.reshape(1, T, F)
    C_yy_ts = mask * op.apply(Jones_img)

    # Dense reference: build G as binary mask matrix  [N_obs_positions, N]
    # then row-sum(G J G^T) = G @ J @ G^T @ ones.
    flat_mask = mask.reshape(N)
    G_rows = []
    for i in range(N):
        if flat_mask[i] > 0:
            row = torch.zeros(N, dtype=torch.float64)
            row[i] = 1.0
            G_rows.append(row)
    G = torch.stack(G_rows, dim=0) if G_rows else torch.zeros(0, N, dtype=torch.float64)

    ones_obs = torch.ones(G.shape[0], dtype=torch.float64)
    expected_obs = G @ J @ G.T @ ones_obs   # row-sum on observed positions

    # Scatter expected_obs back onto masked [1, T, F] layout for comparison.
    expected = torch.zeros(N, dtype=torch.float64)
    obs_idx = torch.nonzero(flat_mask, as_tuple=False).flatten()
    expected[obs_idx] = expected_obs
    expected = expected.reshape(1, T, F)

    assert torch.allclose(C_yy_ts, expected, atol=1e-10), (
        f"L-TMPD row-sum diagonal (linear J) mismatch: "
        f"max err = {(C_yy_ts - expected).abs().max().item()}"
    )
