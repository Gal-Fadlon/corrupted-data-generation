from omegaconf import OmegaConf
import argparse
import sys
import torch
from omegaconf import OmegaConf
import argparse

def parse_args_regular():
    """
    Parse arguments for unconditional models
    Returns: unconditioanl generation args namespace

    """
    parser = argparse.ArgumentParser()
    # --- general ---
    # NOTE: the following arguments are general, they are not present in the config file:
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use for dataloader')
    parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
    parser.add_argument('--log_dir', default='./logs', help='path to save logs')
    parser.add_argument('--wandb', type=bool, default=True, help='use wandb logger')
    parser.add_argument('--missing_rate', type=float, default=0.0)
    parser.add_argument('--tags', type=str, default=['0 missing rate'], help='tags for wandb logger', nargs='+')

    # --- diffusion process --- #
    parser.add_argument('--beta1', type=float, default=1e-5, help='value of beta 1')
    parser.add_argument('--betaT', type=float, default=1e-2, help='value of beta T')
    parser.add_argument('--deterministic', action='store_true', default=False, help='deterministic sampling')

    # ## --- config file --- # ##
    # NOTE: the below configuration are arguments. if given as CLI argument, they will override the config file values
    parser.add_argument('--config', type=str, default='./configs/seq_len_24/stock.yaml', help='config file')
    parser.add_argument('--model_save_path', type=str, default='./saved_models', help='path to save the model')


    # --- training ---
    parser.add_argument('--epochs', type=int, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, help='training batch size')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight decay')

    # --- data ---:
    parser.add_argument('--dataset',
                        choices=['sine', 'energy', 'mujoco', 'stock', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'electricity'], help='training dataset')

    parser.add_argument('--seq_len', type=int,
                        help='input sequence length,'
                             ' only needed if using short-term datasets(stocks,sine,energy,mujoco)')

    # --- image transformations ---:
    parser.add_argument('--delay', type=int,
                        help='delay for the delay embedding transformation, only needed if using delay embedding')
    parser.add_argument('--embedding', type=int,
                        help='embedding for the delay embedding transformation, only needed if using delay embedding')
    parser.add_argument('--embedder', type=str, default='delay',
                        choices=['delay', 'stft'],
                        help='Image embedding: delay (default, non-linear) or stft (linear, near-invertible). '
                             'Used by the STFT validation experiment (Paper v3 §7.10).')
    parser.add_argument('--stft_n_fft', type=int, default=8,
                        help='FFT size for the STFT embedder (ignored when --embedder=delay)')
    parser.add_argument('--stft_hop_length', type=int, default=4,
                        help='Hop length for the STFT embedder (ignored when --embedder=delay)')

    # --- model--- :
    parser.add_argument('--img_resolution', type=int, help='image resolution')
    parser.add_argument('--input_channels', type=int,
                        help='number of image channels, 2 if stft is used, 1 for delay embedding')
    parser.add_argument('--unet_channels', type=int, help='number of unet channels')
    parser.add_argument('--ch_mult', type=int, help='ch mut', nargs='+')
    parser.add_argument('--attn_resolution', type=int, help='attn_resolution', nargs='+')
    parser.add_argument('--diffusion_steps', type=int, help='number of diffusion steps')
    parser.add_argument('--ema', type=bool, help='use ema')
    parser.add_argument('--ema_warmup', type=int, help='ema warmup')

    # --- TST ---first_epoch
    parser.add_argument('--hidden_dim', type=int, default=40, help='dimension of the hidden layer')
    parser.add_argument('--r_layer', type=int, default=2, help='number of RNN layers')
    parser.add_argument('--last_activation_r', type=str, default='sigmoid', help='last activation function for RNN layers')
    parser.add_argument('--first_epoch', type=int, default=10, help='number of first epoch to start training')
    parser.add_argument('--x_hidden', type=int, default=48, help='dimension of x hidden layer')
    parser.add_argument('--input_size', type=int, default=1, help='input size of the model')

    # Adding new arguments for tst_config
    parser.add_argument('--n_heads', type=int, default=5, help='number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='dimension of feedforward layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--pos_encoding', type=str, choices=['fixed', 'learnable'], default='fixed',
                        help='positional encoding type')
    parser.add_argument('--activation', type=str, choices=['relu', 'gelu'], default='gelu', help='activation function')
    parser.add_argument('--norm', type=str, choices=['BatchNorm', 'LayerNorm'], default='BatchNorm',
                        help='normalization type')
    parser.add_argument('--freeze', type=bool, default=False, help='freeze transformer layers')

    parser.add_argument('--ts_rate', type=float, default=0, help='teacher forcing rate for tst')
    parser.add_argument('--save_model', type=bool, default=False, help='save model')
    parser.add_argument('--gaussian_noise_level', type=float, default=0.0, help='noise level injected to the original data')
    parser.add_argument('--new_metrics', type=int, default=1, help='save model')

    # --- logging ---s
    parser.add_argument('--logging_iter', type=int, default=10, help='number of iterations between logging')
    parser.add_argument('--percent', type=int, default=100)
    parsed_args = parser.parse_args()

    # load config file
    config = OmegaConf.to_object(OmegaConf.load(parsed_args.config))
    # override config file with command line args
    for k, v in vars(parsed_args).items():
        if v is None:
            setattr(parsed_args, k, config.get(k, None))
    # add to the parsed args, configs that are not in the parsed args but do in the config file
    # this is needed since multiple config files setups may be used
    for k, v in config.items():
        if k not in vars(parsed_args):
            setattr(parsed_args, k, v)

    parsed_args.input_size = parsed_args.input_channels
    return parsed_args


def parse_args_irregular():
    """
    Parse arguments for unconditional models
    Returns: unconditioanl generation args namespace

    """
    parser = argparse.ArgumentParser()
    # --- general ---
    # NOTE: the following arguments are general, they are not present in the config file:
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use for dataloader')
    parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
    parser.add_argument('--log_dir', default='./logs', help='path to save logs')
    parser.add_argument('--wandb', type=bool, default=True, help='use wandb logger')
    parser.add_argument('--missing_rate', type=float, default=0.0)
    parser.add_argument('--missing_type', type=str, default='random',
                        choices=['random', 'block'],
                        help='Missing pattern: random (MCAR point-wise) or block (contiguous gaps)')
    parser.add_argument('--block_length', type=int, default=12,
                        help='Length of contiguous block gaps when --missing_type=block')
    parser.add_argument('--tags', type=str, default=['30 missing rate'], help='tags for wandb logger', nargs='+')

    # --- diffusion process --- #
    parser.add_argument('--beta1', type=float, default=1e-5, help='value of beta 1')
    parser.add_argument('--betaT', type=float, default=1e-2, help='value of beta T')
    parser.add_argument('--deterministic', action='store_true', default=False, help='deterministic sampling')
    parser.add_argument(
        '--allow_tf32', action='store_true', default=True,
        help='Enable TF32 matmul/cudnn on Ampere+ (faster). Disable with --no_allow_tf32 if you hit CUDA illegal-instruction errors.',
    )
    parser.add_argument(
        '--no_allow_tf32', action='store_false', dest='allow_tf32',
        help='Disable TF32 matmul/cudnn',
    )

    # ## --- config file --- # ##
    # NOTE: the below configuration are arguments. if given as CLI argument, they will override the config file values
    parser.add_argument('--config', type=str, default='./configs/seq_len_24/stock.yaml', help='config file')
    parser.add_argument('--model_save_path', type=str, default='./saved_models', help='path to save the model')


    # --- training ---
    parser.add_argument('--epochs', type=int, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, help='training batch size')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight decay')

    # --- data ---:
    parser.add_argument('--dataset',
                        choices=['sine', 'energy', 'mujoco', 'stock', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'electricity'], help='training dataset')

    parser.add_argument('--seq_len', type=int,
                        help='input sequence length,'
                             ' only needed if using short-term datasets(stocks,sine,energy,mujoco)')

    # --- image transformations ---:
    parser.add_argument('--delay', type=int,
                        help='delay for the delay embedding transformation, only needed if using delay embedding')
    parser.add_argument('--embedding', type=int,
                        help='embedding for the delay embedding transformation, only needed if using delay embedding')
    parser.add_argument('--embedder', type=str, default='delay',
                        choices=['delay', 'stft'],
                        help='Image embedding: delay (default, non-linear) or stft (linear, near-invertible). '
                             'Used by the STFT validation experiment (Paper v3 §7.10).')
    parser.add_argument('--stft_n_fft', type=int, default=8,
                        help='FFT size for the STFT embedder (ignored when --embedder=delay)')
    parser.add_argument('--stft_hop_length', type=int, default=4,
                        help='Hop length for the STFT embedder (ignored when --embedder=delay)')

    # --- model--- :
    parser.add_argument('--img_resolution', type=int, help='image resolution')
    parser.add_argument('--input_channels', type=int,
                        help='number of image channels, 2 if stft is used, 1 for delay embedding')
    parser.add_argument('--unet_channels', type=int, help='number of unet channels')
    parser.add_argument('--ch_mult', type=int, help='ch mut', nargs='+')
    parser.add_argument('--attn_resolution', type=int, help='attn_resolution', nargs='+')
    parser.add_argument('--diffusion_steps', type=int, help='number of diffusion steps')
    parser.add_argument('--ema', type=bool, help='use ema')
    parser.add_argument('--ema_warmup', type=int, help='ema warmup')

    # --- TST ---
    parser.add_argument('--hidden_dim', type=int, default=40, help='dimension of the hidden layer')
    parser.add_argument('--r_layer', type=int, default=2, help='number of RNN layers')
    parser.add_argument('--last_activation_r', type=str, default='sigmoid', help='last activation function for RNN layers')
    parser.add_argument('--first_epoch', type=int, default=2, help='number of first epoch to start training')
    parser.add_argument('--x_hidden', type=int, default=48, help='dimension of x hidden layer')
    parser.add_argument('--input_size', type=int, default=1, help='input size of the model')

    # Adding new arguments for tst_config
    parser.add_argument('--n_heads', type=int, default=5, help='number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='dimension of feedforward layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--pos_encoding', type=str, choices=['fixed', 'learnable'], default='fixed',
                        help='positional encoding type')
    parser.add_argument('--activation', type=str, choices=['relu', 'gelu'], default='gelu', help='activation function')
    parser.add_argument('--norm', type=str, choices=['BatchNorm', 'LayerNorm'], default='BatchNorm',
                        help='normalization type')
    parser.add_argument('--freeze', type=bool, default=False, help='freeze transformer layers')

    parser.add_argument('--ts_rate', type=float, default=0, help='teacher forcing rate for tst')
    parser.add_argument('--save_model', type=bool, default=False, help='save model')
    parser.add_argument('--gaussian_noise_level', type=float, default=0.0, help='noise level injected to the original data')
    parser.add_argument('--noise_timestep', type=float, default=None, 
                        help='timestep fraction [0,1] to compute noise level from diffusion schedule. '
                             'If set, overrides gaussian_noise_level. 0=max noise (sigma_max), 1=min noise (sigma_min)')
    parser.add_argument('--new_metrics', type=int, default=1, help='save model')

    # --- DiffEM Configuration ---
    parser.add_argument('--use_diffem', action='store_true', default=False,
                        help='Enable DiffEM-style EM training loop (with TST conditioning)')
    parser.add_argument('--pure_diffem', action='store_true', default=False,
                        help='Enable PURE DiffEM (no TST, condition on zero-filled + mask only)')
    parser.add_argument('--em_iters', type=int, default=10,
                        help='Number of EM iterations for DiffEM training')
    parser.add_argument('--m_step_epochs', type=int, default=50,
                        help='Number of training epochs per M-step')
    parser.add_argument('--e_step_batch_size', type=int, default=64,
                        help='Batch size for E-step posterior sampling')
    parser.add_argument('--recon_cache_dir', type=str, default='./recon_cache',
                        help='Directory to cache reconstructed datasets per EM iteration')
    parser.add_argument('--num_posterior_samples', type=int, default=1,
                        help='Number of posterior samples per observation in E-step')
    parser.add_argument('--use_tst_conditioning', type=bool, default=True,
                        help='Use TST completion as conditioning signal')
    parser.add_argument('--freeze_tst_after_pretrain', type=bool, default=True,
                        help='Freeze TST encoder/decoder after initial pretraining')
    parser.add_argument('--no_freeze_tst', action='store_true', default=False,
                        help='Do NOT freeze TST encoder/decoder after initial pretraining')
    parser.add_argument('--em_corruption_rate', type=float, default=None,
                        help='Corruption rate for EM M-step pairs (default: same as missing_rate)')
    parser.add_argument('--em_noise_level', type=float, default=None,
                        help='Gaussian noise level for EM M-step pairs (default: same as gaussian_noise_level)')
    parser.add_argument('--train_uncond_after_em', type=bool, default=True,
                        help='Train unconditional model on final reconstructions after EM')
    parser.add_argument('--no_train_uncond', action='store_true', default=False,
                        help='Do NOT train unconditional model after EM')
    parser.add_argument('--uncond_epochs', type=int, default=100,
                        help='Number of epochs to train unconditional model after EM')
    parser.add_argument('--uncond_epochs_per_iter', type=int, default=None,
                        help='Number of epochs to train unconditional model per EM iteration (default: same as m_step_epochs)')
    parser.add_argument('--em_eval_interval', type=int, default=1,
                        help='Evaluate metrics every N EM iterations')

    # --- General corruption (run_diffem_mmps_general_corruption.py) ---
    parser.add_argument('--corruption_type', type=str, default='missing',
                        choices=['missing', 'gaussian_noise', 'gaussian_blur', 'random_projection',
                                 'ts_gaussian_noise', 'ts_temporal_smoothing', 'ts_missing_noise',
                                 'temporal_smoothing', 'combined_missing_noise'],
                        help='Type of forward operator / corruption model')
    parser.add_argument('--corruption_noise_level', type=float, default=0.01,
                        help='Observation noise sigma_y. For gaussian_noise this IS the corruption level.')
    parser.add_argument(
        '--align_irregular_gaussian_baseline', action=argparse.BooleanOptionalAction, default=True,
        help='For gaussian_noise / ts_gaussian_noise with missing_rate=0 on CSV datasets: build corruption '
             'via utils_data.real_data_loading (same TS noise as run_irregular_gaussian_baseline). '
             'Use --no_align_irregular_gaussian_baseline for legacy image-space or per-window TS noise.')
    parser.add_argument(
        '--gaussian_ts_init', type=str, default='img_to_ts',
        choices=['img_to_ts', 'stl_residual_denoise', 'temporal_smooth'],
        help='Initial reconstruction for gaussian_noise / ts_gaussian_noise / gaussian_blur '
             '(image-space corruptions): plain img_to_ts(y), STL residual attenuation, '
             'or temporal moving average.')
    parser.add_argument(
        '--mmps_freqgate_decomp_gaussian', action='store_true', default=False,
        help='For gaussian_noise / ts_gaussian_noise: freq-gate MMPS residual + TS decomp '
             'projection (A=I). Disabled if use_ppca_posterior is set.')
    parser.add_argument('--freqgate_projection_strength', type=float, default=0.5,
                        help='Strength of decomposition projection in freq-gate Gaussian MMPS')
    parser.add_argument('--freqgate_sigma_threshold_frac', type=float, default=0.5,
                        help='Sigma ratio threshold for projection blending')
    parser.add_argument('--freqgate_schedule', type=str, default='linear',
                        choices=['linear', 'cosine'],
                        help='Frequency gate schedule vs diffusion sigma')
    parser.add_argument(
        '--use_ambient_tweedie_mstep', action='store_true', default=False,
        help='Add Ambient DSM auxiliary loss in M-step (ICML 2024; prototype)')
    parser.add_argument('--ambient_tweedie_loss_weight', type=float, default=0.1,
                        help='Weight for Ambient DSM term in M-step')
    parser.add_argument('--ambient_tweedie_consistency_weight', type=float, default=0.0,
                        help='Weight for lightweight consistency stub in M-step')
    parser.add_argument('--blur_sigma', type=float, default=2.0,
                        help='Std of Gaussian blur kernel (for gaussian_blur corruption)')
    parser.add_argument('--blur_kernel_size', type=int, default=None,
                        help='Kernel size for Gaussian blur (auto = 4*blur_sigma+1 if None)')
    parser.add_argument('--projection_dim', type=int, default=None,
                        help='Observation dimension for random_projection (default: d_full // 2)')
    parser.add_argument('--smoothing_window', type=int, default=5,
                        help='Window size for temporal smoothing corruption (odd number)')
    parser.add_argument('--curriculum_warmup_frac', type=float, default=0.5,
                        help='Fraction of EM iters using curriculum (easy samples only)')
    parser.add_argument('--curriculum_easy_frac', type=float, default=0.7,
                        help='Fraction of easiest samples used during curriculum warmup')

    # --- DiffEM E-step variants ---
    # RePaint (run_diffem_uncond.py)
    parser.add_argument('--repaint_n_resample', type=int, default=1,
                        help='Number of resampling iterations per step (1=no resampling)')
    parser.add_argument('--repaint_jump_length', type=int, default=10,
                        help='Number of sigma steps to jump back during resampling')
    # DPS (run_diffem_dps.py)
    parser.add_argument('--dps_guidance_scale', type=float, default=1.0,
                        help='Scale factor for DPS likelihood gradient')
    parser.add_argument('--dps_sigma_y', type=float, default=0.01,
                        help='Observation noise std for DPS likelihood')
    # PiGDM (run_diffem_pigdm.py)
    parser.add_argument('--pigdm_sigma_y', type=float, default=0.01,
                        help='Observation noise std for PiGDM diagonal covariance correction')
    # CSDI baseline (run_csdi_impute_baseline.py)
    parser.add_argument('--csdi_epochs', type=int, default=200,
                        help='Number of epochs to train CSDI imputer')
    parser.add_argument('--csdi_nsample', type=int, default=1,
                        help='Number of CSDI posterior samples per sequence (1=single, >1=median)')
    # MMPS (run_diffem_mmps.py)
    parser.add_argument('--mmps_sigma_y', type=float, default=0.01,
                        help='Observation noise std for MMPS likelihood')
    parser.add_argument('--mmps_cg_iters', type=int, default=1,
                        help='Number of conjugate gradient iterations (1 typical for inpainting)')
    # Co-Evolving EM constraints (run_co_evolving_em.py, paper Sections 5.1–5.7)
    # Section 5.3: adaptive σ_y
    parser.add_argument('--sigma_y_ratio', type=float, default=0.1,
                        help='Ratio c in σ_y = c · σ_t (Prop 2)')
    parser.add_argument('--adaptive_sigma_y', action='store_true', default=True,
                        help='Use σ_y = c · σ_t instead of fixed σ_y')
    parser.add_argument('--no_adaptive_sigma_y', action='store_false', dest='adaptive_sigma_y',
                        help='Disable adaptive σ_y')
    parser.add_argument('--sigma_y_floor', type=float, default=0.0,
                        help='Minimum σ_y: σ_y = max(c·σ_t, floor)')
    # Section 5.4: manifold projection
    parser.add_argument('--consistency_projection', action='store_true', default=True,
                        help='Project onto delay-embedding manifold (Prop 1)')
    parser.add_argument('--no_consistency_projection', action='store_false', dest='consistency_projection',
                        help='Disable manifold projection')
    # Section 5.2: observation-space CG
    parser.add_argument('--obs_space_cg', action='store_true', default=True,
                        help='Run CG in observation (TS) space instead of image space (Prop 4)')
    parser.add_argument('--no_obs_space_cg', action='store_false', dest='obs_space_cg',
                        help='Disable observation-space CG (use image-space CG)')
    # Section 5.7: warm-started CG
    parser.add_argument('--warm_start_cg', action='store_true', default=True,
                        help='Initialize CG from previous reverse step solution')
    parser.add_argument('--no_warm_start_cg', action='store_false', dest='warm_start_cg',
                        help='Disable warm-started CG')
    # Section 5.6: L_obs
    parser.add_argument('--lambda_obs', type=float, default=0.5,
                        help='Weight for L_obs (observation grounding, Prop 3)')
    parser.add_argument('--no_snr_gate', action='store_true', default=False,
                        help='Disable SNR gating on L_obs/L_rep')
    parser.add_argument('--lobs_use_sm_weight', action='store_true', default=False,
                        help='Apply score-matching weight to L_obs')
    parser.add_argument('--snr_gate_sigma_d', type=float, default=None,
                        help='Override σ_d in SNR gate (default: model sigma_data=0.5)')
    # Section 5.5: L_rep
    parser.add_argument('--lambda_rep', type=float, default=0.0,
                        help='Weight for L_rep (manifold penalty in M-step)')
    # Phase 3
    parser.add_argument('--phase3', action='store_true', default=True,
                        help='Run Phase 3: fresh model on final completions')
    parser.add_argument('--no_phase3', action='store_false', dest='phase3',
                        help='Skip Phase 3')
    parser.add_argument('--eval_all_metrics', action='store_true', default=False,
                        help='Compute all metrics at final EM iteration')
    parser.add_argument('--gmrf_lambda', type=float, default=0.1,
                        help='GMRF smoothness prior strength (discrete Laplacian in TS space, 0 = disabled)')
    parser.add_argument('--spectral_filter_order', type=int, default=2,
                        help='Butterworth order for spectral MMPS variants '
                             '(run_diffem_mmps_spectral_post.py, run_diffem_mmps_spectral_cg.py)')
    parser.add_argument('--bootstrap_method', type=str, default='irregular',
                        choices=['irregular', 'gaussian'],
                        help='Bootstrap strategy before MMPS EM in run_diffem_mmps.py')
    parser.add_argument('--bootstrap_pretrain_epochs', type=int, default=None,
                        help='Observed-only TST pretrain epochs for the irregular MMPS bootstrap '
                             '(default: reuse first_epoch)')
    parser.add_argument('--bootstrap_diffusion_epochs', type=int, default=None,
                        help='Masked diffusion warm-start epochs for the irregular MMPS bootstrap '
                             '(default: reuse epochs, else m_step_epochs)')
    parser.add_argument(
        '--eval_bootstrap',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Evaluate the bootstrap model before the first EM iteration to track disc_mean trajectory',
    )
    # Decomposition-enhanced variant (run_diffem_mmps_decomposed.py)
    parser.add_argument('--stl_period', type=int, default=None,
                        help='Period for STL decomposition (auto-detected if None)')
    parser.add_argument('--lambda_trend', type=float, default=0.1,
                        help='Weight for trend total-variation smoothness penalty')
    parser.add_argument('--lambda_spectral', type=float, default=0.05,
                        help='Weight for seasonal spectral matching penalty')
    parser.add_argument('--use_component_loss', type=bool, default=True,
                        help='Use component-aware loss (trend TV + spectral) in M-step')

    # --- AmbientEM Configuration ---
    parser.add_argument('--ambient_em', action='store_true', default=False,
                        help='Enable AmbientEM training (DiffEM + Ambient-Omni aware unconditional training)')
    parser.add_argument('--ambient_alpha', type=float, default=0.7,
                        help='Fraction of EM reconstructions vs corrupted data per batch in unconditional training')
    parser.add_argument('--ambient_delta', type=float, default=0.3,
                        help='Further corruption probability for ambient masked loss')
    parser.add_argument('--uncond_eval_epochs', type=int, default=100,
                        help='Quick unconditional training epochs for per-EM-iteration disc_mean eval')
    parser.add_argument('--use_ppca_init', action='store_true', default=True,
                        help='Use PPCA-based Gaussian initialisation for EM')
    parser.add_argument('--observation_consistency', action='store_true', default=True,
                        help='Enforce observation consistency after E-step sampling')
    parser.add_argument('--e_step_sample_steps', type=int, default=64,
                        help='Number of diffusion steps for E-step sampling')
    parser.add_argument('--cond_em_iters', type=int, default=None,
                        help='Number of conditional EM iterations (run_diffem_cond_mmps_init.py)')
    parser.add_argument('--cond_m_step_epochs', type=int, default=None,
                        help='Epochs per conditional M-step (run_diffem_cond_mmps_init.py)')
    parser.add_argument('--mmps_bootstrap_iters', type=int, default=None,
                        help='Number of MMPS bootstrap EM iterations')
    parser.add_argument('--mmps_bootstrap_epochs', type=int, default=None,
                        help='Epochs per MMPS bootstrap M-step')
    parser.add_argument('--ppca_rank', type=int, default=32,
                        help='Rank for PPCA initialisation')
    parser.add_argument('--ppca_iters', type=int, default=8,
                        help='Number of PPCA EM iterations for initialisation')

    # --- Enhanced MMPS for high corruption (run_diffem_mmps_general_corruption.py) ---
    parser.add_argument('--use_ppca_posterior', action='store_true', default=False,
                        help='Use PPCA covariance in MMPS posterior denoiser (stabilises high corruption)')
    parser.add_argument('--ppca_posterior_rank', type=int, default=32,
                        help='Rank for PPCA covariance in posterior denoiser')
    parser.add_argument('--sigma_y_anneal', action='store_true', default=False,
                        help='Anneal sigma_y from sigma_y_start to sigma_y_end over EM iterations')
    parser.add_argument('--sigma_y_start', type=float, default=0.1,
                        help='Starting sigma_y for annealing (high = diffuse posterior)')
    parser.add_argument('--sigma_y_end', type=float, default=0.01,
                        help='Ending sigma_y for annealing (low = tight data fit)')
    parser.add_argument('--obs_consistency_mmps', action='store_true', default=False,
                        help='Overwrite observed positions with true values after MMPS E-step')

    # --- Component-wise sigma_y (run_diffem_mmps_component_sigma.py) ---
    parser.add_argument('--sigma_y_trend', type=float, default=0.005,
                        help='Observation noise for trend component (tight = reliable)')
    parser.add_argument('--sigma_y_seasonal', type=float, default=0.01,
                        help='Observation noise for seasonal component (medium)')
    parser.add_argument('--sigma_y_residual', type=float, default=0.05,
                        help='Observation noise for residual component (loose = noisy)')

    # --- Gibbs decomposition (run_diffem_mmps_gibbs_decomp.py) ---
    parser.add_argument('--gibbs_synthetic_confidence', type=float, default=0.3,
                        help='Confidence weight for synthetic trend+seasonal observations in Gibbs Pass 2')

    # --- TimeMAE-inspired E-step experiments ---
    # Codebook-regularized E-step (run_diffem_mmps_codebook_estep.py)
    parser.add_argument('--codebook_patch_size', type=int, default=4,
                        help='Sub-series patch length for codebook (TimeMAE window sigma)')
    parser.add_argument('--codebook_n_codes', type=int, default=256,
                        help='Number of codebook entries K (TimeMAE vocab size)')
    parser.add_argument('--codebook_blend_start', type=float, default=0.4,
                        help='Codebook blend strength at EM iter 0')
    parser.add_argument('--codebook_blend_end', type=float, default=0.05,
                        help='Codebook blend strength at final EM iter')
    # Momentum-teacher stabilized E-step (run_diffem_mmps_momentum_teacher.py)
    parser.add_argument('--teacher_momentum', type=float, default=0.95,
                        help='Momentum coefficient eta for teacher update (TimeMAE uses 0.99)')
    parser.add_argument('--teacher_blend_start', type=float, default=0.3,
                        help='Teacher blend weight at EM iter 0')
    parser.add_argument('--teacher_blend_end', type=float, default=0.0,
                        help='Teacher blend weight at final EM iter')

    # --- Multi-sample E-step ---
    parser.add_argument('--estep_n_samples', type=int, default=3,
                        help='Number of MMPS samples to average per E-step (variance reduction)')

    # --- Spectral MMPS (Proposals A/B/C) ---
    parser.add_argument('--spectral_n_probes', type=int, default=1,
                        help='Number of Hutchinson probes for PSD estimation in spectral MMPS')
    parser.add_argument('--spectral_cg_fallback', action='store_true', default=False,
                        help='Use standard CG instead of FFT solve in spectral Gaussian MMPS (ablation)')
    parser.add_argument('--spectral_no_precond', action='store_true', default=False,
                        help='Disable spectral preconditioner in preconditioned CG MMPS (ablation)')
    parser.add_argument('--bandlimited_snr_threshold', type=float, default=1.0,
                        help='SNR threshold for adaptive frequency mask in band-limited MMPS')
    parser.add_argument('--bandlimited_min_cutoff', type=float, default=0.1,
                        help='Minimum frequency cutoff fraction in band-limited MMPS')
    parser.add_argument('--warmstart_epochs', type=int, default=200,
                        help='Number of warmstart training epochs before EM loop')
    parser.add_argument('--phase3_epochs', type=int, default=200,
                        help='Number of Phase 3 continued training epochs after EM')

    # --- Agentic EM (v3) ---
    parser.add_argument('--agentic_em', action='store_true', default=False,
                        help='Enable agentic EM loop (adaptive spectral filter, uniform M-step, early stopping)')
    parser.add_argument('--adaptive_f_cutoff_init', type=float, default=0.5,
                        help='Initial f_cutoff_base before any spectral diagnosis')
    parser.add_argument('--spectral_diag_threshold', type=float, default=0.3,
                        help='Raw CV threshold for determining reliable frequency bands')
    parser.add_argument('--em_patience', type=int, default=2,
                        help='Early stopping patience: stop EM if disc_mean does not improve for this many evals')

    # --- Initialization method ---
    parser.add_argument('--init_method', type=str, default='stl',
                        choices=['stl', 'kalman', 'linear', 'random'],
                        help='Initialization method for EM: stl (iterative STL decomposition), '
                             'kalman (Kalman filter), linear (linear interpolation), '
                             'random (Gaussian fill from observed stats)')

    # --- Curriculum and scaling ---
    parser.add_argument('--curriculum_reveal_max', type=float, default=0.3,
                        help='Max fraction of missing positions revealed in curriculum (increase for high missing rates)')
    parser.add_argument('--kalman_fit_timeout', type=int, default=5,
                        help='Per-series Kalman fit timeout in seconds (increase for long sequences)')
    parser.add_argument('--kalman_global_timeout', type=int, default=1800,
                        help='Global Kalman init budget in seconds')

    # --- Early stopping ---
    parser.add_argument('--early_stop_patience', type=int, default=20,
                        help='Epochs with no loss improvement before stopping M-step/uncond training')
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-4,
                        help='Minimum loss decrease to count as improvement')

    # --- Warm-start DiffEM (Phase 2: conditional refinement after MMPS) ---
    parser.add_argument('--warmstart_iters', type=int, default=5,
                        help='Number of conditional DiffEM iterations after MMPS converges')
    parser.add_argument('--warmstart_cond_epochs', type=int, default=100,
                        help='Epochs to train the conditional model per warm-start iteration')
    parser.add_argument('--warmstart_cond_lr', type=float, default=None,
                        help='Learning rate for conditional model (default: same as learning_rate)')
    parser.add_argument('--warmstart_stop_on_degradation', action='store_true', default=True,
                        help='Stop Phase 2 if disc_mean gets worse than best MMPS result')

    # --- Ambient-MMPS (run_diffem_ambient_mmps.py) ---
    parser.add_argument('--ambient_pretrain_epochs', type=int, default=100,
                        help='Epochs for Phase 0 ambient pre-training on corrupted observations')
    parser.add_argument('--lambda_obs_start', type=float, default=1.0,
                        help='Initial weight for observation-space loss in M-step (1.0 = obs only)')
    parser.add_argument('--lambda_obs_end', type=float, default=0.3,
                        help='Final weight for observation-space loss in M-step')
    parser.add_argument('--further_corrupt_delta', type=float, default=0.1,
                        help='Probability of dropping an observed position in Ambient further corruption')
    parser.add_argument('--ambient_delta_start', type=float, default=0.3,
                        help='Initial delta for ambient-anneal experiments')
    parser.add_argument('--ambient_delta_end', type=float, default=0.05,
                        help='Final delta for ambient-anneal experiments')
    parser.add_argument(
        '--ambient_concat_further_mask',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Concatenate further-corruption mask as an extra UNet channel (Ambient Diffusion paper). '
             'Controlled from CLI for parity/ablation experiments.',
    )
    parser.add_argument('--temporal_block_prob', type=float, default=0.1,
                        help='Probability that an observed timestep becomes a temporal-block center')
    parser.add_argument('--temporal_block_width', type=int, default=4,
                        help='Width of temporal block corruption for block-ambient experiments')
    parser.add_argument('--trust_schedule', type=str, default='linear',
                        choices=['linear', 'cosine'],
                        help='Interpolation schedule for trust-related start/end hyperparameters')
    parser.add_argument('--trust_mstep_mode', type=str, default='soft_loss',
                        choices=['soft_loss', 'input_dropout', 'two_branch'],
                        help='Trust-aware M-step variant for hidden-distribution experiments')
    parser.add_argument('--imputed_loss_weight_start', type=float, default=0.0,
                        help='Initial supervision weight on imputed pixels during trust-aware M-step')
    parser.add_argument('--imputed_loss_weight_end', type=float, default=0.1,
                        help='Final supervision weight on imputed pixels during trust-aware M-step')
    parser.add_argument('--imputed_keep_ratio_start', type=float, default=1.0,
                        help='Initial probability of keeping imputed pixels visible in the UNet input')
    parser.add_argument('--imputed_keep_ratio_end', type=float, default=1.0,
                        help='Final probability of keeping imputed pixels visible in the UNet input')
    parser.add_argument('--observed_keep_ratio_start', type=float, default=1.0,
                        help='Initial probability of keeping observed pixels visible in conservative trust branches')
    parser.add_argument('--observed_keep_ratio_end', type=float, default=1.0,
                        help='Final probability of keeping observed pixels visible in conservative trust branches')
    parser.add_argument('--full_context_prob_start', type=float, default=0.2,
                        help='Initial probability of using the full-context branch in two-branch trust training')
    parser.add_argument('--full_context_prob_end', type=float, default=0.8,
                        help='Final probability of using the full-context branch in two-branch trust training')

    # --- Fast mode (overrides multiple settings for quicker iteration) ---
    parser.add_argument('--fast_mode', action='store_true', default=False,
                        help='Quick iteration mode: m_step_epochs=60, uncond_eval_epochs=50, '
                             'e_step_sample_steps=32, em_iters=10, patience=15')

    # --- logging ---s
    parser.add_argument('--logging_iter', type=int, default=10, help='number of iterations between logging')
    parser.add_argument('--percent', type=int, default=100)
    parsed_args = parser.parse_args()

    # load config file
    config = OmegaConf.to_object(OmegaConf.load(parsed_args.config))
    # override config file with command line args
    for k, v in vars(parsed_args).items():
        if v is None:
            setattr(parsed_args, k, config.get(k, None))
    # add to the parsed args, configs that are not in the parsed args but do in the config file
    # this is needed since multiple config files setups may be used
    for k, v in config.items():
        if k not in vars(parsed_args):
            setattr(parsed_args, k, v)

    parsed_args.input_size = parsed_args.input_channels

    if getattr(parsed_args, 'fast_mode', False):
        fast_overrides = dict(
            m_step_epochs=60,
            uncond_eval_epochs=50,
            uncond_epochs_per_iter=50,
            e_step_sample_steps=32,
            em_iters=10,
            early_stop_patience=15,
        )
        for k, v in fast_overrides.items():
            if k not in [a.lstrip('-') for a in sys.argv]:
                setattr(parsed_args, k, v)
        print(f"[fast_mode] overrides applied: {fast_overrides}")

    # TF32: optional only (--allow_tf32). Unconditional TF32 caused CUDA "illegal instruction"
    # in cudnn conv2d on some cluster RTX 6000 nodes (see run_diffem_ambient_mmps Phase 0).
    if torch.cuda.is_available():
        use_tf32 = getattr(parsed_args, 'allow_tf32', False)
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32

    return parsed_args
