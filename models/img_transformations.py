from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import math

class TsImgEmbedder(ABC):
    """
    Abstract class for transforming time series to images and vice versa
    """

    def __init__(self, device, seq_len):
        self.device = device
        self.seq_len = seq_len

    @abstractmethod
    def ts_to_img(self, signal):
        """

        Args:
            signal: given time series

        Returns:
            image representation of the signal

        """
        pass

    @abstractmethod
    def img_to_ts(self, img):
        """

        Args:
            img: given generated image

        Returns:
            time series representation of the generated image
        """
        pass

    def ts_to_img_adjoint(self, x_img):
        """Exact transpose of ts_to_img.

        Default: fall back to img_to_ts (correct up to a scalar for delay
        embedders since their img_to_ts averages over duplicate positions,
        which is proportional to the transpose). Subclasses with a non-local
        linear lift (e.g. STFTEmbedder) override this with an autograd-based
        VJP that is the exact transpose.
        """
        return self.img_to_ts(x_img)

    def img_to_ts_adjoint(self, v_ts):
        """Exact transpose of img_to_ts.

        For a linear lift L, img_to_ts is L^{-1} (left-inverse). Its adjoint
        L^{-T} is DIFFERENT from L (= ts_to_img) whenever L is non-unitary,
        which is true for STFT with Hann window + per-bin scaling. For the
        delay embedder L is approximately unitary (orthogonal placements +
        average pooling), so ts_to_img is an acceptable approximation and
        this method falls back to it by default.

        Subclasses with a strongly non-unitary lift (STFTEmbedder) should
        override with an exact autograd-based VJP of img_to_ts.
        """
        return self.ts_to_img(v_ts)

    @property
    def pad_mask(self):
        """Binary mask marking valid (non-pad) pixels in ts_to_img output.

        Shape: (1, C, H, W) with 1 inside the native image region and 0
        in the zero-padded border. Default None means no padding is applied
        (or the entire image is structurally valid), which is the case for
        the delay embedder. Subclasses with structural padding (STFT) override
        this to return an explicit mask.
        """
        return None


class DelayEmbedder(TsImgEmbedder):
    """
    Delay embedding transformation
    """

    def __init__(self, device, seq_len, delay, embedding, batch_size, num_features):
        super().__init__(device, seq_len)
        self.device = device
        self.seq_len = seq_len
        self.delay = delay
        self.embedding = embedding
        self.batch_size = batch_size
        self.num_features = num_features
        self.img_shape = None
        self.mapping = None  # Mapping from TS positions to image positions

        # Create the helper series and image to build the mapping
        self.create_mapping()


    def pad_to_square(self, x, mask=0):
        """
        Pads the input tensor x to make it square along the last two dimensions.
        """
        _, _, cols, rows = x.shape
        max_side = max(cols, rows)
        padding = (
            0, max_side - rows, 0, max_side - cols)  # Padding format: (pad_left, pad_right, pad_top, pad_bottom)

        # Padding the last two dimensions to make them square
        x_padded = torch.nn.functional.pad(x, padding, mode='constant', value=mask)
        return x_padded

    def unpad(self, x, original_shape):
        """
        Removes the padding from the tensor x to get back to its original shape.
        """
        _, _, original_cols, original_rows = original_shape
        return x[:, :, :original_cols, :original_rows]

    def ts_to_img(self, signal, pad=True, mask=0):

        batch, length, features = signal.shape
        #  if our sequences are of different lengths, this can happen with physionet and climate datasets
        if self.seq_len != length:
            self.seq_len = length

        x_image = torch.zeros((batch, features, self.embedding, self.embedding),
                              dtype=signal.dtype)
        i = 0
        while (i * self.delay + self.embedding) <= self.seq_len:
            start = i * self.delay
            end = start + self.embedding
            x_image[:, :, :, i] = signal[:, start:end].permute(0, 2, 1)
            i += 1

        ### SPECIAL CASE
        if i * self.delay != self.seq_len and i * self.delay + self.embedding > self.seq_len:
            start = i * self.delay
            end = signal[:, start:].permute(0, 2, 1).shape[-1]
            x_image[:, :, :end, i] = signal[:, start:].permute(0, 2, 1)
            i += 1

        # cache the shape of the image before padding
        self.img_shape = (batch, features, self.embedding, i)
        x_image = x_image.to(self.device)[:, :, :, :i]

        if pad:
            x_image = self.pad_to_square(x_image, mask)

        return x_image


    def create_mapping(self):
        """
        Creates the mapping from time series positions to image positions.
        """
        # Create the helper series
        helper_series = torch.arange(1, self.seq_len + 1, dtype=torch.float32, device=self.device)
        helper_series = helper_series.unsqueeze(0).unsqueeze(-1)  # Shape: (1, seq_len, 1)
        helper_series = helper_series.repeat(self.batch_size, 1, self.num_features)  # Shape: (batch_size, seq_len, num_features)

        # Convert the helper series to image
        helper_image = self.ts_to_img(helper_series, pad=True, mask=0)

        # Unpad the helper image to get the original shape
        helper_image_non_square = self.unpad(helper_image, self.img_shape)
        batch_size, channels, rows, cols = helper_image_non_square.shape

        # Create the mapping
        self.mapping = {}
        # Since the helper series contains unique values from 1 to seq_len, and is the same across batch and features,
        # we can use the first sample and first feature to create the mapping
        for row in range(rows):
            for col in range(cols):
                val = helper_image_non_square[0, 0, row, col].item()
                if val != 0:  # Ignore padding or mask values
                    ts_idx = int(val) - 1  # Convert value back to TS index (0-based)
                    if ts_idx not in self.mapping:
                        self.mapping[ts_idx] = []
                    self.mapping[ts_idx].append((row, col))

    def img_to_ts(self, x_image_square):
        """
        Reconstructs the time series from its image representation.
        """
        # Unpad the image to get back to original shape
        x_image_non_square = self.unpad(x_image_square, self.img_shape)
        batch_size, channels, rows, cols = x_image_non_square.shape

        # Initialize the reconstructed TS tensor
        reconstructed_ts = torch.zeros((batch_size, self.seq_len, channels),
                                       device=self.device,
                                       dtype=x_image_non_square.dtype)

        # Use the mapping to reconstruct the TS
        for ts_idx in range(self.seq_len):
            image_positions = self.mapping.get(ts_idx, [])
            if not image_positions:
                continue  # Skip if there are no corresponding image positions

            values = []
            for row, col in image_positions:
                value = x_image_non_square[:, :, row, col]
                values.append(value)

            # Stack and average the values
            values_tensor = torch.stack(values, dim=0)
            mean_values = values_tensor.mean(dim=0)
            reconstructed_ts[:, ts_idx, :] = mean_values

        # Permute to get shape (batch_size, seq_len, channels)
        reconstructed_ts = reconstructed_ts.permute(0, 1, 2)

        return reconstructed_ts.cuda()


class SpectrogramEmbedder(TsImgEmbedder):
    """
    STFT-based spectrogram transform: (B, T, C) -> (B, C, H, W).

    Computes per-feature STFT magnitude, then resizes to a fixed
    (img_resolution x img_resolution) image via bilinear interpolation.
    Inverse uses Griffin-Lim.
    """

    def __init__(self, device, seq_len, n_fft=8, hop_length=3,
                 img_resolution=8, n_griffin_lim_iters=32):
        super().__init__(device, seq_len)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.img_resolution = img_resolution
        self.n_griffin_lim_iters = n_griffin_lim_iters
        self.window = torch.hann_window(n_fft, device=device)

        freq_bins = n_fft // 2 + 1
        n_frames = (seq_len - n_fft) // hop_length + 1
        if (seq_len - n_fft) % hop_length != 0:
            n_frames += 1
        self.raw_shape = (freq_bins, n_frames)

    def ts_to_img(self, signal, pad=True, mask=0):
        """
        Args:
            signal: (B, T, C) tensor
        Returns:
            (B, C, img_resolution, img_resolution) magnitude spectrogram
        """
        B, T, C = signal.shape
        device = signal.device
        window = self.window.to(device)

        signal_flat = signal.permute(0, 2, 1).reshape(B * C, T)

        spec = torch.stft(
            signal_flat, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window=window, return_complex=True,
            center=False,
        )
        mag = spec.abs()  # (B*C, freq_bins, n_frames)

        mag = mag.unsqueeze(1)  # (B*C, 1, freq, time)
        mag_resized = F.interpolate(
            mag,
            size=(self.img_resolution, self.img_resolution),
            mode='bilinear', align_corners=False,
        )
        mag_resized = mag_resized.squeeze(1)  # (B*C, H, W)

        return mag_resized.reshape(B, C, self.img_resolution, self.img_resolution)

    def img_to_ts(self, img):
        """
        Approximate inverse via Griffin-Lim on the resized magnitude.

        Args:
            img: (B, C, H, W) — magnitude spectrogram image
        Returns:
            (B, T, C) time series
        """
        B, C, H, W = img.shape
        device = img.device
        window = self.window.to(device)

        mag_resized = img.reshape(B * C, 1, H, W)
        mag = F.interpolate(
            mag_resized,
            size=self.raw_shape,
            mode='bilinear', align_corners=False,
        ).squeeze(1)  # (B*C, freq, time)

        # Griffin-Lim
        angles = torch.randn_like(mag) * 2 * math.pi
        for _ in range(self.n_griffin_lim_iters):
            complex_spec = mag * torch.exp(1j * angles)
            signal_est = torch.istft(
                complex_spec, n_fft=self.n_fft,
                hop_length=self.hop_length, win_length=self.n_fft,
                window=window, center=False, length=self.seq_len,
            )
            re_spec = torch.stft(
                signal_est, n_fft=self.n_fft,
                hop_length=self.hop_length, win_length=self.n_fft,
                window=window, return_complex=True, center=False,
            )
            angles = re_spec.angle()

        complex_spec = mag * torch.exp(1j * angles)
        signal_out = torch.istft(
            complex_spec, n_fft=self.n_fft,
            hop_length=self.hop_length, win_length=self.n_fft,
            window=window, center=False, length=self.seq_len,
        )  # (B*C, T)

        return signal_out.reshape(B, C, self.seq_len).permute(0, 2, 1).to(device)


class STFTEmbedder(TsImgEmbedder):
    """
    Complex STFT transformation storing real and imaginary parts as channels.

    Unlike SpectrogramEmbedder (magnitude-only, lossy Griffin-Lim inverse),
    this embedder preserves phase, making ts_to_img / img_to_ts a
    near-invertible linear pair: ISTFT exactly inverts STFT when the window
    satisfies COLA.  This is the linear lift L used in Theorem 1 (Section 3.6)
    for the STFT validation experiment (Section 7.10).

    Input:  (B, T, F)  -- multivariate time series
    Output: (B, 2F, freq_bins, n_frames) -- real and imag channels stacked
    """

    def __init__(self, device, seq_len, n_fft, hop_length, img_resolution=None,
                 scale_mode='zscore', pad_mode='reflect',
                 global_rescale_enabled=False, target_pixel_std=0.5,
                 dead_bin_quantile=0.0):
        super().__init__(device, seq_len)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.img_resolution = img_resolution
        # scale_mode: 'zscore' (per-bin std) or 'absmax' (per-bin max abs).
        # Both are strictly multiplicative (y = x / scale), preserving linearity
        # of L and the CG adjoint identity <Gx, y> = <x, G^T y>. zscore makes
        # the image have pixel std ~= sigma_data = 0.5, matching EDM's noise
        # schedule. absmax compresses std to ~0.15, which causes the denoiser
        # to collapse to the trivial zero function (Section 7.10 analysis).
        self.scale_mode = scale_mode
        # pad_mode: 'reflect' (native 7x7 -> 8x8 reflection padding, all
        # pixels data-bearing) or 'zero' (native 5x7 -> 8x8 with deterministic
        # zeros in 45% of the image, kept for backward-compat). Both are
        # linear and have exact autograd-VJP adjoints.
        self.pad_mode = pad_mode
        # Cached per-bin scales (std under zscore, abs-max under absmax).
        self.scale_real = None
        self.scale_imag = None
        self.img_shape = None
        self._pad_mask_cache = None
        self._pad_mask_shape = None

        # Fix 2: scalar global rescale. After the per-bin zscore, the aggregate
        # pixel std on TRAINING data can still be ~0.9 (short L) to ~53 (long L)
        # because a few high-energy bins dominate the mix. A single scalar
        # folded INTO scale_real/scale_imag commutes through every inner
        # product, preserving linearity and the adjoint identity exactly, and
        # lands pixel_std at `target_pixel_std` (EDM's sigma_data=0.5).
        self.global_rescale_enabled = global_rescale_enabled
        self.target_pixel_std = target_pixel_std
        self.global_scale = 1.0
        # Fix 3: replace the hard rms<1e-5 dead-bin floor with a quantile floor
        # so CG condition number stays <1e4 instead of drifting to >1e8 at
        # long sequences. Value 0.0 preserves the legacy hard-floor behaviour.
        self.dead_bin_quantile = float(dead_bin_quantile)

    @staticmethod
    def pad_to_square(x, target_side, pad_mode='zero', mask=0):
        """Pad (B, C, H, W) to (B, C, target_side, target_side).

        pad_mode:
            'zero'    -> constant padding with value `mask` (default 0).
                         Deterministic zeros; breaks when the pad fraction is
                         large (~45% for 5x7 -> 8x8) because the U-Net learns
                         to predict zero in the pad region and the bias leaks
                         into the live region.
            'reflect' -> reflection padding, which is linear and keeps every
                         pixel data-bearing. Requires h >= 2 and w >= 2 along
                         each padded axis (always true here).

        H and W must both be <= target_side.
        """
        _, _, h, w = x.shape
        pad_h = target_side - h
        pad_w = target_side - w
        if pad_h < 0 or pad_w < 0:
            raise ValueError(
                f"STFT output ({h}x{w}) exceeds img_resolution={target_side}; "
                "increase img_resolution or adjust stft_n_fft / stft_hop_length."
            )
        if pad_mode == 'reflect':
            return torch.nn.functional.pad(
                x, (0, pad_w, 0, pad_h), mode='reflect',
            )
        return torch.nn.functional.pad(
            x, (0, pad_w, 0, pad_h), mode='constant', value=mask,
        )

    @staticmethod
    def unpad(x, original_shape):
        _, _, orig_h, orig_w = original_shape
        return x[:, :, :orig_h, :orig_w]

    def cache_min_max_params(self, train_data, chunk_size=512):
        """Cache per-bin scales for ts_to_img / img_to_ts.

        Dispatches on `self.scale_mode`:
          - 'zscore' -> per-bin standard deviation (see cache_zscore_params);
                         produces image pixel std close to 1, so with
                         `sigma_data = 0.5` EDM's noise schedule is balanced.
          - 'absmax' -> per-bin abs-max (legacy behaviour used by the original
                         STFT validation runs; image std ~= 0.15 which causes
                         the denoiser to collapse to zero).

        All downstream code uses `self.scale_real` / `self.scale_imag`
        identically regardless of mode, so the lift remains strictly linear
        in both cases.
        """
        if self.scale_mode == 'zscore':
            return self.cache_zscore_params(train_data, chunk_size=chunk_size)
        return self._cache_absmax_params(train_data, chunk_size=chunk_size)

    def _cache_absmax_params(self, train_data, chunk_size=512):
        """Legacy per-bin ABS-MAX scaling (kept for backward compatibility).

        See cache_min_max_params' docstring for why this collapses the
        denoiser when used with sigma_data = 0.5.
        """
        train_data = train_data.detach().cpu()

        n = train_data.shape[0]
        abs_max_r = abs_max_i = None
        for start in range(0, n, chunk_size):
            chunk = train_data[start:start + chunk_size]
            real, imag = self._stft_transform(chunk)
            cur_abs_r = real.abs().amax(dim=0, keepdim=True)
            cur_abs_i = imag.abs().amax(dim=0, keepdim=True)
            if abs_max_r is None:
                abs_max_r, abs_max_i = cur_abs_r, cur_abs_i
            else:
                abs_max_r = torch.maximum(abs_max_r, cur_abs_r)
                abs_max_i = torch.maximum(abs_max_i, cur_abs_i)

        # Dead bins (structurally zero: imag of DC/Nyquist for real input)
        # get scale=1.0 so y = x / 1.0 doesn't amplify fp32 noise.
        dead_r = abs_max_r < 1e-5
        dead_i = abs_max_i < 1e-5
        abs_max_r = torch.where(dead_r, torch.ones_like(abs_max_r), abs_max_r)
        abs_max_i = torch.where(dead_i, torch.ones_like(abs_max_i), abs_max_i)

        self.scale_real = abs_max_r.to(self.device)
        self.scale_imag = abs_max_i.to(self.device)

    def cache_zscore_params(self, train_data, chunk_size=512):
        """Cache per-bin ROOT-MEAN-SQUARE (RMS) scaling factors.

        Linear scaling:
            y = x / rms      (forward)
            x = y * rms      (inverse)
        where rms = sqrt(E[x^2]) over the training set per (feature, freq_bin,
        frame). For STFT of zero-mean time series, RMS equals std; we use the
        second-moment form to avoid a mean subtraction and keep the lift L(0)=0
        strictly linear so the CG adjoint identity holds exactly.

        The resulting normalized image has per-bin std ~= 1 on train data,
        and aggregated pixel std ~= 0.5-0.7 across the image - matching EDM's
        `sigma_data = 0.5` so the denoising objective is non-trivial.

        When `self.global_rescale_enabled` (Fix 2 in the STFT-EM plan) is on,
        a single scalar `global_scale = measured_pixel_std / target_pixel_std`
        is folded INTO scale_real / scale_imag so that the image pixel std
        measured on the actual training data (not a random probe) matches
        EDM's `sigma_data = 0.5`. Because it's a scalar multiplier on both
        forward and adjoint paths, it commutes through every inner product
        and preserves the CG adjoint identity exactly.

        When `self.dead_bin_quantile > 0` (Fix 3) we floor tiny RMS values
        at `q * median(rms)` instead of the hard `1e-5` threshold, keeping
        the CG condition number bounded for long-L/high-resolution tiles.

        Args:
            train_data: (B, T, F) training time series tensor.
                        Call once before the training loop starts.
            chunk_size: process the input in chunks of this many sequences
                        to avoid `CUDA error: invalid configuration argument`
                        from torch.stft on very large batches. On such a
                        CUDA error we halve the chunk size and retry (Fix 4).
        """
        train_data = train_data.detach().cpu()

        n = train_data.shape[0]
        sum_sq_r = sum_sq_i = None
        count = 0
        # Fix 4: wrap the STFT loop with a halving retry so we never crash
        # the whole EM run on a CUDA kernel launch configuration error.
        min_chunk = 4
        cur_chunk = int(chunk_size)
        start = 0
        while start < n:
            chunk = train_data[start:start + cur_chunk]
            try:
                real, imag = self._stft_transform(chunk)
            except RuntimeError as exc:
                msg = str(exc)
                if ('CUDA error' in msg or 'invalid configuration' in msg or
                        'out of memory' in msg.lower()) and cur_chunk > min_chunk:
                    cur_chunk = max(min_chunk, cur_chunk // 2)
                    print(f'[STFT.cache] CUDA/STFT error ({msg.splitlines()[0]!r}); '
                          f'halving chunk_size -> {cur_chunk} and retrying.')
                    continue
                raise
            sq_r = (real * real).sum(dim=0, keepdim=True)
            sq_i = (imag * imag).sum(dim=0, keepdim=True)
            if sum_sq_r is None:
                sum_sq_r, sum_sq_i = sq_r, sq_i
            else:
                sum_sq_r = sum_sq_r + sq_r
                sum_sq_i = sum_sq_i + sq_i
            count += chunk.shape[0]
            start += cur_chunk

        rms_r = torch.sqrt(sum_sq_r / max(count, 1))
        rms_i = torch.sqrt(sum_sq_i / max(count, 1))

        # Fix 3: quantile floor. Structurally-dead bins (imag of DC/Nyquist)
        # still hit the hard 1e-5 floor, but any "nearly dead" bin whose
        # rms is far below the median of live bins is raised to
        # `q * median(rms_live)` to cap the CG condition number.
        hard_dead_r = rms_r < 1e-5
        hard_dead_i = rms_i < 1e-5
        rms_r = torch.where(hard_dead_r, torch.ones_like(rms_r), rms_r)
        rms_i = torch.where(hard_dead_i, torch.ones_like(rms_i), rms_i)
        if self.dead_bin_quantile > 0.0:
            live_r = rms_r[~hard_dead_r]
            live_i = rms_i[~hard_dead_i]
            if live_r.numel() > 0:
                floor_r = self.dead_bin_quantile * live_r.median()
                rms_r = torch.where(rms_r < floor_r,
                                    torch.full_like(rms_r, float(floor_r)),
                                    rms_r)
            if live_i.numel() > 0:
                floor_i = self.dead_bin_quantile * live_i.median()
                rms_i = torch.where(rms_i < floor_i,
                                    torch.full_like(rms_i, float(floor_i)),
                                    rms_i)

        self.scale_real = rms_r.to(self.device)
        self.scale_imag = rms_i.to(self.device)
        # Reset the global scale so the measurement below is the post-per-bin,
        # pre-global value. Without this line, subsequent re-calls would
        # compound the scalar.
        self.global_scale = 1.0

        # Fix 2: measure the aggregate pixel std on training data (capped
        # subsample for speed), then fold `pixel_std / target_pixel_std` into
        # the per-bin scales. `ts_to_img` divides by `scale_real*global_scale`
        # so the output pixel std becomes exactly `target_pixel_std`.
        if self.global_rescale_enabled:
            probe_n = min(n, 256)
            probe = train_data[:probe_n].to(self.device)
            with torch.no_grad():
                img_probe = self.ts_to_img(probe)
                measured = img_probe.std().item()
            if measured > 1e-8 and self.target_pixel_std > 1e-8:
                self.global_scale = float(measured / self.target_pixel_std)
                self.scale_real = self.scale_real * self.global_scale
                self.scale_imag = self.scale_imag * self.global_scale
                print(f'[STFT.cache] global_rescale: pixel_std {measured:.4f} '
                      f'-> target {self.target_pixel_std:.4f}, '
                      f'global_scale={self.global_scale:.4f}')

    def _stft_transform(self, data):
        """Raw complex STFT, returning real and imaginary tensors.

        Args:
            data: (B, T, F)
        Returns:
            real: (B, F, freq_bins, n_frames)
            imag: (B, F, freq_bins, n_frames)
        """
        data = data.permute(0, 2, 1)  # (B, F, T)
        spec = T.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length,
            center=True, power=None,
        ).to(data.device)
        transformed = spec(data)  # (B, F, freq_bins, n_frames) complex
        return transformed.real, transformed.imag

    @staticmethod
    def _linear_scale(x, scale):
        """Pure multiplicative scaling: y = x / scale.  Linear in x."""
        return x / scale

    @staticmethod
    def _linear_unscale(x, scale):
        """Inverse of _linear_scale: x = y * scale.  Linear in y."""
        return x * scale

    def ts_to_img(self, signal, pad=True, mask=0):
        """
        Args:
            signal: (B, T, F)
            pad:    if True and img_resolution is set, pad to a square image
            mask:   fill value for constant padding (only used when
                    self.pad_mode == 'zero'). Default 0.
        Returns:
            (B, 2F, H, W) -- real and imag channels concatenated.
            If pad=True and img_resolution is set, H = W = img_resolution.
        """
        assert self.scale_real is not None, (
            "Call cache_min_max_params(train_data) before using ts_to_img"
        )
        real, imag = self._stft_transform(signal)
        real = self._linear_scale(real, self.scale_real.to(signal.device))
        imag = self._linear_scale(imag, self.scale_imag.to(signal.device))
        stft_out = torch.cat((real, imag), dim=1)

        self.img_shape = tuple(stft_out.shape)

        if pad and self.img_resolution is not None:
            stft_out = self.pad_to_square(
                stft_out, self.img_resolution,
                pad_mode=self.pad_mode, mask=mask,
            )

        return stft_out

    def _expected_spec_shape(self):
        """Deterministic (freq_bins, n_frames) for torchaudio Spectrogram(center=True)."""
        freq_bins = self.n_fft // 2 + 1
        n_frames = self.seq_len // self.hop_length + 1
        return freq_bins, n_frames

    def img_to_ts(self, x_image):
        """
        Args:
            x_image: (B, 2F, H, W) - either the raw STFT image or the padded square image
        Returns:
            (B, T, F)
        """
        if self.img_resolution is not None and (
            x_image.shape[-1] == self.img_resolution
            and x_image.shape[-2] == self.img_resolution
        ):
            freq_bins, n_frames = self._expected_spec_shape()
            if freq_bins != self.img_resolution or n_frames != self.img_resolution:
                x_image = x_image[:, :, :freq_bins, :n_frames]

        half = x_image.shape[1] // 2
        real, imag = x_image[:, :half], x_image[:, half:]

        real = self._linear_unscale(real, self.scale_real.to(x_image.device))
        imag = self._linear_unscale(imag, self.scale_imag.to(x_image.device))

        complex_spec = torch.complex(real, imag)
        ispec = T.InverseSpectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, center=True,
        ).to(x_image.device)
        x_ts = ispec(complex_spec, self.seq_len)  # (B, F, T)
        return x_ts.permute(0, 2, 1)  # (B, T, F)

    def ts_to_img_adjoint(self, x_img):
        """Exact transpose of ts_to_img computed via an autograd VJP.

        For any linear forward pipeline (stft -> per-bin scale -> zero-pad),
        `torch.autograd.grad(ts_to_img(x_ts_zero), x_ts_zero, x_img)` returns
        `(ts_to_img)^T @ x_img`. This is the true adjoint — unlike istft, which
        is the left-inverse under COLA but not the matrix transpose — and is
        what observation-space CG requires for an SPD operator.
        """
        assert self.scale_real is not None, (
            "Call cache_min_max_params(train_data) before using ts_to_img_adjoint"
        )
        B = x_img.shape[0]
        F_feat = int(self.scale_real.shape[1])

        was_training = torch.is_grad_enabled()
        with torch.enable_grad():
            x_ts_probe = torch.zeros(
                B, self.seq_len, F_feat,
                device=x_img.device, dtype=torch.float32, requires_grad=True,
            )
            img = self.ts_to_img(x_ts_probe)
            grad_ts, = torch.autograd.grad(
                outputs=img, inputs=x_ts_probe,
                grad_outputs=x_img.to(img.dtype),
                create_graph=False, retain_graph=False,
            )
        if not was_training:
            grad_ts = grad_ts.detach()
        return grad_ts.to(x_img.dtype)

    def img_to_ts_adjoint(self, v_ts):
        """Exact transpose of img_to_ts computed via an autograd VJP.

        For the linear inverse pipeline (crop -> per-bin unscale -> ISTFT),
        `torch.autograd.grad(img_to_ts(probe_img), probe_img, v_ts)` returns
        `(img_to_ts)^T @ v_ts`. This is the true adjoint L^{-T}.

        MMPS obs-space CG requires the operator
            sigma_y^2 I + sigma_t^2 G J^T G^T
        where G(D) = mask_ts * img_to_ts(D) maps image -> TS. Thus
            G^T(v_ts) = img_to_ts_adjoint(mask_ts * v_ts)
        maps TS -> image. Using ts_to_img (= L) instead of L^{-T} is only
        correct for unitary lifts, which STFT with Hann window + per-bin
        scaling is not. See utils/stft_preflight.py for the adjoint audit.
        """
        assert self.scale_real is not None, (
            "Call cache_min_max_params(train_data) before using img_to_ts_adjoint"
        )
        B = v_ts.shape[0]
        F_feat = int(self.scale_real.shape[1])
        freq_bins, n_frames = self._expected_spec_shape()
        if self.img_resolution is not None:
            H = W = self.img_resolution
        else:
            H, W = freq_bins, n_frames

        was_training = torch.is_grad_enabled()
        with torch.enable_grad():
            probe_img = torch.zeros(
                B, 2 * F_feat, H, W,
                device=v_ts.device, dtype=torch.float32, requires_grad=True,
            )
            ts_out = self.img_to_ts(probe_img)
            grad_img, = torch.autograd.grad(
                outputs=ts_out, inputs=probe_img,
                grad_outputs=v_ts.to(ts_out.dtype),
                create_graph=False, retain_graph=False,
            )
        if not was_training:
            grad_img = grad_img.detach()
        return grad_img.to(v_ts.dtype)

    @property
    def pad_mask(self):
        """(1, 2F, H, W) mask: 1 inside the native STFT crop, 0 in the pad.

        Only meaningful for `pad_mode == 'zero'`, where the padded region
        contains deterministic zeros that should be excluded from the M-step
        loss to prevent the denoiser from latching onto them.

        Returns None when:
          - scale_real / img_resolution are not yet set, or
          - `pad_mode == 'reflect'` (reflection-padded pixels are deterministic
            functions of native pixels and thus data-bearing; excluding them
            from the loss would break the lift's linear consistency).
        """
        if self.scale_real is None or self.img_resolution is None:
            return None
        if self.pad_mode == 'reflect':
            return None
        F_feat = int(self.scale_real.shape[1])
        freq_bins, n_frames = self._expected_spec_shape()
        target_shape = (1, 2 * F_feat, self.img_resolution, self.img_resolution)
        crop = (freq_bins, n_frames)
        if self._pad_mask_cache is None or self._pad_mask_shape != (target_shape, crop):
            m = torch.zeros(*target_shape, device=self.device)
            m[:, :, :freq_bins, :n_frames] = 1.0
            self._pad_mask_cache = m
            self._pad_mask_shape = (target_shape, crop)
        return self._pad_mask_cache


class GAFEmbedder(TsImgEmbedder):
    """
    Gramian Angular Sum Field (GASF): (B, T, C) -> (B, C, H, W).

    Each feature is rescaled to [0, 1], mapped to polar coordinates via
    arccos, then the outer sum GASF[t,t'] = cos(phi_t + phi_t') is
    computed.  The resulting T x T matrix is resized to (H, W).
    """

    def __init__(self, device, seq_len, img_resolution=8):
        super().__init__(device, seq_len)
        self.img_resolution = img_resolution

    def ts_to_img(self, signal, pad=True, mask=0):
        """
        Args:
            signal: (B, T, C) tensor with values ideally in [0, 1]
        Returns:
            (B, C, img_resolution, img_resolution) GASF image
        """
        B, T, C = signal.shape
        device = signal.device

        x = signal.permute(0, 2, 1)  # (B, C, T)
        x = x.clamp(0.0, 1.0)

        phi = torch.arccos(x)  # (B, C, T)

        # GASF: cos(phi_t + phi_t') = x_t * x_t' - sqrt(1-x_t^2)*sqrt(1-x_t'^2)
        # Equivalent to outer-sum of angles then cos
        phi_i = phi.unsqueeze(-1)   # (B, C, T, 1)
        phi_j = phi.unsqueeze(-2)   # (B, C, 1, T)
        gasf = torch.cos(phi_i + phi_j)  # (B, C, T, T)

        # Resize to target resolution
        gasf_flat = gasf.reshape(B * C, 1, T, T)
        gasf_resized = F.interpolate(
            gasf_flat,
            size=(self.img_resolution, self.img_resolution),
            mode='bilinear', align_corners=False,
        )
        return gasf_resized.reshape(B, C, self.img_resolution, self.img_resolution)

    def img_to_ts(self, img):
        """
        Approximate inverse: resize back to (T, T), extract diagonal as the
        time series values via arccos of the diagonal (GASF[t,t] = cos(2*phi_t)).

        Args:
            img: (B, C, H, W)
        Returns:
            (B, T, C)
        """
        B, C, H, W = img.shape
        device = img.device

        gasf_flat = img.reshape(B * C, 1, H, W)
        gasf_full = F.interpolate(
            gasf_flat,
            size=(self.seq_len, self.seq_len),
            mode='bilinear', align_corners=False,
        ).squeeze(1)  # (B*C, T, T)

        # Diagonal: GASF[t,t] = cos(2*phi_t) => phi_t = arccos(diag)/2
        # x_t = cos(phi_t)
        diag = torch.diagonal(gasf_full, dim1=-2, dim2=-1)  # (B*C, T)
        diag = diag.clamp(-1.0, 1.0)
        phi = torch.arccos(diag) / 2.0
        x = torch.cos(phi)  # (B*C, T)

        return x.reshape(B, C, self.seq_len).permute(0, 2, 1).to(device)


class MultiViewEmbedder(TsImgEmbedder):
    """
    Stacks multiple image representations along the channel dimension.

    Given N embedders that each produce (B, C_i, H, W), the output is
    (B, sum(C_i), H, W).  Inverse uses the primary (first) embedder only,
    since auxiliary views (spectrogram, GAF) have lossy inverses.
    """

    def __init__(self, embedders, device, seq_len):
        """
        Args:
            embedders: list of TsImgEmbedder instances.
                       The first embedder is the primary one used for img_to_ts.
            device: target device
            seq_len: sequence length
        """
        super().__init__(device, seq_len)
        self.embedders = embedders
        self.primary = embedders[0]
        self.n_views = len(embedders)

        self.channels_per_view = None
        self.img_shape = None

    def ts_to_img(self, signal, pad=True, mask=0):
        """
        Args:
            signal: (B, T, C)
        Returns:
            (B, n_views * C, H, W) — stacked along channel dim
        """
        imgs = []
        for emb in self.embedders:
            if isinstance(emb, DelayEmbedder):
                img = emb.ts_to_img(signal, pad=pad, mask=mask)
            else:
                img = emb.ts_to_img(signal, pad=pad, mask=mask)
            imgs.append(img)

        self.channels_per_view = [img.shape[1] for img in imgs]

        if hasattr(self.primary, 'img_shape'):
            self.img_shape = self.primary.img_shape

        return torch.cat(imgs, dim=1)

    def img_to_ts(self, img):
        """
        Inverse using the primary (delay) embedder on its channel slice.

        Args:
            img: (B, n_views * C, H, W)
        Returns:
            (B, T, C)
        """
        if self.channels_per_view is not None:
            primary_ch = self.channels_per_view[0]
        else:
            primary_ch = img.shape[1] // self.n_views

        primary_img = img[:, :primary_ch, :, :]
        return self.primary.img_to_ts(primary_img)

    @property
    def mapping(self):
        return self.primary.mapping
