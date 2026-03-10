from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
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
