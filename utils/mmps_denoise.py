"""Helpers for MMPS / diffusion when the UNet uses extra input channels (e.g. ambient mask)."""


def slice_denoised_to_data_channels(denoised, data_c: int):
    """Reduce UNet output to ``data_c`` channels (observation / latent channels).

    When ``ambient_concat_further_mask`` is enabled, the network input has
    ``data_c + 1`` channels and the denoiser output may match; MMPS residuals
    and observations stay ``data_c``-wide.
    """
    if denoised.shape[1] <= data_c:
        return denoised
    return denoised[:, :data_c]
