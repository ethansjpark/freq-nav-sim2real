from __future__ import annotations

import torch


def _radial_lowpass_mask(height: int, width: int, radius: int, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    cy, cx = height // 2, width // 2
    dist = torch.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2)
    return (dist <= float(radius)).float()


def freq_adapt(obs: torch.Tensor, radius: int = 16, noise_std: float = 1.0) -> torch.Tensor:
    """
    Applies frequency-domain perturbation by preserving low-frequency magnitude
    and replacing high-frequency magnitude with noise.

    Args:
        obs: Tensor with shape (B, C, H, W), float in [0, 1].
        radius: Low-frequency radial cutoff.
        noise_std: Multiplicative standard deviation for HF noise.
    """
    if radius <= 0:
        return obs
    if obs.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape={tuple(obs.shape)}")

    b, c, h, w = obs.shape
    freq = torch.fft.fft2(obs, dim=(-2, -1))
    freq = torch.fft.fftshift(freq, dim=(-2, -1))

    mag = torch.abs(freq)
    phase = torch.angle(freq)

    mask = _radial_lowpass_mask(h, w, radius=radius, device=obs.device)
    mask = mask.view(1, 1, h, w).expand(b, c, h, w)

    # Scale noise by per-sample magnitude statistics for numerical stability.
    scale = mag.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    noise = torch.randn_like(mag) * scale * float(noise_std)
    mixed_mag = mag * mask + noise * (1.0 - mask)

    mixed_freq = mixed_mag * torch.exp(1j * phase)
    mixed_freq = torch.fft.ifftshift(mixed_freq, dim=(-2, -1))
    rec = torch.fft.ifft2(mixed_freq, dim=(-2, -1)).real
    return rec.clamp(0.0, 1.0)
