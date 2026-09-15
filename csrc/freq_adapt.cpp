#include <torch/extension.h>
#include <cmath>

static torch::Tensor radial_lowpass_mask(
    int64_t height, int64_t width, int64_t radius, torch::Device device) {

  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto yy = torch::arange(height, opts).unsqueeze(1).expand({height, width});
  auto xx = torch::arange(width, opts).unsqueeze(0).expand({height, width});

  float cy = static_cast<float>(height / 2);
  float cx = static_cast<float>(width / 2);

  auto dist = torch::sqrt((yy - cy).square() + (xx - cx).square());
  return (dist <= static_cast<float>(radius)).to(torch::kFloat32);
}

torch::Tensor freq_adapt(
    torch::Tensor obs,
    int64_t radius,
    double noise_std) {

  TORCH_CHECK(obs.dim() == 4, "Expected BCHW tensor, got ndim=", obs.dim());

  if (radius <= 0) {
    return obs;
  }

  const int64_t b = obs.size(0);
  const int64_t c = obs.size(1);
  const int64_t h = obs.size(2);
  const int64_t w = obs.size(3);

  // FFT over spatial dims, shift DC to center.
  auto freq = torch::fft::fft2(obs, /*s=*/c10::nullopt, /*dim=*/{-2, -1});
  freq = torch::fft::fftshift(freq, /*dim=*/{-2, -1});

  auto mag = torch::abs(freq);
  auto phase = torch::angle(freq);

  // Build lowpass mask: 1 inside radius, 0 outside.
  auto mask = radial_lowpass_mask(h, w, radius, obs.device());
  mask = mask.view({1, 1, h, w}).expand({b, c, h, w});

  // Scale noise by per-sample mean magnitude for numerical stability.
  auto scale = mag.mean(/*dim=*/{-2, -1}, /*keepdim=*/true).clamp_min(1e-6);
  auto noise = torch::randn_like(mag) * scale * static_cast<float>(noise_std);

  // Preserve LF magnitude, replace HF with scaled noise.
  auto mixed_mag = mag * mask + noise * (1.0f - mask);

  // Reconstruct complex spectrum and invert.
  auto mixed_freq = mixed_mag * torch::polar(
      torch::ones_like(phase), phase);
  mixed_freq = torch::fft::ifftshift(mixed_freq, /*dim=*/{-2, -1});
  auto rec = torch::fft::ifft2(mixed_freq, /*s=*/c10::nullopt, /*dim=*/{-2, -1});

  return torch::real(rec).clamp(0.0, 1.0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("freq_adapt", &freq_adapt,
        "Frequency-domain perturbation: preserve LF, replace HF with noise (C++)",
        py::arg("obs"),
        py::arg("radius") = 16,
        py::arg("noise_std") = 1.0);
}
