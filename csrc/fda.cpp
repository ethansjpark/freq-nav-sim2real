#include <torch/extension.h>

torch::Tensor fourier_swap(
    torch::Tensor src,
    torch::Tensor tgt,
    double beta) {

  TORCH_CHECK(src.dim() == 3 && tgt.dim() == 3,
              "src and tgt must be 3-D (H, W, C)");
  TORCH_CHECK(src.sizes() == tgt.sizes(),
              "src and tgt must have the same shape");

  // Work in float64 for FFT precision (matches numpy default).
  auto src_f = src.to(torch::kFloat64);
  auto tgt_f = tgt.to(torch::kFloat64);

  const int64_t h = src_f.size(0);
  const int64_t w = src_f.size(1);

  // 2-D FFT over spatial dims (0, 1), then shift DC to center.
  auto src_fft = torch::fft::fftshift(
      torch::fft::fft2(src_f, /*s=*/c10::nullopt, /*dim=*/{0, 1}),
      /*dim=*/{0, 1});
  auto tgt_fft = torch::fft::fftshift(
      torch::fft::fft2(tgt_f, /*s=*/c10::nullopt, /*dim=*/{0, 1}),
      /*dim=*/{0, 1});

  auto mixed_fft = src_fft.clone();

  int64_t b = static_cast<int64_t>(std::min(h, w) * beta);
  int64_t h_mid = h / 2;
  int64_t w_mid = w / 2;

  // Swap low-frequency region from target into source.
  using torch::indexing::Slice;
  mixed_fft.index_put_(
      {Slice(h_mid - b, h_mid + b), Slice(w_mid - b, w_mid + b)},
      tgt_fft.index({Slice(h_mid - b, h_mid + b), Slice(w_mid - b, w_mid + b)}));

  // Inverse FFT, take real part, clamp to [0, 255].
  auto mixed = torch::fft::ifft2(
      torch::fft::ifftshift(mixed_fft, /*dim=*/{0, 1}),
      /*s=*/c10::nullopt, /*dim=*/{0, 1});
  auto result = torch::real(mixed);
  result = result.clamp(0.0, 255.0).to(torch::kUInt8);

  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fourier_swap", &fourier_swap,
        "Fourier Domain Adaptation — low-frequency swap (C++)",
        py::arg("src"),
        py::arg("tgt"),
        py::arg("beta") = 0.01);
}
