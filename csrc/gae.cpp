#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> compute_gae(
    torch::Tensor rewards,
    torch::Tensor dones,
    torch::Tensor values,
    torch::Tensor next_value,
    double gamma,
    double gae_lambda) {

  TORCH_CHECK(rewards.dim() == 1, "rewards must be 1-D");
  TORCH_CHECK(dones.dim() == 1, "dones must be 1-D");
  TORCH_CHECK(values.dim() == 1, "values must be 1-D");

  const int64_t t = rewards.size(0);
  TORCH_CHECK(dones.size(0) == t && values.size(0) == t,
              "rewards, dones, values must have the same length");

  auto advantages = torch::zeros_like(rewards);

  auto r_a = rewards.accessor<float, 1>();
  auto d_a = dones.accessor<float, 1>();
  auto v_a = values.accessor<float, 1>();
  auto adv_a = advantages.accessor<float, 1>();
  float next_val = next_value.reshape({1}).item<float>();

  float last_adv = 0.0f;
  for (int64_t step = t - 1; step >= 0; --step) {
    float non_terminal = 1.0f - d_a[step];
    float delta =
        r_a[step] + static_cast<float>(gamma) * next_val * non_terminal - v_a[step];
    last_adv =
        delta + static_cast<float>(gamma * gae_lambda) * non_terminal * last_adv;
    adv_a[step] = last_adv;
    next_val = v_a[step];
  }

  auto returns = advantages + values;
  return {advantages, returns};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_gae", &compute_gae,
        "Generalized Advantage Estimation (C++)",
        py::arg("rewards"),
        py::arg("dones"),
        py::arg("values"),
        py::arg("next_value"),
        py::arg("gamma"),
        py::arg("gae_lambda"));
}
