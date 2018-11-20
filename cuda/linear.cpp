#include <torch/torch.h>
#include <vector>

at::Tensor linear_forward(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias={}) {
  auto output = at::matmul(input, weight.t());
  if (bias.defined()) {
    output.add_(bias);
  }
  return output;
}

std::vector<at::Tensor> linear_backword(const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias) {
  auto grad_input = at::matmul(grad_output, weight);
  auto grad_weight = at::matmul(grad_output.t(), input);
  auto grad_bias = bias.defined() ? grad_output.sum(0, /*keepdim=*/false) : at::Tensor{};
  return {
    grad_input,
    grad_weight,
    grad_bias
  };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &linear_forward, "linear forward");
  m.def("backward", &linear_backword, "linear backward");
}