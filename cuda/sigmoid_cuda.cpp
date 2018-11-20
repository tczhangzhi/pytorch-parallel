#include <torch/torch.h>
#include <vector>

at::Tensor sigmoid_cuda_forward(
    at::Tensor input);

at::Tensor sigmoid_cuda_backward(
    at::Tensor grad_output,
    at::Tensor output);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor sigmoid_forward(
    at::Tensor input) {
  CHECK_INPUT(input);
  return sigmoid_cuda_forward(input);
}

at::Tensor sigmoid_backward(
    at::Tensor grad_output,
    at::Tensor output) {
  CHECK_INPUT(grad_output);
  CHECK_INPUT(output);
  return sigmoid_cuda_backward(
    grad_output,
    output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sigmoid_forward, "sigmoid forward (CUDA)");
  m.def("backward", &sigmoid_backward, "sigmoid backward (CUDA)");
}
