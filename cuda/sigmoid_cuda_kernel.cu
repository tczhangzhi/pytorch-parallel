#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {
  template <typename scalar_t>
  __device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
    return 1.0 / (1.0 + exp(-z));
  }

  template <typename scalar_t>
  __device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
    return (1.0 - z) * z;
  }

  template <typename scalar_t>
  __global__ void sigmoid_cuda_forward_kernel(
      const scalar_t* __restrict__ input,
      scalar_t* __restrict__ output) {
    const int index = blockIdx.x * blockDim.x + blockIdx.y;
    output[index] = sigmoid(input[index]);
  }

  template <typename scalar_t>
  __global__ void sigmoid_cuda_backward_kernel(
      const scalar_t* __restrict__ grad_output,
      const scalar_t* __restrict__ output,
      scalar_t* __restrict__ new_grad_output) {
    const int index = blockIdx.x * blockDim.x + blockIdx.y;
    new_grad_output[index] = d_sigmoid(output[index] * grad_output[index]);
  }
} // namespace

at::Tensor sigmoid_cuda_forward(
    at::Tensor input) {
  auto output = at::zeros_like(input);
  const dim3 blocks(input.size(0), input.size(1));
  const int threads = 1;

  AT_DISPATCH_FLOATING_TYPES(input.type(), "sigmoid_forward_cuda", ([&] {
    sigmoid_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
      input.data<scalar_t>(),
      output.data<scalar_t>());
  }));

  return output;
}

at::Tensor sigmoid_cuda_backward(
    at::Tensor grad_output,
    at::Tensor output) {
  auto new_grad_output = at::zeros_like(grad_output);
  const dim3 blocks(grad_output.size(0), grad_output.size(1));
  const int threads = 1;

  AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "sigmoid_backward_cuda", ([&] {
    sigmoid_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
      grad_output.data<scalar_t>(),
      output.data<scalar_t>(),
      new_grad_output.data<scalar_t>());
  }));

  return new_grad_output;
}
