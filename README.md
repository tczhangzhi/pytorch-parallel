# Custom Dense layer in PyTorch

The project customizes the dense layer in pytorch following the official tutorial. In the project, we first write python code, and then gradually use C++ and CUDA to optimize key operations. I hope this project will help your Pytorch, ATen, CUDA and PTX learning.

## How to run

### Python Extensions

Check the grad.

```
# ./
python grad_check.py py
```

### C++ Extensions

Pybind11 is used for Python and C++ interactions. Install these packages:

```
conda install pytest pybind11
```

Enter the C++ folder and compile the code.

```
# ./cpp
python setup.py install
```

Check the grad.

```
# ./
python grad_check.py cpp
```

### CUDA Extensions

Enter the CUDA folder and compile the code.

```
# ./cuda
python setup.py install
```

Check the grad.

```
# ./
python grad_check.py cuda
```

### PTX Example

Enter the PTX folder and compile the code.

```
# ./ptx
sh compile.sh
```

After changing the ` sigmoid_cuda_kernal.ptx` file, recompile your code.

```
# ./ptx
sh recompile.sh
```

Test your result.

```
./sigmoid_cuda_kernal
```

## How to write

After Reading the example of the pytorch official website, I feel that it is really a little difficult for novices to learn CUDA. So I wrote a simple Demo for the students who just started. What we want to achieve is a Dense layer in Tensorflow. If it is not for teaching, you can use Linear + activation functions directly. But this time, we will start with Python and gradually use CPP and CUDA to optimize key operations.

There are two steps to implementing a Python extension:

* Implement a Function that completes the definition of forward and backward operations
* Implement a Module that completes the parameters' initialization according to the hyperparameter, and then calls the Function to calculate

For the operations provided in Pytorch, we don't need a Function and Module  will help us to automatically backward. But in order to make us understand better, let's write a Function is defined as follows:

```
class DenseFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        output = torch.sigmoid(output)
        ctx.save_for_backward(input, weight, bias, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, output = ctx.saved_tensors
        grad_sigmoid = (1.0 - output) * output
        grad_output = grad_sigmoid * grad_output
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias
```

We need to remember that the number of  parameters for forward is the number of outputs for backward. And the number of  outputs for forward is the number of parameters for backward. Don't loss them.
After completing the Function definition, the calculations are clear. All the Module has to do is initialize the training parameters based on the hyperparameters.

```
class Dense(Module):
    def __init__(self, input_features, output_features, bias=True):
        super(Dense, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        return DenseFunction.apply(input, self.weight, self.bias)
```

Based on the Python version of the custom layer, we extracted the linear part and accelerated it with CPP.

```
import linear_cpp

class DenseFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        output = linear_cpp.forward(input, weight, bias)
        output = torch.sigmoid(output)
        ctx.save_for_backward(input, weight, bias, output)
        return output

    @staticmethod
    def backward(ctx, grad_output): 
        input, weight, bias, output = ctx.saved_variables
        grad_sigmoid = (1.0 - output) * output
        grad_output = grad_sigmoid * grad_output
        grad_input, grad_weight, grad_bias = linear_cpp.backward(grad_output, input, weight, bias)
        return grad_input, grad_weight, grad_bias
```

linear_cpp is a  CPP library that pybind11 compiled and introduced into. We pass  the linear operation to CPP by calling forward and backward function of linear_cpp. You will find that the activation part is still achieved by Python (we will use CUDA later).

The code for CPP is not difficult, we can make it directly with `matmul` function and `add` function provided by ATen. Of course, Pytorch's source code is different, because there are more efficient APIs.

```
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
```

It should be noted that the CPP cannot save the input and output during forward and keep them for backward. CPP can only accept input and produce output as a pure function. The dirty work of saving variables is done with Python. Parameters is passed to CPP by Python when CPP is called in backward.

We left the activation function to Python in the previous section. It's time to hand it over to CUDA. To call CUDA we need to use CPP. We can calls CUDA functions to get the result of CUDA calculation with CPP .

```
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
```

The data type of CPP is slightly different from the data type of CUDA. We use the `AT_DISPATCH_FLOATING_TYPES` function to help us pass the CPP parameters to CUDA, and then pass the CUDA results to CPP without manual data type conversion.

```
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
```

It is worth noting that the operation needs to be called by global and executed by device.

```
...  
  template <typename scalar_t>
  __device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
    return 1.0 / (1.0 + exp(-z));
  }

  template <typename scalar_t>
  __global__ void sigmoid_cuda_forward_kernel(
      const scalar_t* __restrict__ input,
      scalar_t* __restrict__ output) {
    const int index = blockIdx.x * blockDim.x + blockIdx.y;
    output[index] = sigmoid(input[index]);
  }
...
```

The same is true for backward delivery. In this way, we can easily complete the CUDA extension.

Good luck for you.

## License

[MIT](http://opensource.org/licenses/MIT)

Copyright (c) 2018-present, Zhi Zhang

