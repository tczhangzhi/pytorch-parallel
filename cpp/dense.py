import torch
from torch.nn import Module, Parameter
from torch.autograd import Function

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