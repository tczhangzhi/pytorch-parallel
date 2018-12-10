import torch
from torch.nn import Module, Parameter
from torch.autograd import Function

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