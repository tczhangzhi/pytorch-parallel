from __future__ import division
from __future__ import print_function

import argparse
import torch

from torch.autograd import Variable, gradcheck

parser = argparse.ArgumentParser()
parser.add_argument('example', choices=['py', 'cpp', 'cuda'])
parser.add_argument('-b', '--batch-size', type=int, default=3)
parser.add_argument('-f', '--feature-size', type=int, default=17)
parser.add_argument('-o', '--output-size', type=int, default=3)
parser.add_argument('-c', '--cuda', action='store_true')
options = parser.parse_args()

if options.example == 'py':
    from python.dense import DenseFunction
elif options.example == 'cpp':
    from cpp.dense import DenseFunction
else:
    from cuda.dense import DenseFunction
    options.cuda = True

X = torch.randn(options.batch_size, options.feature_size)
W = torch.randn(options.output_size, options.feature_size)
b = torch.randn(options.output_size)

variables = [X, W, b]

for i, var in enumerate(variables):
    if options.cuda:
        var = var.cuda()
    variables[i] = Variable(var.double(), requires_grad=True)

if gradcheck(DenseFunction.apply, variables, eps=1e-6, atol=1e-4):
    print('Ok')
