from __future__ import division
from __future__ import print_function

import time
import argparse
import torch
import torch.nn as nn
from torch.nn import Module, Parameter
from torch.nn import functional as F

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', choices=['py', 'cpp', 'cuda'])
parser.add_argument('-e', '--epoch', type=int, default=100)
parser.add_argument('-s', '--size', type=int, default=100)
options = parser.parse_args()

if options.mode == 'py':
    from python.dense import Dense
elif options.mode == 'cpp':
    from cpp.dense import Dense
elif options.mode == 'cuda':
    from cuda.dense import Dense

inputs = torch.randn(options.size, 256)
labels = torch.rand(options.size).mul(10).long()

class Model(Module):

    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = Dense(256, 64)
        self.dense2 = Dense(64, 16)
        self.dense3 = Dense(16, 10)

    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return F.log_softmax(x, dim=1)

model = Model()

inputs = inputs.cuda()
labels = labels.cuda()
model = model.cuda()

# dataparallel = DataParallel(model, device_ids=[0, 1, 2, 3])

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(dataparallel.module.parameters(), lr=1e-4)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

forward_time = 0
backward_time = 0
for _ in range(options.epoch):
    optimizer.zero_grad()

    start = time.time()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    elapsed = time.time() - start
    forward_time += elapsed

    start = time.time()
    loss.backward()
    optimizer.step()
    elapsed = time.time() - start
    backward_time += elapsed

print('Forward: {0:.3f} | Backward {1:.3f}'.format(forward_time, backward_time))