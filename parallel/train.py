import pdb
import os
import torch
import torch.distributed as dist

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

class RingAllReduce(object):

    def __init__(self, model, criterion, optimizer, dataset, addr='127.0.0.1', port='29500', backend='gloo'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        self.addr = addr
        self.port = port
        self.backend = backend

    def partition_dataset(self):
        size = dist.get_world_size()
        bsz = 128 // size
        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(self.dataset, partition_sizes)
        partition = partition.use(dist.get_rank())
        train_set = torch.utils.data.DataLoader(
            partition, batch_size=bsz, shuffle=True)
        return train_set, bsz

    def average_gradients(self):
        """ Gradient averaging. """
        size = float(dist.get_world_size())
        for param in self.model.parameters():
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
            param.grad.data /= size

    def run(self, rank, size):
        """ Distributed Synchronous SGD Example """
        torch.manual_seed(1234)
        train_set, bsz = self.partition_dataset()
        optimizer = self.optimizer
        criterion = self.criterion

        num_batches = ceil(len(train_set.dataset) / float(bsz))
        for epoch in range(10):
            epoch_loss = 0.0
            for data, target in train_set:
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                epoch_loss += loss.data[0]
                loss.backward()
                self.average_gradients()
                optimizer.step()
            print('Rank ',
                dist.get_rank(), ', epoch ', epoch, ': ',
                epoch_loss / num_batches)

    def init_processes(self, rank, size, fn):
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = self.addr
        os.environ['MASTER_PORT'] = self.port
        dist.init_process_group(self.backend, rank=rank, world_size=size)
        fn(rank, size)

    def train(self, size=2):
        processes = []
        for rank in range(size):
            p = Process(target=self.init_processes, args=(rank, size, self.run))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()