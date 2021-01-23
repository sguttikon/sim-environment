#!/usr/bin/env python3

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
import torch
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    #dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size,rank=rank)

def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def training(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    torch.cuda.set_device(rank)
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # train_dataset = TensorDataset(config.masked_sentences, config.original_sentences)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=config.world_size)
    # train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,sampler=train_sampler, shuffle=False)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(f"{n_gpus} is gpu available.")
    mp.spawn(training, nprocs=world_size, args=(world_size,))
