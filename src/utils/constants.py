#!/usr/bin/env python3
import torch

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    IS_CUDA = True
else:
    DEVICE = torch.device('cpu')
    IS_CUDA = False

VISUAL_FEATURES = 512
STATE_DIMS = 3
ACTION_DIMS = 2
NUM_PARTICLES = 500
RANDOM_SEED = 42
WIDTH, HEIGHT = 256, 256
BATCH_SIZE = 8
is_acts_disc = True
SEQ_LEN = 16
