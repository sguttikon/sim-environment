#!/usr/bin/env python3
import torch

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    IS_CUDA = True
else:
    DEVICE = torch.device('cpu')
    IS_CUDA = False

VISUAL_FEATURES = 64
STATE_DIMS = 3
ACTION_DIMS = 2
NUM_PARTICLES = 5000
RANDOM_SEED = 42
