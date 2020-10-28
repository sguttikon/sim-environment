#!/usr/bin/env python3
import numpy as np

def wrap_angle(angle) -> float:
    """
    Wrap the angle to [-np.pi, np.pi]
    """
    return ( (angle-np.pi) % (2*np.pi) ) - np.pi

def compute_sq_distance(pose1, pose2):
    """
    calculate squared distance between pose

    :param torch.Tensor pose1:
    :param torch.Tensor pose2:
    :return torch.Tensor
    """
    result = 0.0

    for i in range(3):
        diff = pose1[..., i] - pose2[..., i]

        # wrap angle for theta
        if i == 2:
            diff = wrap_angle(diff)
        result += diff **2

    return result
