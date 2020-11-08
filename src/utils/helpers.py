#!/usr/bin/env python3
from transforms3d.euler import quat2euler
import numpy as np
import torch

def get_gt_pose(robot):
    """
    """
    position = robot.get_position()
    euler = quat2euler(robot.get_orientation())
    gt_pose = np.array([
        position[0],
        position[1],
        wrap_angle(euler[0])
    ])
    return gt_pose

def wrap_angle(angle):
    """
    wrap the give angle to range [-np.pi, np.pi]
    """
    return ( (angle-np.pi) % (2*np.pi) ) - np.pi

def eucld_dist(pose1, pose2):
    """
    """
    return torch.norm(pose1-pose2)
