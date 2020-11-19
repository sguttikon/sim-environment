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
    diff = pose1-pose2
    if len(diff.shape) == 1:
        return torch.norm(diff)
    else:
        return torch.norm(diff, dim=1)

def motion_model_velocity(new_pose, vel_cmd, old_pose):
    x, y, theta = old_pose
    x_prime, y_prime, theta_prime = new_pose
    lin_vel, ang_vel = vel_cmd
    delta_time = 1.

    mu = .5 * ( (x-x_prime)*np.cos(theta) + (y-y_prime)*np.sin(theta) ) / \
              ( (y-y_prime)*np.cos(theta) - (x-x_prime)*np.sin(theta) )

    x_star = .5 * (x + x_prime) + mu * (y-y_prime)
    y_star = .5 * (y + y_prime) + mu * (x_prime-x)
    r_star = np.sqrt( (x-x_star)**2 + (y-y_star)**2 )

    delta_theta = np.arctan2(y_prime-y_star, x_prime-x_star) - \
                  np.arctan2(y-y_star, x-x_star)
    delta_theta = helpers.wrap_angle(delta_theta)

    lin_vel_hat = (delta_theta * r_star) / delta_time
    ang_vel_hat = delta_theta / delta_time
    bearing_hat = (theta_prime - theta) / delta_time - ang_vel_hat

    vel_cmd_hat = np.array([lin_vel_hat, ang_vel_hat])
    print(new_pose, sample_motion_model_velocity(vel_cmd_hat, old_pose))

def sample_motion_model_velocity(vel_cmd, old_pose):
    x, y, theta = old_pose
    lin_vel, ang_vel = vel_cmd
    delta_time = 1.

    radius = lin_vel/ang_vel
    x_prime = x - radius*np.sin(theta) + \
                  radius*np.sin(theta + ang_vel*delta_time)
    y_prime = y + radius*np.cos(theta) - \
                  radius*np.cos(theta + ang_vel*delta_time)
    theta_prime = theta + ang_vel*delta_time

    return np.array([x_prime, y_prime, theta_prime])
