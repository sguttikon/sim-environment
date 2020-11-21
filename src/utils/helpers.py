#!/usr/bin/env python3
from transforms3d.euler import quat2euler
import numpy as np
import torch
import utils.constants as constants

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

def wrap_angle(angle, use_numpy=False):
    """
    wrap the give angle to range [-np.pi, np.pi]
    """
    if use_numpy:
        return np.arctan2(np.sin(angle), np.cos(angle))
    else:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

def wrap_angle2(angle):
    """
    wrap the give angle to range [0, 2*np.pi]
    """
    return np.mod(angle, 2*np.pi)

def to_tensor(array):
    return torch.from_numpy(array.copy()).float().to(constants.DEVICE)

def to_numpy(tensor):
    return tensor.cpu().detach().numpy()

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
    #delta_theta = wrap_angle2(delta_theta)
    print(delta_theta)

    lin_vel_hat = (delta_theta * r_star) / delta_time
    ang_vel_hat = delta_theta / delta_time
    bearing_hat = (theta_prime - theta) / delta_time - ang_vel_hat

    vel_cmd_hat = np.array([lin_vel_hat, ang_vel_hat])
    print(vel_cmd, vel_cmd_hat, new_pose, sample_motion_model_velocity(vel_cmd_hat, old_pose))

def sample_motion_model_velocity(vel_cmd, old_pose, delta_t=1., use_noise=False):
    '''
    '''
    x, y, theta = old_pose
    lin_vel, ang_vel = vel_cmd

    if use_noise:
        # reference: probabilistic robotics book
        alpha1 = alpha2 = alpha3 = alpha4 = alpha5 = alpha6 = 0.02

        std1 = np.sqrt(alpha1*lin_vel*lin_vel + alpha2*ang_vel*ang_vel)
        lin_vel_hat = lin_vel + np.random.normal(loc=.0, scale=std1)

        std2 = np.sqrt(alpha3*lin_vel*lin_vel + alpha4*ang_vel*ang_vel)
        ang_vel_hat = ang_vel + np.random.normal(loc=.0, scale=std2)

        std3 = np.sqrt(alpha5*lin_vel*lin_vel + alpha6*ang_vel*ang_vel)
        gamma_hat = np.random.normal(loc=.0, scale=std3)

        radius = lin_vel_hat/(ang_vel_hat + 1e-8)
        x_prime = x + radius*(np.sin(theta + ang_vel_hat*delta_t) - np.sin(theta))
        y_prime = y + radius*(np.cos(theta) - np.cos(theta + ang_vel_hat*delta_t))
        theta_prime = wrap_angle(theta + ang_vel_hat*delta_t + gamma_hat*delta_t)

        new_pose = np.array([x_prime, y_prime, theta_prime])
    else:
        # simplified
        est_x = x + lin_vel*np.cos(theta)*delta_t
        est_y = y + lin_vel*np.sin(theta)*delta_t
        est_theta = wrap_angle(theta + ang_vel*delta_t)

        new_pose = np.array([est_x, est_y, est_theta])

    return new_pose

def transform_poses(poses):
    return torch.cat([
        poses[:, 0:2], torch.cos(poses[:, 2:3]), torch.sin(poses[:, 2:3])
    ], axis=-1)
