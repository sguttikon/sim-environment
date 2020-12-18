#!/usr/bin/env python3
from transforms3d.euler import quat2euler
import numpy as np
import torch
import utils.constants as constants
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def get_gt_pose(robot):
    """
    """
    position = robot.get_position()
    euler = quat2euler(robot.get_orientation())
    gt_pose = np.array([
        position[0],
        position[1],
        wrap_angle(euler[0], use_numpy=True)
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

def eucld_dist(pose1, pose2, use_numpy=False):
    """
    """
    diff = pose1-pose2
    if len(diff.shape) == 1:
        if use_numpy:
            return np.linalg.norm(diff)
        else:
            return torch.norm(diff)
    if len(diff.shape) == 2:
        if use_numpy:
            return np.linalg.norm(diff, axis=1, keepdims=True)
        else:
            return torch.norm(diff, dim=1, keepdim=True)

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

def transform_poses(poses, use_numpy=False):
    if len(poses.shape) == 1:
        transformed_poses = torch.cat([
                        poses[0:2],
                        torch.cos(poses[2:3]),
                        torch.sin(poses[2:3])
                    ], axis=-1)
    elif len(poses.shape) == 2:
        if use_numpy:
            transformed_poses = np.concatenate([
                            poses[:, 0:2],
                            np.cos(poses[:, 2:3]),
                            np.sin(poses[:, 2:3]),
                        ], axis=-1)
        else:
            transformed_poses = torch.cat([
                            poses[:, 0:2],
                            torch.cos(poses[:, 2:3]),
                            torch.sin(poses[:, 2:3])
                        ], axis=-1)
    elif len(poses.shape) == 3:
        trans_b_poses = []
        for b_idx in range(poses.shape[0]):
            trans_pose = torch.cat([
                        poses[b_idx][:, 0:2],
                        torch.cos(poses[b_idx][:, 2:3]),
                        torch.sin(poses[b_idx][:, 2:3])
                    ], axis=-1)
            trans_b_poses.append(trans_pose)
        transformed_poses = torch.stack(trans_b_poses)
    return transformed_poses

def get_triplet_labels(gt_pose, particles, desc=False):
    dist = eucld_dist(gt_pose, particles)
    _, indices = torch.sort(dist, descending=desc)
    return indices

def get_mse_loss(gt_pose, particles, std = 0.5):
    sqrt_dist = eucld_dist(gt_pose, particles)
    activations = (1/(particles.shape[0]*np.sqrt(2 *np.pi * std**2))) * torch.exp(-sqrt_dist/(2 * std**2))
    loss = torch.mean(-torch.log(1e-16 + activations))
    return loss

# reference https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()
