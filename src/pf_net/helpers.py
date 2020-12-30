#!/usr/bin/env python3

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import torch

# reference: https://stackoverflow.com/questions/64554658/calculate-covariance-of-torch-tensor-2d-feature-map
def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()

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

def normalize(angle, isTensor=False):
    '''
        wrap the give angle to range [-np.pi, np.pi]
    '''
    if isTensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))
    else:
        return np.arctan2(np.sin(angle), np.cos(angle))

def angle_diff(th1, th2):
    th1 = normalize(th1)
    th2 = normalize(th2)

    d1 = th1 - th2
    d2 = 2 * np.pi - np.fabs(d1)
    if d1 > 0:
        d2 = d2 * -1.0

    if np.fabs(d1) < np.fabs(d2):
        return d1
    else:
        return d2

def compute_odometry(old_pose, new_pose):
    x1, y1, th1 = old_pose
    x2, y2, th2 = new_pose

    abs_x = (x2 - x1)
    abs_y = (y2 - y1)
    abs_th = angle_diff(th2, th1)

    odometry = np.array([abs_x, abs_y, abs_th])
    return odometry

def sample_motion_odometry(old_pose, odometry):
    x1, y1, th1 = old_pose
    abs_x, abs_y, abs_th = odometry

    delta_trans = np.sqrt(abs_y**2 + abs_x**2)
    if delta_trans < 0.01:
        delta_rot1 = 0.0
    else:
        delta_rot1 = angle_diff(np.arctan2(abs_y, abs_x), th1)
    delta_rot2 = angle_diff(abs_th, delta_rot1);

    delta_rot1_noise = np.minimum(
                                np.fabs(angle_diff(delta_rot1, 0.0)),
                                np.fabs(angle_diff(delta_rot1, np.pi))
                            )
    delta_rot2_noise = np.minimum(
                                np.fabs(angle_diff(delta_rot2, 0.0)),
                                np.fabs(angle_diff(delta_rot2, np.pi))
                            )

    alpha1 = 0.1
    alpha2 = 0.1
    alpha3 = 0.1
    alpha4 = 0.1

    scale = alpha1*delta_rot1_noise*delta_rot1_noise + alpha2*delta_trans*delta_trans
    delta_rot1_hat = angle_diff(delta_rot1, np.random.normal(0.0, scale))

    scale = alpha4*delta_rot1_noise*delta_rot1_noise + alpha3*delta_trans*delta_trans + \
            alpha4*delta_rot2_noise*delta_rot2_noise
    delta_trans_hat = delta_trans - np.random.normal(0.0, scale)

    scale = alpha1*delta_rot2_noise*delta_rot2_noise + alpha2*delta_trans*delta_trans
    delta_rot2_hat = angle_diff(delta_rot2, np.random.normal(0.0, scale))

    x_new = x1 + delta_trans_hat * np.cos(th1 + delta_rot1_hat)
    y_new = y1 + delta_trans_hat * np.sin(th1 + delta_rot1_hat)
    th_new = th1 + delta_rot1_hat + delta_rot2_hat
    new_pose = np.array([x_new, y_new, th_new])

    return new_pose

if __name__ == '__main__':
    old_pose = np.array([ 0.5757,  0.4682, -1.3175])
    new_pose = np.array([ 0.5892,  0.4159, -1.3187])
    odometry = compute_odometry(old_pose, new_pose)
    print(odometry)

    new_pose = sample_motion_odometry(old_pose, odometry)
    print(new_pose)
