#!/usr/bin/env python3

import numpy as np
from torch import nn
import datautils

class TransitionModel(nn.Module):

    def __init__(self, params):
        super(TransitionModel, self).__init__()

        self.params = params

    def forward(self, particle_states, odometry):

        translation_std = self.params.transition_std[0]
        rotation_std = self.params.transition_std[1]

        parts_x, parts_y, parts_th = particle_states

        odom_x, odom_y, odom_th = odometry

        # add orientation noise
        noise_th = np.random.normal(loc=0, scale=1.0, size=1) * rotation_std
        parts_th = parts_th + noise_th

        sin_th = np.sin(parts_th)
        cos_th = np.cos(parts_th)
        delta_x = odom_x * cos_th + odom_y * sin_th
        delta_y = odom_x * sin_th - odom_y * cos_th
        delta_th = odom_th

        noise_x = np.random.normal(loc=0, scale=1.0, size=1) * translation_std
        noise_y = np.random.normal(loc=0, scale=1.0, size=1) * translation_std
        delta_x = delta_x + noise_x
        delta_y = delta_y + noise_y

        x = parts_x + delta_x
        y = parts_y + delta_y
        th = datautils.wrap_angle(parts_th + delta_th)
        print(x, y, th)


if __name__ == '__main__':
    trans_model = TransitionModel(None)
    # old_pose = np.array([ 0.5757,  0.4682, -1.3175])
    # new_pose = np.array([ 0.5892,  0.4159, -1.3187])
    old_pose = np.array([ 0.5892,  0.4159, -1.3187])
    new_pose = np.array([ 0.6027,  0.3636, -1.3201])

    odometry = datautils.calc_odometry(old_pose, new_pose)
    trans_model(old_pose, odometry)
