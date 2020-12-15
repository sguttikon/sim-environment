#!/usr/bin/env python3

from gibson2.envs.locomotor_env import NavigateRandomEnv
from gibson2.utils.utils import parse_config
from gibson2.utils.assets_utils import get_model_path

from transforms3d.euler import quat2euler
from display import Render
from skimage import io
import numpy as np
import os

class PFNet(object):

    def __init__(self, params):

        self.env = NavigateRandomEnv(config_file = params.config_filepath,
                    mode = params.env_mode,
        )

        # set common seed value
        # self.env.seed(params.seed)
        # np.random.seed(params.seed)

        self.robot = self.env.robots[0] #hardcoded

        config_data = parse_config(params.config_filepath)
        self.model_id = config_data['model_id']
        self.pixel_to_mts = config_data['trav_map_resolution'] # each map pixel in meter

        self.env.reset()
        self.num_particles = params.num_particles
        self.transition_std = params.transition_std
        rnd_particles = self.init_particles()
        self.curr_gt_pose = self.get_gt_pose()
        self.global_floor_map = self.get_env_map()

        self.render = Render()
        self.render.plot_map(self.global_floor_map)
        self.render.plot_robot(self.curr_gt_pose, 'green')
        self.render.plot_particles(rnd_particles, 'coral')

    def init_particles(self):
        rnd_particles = []

        lmt = 1
        gt_pose = self.get_gt_pose()
        bounds = np.array([
            [gt_pose[0] - lmt, gt_pose[0] + lmt],
            [gt_pose[1] - lmt, gt_pose[1] + lmt],
        ])

        cnt = 0
        translation_std = self.transition_std[0]
        while cnt < self.num_particles:
            _, self.initial_pos = self.env.scene.get_random_point_floor(self.env.floor_num, self.env.random_height)#
            self.initial_pos = self.initial_pos + np.random.uniform(0, 1.0, size=self.initial_pos.shape) * translation_std
            self.initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
            rnd_pose = [self.initial_pos[0], self.initial_pos[1], self.initial_orn[2]]

            if bounds[0][0] <= rnd_pose[0] <= bounds[0][1] and \
               bounds[1][0] <= rnd_pose[1] <= bounds[1][1]:
                rnd_particles.append(rnd_pose)
                cnt = cnt + 1
            else:
                continue

        rnd_particles = np.array(rnd_particles)
        return rnd_particles

    def get_gt_pose(self):

        position = self.robot.get_position()
        euler_orientation = quat2euler(self.robot.get_orientation())

        gt_pose = np.array([
            position[0],
            position[1],
            euler_orientation[0]
        ])

        return gt_pose

    def get_env_map(self):

        model_path = get_model_path(self.model_id)
        floor_idx = self.env.floor_num
        filename = os.path.join(model_path, 'floor_{}.png'.format(floor_idx))

        floor_map = io.imread(filename)
        return floor_map

    def __del__(self):
        del self.render
