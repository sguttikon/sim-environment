#!/usr/bin/env python3

import sys
def set_path(path: str):
    try:
        sys.path.index(path)
    except ValueError:
        sys.path.insert(0, path)

# set programatically the path to 'sim-environment' directory (alternately can also set PYTHONPATH)
set_path('/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/sim-environment/src')

from utils import display, helpers
import torch.nn.functional as F
import networks.networks as nets
import numpy as np
import torch
import cv2
import os

from skimage.viewer import ImageViewer
from skimage import io, transform

def spatial_transform():

    floor_idx = 0
    model_path = '/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/sim-environment/envs/iGibson/gibson2/dataset/Rs'
    img_name = os.path.join(model_path, 'floor_{0}.png'.format(floor_idx))
    floor_map = cv2.flip(io.imread(img_name), 0)

    # rescale
    new_h, new_w = (256, 256)
    floor_map = transform.resize(floor_map, (new_h, new_w))

    floor_map = torch.from_numpy(floor_map).float().unsqueeze(0).unsqueeze(1)
    floor_map_res = torch.tensor([0.1], dtype=torch.float)

    # position and orientation
    pose_x = 1
    pose_y = -1
    pose_th = helpers.wrap_angle(- 1 * np.pi/4, use_numpy=True)
    pose = torch.tensor([[[pose_x, pose_y, pose_th]]], dtype=torch.float)

    # affine transformation
    scale = 2 * floor_map_res[0]
    t_x = pose_x * scale
    t_y = -pose_y * scale
    r_c = np.cos(pose_th)
    r_s = np.sin(pose_th)

    # theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
    theta = torch.tensor([r_c, r_s, t_x, -r_s, r_c, t_y], dtype=torch.float)
    theta = theta.view(-1, 2, 3)
    grid_size = torch.Size((floor_map.shape[0], floor_map.shape[1], floor_map.shape[2], floor_map.shape[3]))

    grid = F.affine_grid(theta, grid_size, align_corners=False)
    st_map = F.grid_sample(floor_map, grid, align_corners=False)
    local_map = st_map.squeeze()

    # crop
    input_row, input_cols = local_map.shape
    output_row, output_cols = (128, 128)
    s_row_idx = int(input_row//2) - int(output_row)//2
    s_col_idx = int(input_cols//2) - int(output_cols)//2
    local_map = local_map[s_row_idx:s_row_idx+output_row, s_col_idx:s_col_idx+output_cols]

    # disp_local_map(floor_map, local_map, floor_map_res, pose)

    stn = nets.SpatialTransformerNet()
    particles = torch.tensor([
                    [[0, -0, helpers.wrap_angle(- 0 * np.pi/4, use_numpy=True)]],
                    [[1, -1, helpers.wrap_angle(- 1 * np.pi/4, use_numpy=True)]],
                    [[1, -1, helpers.wrap_angle(+ 1 * np.pi/4, use_numpy=True)]],
                ], dtype=torch.float)

    particles_local_map = stn(floor_map, floor_map_res[0], particles)
    disp_local_map(floor_map, particles_local_map[0].squeeze(), floor_map_res, pose)

def disp_local_map(floor_map, local_map, floor_map_res, pose):

    viewer = ImageViewer(helpers.to_numpy(local_map))
    viewer.show()

    data = {
        #'occ_map': local_map.unsqueeze(0),
        'occ_map': floor_map.squeeze(1),
        'occ_map_res': floor_map_res,
        'robot_gt_pose': pose,
    }

    # render = display.Render()
    # render.update_figures(data)


if __name__ == '__main__':
    spatial_transform()
