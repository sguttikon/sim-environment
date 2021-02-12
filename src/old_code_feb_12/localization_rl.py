#!/usr/bin/env python3

import os
from gibson2.envs.locomotor_env import NavigateEnv, NavigateRandomEnv
from gibson2.utils.utils import parse_config
from gibson2.utils.assets_utils import get_model_path
from transforms3d.euler import quat2euler
from DPF.dpf import DMCL
import numpy as np
import torch
import utils
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

curr_dir_path = os.path.dirname(os.path.abspath(__file__))
dmcl = DMCL()

fig = plt.figure(figsize=(7, 7))
plt_ax = fig.add_subplot(111)
plt.ion()
plt.show()

plots = {
    'map': None,
    'robot': None,
    'heading': None,
    'particles': None,
}

def get_pose(robot) -> np.ndarray:
    """
    """
    position = robot.get_position()
    euler = quat2euler(robot.get_orientation())
    pose = np.array([position[0], position[2], utils.wrap_angle(euler[0])])
    return pose

def train_network(env):
    """
    """
    num_epochs = 1
    epoch_len = 10
    epoch = 0

    turtlebot = env.robots[0]
    while epoch < num_epochs:
        epoch += 1
        obs = env.reset()
        init_pose = get_pose(turtlebot)
        particles, particle_probs = dmcl.initialize_particles(init_pose)

        for step in range(epoch_len):
            plots['robot'], plots['heading'] = plot_robot(get_pose(turtlebot), plots['robot'], plots['heading'])
            plots['particles'] = plot_particles(particles, plots['particles'])

            action = env.action_space.sample()
            obs, reward, done, info = nav_env.step(action)

            ## motion update
            particles = dmcl.motion_update(action, particles)

            # define motion model loss and optimizer
            curr_pose = get_pose(turtlebot)
            new_pose = torch.from_numpy(curr_pose).to(dmcl.get_device())
            sq_distance = utils.compute_sq_distance(particles, new_pose)
            std = 0.01
            pdf = (1/dmcl.get_num_particles()) * (1/np.sqrt(2*np.pi*std**2)) \
                    * torch.exp(-sq_distance / (2*np.pi*std**2))

            motion_loss = torch.mean(-torch.log(1e-16 + pdf), axis=0)

            ## measurement update
            particle_probs *= dmcl.measurement_update(obs, particles)
            particle_probs /= torch.sum(particle_probs, axis=0) # normalize probabilities

            state = dmcl.particles_to_state(particles, particle_probs)

            ## resample
            particles = dmcl.resample_particles(particles, particle_probs)
    #print(particles)

def plot_robot(pose, pose_plt, heading_plt, scale= 0.1):
    """
    """

    pose_x, pose_y, yaw = pose
    pose_x = pose_x * scale
    pose_y = pose_y * scale

    line_len = 10.0 * scale
    robot_radius = 10.0 * scale
    color = 'blue'

    xdata = [pose_x, pose_x + (robot_radius + line_len) * np.cos(yaw)]
    ydata = [pose_y, pose_y + (robot_radius + line_len) * np.sin(yaw)]

    if pose_plt == None:
        pose_plt = Wedge((pose_x, pose_y), robot_radius, 0, 360, color=color, alpha=0.5)
        plt_ax.add_artist(pose_plt)
        heading_plt, = plt_ax.plot(xdata, ydata, color=color, alpha=0.5)
    else:
        pose_plt.update({'center': [pose_x, pose_y]})
        heading_plt.update({'xdata': xdata, 'ydata': ydata})

    plt.draw()
    plt.pause(0.00000000001)

    return pose_plt, heading_plt

def plot_map(config_data: dict, map_plt):
    """
    """

    model_id = config_data['model_id']
    erosion = config_data['trav_map_erosion']
    scale = config_data['trav_map_resolution']
    floors = 0

    model_path = get_model_path(model_id)
    with open(os.path.join(model_path, 'floors.txt'), 'r') as f:
        floors = sorted(list( map(float, f.readlines()) ))

    # default assuming ground floor map
    idx = 0
    trav_map = cv2.imread(os.path.join(model_path,
                            'floor_trav_{0}.png'.format(idx))
                            )
    obs_map = cv2.imread(os.path.join(model_path,
                            'floor_{0}.png'.format(idx))
                            )
    orign_x, orign_y = 0, 0

    shape = trav_map.shape
    x_max = (shape[0]/2 + orign_x) * scale
    x_min = (-shape[0]/2 + orign_x) * scale
    y_max = (shape[1]/2 + orign_y) * scale
    y_min = (-shape[1]/2 + orign_y) * scale
    extent = [x_min, x_max, y_min, y_max]

    if map_plt == None:
        map_plt = plt_ax.imshow(trav_map, cmap=plt.cm.binary, origin='upper', extent=extent)
        plt_ax.plot(orign_x, orign_y, 'm+', markersize=14)
        plt_ax.grid()
        plt_ax.set_xlim([x_min, x_max])
        plt_ax.set_ylim([y_min, y_max])

        ticks_x = np.linspace(x_min, x_max)
        ticks_y = np.linspace(y_min, y_max)
        plt_ax.set_xticks(ticks_x, ' ')
        plt_ax.set_yticks(ticks_y, ' ')

        plt_ax.set_xlabel('x coords')
        plt_ax.set_ylabel('y coords')
    else:
        pass

    plt.draw()
    plt.pause(0.00000000001)

    return map_plt

def plot_particles(particles, particles_plt):
    """
    """
    color = 'C0'
    particles = particles.cpu().detach().numpy()
    xdata = particles[:, 0]
    ydata = particles[:, 1]
    if particles_plt == None:
        particles_plt = plt.scatter(xdata, ydata, s=14, c=color)
    else:
        particles_plt.set_offsets(particles[:, 0:2])

    plt.draw()
    plt.pause(0.00000000001)

    return particles_plt

if __name__ == '__main__':
    # configuration file contains: robot, scene, etc. details
    config_file_path = os.path.join(curr_dir_path, 'turtlebot.yaml')
    config_data = parse_config(config_file_path)

    mode = 'headless' # []'headless', 'gui']
    render_to_tensor=True
    nav_env = NavigateEnv(config_file=config_file_path,
                            mode=mode,
                            render_to_tensor=render_to_tensor)

    plot_map(config_data, plots['map'])
    train_network(nav_env)

    # to prevent plot from closing
    plt.ioff()
    plt.show()
