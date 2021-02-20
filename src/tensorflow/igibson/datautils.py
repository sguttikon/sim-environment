#!/usr/bin/env python3

import numpy as np
import pybullet as p

def normalize(angle):
    """
    Normalize the angle to [-pi, pi]
    :param float angle: input angle to be normalized
    :return float: normalized angle
    """
    quaternion = p.getQuaternionFromEuler(np.array([0, 0, angle]))
    euler = p.getEulerFromQuaternion(quaternion)
    return euler[2]

def calc_odometry(old_pose, new_pose):
    """
    Calculate the odometry between two poses
    :param ndarray old_pose: pose1 (x, y, theta)
    :param ndarray new_pose: pose2 (x, y, theta)
    :return ndarray: odometry (odom_x, odom_y, odom_th)
    """
    x1, y1, th1 = old_pose
    x2, y2, th2 = new_pose

    abs_x = (x2 - x1)
    abs_y = (y2 - y1)

    th1 = normalize(th1)
    sin = np.sin(th1)
    cos = np.cos(th1)

    th2 = normalize(th2)
    odom_th = normalize(th2 - th1)
    odom_x = cos * abs_x + sin * abs_y
    odom_y = cos * abs_y - sin * abs_x

    odometry = np.array([odom_x, odom_y, odom_th])
    return odometry

def sample_motion_odometry(old_pose, odometry):
    """
    Sample new pose based on give pose and odometry
    :param ndarray old_pose: given pose (x, y, theta)
    :param ndarray odometry: given odometry (odom_x, odom_y, odom_th)
    :return ndarray: new pose (x, y, theta)
    """
    x1, y1, th1 = old_pose
    odom_x, odom_y, odom_th = odometry

    th1 = normalize(th1)
    sin = np.sin(th1)
    cos = np.cos(th1)

    x2 = x1 + (cos * odom_x - sin * odom_y)
    y2 = y1 + (sin * odom_x + cos * odom_y)
    th2 = normalize(th1 + odom_th)

    new_pose = np.array([x2, y2, th2])
    return new_pose

def gather_episode_stats(env, trajlen, num_particles):
    """
    Run the gym environment and collect the required stats
    :param int trajlen: length of trajectory (episode steps)
    :param num_particles: number of particles
    :return dict: episode stats data containing:
        odometry, true poses, observation, particles, particles weights, floor map
    """
    global_map = None
    true_poses = []
    observation = []
    odometry = []

    state = env.reset()

    global_map = state['floor_map']

    rgb = state['rgb']
    observation.append(rgb)

    old_pose = state['pose']
    true_poses.append(old_pose)

    for _ in range(trajlen-1):
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)

        rgb = state['rgb']
        observation.append(rgb)

        new_pose = state['pose']
        true_poses.append(new_pose)

        odom = calc_odometry(old_pose, new_pose)
        odometry.append(odom)
        old_pose = new_pose

    odom = calc_odometry(old_pose, new_pose)
    odometry.append(odom)

    init_particles = env.get_random_particles(num_particles)
    init_particles_weights = np.full((num_particles, ), (1./num_particles))

    episode_data = {}
    episode_data['global_map'] = global_map # (height, width)
    episode_data['odometry'] = np.stack(odometry)  # (trajlen, 3)
    episode_data['true_states'] = np.stack(true_poses)  # (trajlen, 3)
    episode_data['observation'] = np.stack(observation) # (trajlen, height, width)
    episode_data['init_particles'] = np.stack(init_particles)   # (num_particles, 3)
    episode_data['init_particles_weights'] = np.stack(init_particles_weights)   # (num_particles,)

    return episode_data

def get_batch_data(env, params):
    """
    Gather batch of episode stats
    :param params: parsed parameters
    :return dict: episode stats data containing:
        odometry, true poses, observation, particles, particles weights, floor map
    """
    trajlen = params.trajlen
    batch_size = params.batch_size
    num_particles = params.num_particles

    odometry = []
    global_map = []
    observation = []
    true_states = []
    init_particles = []
    init_particles_weights = []

    for _ in range(batch_size):
        episode_data = gather_episode_stats(env, trajlen, num_particles)

        odometry.append(episode_data['odometry'])
        global_map.append(episode_data['global_map'])
        true_states.append(episode_data['true_states'])
        observation.append(episode_data['observation'])
        init_particles.append(episode_data['init_particles'])
        init_particles_weights.append(episode_data['init_particles_weights'])

    batch_data = {}
    batch_data['odometry'] = np.stack(odometry)
    batch_data['global_map'] = np.stack(global_map)
    batch_data['true_states'] = np.stack(true_states)
    batch_data['observation'] = np.stack(observation)
    batch_data['init_particles'] = np.stack(init_particles)
    batch_data['init_particles_weights'] = np.stack(init_particles_weights)

    return batch_data

if __name__ == '__main__':
    print(normalize(3*np.pi))
    print(normalize(-3*np.pi))
