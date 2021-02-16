#!/usr/bin/env python3

import cv2
import pfnet
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.patches import Arrow
from utils import datautils, arguments, pfnet_loss
from matplotlib.backends.backend_agg import FigureCanvasAgg

def draw_map(plt_ax, global_map):
    map_plt = plt_ax.imshow(global_map)
    return map_plt

def draw_particles(robot_plt, particles, particle_weights, params):
    colors = cm.rainbow(particle_weights)
    size = 1 #params.rows
    res = 1 #params.res

    positions = particles[:, 0:2] * size * res
    if 'particles' not in robot_plt:
        robot_plt['particles'] = plt.scatter(positions[:, 0], positions[:, 1], s=10, c=colors, alpha=.25)
    else:
        robot_plt['particles'].set_offsets(positions[:, 0:2])
        robot_plt['particles'].set_color(colors)
    return robot_plt

def draw_robot(plt_ax, robot_plt, robot_state, clr, params):
    x, y, theta = robot_state
    size = 1 #params.rows
    res = 1 #params.map_pixel_in_meters

    x *= size * res
    y *= size * res
    radius = 10 * size * res
    length = 2 * radius
    dx = length * np.cos(theta)
    dy = length * np.sin(theta)
    if 'robot' not in robot_plt:
        robot_plt['robot'] = Wedge((x, y), radius, 0, 360, color=clr, alpha=.9)
        plt_ax.add_artist(robot_plt['robot'])
    else:
        robot_plt['robot'].set_center((x, y))
        # oldpose
        robot_plt['heading'].set_alpha(.0)

    robot_plt['heading'] = Arrow(x, y, dx, dy, width=radius, fc=clr, alpha=.9)
    plt_ax.add_artist(robot_plt['heading'])    # newpose

    return robot_plt

def display_results(params):
    """
    display results with the parsed arguments
    """

    batch_size = 1
    num_batches = 2
    num_particles = params.num_particles
    trajlen = params.trajlen
    testfiles = params.testfiles

    # evaluation data
    test_ds = datautils.get_dataflow(testfiles, batch_size, is_training=True)

    # pf model
    model = pfnet.pfnet_model(params)

    # load model from checkpoint file
    if params.load:
        print("=====> Loading model from " + params.load)
        model.load_weights(params.load)

    fig = plt.figure(figsize=(7, 7), dpi=300)
    plt_ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)

    itr = test_ds.as_numpy_iterator()
    # run over all evaluation samples in an epoch
    for idx in tqdm(range(num_batches)):
        #clear subplots
        plt_ax.cla()

        raw_record = next(itr)
        data_sample = datautils.transform_raw_record(raw_record, params)

        observations = tf.convert_to_tensor(data_sample['observation'], dtype=tf.float32)
        odometry = tf.convert_to_tensor(data_sample['odometry'], dtype=tf.float32)
        true_states = tf.convert_to_tensor(data_sample['true_states'], dtype=tf.float32)
        global_map = tf.convert_to_tensor(data_sample['global_map'], dtype=tf.float32)
        global_map_shape = data_sample['org_map_shapes']
        init_particles = tf.convert_to_tensor(data_sample['init_particles'], dtype=tf.float32)
        init_particle_weights = tf.constant(np.log(1.0/float(num_particles)),
                                    shape=(batch_size, num_particles), dtype=tf.float32)

        # start trajectory with initial particles and weights
        state = [init_particles, init_particle_weights, global_map]

        # if stateful: reset RNN s.t. initial_state is set to initial particles and weights
        # if non-stateful: pass the state explicity every step
        if params.stateful:
            model.layers[-1].reset_states(state)    # RNN layer

        input = [observations, odometry]
        model_input = (input, state)

        # forward pass
        output, state = model(model_input, training=False)

        particle_states, particle_weights = output
        lin_weights = tf.nn.softmax(particle_weights, axis=-1)
        est_states = tf.math.reduce_sum(tf.math.multiply(
                            particle_states[:, :, :, :], lin_weights[:, :, :, None]
                        ), axis=2)

        # plot map
        map = global_map[0, :global_map_shape[0][0], :global_map_shape[0][1], 0].numpy()
        map_plt = draw_map(plt_ax, map)

        images = []
        gt_plt = {}
        est_plt = {}
        for traj in range(trajlen):
            true_state = true_states[:, traj, :]
            est_state = est_states[:, traj, :]
            particle_state = particle_states[:, traj, :, :]
            lin_weight = lin_weights[:, traj, :]

            # plot true robot pose
            gt_plt = draw_robot(plt_ax, gt_plt, true_state[0], '#7B241C', params)

            # plot est robot pose
            est_plt = draw_robot(plt_ax, est_plt, est_state[0], '#515A5A', params)

            # plot est pose particles
            draw_particles(est_plt, particle_state[0], lin_weight[0], params)

            plt_ax.legend([gt_plt['robot'], est_plt['robot']], ["gt_pose", "est_pose"])

            canvas.draw()
            img = np.array(canvas.renderer._renderer)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            images.append(img)

        size = (images[0].shape[0], images[0].shape[1])
        out = cv2.VideoWriter(params.out_folder + f'result_{idx}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, size)

        for i in range(len(images)):
            out.write(images[i])
            # cv2.imwrite(params.out_folder + f'result_img_{i}.png', images[i])
        out.release()

    print('testing finished')

if __name__ == '__main__':
    params = arguments.parse_args()

    params.out_folder = './output/'
    Path(params.out_folder).mkdir(parents=True, exist_ok=True)

    display_results(params)
