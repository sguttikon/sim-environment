#!/usr/bin/env python3

import cv2
import glob
import argparse
import numpy as np
import tensorflow as tf

def read_tfrecord(example_proto):
    """
    parse the raw tfrecord input based on feature_description
    :param Tensor:
    """
    # create a description of the features
    feature_description = {
        'map_wall': tf.io.FixedLenFeature([], tf.string),
        'map_roomid': tf.io.FixedLenFeature([], tf.string),
        'states': tf.io.FixedLenFeature([], tf.string),
        'odometry': tf.io.FixedLenFeature([], tf.string),
        'rgb': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    }

    # Parse the input `tf.train.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)

def decode_image(img_str, resize=None):
    """
    decode image from tfrecord data
    :param img_str: image encoded as a png in a string
    :param resize: tuple of width, height, new size of image (optional)
    :return np.ndarray: image (k, H, W, 1)
    """
    nparr = np.frombuffer(img_str, np.uint8)
    img_str = cv2.imdecode(nparr, -1)
    if resize is not None:
        img_str = cv2.resize(img_str, resize)
    return img_str

def raw_images_to_array(images):
    """
    decode and normalize multiple images from tfrecord data
    :param images: list of images encoded as a png in a string
    :return np.ndarray: images (N, 56, 56, ch) normalized for training
    """
    image_list = []
    for image_str in images:
        image = decode_image(image_str, (56, 56))
        image = normalize_observation(np.atleast_3d(image.astype(np.float32)))
        image_list.append(image)
    return np.stack(image_list)

def normalize_observation(x):
    """
    normalize observation input: an rgb image or a depth image
    :param x: observation input (56, 56, ch)
    :return np.ndarray: normalized observation (56, 56, ch)
    """
    # resale to [-1, 1]
    if x.ndim == 2 or x.shape[2] == 1:  # depth
        return x * (2.0 / 100.0) - 1.0
    else:   # rgb
        return x * (2.0 / 255.0) - 1.0

def process_roomid_map(roomidmap_feature):
    """
    decode room image from tfrecord data
    :param roomidmap_feature: room image encoded as a png in a string
    :return np.ndarray: image (H, W, 1)
    """
    # axes is not transposed unlike others
    roomidmap = np.atleast_3d(decode_image(roomidmap_feature))
    return roomidmap

def process_wall_map(wallmap_feature):
    """
    decode wall map image from tfrecord data
    :param wallmap_feature: wall map image encoded as a png in a string
    :return np.ndarray: image (H, W, 1)
    """
    floormap = np.atleast_3d(decode_image(wallmap_feature))
    # wall map image need to be transposed and inverted here
    floormap = 255 - np.transpose(floormap, axes=[1, 0, 2])
    floormap = normalize_map(floormap.astype(np.float32))
    return floormap

def normalize_map(x):
    """
    normalize map input
    :param x: map input (H, W, ch)
    :return np.ndarray: normalized map (H, W, ch)
    """
    # rescale to [0, 2], later zero padding will produce equivalent obstacle
    return x * (2.0/255.0)

def pad_images(images, new_shape):
    """
    zero-pad right and bottom of image to match new shape (largest in batch)
    :param images: list of np.ndarray map images
    :param new_shape: tuple of new (width, height, channel) of image
    :return np.ndarray: zero-paded map images (N, new_H, new_W, new_ch)
    """
    pad_images = []
    for img in images:
        new_img = np.zeros(new_shape, np.float32)
        old_shape = img.shape
        new_img[:old_shape[0], :old_shape[1], :old_shape[2]] = img
        pad_images.append(new_img)
    return pad_images

def transform_raw_record(raw_record, params):
    """
    process tfrecords data of House3D trajectories
    :param raw_record: raw tfrecord data
    :param params: parsed arguments
    :return dict: processed data containing: true_states, odometries, observations, global map, initial particles
    """
    trans_record = {}

    trajlen = params.trajlen
    batch_size = params.batch_size
    num_particles = params.num_particles
    init_particles_cov = params.init_particles_cov
    init_particles_distr = params.init_particles_distr

    # process true states
    states = []
    for raw_state in raw_record['states']:
        state = np.frombuffer(raw_state, np.float32).reshape(-1, 3)[:trajlen, :]
        # states.append([state[i:i+bptt_steps] for i in seq_indices])
        states.append(state)
    trans_record['true_states'] = np.stack(states)  # (batch_size, trajlen, 3)

    # # process is_first_step
    # is_first_step = []
    # for split_i in range(num_segments):
    #     is_first_step.append(split_i == 0)
    # trans_record['is_first_step'] = np.stack([is_first_step] * batch_size)

    # process odometry
    odometry = []
    for raw_odom in raw_record['odometry']:
        odom = np.frombuffer(raw_odom, np.float32).reshape(-1, 3)[:trajlen, :]
        # odometry.append([odom[i:i+bptt_steps] for i in seq_indices])
        odometry.append(odom)
    trans_record['odometry'] = np.stack(odometry)   # (batch_size, trajlen, 3)

    # process rgb observation
    rgbs = []
    for raw_rgb in raw_record['rgb']:
        rgb = raw_images_to_array(raw_rgb[:trajlen])
        # rgbs.append([rgb[i:i+bptt_steps] for i in seq_indices])
        rgbs.append(rgb)
    trans_record['observation'] = np.stack(rgbs)    # (batch_size, trajlen, 56, 56, 3)

    # # process map room id
    # map_roomids = []
    # for map_roomid in raw_record['map_roomid']:
    #     map_roomids.append(process_roomid_map(map_roomid))

    # process wall map
    map_walls = []
    max_height, max_width, max_channel = (0, 0, 0)
    for map_wall in raw_record['map_wall']:
        wall_img = process_wall_map(map_wall)
        map_walls.append(wall_img)
        old_shape = wall_img.shape
        if max_height < old_shape[0]:
            max_height = old_shape[0]
        if max_width < old_shape[1]:
            max_width = old_shape[1]
        if max_channel < old_shape[2]:
            max_channel = old_shape[2]

    # generate random particle states
    trans_record['init_particles'] = random_particles(
                                        num_particles,
                                        init_particles_distr,
                                        trans_record['true_states'][:, 0, :],
                                        init_particles_cov
                                    )   # (batch_size, num_particles, 3)

    # zero pad map wall image
    new_shape = (max_height, max_width, max_channel)
    pad_map_walls = pad_images(map_walls, new_shape)
    trans_record['global_map'] = np.stack(pad_map_walls)  # (batch_size, H, W, 1)

    return trans_record

def random_particles(num_particles, particles_distr, state, particles_cov):
    """
    generate a random set of particles
    :param num_particles: number of particles
    :param distr: string type of distribution, possible value: [tracking, one-room]
        tracking - the distribution is a Gaussian centered near the true state
        one-room - the distribution is uniform over states in room defined by the true state
    :param state: true state (batch_size, 3)
    :param particle_cov: for tracking Gaussian covariance matrix (3, 3)
    :return np.ndarray: random particles (batch_size, num_particles, 3)
    """

    particles = []
    if particles_distr == 'tracking':

        # iterate per batch_size
        for b_idx in range(state.shape[0]):
            # sample offset from the Gaussian
            center = np.random.multivariate_normal(mean=state[b_idx], cov=particles_cov)

            # sample particles from the Gaussian, centered around the offset
            particles.append(np.random.multivariate_normal(mean=center, cov=particles_cov, size=num_particles))
    else:
        raise ValueError

    particles = np.stack(particles)
    return particles

def get_dataflow(filenames, batch_size, s_buffer_size=100, is_training=False):

    ds = tf.data.TFRecordDataset(filenames)
    if is_training:
        ds = ds.shuffle(s_buffer_size, reshuffle_each_iteration=True)
    ds = ds.map(read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    # ds = ds.repeat(2)

    return ds

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    params = argparser.parse_args()

    filenames = list(glob.glob('/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/sim-environment/src/tensorflow/pfnet/data/valid.tfrecords'))
    params.filenames = filenames
    params.batch_size = 8
    params.epochs = 1
    params.trajlen = 24
    params.bptt_steps = 4
    params.num_particles = 30
    params.particles_distr = 'tracking'
    params.particles_std = np.array([0.3, 0.523599])    # 30cm, 30degrees
    params.map_pixel_in_meters = 0.02

    # build initial covariance matrix of particles, in pixels and radians
    particle_std = params.particles_std.copy()
    particle_std[0] = particle_std[0] / params.map_pixel_in_meters  # convert meters to pixels
    particle_std2 = np.square(particle_std)  # variance
    params.particles_cov = np.diag(particle_std2[(0, 0, 1),])

    ds = get_dataflow(params.filenames, params.batch_size)
    for epoch in range(params.epochs):
        for raw_record in ds.as_numpy_iterator():
            trans_record = transform_raw_record(raw_record, params)
            print(trans_record['true_states'].shape,
                trans_record['odometry'].shape,
                trans_record['observation'].shape,
                trans_record['global_map'].shape,
                trans_record['init_particles'].shape,
                )
            break
