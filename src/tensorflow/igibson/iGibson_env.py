#!/usr/bin/env python3

import gym
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from collections import OrderedDict
from utils import render, datautils
from gibson2.envs.env_base import BaseEnv
from gibson2.sensors.vision_sensor import VisionSensor
from gibson2.termination_conditions.timeout import Timeout
from gibson2.external.pybullet_tools.utils import stable_z_on_aabb
from gibson2.reward_functions.collision_reward import CollisionReward

class iGibsonEnv(BaseEnv):
    """
    openai custom env of Interactive Gibson 3D Environment
    reference - https://github.com/openai/gym/blob/master/docs/creating-environments.md
    """

    metadata = {'render.modes': ['human']}

    def __init__(self,
        config_file,
        scene_id=None,
        mode='headless',
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        device_idx=0,
        render_to_tensor=False,
        automatic_reset=False,
        num_particles=100,
        show_plot=False,
        max_step=50
    ):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless, gui, iggui
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device_idx: which GPU to run the simulation and rendering on
        :param render_to_tensor: whether to render directly to pytorch tensors
        :param automatic_reset: whether to automatic reset after an episode finishes
        :param num_particles: number of particles
        :param show_plot: whether to render plots or not
        """
        # self.num_particles = num_particles
        super(iGibsonEnv, self).__init__(config_file=config_file,
                                         scene_id=scene_id,
                                         mode=mode,
                                         action_timestep=action_timestep,
                                         physics_timestep=physics_timestep,
                                         device_idx=device_idx,
                                         render_to_tensor=render_to_tensor)
        self.use_rnd_floor = False
        self.automatic_reset = automatic_reset
        self.trav_map_resolution = self.config.get('trav_map_resolution', 0.1)

        self.reward_functions = [
            CollisionReward(self.config),
        ]
        self.termination_conditions = [
            Timeout(self.config),
        ]
        self.termination_conditions[0].max_step = max_step

        self.show_plot = show_plot
        if self.show_plot:
            # code related to displaying results in matplotlib
            fig = plt.figure(figsize=(7, 7))
            self.plt_ax = fig.add_subplot(111)
            plt.ion()
            plt.show()

            self.trav_map_plt = None
            self.robot_gt_plts = {
                'robot_position': None,
                'robot_heading': None,
            }
            self.robot_est_plts = {
                'robot_position': None,
                'robot_heading': None,
                'particles': None,
            }

            plt.draw()
            plt.pause(0.00000000001)

    def load(self):
        """
        Load environment
        """
        super(iGibsonEnv, self).load()

        self.load_task_setup()
        self.load_observation_space()
        self.load_action_space()
        self.load_miscellaneous_variables()

    def load_miscellaneous_variables(self):
        """
        Load miscellaneous variables for book keeping
        """
        self.current_step = 0
        self.collision_links = []

    def load_task_setup(self):
        """
        Load task setup
        """
        self.initial_pos_z_offset = self.config.get('initial_pos_z_offset', 0.1)

        # ignore the agent's collision with these body ids
        self.collision_ignore_body_b_ids = set(
            self.config.get('collision_ignore_body_b_ids', []))
        # ignore the agent's collision with these link ids of itself
        self.collision_ignore_link_a_ids = set(
            self.config.get('collision_ignore_link_a_ids', []))

    def load_observation_space(self):
        """
        Load observation space
        """
        self.output = self.config['output']
        self.image_width = self.config.get('image_width', 128)
        self.image_height = self.config.get('image_height', 128)

        # observation_space = OrderedDict()
        # sensors = OrderedDict()
        # vision_modalities = []
        # scan_modalities = []
        #
        # # initialize observation space
        # observation_space['pose'] = self.build_obs_space(
        #         shape=(3,), low=-np.inf, high=-np.inf)
        #
        # # observation_space['particles'] = self.build_obs_space(
        # #         shape=(self.num_particles, 3), low=-np.inf, high=-np.inf)
        # #
        # # observation_space['particles_weights'] = self.build_obs_space(
        # #         shape=(self.num_particles,), low=-np.inf, high=-np.inf)
        #
        # if 'rgb' in self.output:
        #     observation_space['rgb'] = self.build_obs_space(
        #         shape=(self.image_height, self.image_width, 3),
        #         low=0.0, high=1.0)
        #     vision_modalities.append('rgb')
        #
        # # initialize sensors
        # if len(vision_modalities) > 0:
        #     sensors['vision'] = VisionSensor(self, vision_modalities)
        #
        # self.observation_space = gym.spaces.Dict(observation_space)
        # self.sensors = sensors

        sensors = OrderedDict()

        vision_modalities = []
        scan_modalities = []

        self.observation_space = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3),
                low=0.0, high=1.0)
        vision_modalities.append('rgb')

        # initialize sensors
        if len(vision_modalities) > 0:
            sensors['vision'] = VisionSensor(self, vision_modalities)
        self.sensors = sensors

    def build_obs_space(self, shape, low, high):
        """
        Helper function that builds individual observation spaces
        """
        return gym.spaces.Box(low=low, high=high,
                        shape=shape, dtype=np.float32)

    def load_action_space(self):
        """
        Load action space
        """
        self.action_space = self.robots[0].action_space

    def get_observation(self):
        """
        Get the current observation
        :return ndarray: rgb observation (H, W, 3)
        """

        obs = OrderedDict()

        if 'vision' in self.sensors:
            vision_obs = self.sensors['vision'].get_obs(self)
            for modality in vision_obs:
                obs[modality] = vision_obs[modality]

        # process image for training
        rgb = datautils.process_raw_image(obs['rgb'])

        return rgb  # rgb image

    def get_floor_map(self):
        """
        Get the scene floor map
        :return ndarray: floor map of current scene (H, W, 1)
        """
        floor_map = self.scene.floor_map[self.floor_num]
        # process image for training
        floor_map = datautils.process_floor_map(floor_map)

        return floor_map

    def get_robot_state(self):
        """
        Get the current robot state
        :return dict: state of robot as a dictionary
            containing: gt pose
        """
        robot_state = OrderedDict()

        # robot_state = self.robots[0].calc_state()
        robot_pos = self.robots[0].get_position()   # [x, y, z]
        robot_orn = p.getEulerFromQuaternion(self.robots[0].get_orientation())  # [r, p, y]
        robot_state['pose'] = np.array([robot_pos[0], robot_pos[1], robot_orn[2]])  # [x, y, theta]

        # robot_state['particles'] = self.particles
        # robot_state['particles_weights'] = self.particles_weights

        return robot_state

    def render(self, mode='human'):
        """
        Render plots
        """

        if self.show_plot:
            floor_map = self.get_floor_map()
            robot_state = self.get_robot_state()

            # environment map
            map_plt = self.trav_map_plt
            floor_map = floor_map[:, :, 0]    # [H, W]
            map_plt = render.draw_floor_map(floor_map, self.plt_ax, map_plt)
            self.trav_map_plt = map_plt

            # ground truth robot pose and heading
            color = '#7B241C'
            robot_pose = robot_state['pose']
            position_plt, heading_plt = self.robot_gt_plts['robot_position'], self.robot_gt_plts['robot_heading']
            position_plt, heading_plt = render.draw_robot_pose(robot_pose, color, self.plt_ax, position_plt, heading_plt, self.trav_map_resolution)
            self.robot_gt_plts['robot_position'], self.robot_gt_plts['robot_heading'] = position_plt, heading_plt

            # # estimated particles pose
            # particles = robot_state['particles']
            # weights = robot_state['particles_weights']
            # particles_plt = self.robot_est_plts['particles']
            # particles_plt = render.draw_particles_pose(particles, weights, particles_plt, self.trav_map_resolution)
            # self.robot_est_plts['particles'] = particles_plt

            # plt_ax.legend([gt_plt['robot_position'], est_plt['robot_position']], ["gt_pose", "est_pose"])

            plt.draw()
            plt.pause(0.00000000001)

    def close(self):
        """
        environment close()
        """
        super(iGibsonEnv, self).close()

        if self.show_plot:
            # to prevent plot from closing after environment is closed
            plt.ioff()
            plt.show()

    def step(self, action):
        """
        Apply robot's action.
        Returns the next state, reward, done and info,
        following OpenAI Gym's convention
        :param action: robot actions
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """

        self.current_step += 1

        robot = self.robots[0]
        body_id = robot.robot_ids[0]
        if action is not None:
            robot.apply_action(action)

        # run simulation
        self.simulator_step()
        collision_links = list(p.getContactPoints(bodyA=body_id))
        self.collision_links = self.filter_collision_links(collision_links)

        info = {}
        state = self.get_observation()
        reward, info = self.get_reward(collision_links, action, info)
        done, info = self.get_termination(collision_links, action, info)
        self.populate_info(info)

        if done and self.automatic_reset:
            info['last_observation'] = state
            state = self.reset()

        return state, reward, done, info

    def reset(self):
        """
        Reset episode
        """

        if self.show_plot:
            #clear subplots
            self.plt_ax.cla()

        # move robot away from the scene
        self.robots[0].set_position([100.0, 100.0, 100.0])

        # scene reset
        self.reset_scene()

        # robot reset
        self.reset_robot()

        # simulator sync
        self.simulator.sync()

        # # initial particles and corresponding weights
        # self.particles = self.get_random_particles(self.num_particles)
        # self.particles_weights = np.full((self.num_particles, ), (1./self.num_particles))

        state = self.get_observation()

        self.reset_variables()

        return state

    def reset_scene(self):
        """
        Reset environment scene floor plane
        """

        # get random floor
        if self.use_rnd_floor:
            self.floor_num = self.scene.get_random_floor()
        else:
            self.floor_num = 0

        # reset scene floor
        self.scene.reset_floor(floor=self.floor_num)

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode
        """
        self.current_step = 0
        self.collision_links = []

    def reset_robot(self):
        """
        Reset the robot initial pose (position and orientation)
            sample random initial pos and orn, check if its valid and land on it.
        """

        reset_success = False
        max_trials = 100
        robot = self.robots[0]

        # cache pybullet state
        # TODO: p.saveState takes a few seconds, need to speed up
        state_id = p.saveState()
        for i in range(max_trials):
            pos, orn = self.sample_random_pose()
            reset_success = self.test_valid_pose(robot, pos, orn)

            p.restoreState(state_id)
            if reset_success:
                break

        if not reset_success:
            print("WARNING: Failed to reset robot without collision")

        p.removeState(state_id)

        initial_pos = pos
        initial_orn = orn
        self.land_robot(robot, initial_pos, initial_orn)

        for reward_function in self.reward_functions:
            reward_function.reset(None, self)

    def sample_random_pose(self):
        """
        Sample robot pose (position and orientation)
        :return (ndarray, ndarray): position (xyz) and orientation (xyzw)
        """
        _, pos = self.scene.get_random_point(floor=self.floor_num)   # [x, y, z]
        orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)]) # [r, p, y]
        orn = p.getQuaternionFromEuler(orn) # [x, y, z, w]

        return pos, orn

    def land_robot(self, robot, pos, orn):
        """
        Land the robot onto the floor, given a valid position and orientation
        :param Turtlebot robot: robot instance
        :param ndarray pos: robot position (xyz)
        :param ndarray orn: robot orientation (xyzw)
        """

        self.set_pos_orn_with_z_offset(robot, pos, orn)

        robot.robot_specific_reset()
        robot.keep_still()

        body_id = robot.robot_ids[0]

        land_success = False
        # land for maximum 1 second, should fall down ~5 meters
        max_simulator_step = int(1.0 / self.action_timestep)
        for _ in range(max_simulator_step):
            self.simulator_step()
            if len(p.getContactPoints(bodyA=body_id)) > 0:
                land_success = True
                break

        if not land_success:
            print("WARNING: Failed to land")

        robot.robot_specific_reset()
        robot.keep_still()

    def test_valid_pose(self, robot, pos, orn):
        """
        Test if the robot can be placed with no collision
        :param Turtlebot robot: robot instance
        :param ndarray pos: robot position (xyz)
        :param ndarray orn: robot orientation (xyzw)
        :return boolean: validity
        """

        self.set_pos_orn_with_z_offset(robot, pos, orn)

        robot.robot_specific_reset()
        robot.keep_still()

        body_id = robot.robot_ids[0]
        has_collision = self.check_collision(body_id)

        return has_collision

    def set_pos_orn_with_z_offset(self, robot, pos, orn, offset=None):
        """
        Reset position and orientation for the robot
        :param Turtlebot robot: robot instance
        :param ndarray pos: robot position (xyz)
        :param ndarray orn: robot orientation (xyzw)
        :param offset: z offset
        """

        if offset is None:
            offset = self.initial_pos_z_offset

        body_id = robot.robot_ids[0]

        # first set the correct orientation (with temporary position)
        robot.set_position_orientation(pos, orn)

        # compute stable z based on this orientation
        stable_z = stable_z_on_aabb(body_id, [pos, pos])

        # change the z-value of position with stable_z + additional offset
        # in case the surface is not perfect smooth (has bumps)
        robot.set_position([pos[0], pos[1], stable_z + offset])

    def check_collision(self, body_id):
        """
        Check with the given body_id has any collision after one simulator step
        :param body_id: pybullet body id
        :return: whether the given body_id has no collision
        """
        self.simulator_step()
        collisions = list(p.getContactPoints(bodyA=body_id))

        return len(collisions) == 0

    def get_random_particles(self, num_particles, robot_pose=None, particles_cov=None):
        """
        Sample random particles based on the scene
        :param robot_pose: ndarray indicating the robot pose (3,)
            if None, random particle poses are sampled using unifrom distribution
            otherwise, sampled using gaussian distribution around the robot_pose
        :param particles_cov: for tracking Gaussian covariance matrix (3, 3)
        :param num_particles: integer indicating the number of random particles
        :return ndarray: random particle poses  (num_particles, 3)
        """

        particles = []
        if robot_pose is None:
            # uniform distribution
            sample_i = 0
            while sample_i < num_particles:
                pos, orn = self.sample_random_pose()
                orn = p.getEulerFromQuaternion(orn) # [r, p, y]
                particles.append([pos[0], pos[1], orn[2]])  # [x, y, theta]

                sample_i = sample_i + 1
            particles = np.array(particles) # [num_particles, 3]
        else:
            # gaussian distribution
            assert list(robot_pose.shape) == [3]

            # sample offset from the Gaussian
            center = np.random.multivariate_normal(mean=robot_pose, cov=particles_cov)

            # sample particles from the Gaussian, centered around the offset
            particles.append(np.random.multivariate_normal(mean=center, cov=particles_cov, size=num_particles))

            particles = np.array(particles).squeeze(axis=0) # [num_particles, 3]
        return particles

    def filter_collision_links(self, collision_links):
        """
        Filter out collisions that should be ignored
        :param collision_links: original collisions, a list of collisions
        :return: filtered collisions
        """
        new_collision_links = []
        for item in collision_links:
            # ignore collision with body b
            if item[2] in self.collision_ignore_body_b_ids:
                continue

            # ignore collision with robot link a
            if item[3] in self.collision_ignore_link_a_ids:
                continue

            # ignore self collision with robot link a (body b is also robot itself)
            if item[2] == self.robots[0].robot_ids[0] and item[4] in self.collision_ignore_link_a_ids:
                continue
            new_collision_links.append(item)
        return new_collision_links

    def get_reward(self, collision_links=[], action=None, info={}):
        """
        Aggreate reward functions
        :param collision_links: collision links after executing action
        :param action: the executed action
        :param info: additional info
        :return reward: total reward of the current timestep
        :return info: additional info
        """
        reward = 0.0
        for reward_function in self.reward_functions:
            reward += reward_function.get_reward(None, self)

        return reward, info

    def get_termination(self, collision_links=[], action=None, info={}):
        """
        Aggreate termination conditions
        :param collision_links: collision links after executing action
        :param action: the executed action
        :param info: additional info
        :return done: whether the episode has terminated
        :return info: additional info
        """
        done = False
        success = False
        for condition in self.termination_conditions:
            d, s = condition.get_termination(None, self)
            done = done or d
            success = success or s
        info['done'] = done
        info['success'] = success
        return done, info

    def populate_info(self, info):
        """
        Populate info dictionary with any useful information
        """
        info['episode_length'] = self.current_step
