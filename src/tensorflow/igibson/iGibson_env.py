#!/usr/bin/env python3

import gym
import numpy as np
import pybullet as p
from collections import OrderedDict
from gibson2.envs.env_base import BaseEnv
from gibson2.sensors.vision_sensor import VisionSensor
from gibson2.external.pybullet_tools.utils import stable_z_on_aabb

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
        render_to_tensor=False
    ):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless, gui, iggui
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device_idx: which GPU to run the simulation and rendering on
        :param render_to_tensor: whether to render directly to pytorch tensors
        """
        super(iGibsonEnv, self).__init__(config_file=config_file,
                                         scene_id=scene_id,
                                         mode=mode,
                                         action_timestep=action_timestep,
                                         physics_timestep=physics_timestep,
                                         device_idx=device_idx,
                                         render_to_tensor=render_to_tensor)
        self.use_rnd_floor = False

    def load(self):
        """
        Load environment
        """
        super(iGibsonEnv, self).load()

        self.load_task_setup()
        self.load_observation_space()
        self.load_action_space()

    def load_task_setup(self):
        """
        Load task setup
        """
        self.initial_pos_z_offset = self.config.get('initial_pos_z_offset', 0.1)

    def load_observation_space(self):
        """
        Load observation space
        """
        self.output = self.config['output']
        self.image_width = self.config.get('image_width', 128)
        self.image_height = self.config.get('image_height', 128)

        observation_space = OrderedDict()
        sensors = OrderedDict()
        vision_modalities = []
        scan_modalities = []

        # initialize observation space
        observation_space['pose'] = self.build_obs_space(
                shape=(3,), low=-np.inf, high=-np.inf)

        if 'rgb' in self.output:
            observation_space['rgb'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3),
                low=0.0, high=1.0)
            vision_modalities.append('rgb')

        # initialize sensors
        if len(vision_modalities) > 0:
            sensors['vision'] = VisionSensor(self, vision_modalities)

        self.observation_space = gym.spaces.Dict(observation_space)
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

    def get_state(self):
        """
        Get the current observation
        :return: observation as a dictionary
        """
        state = OrderedDict()

        if 'vision' in self.sensors:
            vision_obs = self.sensors['vision'].get_obs(self)
            for modality in vision_obs:
                state[modality] = vision_obs[modality]

        # robot_state = self.robots[0].calc_state()
        robot_pos = self.robots[0].get_position()   # [x, y, z]
        robot_orn = p.getEulerFromQuaternion(self.robots[0].get_orientation())  # [r, p, y]
        state['pose'] = np.array([robot_pos[0], robot_pos[1], robot_orn[2]])  # [x, y, theta]

        return state

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

        robot = self.robots[0]
        body_id = robot.robot_ids[0]
        if action is not None:
            robot.apply_action(action)

        # run simulation
        self.simulator_step()
        collision_links = list(p.getContactPoints(bodyA=body_id))

        state = self.get_state()
        reward = 0
        info = {}
        done = False

        return state, reward, done, info

    def reset(self):
        """
        Reset episode
        """

        # move robot away from the scene
        self.robots[0].set_position([100.0, 100.0, 100.0])

        # scene reset
        self.reset_scene()

        # robot reset
        self.reset_robot()

        # simulator sync
        self.simulator.sync()

        state = self.get_state()

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
