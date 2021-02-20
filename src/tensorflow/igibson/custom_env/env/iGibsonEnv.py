#!/usr/bin/env python3

import os
import gym
import cv2
import numpy as np
import pybullet as p
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from gibson2.simulator import Simulator
from transforms3d.euler import euler2quat
from gibson2.utils.utils import quatToXYZW
from gibson2.utils.utils import parse_config
from gibson2.robots.turtlebot_robot import Turtlebot
from gibson2.utils.assets_utils import get_scene_path
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.external.pybullet_tools.utils import stable_z_on_aabb
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings

class iGibsonEnv(gym.Env):
    """
    openai custom env of Interactive Gibson 3D Environment
    reference - https://github.com/openai/gym/blob/master/docs/creating-environments.md
    """

    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Initialize our iGibsonEnv
        """
        super(iGibsonEnv, self).__init__()

        config = parse_config(os.path.join('/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/sim-environment/src/tensorflow/igibson/custom_env/configs/', 'turtlebot_demo.yaml'))
        self.__resolution = config['trav_map_resolution']
        self.__erosion = config['trav_map_erosion']
        self.__initial_pos_z_offset = config['initial_pos_z_offset']
        settings = MeshRendererSettings(enable_shadow=False, msaa=False)

        # initialize our simulator
        self.__action_timestep = 1./10.0
        self.__physics_timestep = 1./240.0
        render = 'gui'
        self.__simulator = Simulator(mode=render,
                        physics_timestep=self.__physics_timestep,
                        render_timestep=self.__action_timestep,
                        image_width=256,
                        image_height=256,
                        rendering_settings=settings)

        # initialize and load scene into simulation
        self.__scene_id = 'Rs'
        self.__scene = StaticIndoorScene(self.__scene_id,
                        trav_map_resolution=self.__resolution,
                        trav_map_erosion=self.__erosion,
                        build_graph=True, pybullet_load_texture=True)
        self.__simulator.import_scene(self.__scene)

        # initialize and load robot into simulation
        self.__robot = Turtlebot(config)
        self.__simulator.import_robot(self.__robot)

        self.__is_rnd_floor = False
        self.__floor_num = self.__get_rnd_floor()
        self.__num_particles = 5000

        is_plot = False
        if is_plot:
            # code related to displaying results in matplotlib
            fig = plt.figure(figsize=(7, 7))
            self.__plt_ax = fig.add_subplot(111)
            plt.ion()
            plt.show()

        self.__trav_map_plt = None
        self.__robot_gt_plts = {
            'robot_position': None,
            'robot_heading': None,
        }
        self.__robot_est_plts = {
            'robot_position': None,
            'robot_heading': None,
            'particles': None,
        }

    def step(self, action):
        """
        Override gym environment step() with custom logic
        """
        turtlebot.apply_action(action)
        s.step()
        rgb = s.renderer.render_robot_cameras(modes=('rgb'))

    def reset(self):
        """
        Override gym environment reset() with custom logic
        """

        # move robot away from the scene
        self.__robot.set_position([100.0, 100.0, 100.0])

        # scene reset
        self.reset_scene()

        # robot reset
        self.reset_robot()
        self.__simulator.sync()

        robot_state = self.__robot.calc_state()
        self.__gt_pose = robot_state[[0, 1, 5]]     # [x, y, theta]

        # get initial random particles
        self.__particles = self.__get_random_particles(self.__num_particles)
        self.__weights = np.full((self.__num_particles,), 1/self.__num_particles)

        # self.__simulator.step()
        # rgb = self.__simulator.renderer.render_robot_cameras(modes=('rgb'))
        return self.__gt_pose, self.__particles

    def render(self, mode='human'):
        """
        Override gym environment render() with custom logic
        """

        # environment map
        floor_map = self.__scene.floor_map[self.__floor_num]
        self.__trav_map_plt = self.__draw_floor_map(floor_map, self.__trav_map_plt)

        # ground truth robot pose and heading
        robot_pose = self.__gt_pose
        color = '#7B241C'
        position_plt, heading_plt = self.__robot_gt_plts['robot_position'], self.__robot_gt_plts['robot_heading']
        position_plt, heading_plt = self.__draw_robot_pose(robot_pose, color, position_plt, heading_plt)
        self.__robot_gt_plts['robot_position'], self.__robot_gt_plts['robot_heading'] = position_plt, heading_plt

        # estimated particles pose
        particles = self.__particles
        weights = self.__weights
        particles_plt = self.__robot_est_plts['particles']
        particles_plt = self.__draw_particle_poses(particles, weights, particles_plt)
        self.__robot_est_plts['particles'] = particles_plt

    def close(self):
        """
        Override gym environment close() with custom logic
        """
        self.__simulator.disconnect()
        super(iGibsonEnv, self).close()

        # to prevent plot from closing after environment is closed
        plt.ioff()
        plt.show()

    def reset_scene(self):
        """
        Reset environment scene to random floor
        """
        # choose random floor
        self.__floor_num = self.__get_rnd_floor()

        # scene reset
        self.__scene.reset_floor(self.__floor_num)

    def reset_robot(self):
        """
        Reset robot to random position and orientation on scene floor
        """

        reset_success = False
        max_trials = 100

        # cache pybullet state
        state_id = p.saveState()
        for i in range(max_trials):
            # get random robot inital pose and orientation
            pos, orn = self.__get_random_pose(self.__floor_num)
            reset_success = self.__test_valid_position(self.__robot, pos, orn)

            p.restoreState(state_id)
            if reset_success:
                break

        if not reset_success:
            print("WARNING: Failed to reset")

        p.removeState(state_id)
        init_pos = pos
        init_orn = orn

        self.__land(self.__robot, init_pos, init_orn)

    ###### private methods ######

    def __get_rnd_floor(self):
        """
        get random floor number
        :return integer: return random floor number or 0
        """

        # randomly sample floor
        if self.__is_rnd_floor:
            floor_num = self.__scene.get_random_floor()
        else:
            floor_num = 0

        return floor_num

    def __test_valid_position(self, robot, pos, orn):
        """
        Test if the robot can be placed with non collision
        :param gibson2.robots.turtlebot_robot.Turtlebot robot: Turtlebot
        :param pos: position of robot (xyz)
        :param orn: orientation of robot (xyzw)
        :return boolean: whether its valid or not
        """

        self.__set_pos_orn_with_z_offset(robot, pos, orn)

        robot.robot_specific_reset()
        robot.keep_still()

        body_id = robot.robot_ids[0]
        has_collision = self.__check_collision(body_id)

        return has_collision

    def __set_pos_orn_with_z_offset(self, robot, pos, orn):
        """
        Reset position and orientation for the robot (with correction of stable z)
        :param gibson2.robots.turtlebot_robot.Turtlebot robot: Turtlebot
        :param pos: position of robot (xyz)
        :param orn: orientation of robot (xyzw)
        """

        offset = self.__initial_pos_z_offset
        body_id = robot.robot_ids[0]

        # first set the correct orientation
        robot.set_position_orientation(pos, orn)

        # compute stable z based on this orientation
        stable_z = stable_z_on_aabb(body_id, [pos, pos])

        # change the z-value of position with stable_z + additional offset
        # in case the surface is not perfect smooth (has bumps)
        robot.set_position([pos[0], pos[1], stable_z + offset])

    def __check_collision(self, body_id):
        """
        Check for collision
        :param int body_id:
        """
        self.__simulator.step()

        collisions = list(p.getContactPoints(bodyA=body_id))

        return len(collisions) == 0

    def __land(self, robot, pos, orn):
        """
        Land the robot onto the floor, given a valid position and orientation
        :param gibson2.robots.turtlebot_robot.Turtlebot robot: Turtlebot
        :param pos: position of robot (xyz)
        :param orn: orientation of robot (xyzw)
        """

        self.__set_pos_orn_with_z_offset(robot, pos, orn)

        robot.robot_specific_reset()
        robot.keep_still()

        body_id = robot.robot_ids[0]

        land_success = False
        # land for maximum 1 second, should fall down ~5 meters
        max_simulator_step = int(1.0 / self.__action_timestep)
        for _ in range(max_simulator_step):
            self.__simulator.step()
            if len(p.getContactPoints(bodyA=body_id)) > 0:
                land_success = True
                break

        if not land_success:
            print("WARNING: Failed to land")

        robot.robot_specific_reset()
        robot.keep_still()

    def __get_random_particles(self, num_particles, robot_pose=None):
        """
        Sample random particles based on the scene
        :param robot_pose: ndarray indicating the robot pose
            if None, random particle poses are sampled using unifrom distribution
            otherwise, sampled using gaussian distribution around the robot_pose
        :param num_particles: integer indicating the number of random particles
        :return ndarray: random particle poses
        """

        particles = []
        if robot_pose is None:
            # uniform distribution
            sample_i = 0
            while sample_i < num_particles:
                pos, orn = self.__get_random_pose(self.__floor_num)
                # convert quaternion to euler
                orn = p.getEulerFromQuaternion(orn)     # [r, p, y]
                particles.append([pos[0], pos[1], orn[2]])  # [x, y, theta]
                sample_i = sample_i + 1
        else:
            # gaussian distribution
            raise ValueError('#TODO')

        particles = np.array(particles)
        return particles

    def __get_random_pose(self, floor_num):
        """
        Returns random position and orientation on the floor
        :param floor_num: integer indicating the floor
        :return tuple(ndarray, ndarray): random postition (xyz) and orientation (xyzw)
        """

        # randomly sample a position on the floor
        pos = self.__scene.get_random_point(floor_num)[1]    # [x, y, z]

        # randomly sample a orientation on the floor
        orn = [0, 0, np.random.uniform(0, np.pi * 2)]   # [r, p, y]
        orn = p.getQuaternionFromEuler(orn)     # [x, y, z, w]

        return pos, orn

    def __draw_floor_map(self, floor_map, map_plt):
        """
        Render the scene floor map
        :param ndarray floor_map: environment scene floor map
        :return matplotlib.image.AxesImage: updated plot of scene floor map
        """

        height, width = floor_map.shape
        orign_x, orign_y = 0, 0

        # offset the map to display correctly w.r.t origin
        x_max = width/2 + orign_x
        x_min = -width/2 + orign_x
        y_max = height/2 + orign_y
        y_min = -height/2 + orign_y
        extent = [x_min, x_max, y_min, y_max]

        if map_plt is None:
            # draw floor map
            floor_map = (255 - floor_map)   # invert image
            map_plt = self.__plt_ax.imshow(floor_map, cmap=plt.cm.binary, origin='lower', extent=extent)
        else:
            # do nothing
            pass
        return map_plt

    def __draw_particle_poses(self, particles, weights, particles_plt):
        """
        Render the particle poses on the scene floor map
        :param ndarray particles: estimates of particle pose
        :param ndarray weights: corresponding weights of particle pose estimates
        :param matplotlib.collections.PathCollection: plot of particle position color coded according to weights
        :return matplotlib.collections.PathCollection: updated plot of particles
        """

        pos = particles[:, 0:2] / self.__resolution # [num, x, y]
        color = cm.rainbow(weights)

        particles_plt = self.__robot_est_plts['particles']
        if particles_plt is None:
            # render particles positions woth color
            particles_plt = plt.scatter(pos[:, 0], pos[:, 1], s=10, c=color, alpha=.4)
        else:
            # update existing particles positions and color
            particles_plt.set_offsets(pos)
            particles_plt.set_color(color)

        return particles_plt

    def __draw_robot_pose(self, robot_pose, color, position_plt, heading_plt):
        """
        Render the robot pose on the scene floor map
        :param ndarray robot_pose: ndarray representing robot position (x, y) and heading (theta)
        :param matplotlib.patches.Wedge position_plt: plot of robot position
        :param matplotlib.lines.Line2D heading_plt: plot of robot heading
        :param str color: color used to render robot position and heading
        :return tuple(matplotlib.patches.Wedge, matplotlib.lines.Line2D): updated position and heading plot of robot
        """

        x, y, heading = robot_pose

        x = x / self.__resolution
        y = y / self.__resolution

        heading_len  = robot_radius = 1.0
        xdata = [x, x + (robot_radius + heading_len) * np.cos(heading)]
        ydata = [y, y + (robot_radius + heading_len) * np.sin(heading)]

        if position_plt == None:
            position_plt = Wedge((x, y), robot_radius, 0, 360, color=color, alpha=0.5)
            self.__plt_ax.add_artist(position_plt)
            heading_plt, = self.__plt_ax.plot(xdata, ydata, color=color, alpha=0.5)
        else:
            position_plt.update({'center': [x, y]})
            heading_plt.update({'xdata': xdata, 'ydata': ydata})

        return position_plt, heading_plt
