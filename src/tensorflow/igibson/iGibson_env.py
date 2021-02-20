#!/usr/bin/env python3

from gibson2.envs.env_base import BaseEnv

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
