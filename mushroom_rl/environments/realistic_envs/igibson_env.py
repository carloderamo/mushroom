import os
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import igibson
    from igibson.envs.igibson_env import iGibsonEnv
    from igibson.utils.utils import parse_config

import gym
import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.environments import Gym
from mushroom_rl.utils.spaces import Discrete, Box
from mushroom_rl.utils.frames import LazyFrames, preprocess_frame


class iGibsonWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = env.observation_space.spaces['rgb']

    def observation(self, observation):
        return observation['rgb'] * 255.


class iGibson(Gym):
    """
    Interface for iGibson https://github.com/StanfordVL/iGibson

    There are both navigation and interaction tasks.
    Observations are pixel images of what the agent sees in front of itself.
    Image resolution is specified in the config file.
    By default, actions are continuous, but can be discretized automatically
    using a flag.

    Scene and task details, are defined in the config yaml file.

    """
    def __init__(self, horizon=None, gamma=0.99, is_discrete=False, width=None, height=None):
        """
        Constructor.

        Args:
             scene_name (str): name of the Replica scene where the agent is placed;
             config_path (str): path to the .yaml file specifying the task (see habitat-lab/configs/tasks/);
             horizon (int, None): the horizon;
             gamma (float, 0.99): the discount factor;

        """
        # MDP creation
        self._not_pybullet = True
        self._first = True

        config_file = os.path.join(igibson.root_path, 'test', 'test_house.yaml')

        env = iGibsonEnv(config_file=config_file, mode='headless')
        config = parse_config(config_file)
        config['is_discrete'] = is_discrete
        if horizon is not None:
            config['max_step'] = horizon
        else:
            horizon = config['max_step']
            config['max_step'] = horizon + 1 # Hack to ignore gym time limit

        if width is not None:
            config['image_width'] = width
        if height is not None:
            config['image_height'] = height

        env.config = config
        env.simulator.reload()
        env.load()

        env = iGibsonWrapper(env)

        self.env = env

        self._img_size = env.observation_space.shape[0:2]

        # MDP properties
        action_space = self.env.action_space
        observation_space = Box(
            low=0., high=255., shape=(3, self._img_size[1], self._img_size[0]))
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        if isinstance(action_space, Discrete):
            self._convert_action = lambda a: a[0]
        else:
            self._convert_action = lambda a: a

        Environment.__init__(self, mdp_info)

    def reset(self, state=None):
        assert state is None, 'Cannot set iGibson state'
        return self._convert_observation(np.atleast_1d(self.env.reset()))

    def step(self, action):
        action = self._convert_action(action)
        obs, reward, absorbing, info = self.env.step(action)
        return self._convert_observation(np.atleast_1d(obs)), reward, absorbing, info

    def stop(self):
        pass

    @staticmethod
    def _convert_observation(observation):
        return observation.transpose((2, 0, 1))