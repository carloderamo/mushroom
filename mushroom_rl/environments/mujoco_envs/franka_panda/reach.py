import math
from pathlib import Path
import random

import numpy as np
import mujoco

from mushroom_rl.environments.mujoco import ObservationType
from mushroom_rl.rl_utils.spaces import Box
from mushroom_rl.environments.mujoco_envs.franka_panda.panda import Panda


class Reach(Panda):

    def __init__(
        self,
        gamma: float = 0.99,
        horizon: int = 200,
        gripper_goal_distance_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        n_substeps: int = 10,
        goal_distance_threshold: float = 0.1,
        **viewer_params,
    ):

        xml_path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "panda"
            / "reach"
            / "reach.xml"
        ).as_posix()

        additional_data_spec = [
            ("goal_pos", "goal", ObservationType.BODY_POS),
        ]

        self._gripper_goal_distance_reward_weight = gripper_goal_distance_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._goal_distance_threshold = goal_distance_threshold

        if viewer_params:
            viewer_params["camera_params"]["static"]["elevation"] = -30
            viewer_params["camera_params"]["static"]["distance"] = 4

        super().__init__(
            xml_path,
            gamma=gamma,
            horizon=horizon,
            additional_data_spec=additional_data_spec,
            n_substeps=n_substeps,
            **viewer_params,
        )

    def _modify_mdp_info(self, mdp_info):
        self.obs_helper.add_obs("goal_pos", 3)
        mdp_info = super()._modify_mdp_info(mdp_info)
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    def _create_observation(self, obs):
        obs = super()._create_observation(obs)
        goal_pos = self._read_data("goal_pos")
        obs = np.concatenate([obs, goal_pos])
        return obs

    def _get_goal_pos(self, obs):
        return self.obs_helper.get_from_obs(obs, "goal_pos")

    def _get_gripper_pos(self, obs):
        return self.obs_helper.get_from_obs(obs, "gripper_pos")

    def _get_gripper_goal_distance(self, obs):
        gripper_pos = self._get_gripper_pos(obs)
        goal_pos = self._get_goal_pos(obs)
        return np.linalg.norm(goal_pos - gripper_pos)

    def _get_gripper_goal_distance_reward(self, obs):
        gripper_goal_distance = self._get_gripper_goal_distance(obs)
        distance_reward = -gripper_goal_distance
        return self._gripper_goal_distance_reward_weight * distance_reward

    def _get_ctrl_cost(self, action):
        # [:-1] to exclude the actions of the fingers
        ctrl_cost = np.sum(np.square(action[:-1]))
        return self._ctrl_cost_weight * ctrl_cost

    def reward(self, obs, action, next_obs, absorbing):
        if absorbing:
            return 100
        gripper_goal_distance_reward = self._get_gripper_goal_distance_reward(next_obs)
        ctrl_cost = self._get_ctrl_cost(action)
        reward = gripper_goal_distance_reward - ctrl_cost
        return reward

    def is_absorbing(self, obs):
        return self._get_gripper_goal_distance(obs) < self._goal_distance_threshold

    def setup(self, obs):
        super().setup(obs)
        self._randomize_goal_position()
        mujoco.mj_forward(self._model, self._data)  # type: ignore

    def _create_info_dictionary(self, obs):
        info = super()._create_info_dictionary(obs)
        info["gripper_goal_distance_reward"] = self._get_gripper_goal_distance(obs)
        # info["ctrl_cost"] = self.ctrl_cost(action)
        return info

    def _randomize_goal_position(self):
        random_workspace_pos = self.sample_workspace()
        self._data.mocap_pos[0][:] = random_workspace_pos

    def sample_workspace(self):
        # Workspace dimensions
        radius = 0.8  # mm
        workspace_height = 1  # mm

        # Generate random angle and radius
        angle = random.uniform(-np.pi / 2, np.pi / 2)
        r = random.uniform(0.3, radius)

        # Convert polar coordinates to Cartesian
        x = r * math.cos(angle)
        y = r * math.sin(angle)

        # Generate random height within the workspace
        z = random.uniform(0, workspace_height)

        return x, y, z
