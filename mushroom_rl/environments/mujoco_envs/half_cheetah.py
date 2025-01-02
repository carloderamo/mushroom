from pathlib import Path

import numpy as np
import mujoco

from mushroom_rl.environments.mujoco import MuJoCo, ObservationType
from mushroom_rl.rl_utils.spaces import Box


class HalfCheetah(MuJoCo):
    """
    The HalfCheetah MuJoCo environment as presented in:
    "A Cat-Like Robot Real-Time Learning to Run". Pawel Wawrzynski. 2009.
    """

    def __init__(
        self,
        gamma=0.99,
        horizon=1000,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
        n_substeps=5,
        exclude_current_positions_from_observation=True,
        **viewer_params,
    ):
        """
        Constructor.

        """
        xml_path = (
            Path(__file__).resolve().parent / "data" / "half_cheetah" / "model.xml"
        ).as_posix()
        actuation_spec = ["bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]

        observation_spec = [
            ("z_pos", "rootz", ObservationType.JOINT_POS),
            ("y_pos", "rooty", ObservationType.JOINT_POS),
            ("bthigh_pos", "bthigh", ObservationType.JOINT_POS),
            ("bshin_pos", "bshin", ObservationType.JOINT_POS),
            ("bfoot_pos", "bfoot", ObservationType.JOINT_POS),
            ("fthigh_pos", "fthigh", ObservationType.JOINT_POS),
            ("fshin_pos", "fshin", ObservationType.JOINT_POS),
            ("ffoot_pos", "ffoot", ObservationType.JOINT_POS),
            ("x_vel", "rootx", ObservationType.JOINT_VEL),
            ("z_vel", "rootz", ObservationType.JOINT_VEL),
            ("y_vel", "rooty", ObservationType.JOINT_VEL),
            ("bthigh_vel", "bthigh", ObservationType.JOINT_VEL),
            ("bshin_vel", "bshin", ObservationType.JOINT_VEL),
            ("bfoot_vel", "bfoot", ObservationType.JOINT_VEL),
            ("fthigh_vel", "fthigh", ObservationType.JOINT_VEL),
            ("fshin_vel", "fshin", ObservationType.JOINT_VEL),
            ("ffoot_vel", "ffoot", ObservationType.JOINT_VEL),
        ]

        additional_data_spec = [
            ("x_pos", "rootx", ObservationType.JOINT_POS),
            ("torso_vel", "torso", ObservationType.BODY_VEL_WORLD),
        ]

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        super().__init__(
            xml_file=xml_path,
            gamma=gamma,
            horizon=horizon,
            observation_spec=observation_spec,
            actuation_spec=actuation_spec,
            additional_data_spec=additional_data_spec,
            n_substeps=n_substeps,
            **viewer_params,
        )

    def _modify_mdp_info(self, mdp_info):
        if not self._exclude_current_positions_from_observation:
            self.obs_helper.add_obs("x_pos", 1)
        mdp_info = super()._modify_mdp_info(mdp_info)
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    def _create_observation(self, obs):
        obs = super()._create_observation(obs)
        if not self._exclude_current_positions_from_observation:
            x_pos = self._read_data("x_pos")
            obs = np.concatenate([obs, x_pos])
        return obs

    def is_absorbing(self, obs):
        return False

    def _get_forward_reward(self):
        forward_reward = self._read_data("torso_vel")[3]
        return self._forward_reward_weight * forward_reward

    def _get_ctrl_cost(self, action):
        ctrl_cost = np.sum(np.square(action))
        return self._ctrl_cost_weight * ctrl_cost

    def reward(self, obs, action, next_obs, absorbing):
        forward_reward = self._get_forward_reward()
        ctrl_cost = self._get_ctrl_cost(action)
        reward = forward_reward - ctrl_cost
        return reward

    def _generate_noise(self):
        self._data.qpos[:] = self._data.qpos + np.random.uniform(
            -self._reset_noise_scale, self._reset_noise_scale, self._model.nq
        )
        self._data.qvel[:] = self._data.qvel + np.random.uniform(
            -self._reset_noise_scale, self._reset_noise_scale, self._model.nv
        )

    def setup(self, obs):
        super().setup(obs)

        self._generate_noise()

        mujoco.mj_forward(self._model, self._data)  # type: ignore

    def _create_info_dictionary(self, obs, action):
        info = {
            "forward_reward": self._get_forward_reward(),
        }
        info["ctrl_cost"] = self._get_ctrl_cost(action)
        return info
