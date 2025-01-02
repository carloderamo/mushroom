from pathlib import Path
from typing import Tuple

import numpy as np
from mushroom_rl.environments.mujoco import MuJoCo, ObservationType
from mushroom_rl.rl_utils.spaces import Box
import mujoco


class Ant(MuJoCo):
    """
    The Ant MuJoCo environment as presented in:
    "High-Dimensional Continuous Control Using Generalized Advantage Estimation". John Schulman et. al.. 2015.
    and implemented in Gymnasium
    """

    def __init__(
        self,
        gamma: float = 0.99,
        horizon: int = 1000,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.5,
        contact_cost_weight: float = 5e-4,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.2, 1.0),
        contact_force_range: Tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 0.1,
        n_substeps: int = 5,
        exclude_current_positions_from_observation: bool = True,
        use_contact_forces: bool = False,
        **viewer_params,
    ):
        """
        Constructor.

        """
        xml_path = (
            Path(__file__).resolve().parent / "data" / "ant" / "model.xml"
        ).as_posix()

        # This order is correct as specified in gymnasium
        actuation_spec = [
            "hip_4",
            "ankle_4",
            "hip_1",
            "ankle_1",
            "hip_2",
            "ankle_2",
            "hip_3",
            "ankle_3",
        ]

        observation_spec = [
            ("root_pose", "root", ObservationType.JOINT_POS),
            ("hip_1_pos", "hip_1", ObservationType.JOINT_POS),
            ("ankle_1_pos", "ankle_1", ObservationType.JOINT_POS),
            ("hip_2_pos", "hip_2", ObservationType.JOINT_POS),
            ("ankle_2_pos", "ankle_2", ObservationType.JOINT_POS),
            ("hip_3_pos", "hip_3", ObservationType.JOINT_POS),
            ("ankle_3_pos", "ankle_3", ObservationType.JOINT_POS),
            ("hip_4_pos", "hip_4", ObservationType.JOINT_POS),
            ("ankle_4_pos", "ankle_4", ObservationType.JOINT_POS),
            ("root_vel", "root", ObservationType.JOINT_VEL),
            ("hip_1_vel", "hip_1", ObservationType.JOINT_VEL),
            ("ankle_1_vel", "ankle_1", ObservationType.JOINT_VEL),
            ("hip_2_vel", "hip_2", ObservationType.JOINT_VEL),
            ("ankle_2_vel", "ankle_2", ObservationType.JOINT_VEL),
            ("hip_3_vel", "hip_3", ObservationType.JOINT_VEL),
            ("ankle_3_vel", "ankle_3", ObservationType.JOINT_VEL),
            ("hip_4_vel", "hip_4", ObservationType.JOINT_VEL),
            ("ankle_4_vel", "ankle_4", ObservationType.JOINT_VEL),
        ]

        additional_data_spec = [
            ("torso_pos", "torso", ObservationType.BODY_POS),
            ("torso_vel", "torso", ObservationType.BODY_VEL_WORLD),
        ]

        collision_groups = [
            ("torso", ["torso_geom"]),
            ("floor", ["floor"]),
        ]

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._use_contact_forces = use_contact_forces

        super().__init__(
            xml_file=xml_path,
            gamma=gamma,
            horizon=horizon,
            observation_spec=observation_spec,
            actuation_spec=actuation_spec,
            collision_groups=collision_groups,
            additional_data_spec=additional_data_spec,
            n_substeps=n_substeps,
            **viewer_params,
        )

    def _modify_mdp_info(self, mdp_info):
        if self._exclude_current_positions_from_observation:
            self.obs_helper.remove_obs("root_pose", 0)
            self.obs_helper.remove_obs("root_pose", 1)
        if self._use_contact_forces:
            self.obs_helper.add_obs("collision_force", 6)
        mdp_info = super()._modify_mdp_info(mdp_info)
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    def _create_observation(self, obs):
        obs = super()._create_observation(obs)
        if self._use_contact_forces:
            collision_force = self._get_collision_force("torso", "floor")
            obs = np.concatenate([obs, collision_force])
        return obs

    def _is_finite(self):
        states = self.get_states()
        return np.isfinite(states).all()

    def _is_within_z_range(self):
        z_pos = self._read_data("torso_pos")[2]
        min_z, max_z = self._healthy_z_range
        return min_z <= z_pos <= max_z

    def _is_healthy(self):
        is_healthy = self._is_finite() and self._is_within_z_range()
        return is_healthy

    def is_absorbing(self, obs):
        absorbing = self._terminate_when_unhealthy and not self._is_healthy()
        return absorbing

    def _get_healthy_reward(self, obs):
        return (
            self._terminate_when_unhealthy and self._is_healthy()
        ) * self._healthy_reward

    def _get_forward_reward(self):
        forward_reward = self._read_data("torso_vel")[3]
        return self._forward_reward_weight * forward_reward

    def _get_ctrl_cost(self, action):
        ctrl_cost = np.sum(np.square(action))
        return self._ctrl_cost_weight * ctrl_cost

    def _get_contact_cost(self, obs):
        collision_force = self.obs_helper.get_from_obs(obs, "collision_force")
        contact_cost = np.sum(
            np.square(np.clip(collision_force, *self._contact_force_range))
        )
        return self._contact_cost_weight * contact_cost

    def reward(self, obs, action, next_obs, absorbing):
        healthy_reward = self._get_healthy_reward(next_obs)
        forward_reward = self._get_forward_reward()
        cost = self._get_ctrl_cost(action)
        if self._use_contact_forces:
            contact_cost = self._get_contact_cost(next_obs)
            cost += contact_cost
        reward = healthy_reward + forward_reward - cost
        return reward

    def _generate_noise(self):
        self._data.qpos[:] = self._data.qpos + np.random.uniform(
            -self._reset_noise_scale, self._reset_noise_scale, size=self._model.nq
        )

        self._data.qvel[:] = (
            self._data.qvel
            + self._reset_noise_scale * np.random.standard_normal(self._model.nv)
        )

    def setup(self, obs):
        super().setup(obs)

        self._generate_noise()

        mujoco.mj_forward(self._model, self._data)  # type: ignore

    def _create_info_dictionary(self, obs, action):
        info = {
            "healthy_reward": self._get_healthy_reward(obs),
            "forward_reward": self._get_forward_reward(),
        }
        info["ctrl_cost"] = self._get_ctrl_cost(action)
        if self._use_contact_forces:
            info["contact_cost"] = self._get_contact_cost(obs)
        return info

    def get_states(self):
        """Return the position and velocity joint states of the model"""
        return np.concatenate([self._data.qpos.flat, self._data.qvel.flat])
