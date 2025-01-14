from pathlib import Path

import numpy as np
import mujoco

from mushroom_rl.environments.mujoco import ObservationType
from mushroom_rl.rl_utils.spaces import Box
from mushroom_rl.environments.mujoco_envs.franka_panda.panda import Panda


class Pick(Panda):
    def __init__(
        self,
        gamma: float = 0.99,
        horizon: int = 200,
        gripper_cube_distance_reward_weight: float = 0.5,
        cube_goal_distance_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0,
        contact_cost_weight: float = 0,
        n_substeps: int = 10,
        cube_reset_noise: float = 0.1,
        **viewer_params,
    ):
        xml_path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "panda"
            / "pick"
            / "pick.xml"
        ).as_posix()

        additional_data_spec = [
            ("cube_pose", "cube", ObservationType.JOINT_POS),
            ("cube_vel", "cube", ObservationType.JOINT_VEL),
            ("goal_pos", "goal", ObservationType.BODY_POS),
        ]

        collision_groups = [
            ("cube", ["cube"]),
            ("table", ["table"]),
        ]

        self._gripper_cube_distance_reward_weight = gripper_cube_distance_reward_weight
        self._cube_goal_distance_reward_weight = cube_goal_distance_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._cube_reset_noise = cube_reset_noise
        self._goal_rot = np.array([1.0, 0.0, 0.0, 0.0])

        super().__init__(
            xml_path,
            gamma=gamma,
            horizon=horizon,
            additional_data_spec=additional_data_spec,
            collision_groups=collision_groups,
            n_substeps=n_substeps,
            **viewer_params,
        )

    def _modify_mdp_info(self, mdp_info):
        self.obs_helper.add_obs("left_fingertip_pos", 3)
        self.obs_helper.add_obs("right_fingertip_pos", 3)
        self.obs_helper.add_obs("left_cube_pos", 3)
        self.obs_helper.add_obs("right_cube_pos", 3)
        self.obs_helper.add_obs("cube_pos", 3)
        self.obs_helper.add_obs("cube_rot", 4)
        self.obs_helper.add_obs("cube_vel", 6)
        self.obs_helper.add_obs("goal_pos", 3)
        self.obs_helper.add_obs("goal_rot", 4)
        self.obs_helper.add_obs("collision_force", 1)
        self.obs_helper.add_obs("cube_in_hand", 1)

        mdp_info = super()._modify_mdp_info(mdp_info)
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    def _create_observation(self, obs):
        obs = super()._create_observation(obs)

        left_fingertip_pos = self._read_data("left_fingertip_pos")
        right_fingertip_pos = self._read_data("right_fingertip_pos")

        left_cube_pos = self._read_data("left_cube_pos")
        right_cube_pos = self._read_data("right_cube_pos")

        cube_pose = self._read_data("cube_pose")
        cube_vel = self._read_data("cube_vel")
        goal_pos = self._read_data("goal_pos")
        goal_rot = self._goal_rot

        collision_force = np.array(
            [
                np.sum(
                    np.square(
                        self._get_collision_force("hand", "floor")
                        + self._get_collision_force("left_finger", "floor")
                        + self._get_collision_force("right_finger", "floor")
                        + self._get_collision_force("robot", "floor")
                        + self._get_collision_force("hand", "robot")
                        + self._get_collision_force("left_finger", "robot")
                        + self._get_collision_force("right_finger", "robot")
                        + self._get_collision_force("hand", "right_finger")
                        + self._get_collision_force("hand", "left_finger")
                        + self._get_collision_force("hand", "cube")
                        + self._get_collision_force("robot", "cube")
                        + self._get_collision_force("hand", "table")
                        + self._get_collision_force("robot", "table")
                        + self._get_collision_force("left_finger", "table")
                        + self._get_collision_force("right_finger", "table")
                    )
                )
            ]
        )

        cube_in_hand = int(self._is_cube_in_hand())
        cube_in_hand = np.array([cube_in_hand])  # type: ignore
        obs = np.concatenate(
            [
                obs,
                left_fingertip_pos,
                right_fingertip_pos,
                left_cube_pos,
                right_cube_pos,
                cube_pose,
                cube_vel,
                goal_pos,
                goal_rot,
                collision_force,
                cube_in_hand,
            ]
        )
        return obs

    def _get_cube_goal_distance(self, obs):
        cube_pos = self.obs_helper.get_from_obs(obs, "cube_pos")
        goal_pos = self.obs_helper.get_from_obs(obs, "goal_pos")
        return np.linalg.norm(cube_pos - goal_pos).item()

    def _get_gripper_cube_distance(self, obs):
        left_fingertip_pos = self.obs_helper.get_from_obs(obs, "left_fingertip_pos")
        left_cube_pos = self.obs_helper.get_from_obs(obs, "left_cube_pos")

        right_fingertip_pos = self.obs_helper.get_from_obs(obs, "right_fingertip_pos")
        right_cube_pos = self.obs_helper.get_from_obs(obs, "right_cube_pos")

        gripper_cube_distance = (
            np.linalg.norm(left_fingertip_pos - left_cube_pos).item()
            + np.linalg.norm(right_fingertip_pos - right_cube_pos).item()
        )
        return gripper_cube_distance

    def _get_gripper_cube_distance_reward(self, obs):
        gripper_cube_distance = self._get_gripper_cube_distance(obs)
        distance_reward = -gripper_cube_distance
        return self._gripper_cube_distance_reward_weight * distance_reward

    def _get_cube_goal_distance_reward(self, obs):
        cube_goal_distance = self._get_cube_goal_distance(obs)
        distance_reward = -cube_goal_distance
        return self._cube_goal_distance_reward_weight * distance_reward

    def _get_ctrl_cost(self, action):
        # [:-1] to exclude the actions of the fingers
        ctrl_cost = -np.sum(np.square(action[:-1]))
        return self._ctrl_cost_weight * ctrl_cost

    def _get_contact_cost(self, obs):
        collision_forces = self.obs_helper.get_from_obs(obs, "collision_force").item()
        contact_cost = -collision_forces
        return self._contact_cost_weight * contact_cost

    def reward(self, obs, action, next_obs, absorbing):
        gripper_cube_distance_reward = self._get_gripper_cube_distance_reward(next_obs)
        cube_goal_distance_reward = self._get_cube_goal_distance_reward(next_obs)
        ctrl_cost = self._get_ctrl_cost(action)
        contact_cost = self._get_contact_cost(next_obs)

        cube_in_hand_reward = 0.05

        reward = (
            gripper_cube_distance_reward
            + cube_goal_distance_reward
            + ctrl_cost
            + contact_cost
            + cube_in_hand_reward
        )
        return reward

    def is_absorbing(self, obs):
        cube_pos = self.obs_helper.get_from_obs(obs, "cube_pos")
        cube_rot = self.obs_helper.get_from_obs(obs, "cube_rot")
        goal_pos = self.obs_helper.get_from_obs(obs, "goal_pos")

        cube_goal_distance = np.linalg.norm(goal_pos - cube_pos)

        is_cube_at_goal = cube_goal_distance < 0.05
        is_cube_aligned = (
            self.quaternion_distance(cube_rot, self._goal_rot) < 0.3 or True
        )
        return (is_cube_at_goal and is_cube_aligned) or self._check_collision(
            "cube", "floor"
        )

    def setup(self, obs):
        super().setup(obs)
        # self._randomize_goal_pos()
        # self._randomize_cube_pos()
        mujoco.mj_forward(self._model, self._data)  # type: ignore

    def _create_info_dictionary(self, obs, action):
        info = super()._create_info_dictionary(obs)
        info["gripper_cube_distance_reward"] = self._get_gripper_cube_distance_reward(
            obs
        )
        info["cube_goal_distance_reward"] = self._get_cube_goal_distance_reward(obs)
        info["ctrl_cost"] = self._get_ctrl_cost(action)
        info["contact_cost"] = self._get_contact_cost(obs)
        info["gripper_cube_distance"] = self._get_gripper_cube_distance(obs)
        info["cube_goal_distance"] = self._get_cube_goal_distance(obs)
        info["is_cube_in_hand"] = self._is_cube_in_hand()
        return info

    def _is_cube_in_hand(self):
        # left_cube_pos = self._read_data("left_cube_pos")
        # right_cube_pos = self._read_data("right_cube_pos")
        # left_fingertip_pos = self._read_data("left_fingertip_pos")
        # right_fingertip_pos = self._read_data("right_fingertip_pos")

        is_cube_in_hand = self._check_collision(
            "cube", "left_finger"
        ) and self._check_collision("cube", "right_finger")
        return is_cube_in_hand

    def _randomize_goal_pos(self):
        self._data.mocap_pos[0][:2] += np.random.uniform(-0.1, 0.1, 2)

    def _randomize_cube_pos(self):
        cube_pos = self._read_data("cube_pose")
        cube_pos[:2] += np.random.uniform(
            -self._cube_reset_noise, self._cube_reset_noise, 2
        )
        self._write_data("cube_pose", cube_pos)

    def normalize_quaternion(self, q):
        norm = np.linalg.norm(q)
        return q / norm

    def quaternion_distance(self, cube_rot, goal_rot):
        cube_rot = self.normalize_quaternion(cube_rot)

        cos_half_angle = np.abs(np.dot(cube_rot, goal_rot))

        theta = 2 * np.arccos(cos_half_angle)
        return theta

    # def _compute_action(self, obs, action):
    #     action = super()._compute_action(obs, action)
    #     action = np.zeros_like(action)
    #     return action
