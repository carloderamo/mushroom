from pathlib import Path

import numpy as np
import mujoco

from mushroom_rl.environments.mujoco import ObservationType
from mushroom_rl.rl_utils.spaces import Box
from mushroom_rl.environments.mujoco_envs.franka_panda.panda import Panda


class Push(Panda):
    def __init__(
        self,
        gamma: float = 0.99,
        horizon: int = 200,
        gripper_cube_distance_reward_weight: float = 0.5,
        cube_goal_distance_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.0,
        contact_cost_weight: float = 0,
        n_substeps: int = 10,
        goal_noise_scale: float = 0.2,
        contact_force_range: tuple[float, float] = (-25.0, 25.0),
        **viewer_params,
    ):

        xml_path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "panda"
            / "push"
            / "push.xml"
        ).as_posix()

        actuation_spec = [
            "actuator1",
            "actuator2",
            "actuator3",
            "actuator4",
            "actuator5",
            "actuator6",
            "actuator7",
        ]

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

        self._goal_noise_scale = goal_noise_scale
        self._contact_force_range = contact_force_range

        if viewer_params:
            viewer_params["camera_params"]["static"]["elevation"] = -30
            viewer_params["camera_params"]["static"]["distance"] = 2
            viewer_params["camera_params"]["static"]["lookat"] = (0.5, 0.0, 0.5)
            viewer_params["camera_params"]["static"]["azimuth"] = 180

        super().__init__(
            xml_path,
            gamma=gamma,
            horizon=horizon,
            actuation_spec=actuation_spec,
            additional_data_spec=additional_data_spec,
            collision_groups=collision_groups,
            n_substeps=n_substeps,
            **viewer_params,
        )

    def _modify_mdp_info(self, mdp_info):
        self.obs_helper.add_obs("cube_pos", 3)
        self.obs_helper.add_obs("cube_rot", 4)
        self.obs_helper.add_obs("cube_vel", 6)
        self.obs_helper.add_obs("goal_pos", 3)
        self.obs_helper.add_obs("collision_force", 1)
        mdp_info = super()._modify_mdp_info(mdp_info)
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    def _create_observation(self, obs):
        obs = super()._create_observation(obs)
        cube_pose = self._read_data("cube_pose")
        cube_vel = self._read_data("cube_vel")
        goal_pos = self._read_data("goal_pos")
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
        obs = np.concatenate([obs, cube_pose, cube_vel, goal_pos, collision_force])
        return obs

    def _get_cube_goal_distance(self, obs):
        cube_pos = self.obs_helper.get_from_obs(obs, "cube_pos")
        goal_pos = self.obs_helper.get_from_obs(obs, "goal_pos")
        return np.linalg.norm(cube_pos - goal_pos).item()

    def _get_gripper_cube_distance(self, obs):
        gripper_pos = self.obs_helper.get_from_obs(obs, "gripper_pos")
        cube_pos = self.obs_helper.get_from_obs(obs, "cube_pos").copy()
        cube_pos[1] += 0.025
        return np.linalg.norm(gripper_pos - cube_pos).item()

    def _get_gripper_cube_distance_reward(self, obs):
        gripper_cube_distance = self._get_gripper_cube_distance(obs)
        distance_reward = -gripper_cube_distance
        return self._gripper_cube_distance_reward_weight * distance_reward

    def _get_cube_goal_distance_reward(self, obs):
        cube_goal_distance = self._get_cube_goal_distance(obs)
        distance_reward = -cube_goal_distance
        distance_reward = max(distance_reward, -0.255)
        return self._cube_goal_distance_reward_weight * distance_reward

    def _get_ctrl_cost(self, action):
        # [:-1] to exclude the actions of the fingers
        ctrl_cost = -np.sum(np.square(action[:-1]))
        return self._ctrl_cost_weight * ctrl_cost

    def _get_contact_cost(self, obs):
        collision_force = self.obs_helper.get_from_obs(obs, "collision_force")
        contact_cost = -collision_force
        return self._contact_cost_weight * contact_cost

    def reward(self, obs, action, next_obs, absorbing):
        gripper_cube_distance_reward = self._get_gripper_cube_distance_reward(next_obs)
        cube_goal_distance_reward = self._get_cube_goal_distance_reward(next_obs)
        ctrl_cost = self._get_ctrl_cost(action)
        contact_cost = self._get_contact_cost(next_obs)
        reward = (
            cube_goal_distance_reward
            + gripper_cube_distance_reward
            + ctrl_cost
            + contact_cost
        )
        return reward

    def is_absorbing(self, obs):
        cube_vel = self.obs_helper.get_from_obs(obs, "cube_vel")
        cube_goal_distance = self._get_cube_goal_distance(obs)
        is_cube_at_goal = cube_goal_distance < 0.05
        if is_cube_at_goal:
            print(np.sum(np.square(cube_vel)))
        is_cube_moving = np.sum(np.square(cube_vel)) > 0.02
        return is_cube_at_goal and not is_cube_moving

    def setup(self, obs):
        super().setup(obs)
        # self._randomize_goal_position()
        mujoco.mj_forward(self._model, self._data)  # type: ignore

    def _create_info_dictionary(self, obs, action):
        info = super()._create_info_dictionary(obs)
        self._get_contact_cost(obs)
        info["gripper_cube_distance_reward"] = self._get_gripper_cube_distance_reward(
            obs
        )
        info["cube_goal_distance_reward"] = self._get_cube_goal_distance_reward(obs)
        info["ctrl_cost"] = self._get_ctrl_cost(action)
        info["contact_cost"] = self._get_contact_cost(obs)
        info["gripper_cube_distance"] = self._get_gripper_cube_distance(obs)
        info["cube_goal_distance"] = self._get_cube_goal_distance(obs)
        return info

    def _randomize_goal_position(self):
        self._data.mocap_pos[0][0] += np.random.uniform(
            -self._goal_noise_scale, self._goal_noise_scale
        )
