import numpy as np
import mujoco

from mushroom_rl.rl_utils.spaces import Box
from mushroom_rl.environments.mujoco_envs.air_hockey.base import AirHockeyBase
from mushroom_rl.utils.mujoco import forward_kinematics


class AirHockeySingle(AirHockeyBase):
    """
    Base class for single agent air hockey tasks.

    """

    def __init__(
        self,
        gamma=0.99,
        horizon=120,
        env_noise=False,
        obs_noise=False,
        timestep=1 / 240.0,
        n_intermediate_steps=1,
        **viewer_params,
    ):
        """
        Constructor.

        """
        self.init_state = np.array([-0.9273, 0.9273, np.pi / 2])
        super().__init__(
            gamma=gamma,
            horizon=horizon,
            env_noise=env_noise,
            n_agents=1,
            obs_noise=obs_noise,
            timestep=timestep,
            n_intermediate_steps=n_intermediate_steps,
            **viewer_params,
        )

        # Remove z position and quaternion from puck pos
        self.obs_helper.remove_obs("puck_pos", 2)
        self.obs_helper.remove_obs("puck_pos", 3)
        self.obs_helper.remove_obs("puck_pos", 4)
        self.obs_helper.remove_obs("puck_pos", 5)
        self.obs_helper.remove_obs("puck_pos", 6)

        # Remove linear z velocity and angular velocity around x and y
        self.obs_helper.remove_obs("puck_vel", 2)
        self.obs_helper.remove_obs("puck_vel", 3)
        self.obs_helper.remove_obs("puck_vel", 4)

        self.obs_helper.add_obs("collision_robot_1_puck", 1, 0, 1)
        self.obs_helper.add_obs("collision_short_sides_rim_puck", 1, 0, 1)

        self._mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())

        self.has_hit = False
        self.has_bounce = False

    def get_puck(self, obs):
        """
        Getting the puck properties from the observations
        Args:
            obs: The current observation

        Returns:
            ([pos_x, pos_y], [lin_vel_x, lin_vel_y], ang_vel_z)

        """
        puck_pos = self.obs_helper.get_from_obs(obs, "puck_pos")
        puck_lin_vel = self.obs_helper.get_from_obs(obs, "puck_vel")[1:]
        puck_ang_vel = self.obs_helper.get_from_obs(obs, "puck_vel")[0]
        return puck_pos, puck_lin_vel, puck_ang_vel

    def get_ee(self):
        """
        Getting the ee properties from the current internal state

        Returns:
            ([pos_x, pos_y, pos_z], [ang_vel_x, ang_vel_y, ang_vel_z, lin_vel_x, lin_vel_y, lin_vel_z])

        """
        ee_pos = self._read_data("robot_1/ee_pos")

        ee_vel = self._read_data("robot_1/ee_vel")

        return ee_pos, ee_vel

    def _modify_observation(self, obs):
        new_obs = obs.copy()
        self._puck_2d_in_robot_frame(
            self.obs_helper.get_from_obs(new_obs, "puck_pos"), self.agents[0]["frame"]
        )

        self._puck_2d_in_robot_frame(
            self.obs_helper.get_from_obs(new_obs, "puck_vel"),
            self.agents[0]["frame"],
            type="vel",
        )

        if self.obs_noise:
            self.obs_helper.get_from_obs(new_obs, "puck_pos")[:] += (
                np.random.randn(2) * 0.001
            )

        return new_obs

    def setup(self, obs):
        self.has_hit = False
        self.has_bounce = False

        for i in range(3):
            self._data.joint("planar_robot_1/joint_" + str(i + 1)).qpos = (
                self.init_state[i]
            )

        super().setup(obs)
        mujoco.mj_fwdPosition(self._model, self._data)

    def _simulation_post_step(self):
        if not self.has_hit:
            self.has_hit = self._check_collision("puck", "robot_1/ee")

        if not self.has_bounce:
            self.has_bounce = self._check_collision("puck", "rim_short_sides")

    def _create_observation(self, state):
        obs = super(AirHockeySingle, self)._create_observation(state)
        return np.append(obs, [self.has_hit, self.has_bounce])

    def _create_info_dictionary(self, obs, action):
        constraints = {}
        q_pos = self.obs_helper.get_joint_pos_from_obs(obs)
        q_vel = self.obs_helper.get_joint_vel_from_obs(obs)

        x_pos, _ = forward_kinematics(
            self.robot_model, self.robot_data, q_pos, "planar_robot_1/body_ee"
        )

        # Translate to table space
        ee_pos = x_pos + self.agents[0]["frame"][:3, 3]

        # ee_constraint: force the ee to stay within the bounds of the table
        # 1 Dimension on x direction: x > x_lb
        # 2 Dimension on y direction: y > y_lb, y < y_ub
        x_lb = -(
            self.env_spec["table"]["length"] / 2 + self.env_spec["mallet"]["radius"]
        )
        y_lb = -(
            self.env_spec["table"]["width"] / 2 - self.env_spec["mallet"]["radius"]
        )
        y_ub = self.env_spec["table"]["width"] / 2 - self.env_spec["mallet"]["radius"]

        constraints["ee_constraints"] = np.array(
            [-ee_pos[0] + x_lb, -ee_pos[1] + y_lb, ee_pos[1] - y_ub]
        )

        # joint_pos_constraint: stay within the robots joint position limits
        constraints["joint_pos_constraints"] = np.zeros(6)
        constraints["joint_pos_constraints"][:3] = (
            q_vel - self.obs_helper.get_joint_pos_limits()[1]
        )
        constraints["joint_pos_constraints"][3:] = (
            self.obs_helper.get_joint_pos_limits()[0] - q_vel
        )

        # joint_vel_constraint: stay within the robots joint velocity limits
        constraints["joint_vel_constraints"] = np.zeros(6)
        constraints["joint_vel_constraints"][:3] = (
            q_vel - self.obs_helper.get_joint_vel_limits()[1]
        )
        constraints["joint_vel_constraints"][3:] = (
            self.obs_helper.get_joint_vel_limits()[0] - q_vel
        )

        return constraints
