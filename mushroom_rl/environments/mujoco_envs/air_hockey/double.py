import numpy as np
import mujoco

from mushroom_rl.rl_utils.spaces import Box
from mushroom_rl.environments.mujoco_envs.air_hockey.base import AirHockeyBase
from mushroom_rl.utils.mujoco import forward_kinematics


class AirHockeyDouble(AirHockeyBase):
    """
    Base class for two agents air hockey tasks.

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
            n_agents=2,
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

        self.obs_helper.remove_obs("robot_2/puck_pos", 2)
        self.obs_helper.remove_obs("robot_2/puck_pos", 3)
        self.obs_helper.remove_obs("robot_2/puck_pos", 4)
        self.obs_helper.remove_obs("robot_2/puck_pos", 5)
        self.obs_helper.remove_obs("robot_2/puck_pos", 6)

        # Remove linear z velocity and angular velocity around x and y
        self.obs_helper.remove_obs("puck_vel", 2)
        self.obs_helper.remove_obs("puck_vel", 3)
        self.obs_helper.remove_obs("puck_vel", 4)

        self.obs_helper.remove_obs("robot_2/puck_vel", 2)
        self.obs_helper.remove_obs("robot_2/puck_vel", 3)
        self.obs_helper.remove_obs("robot_2/puck_vel", 4)

        self.obs_helper.add_obs("collision_robot_1_puck", 1, 0, 1)
        self.obs_helper.add_obs("collision_robot_2_puck", 1, 0, 1)
        self.obs_helper.add_obs("collision_short_sides_rim_puck", 1, 0, 1)

        self._mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())

        self.robot_1_hit = False
        self.robot_2_hit = False
        self.has_bounce = False

    def get_puck(self, obs):
        puck_pos = self.obs_helper.get_from_obs(obs, "puck_pos")
        puck_lin_vel = self.obs_helper.get_from_obs(obs, "puck_vel")[1:]
        puck_ang_vel = self.obs_helper.get_from_obs(obs, "puck_vel")[0]
        return puck_pos, puck_lin_vel, puck_ang_vel

    def get_ee(self, robot=1):
        ee_pos = self._read_data("planar_robot_" + str(robot) + "/ee_pos")

        ee_vel = self._read_data("planar_robot_" + str(robot) + "/ee_vel")

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

        self._puck_2d_in_robot_frame(
            self.obs_helper.get_from_obs(new_obs, "robot_2/puck_pos"),
            self.agents[1]["frame"],
        )

        self._puck_2d_in_robot_frame(
            self.obs_helper.get_from_obs(new_obs, "robot_2/puck_vel"),
            self.agents[1]["frame"],
            type="vel",
        )

        if self.obs_noise:
            noise = np.random.randn(2) * 0.001
            self.obs_helper.get_from_obs(new_obs, "puck_pos")[:] += noise
            self.obs_helper.get_from_obs(new_obs, "robot_2/puck_pos")[:] += noise

        return new_obs

    def reward(self, state, action, next_state, absorbing):
        return 0

    def setup(self, obs):
        self.robot_1_hit = False
        self.robot_2_hit = False
        self.has_bounce = False

        for i in range(3):
            self._data.joint("planar_robot_1/joint_" + str(i + 1)).qpos = (
                self.init_state[i]
            )

        for i in range(3):
            self._data.joint("planar_robot_2/joint_" + str(i + 1)).qpos = (
                self.init_state[i]
            )

        super().setup(obs)
        mujoco.mj_fwdPosition(self._model, self._data)

    def _simulation_post_step(self):
        if not self.robot_1_hit:
            self.robot_1_hit = self._check_collision("puck", "robot_1/ee")

        if not self.robot_2_hit:
            self.robot_2_hit = self._check_collision("puck", "robot_1/ee")

        if not self.has_bounce:
            self.has_bounce = self._check_collision("puck", "rim_short_sides")

    def _create_observation(self, state):
        obs = super(AirHockeyDouble, self)._create_observation(state)
        return np.append(obs, [self.robot_1_hit, self.robot_2_hit, self.has_bounce])

    def _create_info_dictionary(self, obs, action):
        constraints = {"agent-1": {}, "agent-2": {}}

        for i, key in enumerate(constraints.keys()):

            q_pos = self.obs_helper.get_joint_pos_from_obs(obs)[i * 3 : (i + 1) * 3]
            q_vel = self.obs_helper.get_joint_vel_from_obs(obs)[i * 3 : (i + 1) * 3]

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
            y_ub = (
                self.env_spec["table"]["width"] / 2 - self.env_spec["mallet"]["radius"]
            )

            constraints[key]["ee_constraints"] = np.array(
                [-ee_pos[0] + x_lb, -ee_pos[1] + y_lb, ee_pos[1] - y_ub]
            )

            # joint_pos_constraint: stay within the robots joint position limits
            constraints[key]["joint_pos_constraints"] = np.zeros(6)
            constraints[key]["joint_pos_constraints"][:3] = (
                q_vel - self.obs_helper.get_joint_pos_limits()[1][i * 3 : (i + 1) * 3]
            )
            constraints[key]["joint_pos_constraints"][3:] = (
                self.obs_helper.get_joint_pos_limits()[0][i * 3 : (i + 1) * 3] - q_vel
            )

            # joint_vel_constraint: stay within the robots joint velocity limits
            constraints[key]["joint_vel_constraints"] = np.zeros(6)
            constraints[key]["joint_vel_constraints"][:3] = (
                q_vel - self.obs_helper.get_joint_vel_limits()[1][i * 3 : (i + 1) * 3]
            )
            constraints[key]["joint_vel_constraints"][3:] = (
                self.obs_helper.get_joint_vel_limits()[0][i * 3 : (i + 1) * 3] - q_vel
            )

        return constraints


if __name__ == "__main__":
    env = AirHockeyDouble(env_noise=False, obs_noise=False, n_intermediate_steps=4)

    env.reset()
    R = 0.0
    J = 0.0
    gamma = 1.0
    steps = 0
    while True:
        action = np.random.randn(6) * 5
        observation, reward, done, info = env.step(action)
        env.render()
        gamma *= env.info.gamma
        J += gamma * reward
        R += reward
        steps += 1
        if done or steps > env.info.horizon:
            print("J: ", J, " R: ", R)
            R = 0.0
            J = 0.0
            gamma = 1.0
            steps = 0
            env.reset()
