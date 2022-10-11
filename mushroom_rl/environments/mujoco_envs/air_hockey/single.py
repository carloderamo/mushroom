import numpy as np

from mushroom_rl.core import MDPInfo
from mushroom_rl.utils.spaces import Box
from mushroom_rl.environments.mujoco_envs.air_hockey import AirHockeyBase
from mushroom_rl.utils.mujoco import ObservationType


class AirHockeySingle(AirHockeyBase):
    """
    Base class for single agent air hockey tasks.
    """
    def __init__(self, gamma=0.99, horizon=120, env_noise=False, obs_noise=False, obs_delay=False,
                 torque_control=True, step_action_function=None, timestep=1 / 240., n_intermediate_steps=1,
                 number_flags=0):

        """
        Constructor.
        Args:
            number_flags(int, 0): Amount of flags which are added to the observation space
        """
        self.init_state = np.array([-0.9273, 0.9273, np.pi / 2])
        self.number_flags = number_flags
        super().__init__(gamma=gamma, horizon=horizon, env_noise=env_noise, n_agents=1, obs_noise=obs_noise,
                         torque_control=torque_control, step_action_function=step_action_function,
                         timestep=timestep, n_intermediate_steps=n_intermediate_steps)

        self.obs_helper.remove_obs("puck_pos", 2)
        self.obs_helper.remove_obs("puck_vel", 0)
        self.obs_helper.remove_obs("puck_vel", 1)
        self.obs_helper.remove_obs("puck_vel", 5)

        self.obs_helper.add_obs("flags", number_flags, 0, 1)

        self._mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())

    def get_puck(self, obs):
        puck_pos = self.obs_helper.get_from_obs(obs, "puck_pos")
        puck_lin_vel = self.obs_helper.get_from_obs(obs, "puck_vel")[1:]
        puck_ang_vel = self.obs_helper.get_from_obs(obs, "puck_vel")[0]
        return puck_pos, puck_lin_vel, puck_ang_vel

    def get_ee(self):
        ee_pos = self._read_data("planar_robot_1/ee_pos")

        ee_vel = self._read_data("planar_robot_1/ee_vel")

        return ee_pos, ee_vel

    def _modify_observation(self, obs):
        self._puck_2d_in_robot_frame(self.obs_helper.get_from_obs(obs, "puck_pos"), self.agents[0]["frame"])

        self._puck_2d_in_robot_frame(self.obs_helper.get_from_obs(obs, "puck_vel"), self.agents[0]["frame"], type='vel')

        if self.obs_noise:
            self.obs_helper.get_from_obs(obs, "puck_pos")[:] += np.random.randn(2) * 0.001

        return obs

    def _puck_2d_in_robot_frame(self, puck_in, robot_frame, type='pose'):
        if type == 'pose':
            puck_frame = np.eye(4)
            puck_frame[:2, 3] = puck_in

            frame_target = np.linalg.inv(robot_frame) @ puck_frame
            puck_in[:] = frame_target[:2, 3]

        if type == 'vel':
            rot_mat = robot_frame[:3, :3]

            vel_lin = np.zeros(3)
            vel_lin[:2] = puck_in[1:]

            vel_target = rot_mat.T @ vel_lin

            puck_in[1:] = vel_target[:2]

    def reward(self, state, action, next_state, absorbing):
        return 0

    def setup(self):
        for i in range(3):
            self._data.joint("planar_robot_1/joint_" + str(i+1)).qpos = self.init_state[i]

if __name__ == "__main__":
    import time
    env = AirHockeySingle()
    state = env.reset()

    env.render()
    #time.sleep(5)
    steps = 0
    while True:
        # time.sleep(5)
        # print(state)
        # state, reward, done, info = env.step(np.array([-1.5708, -1.5708,  1.5708, -1.5708, -1.5708,  0.]))
        state, reward, done, info = env.step(np.array([0, 0, 0]))

        # print(env._get_collision_force("puck", "rim"))
        env.render()
        steps += 1

        if steps > 560:
            env.reset()
            steps = 0
        #print(env.dt)
        time.sleep(env.dt)