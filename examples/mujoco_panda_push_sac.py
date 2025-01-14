import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments.mujoco_envs.franka_panda.push import Push

from tqdm import trange


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(
            self._h1.weight, gain=nn.init.calculate_gain("relu") / 10
        )
        nn.init.xavier_uniform_(
            self._h2.weight, gain=nn.init.calculate_gain("relu") / 10
        )
        nn.init.xavier_uniform_(
            self._h3.weight, gain=nn.init.calculate_gain("linear") / 10
        )

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(
            self._h1.weight, gain=nn.init.calculate_gain("relu") / 10
        )
        nn.init.xavier_uniform_(
            self._h2.weight, gain=nn.init.calculate_gain("relu") / 10
        )
        nn.init.xavier_uniform_(
            self._h3.weight, gain=nn.init.calculate_gain("linear") / 10
        )

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a


def experiment(alg, n_epochs, n_steps, n_episodes_test):
    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info("Experiment Algorithm: " + alg.__name__)

    # MDP
    mdp = Push()

    # Settings
    initial_replay_size = 5_000
    max_replay_size = 500_000
    batch_size = 256
    n_features = 256
    warmup_transitions = 10_000
    tau = 5e-3
    lr_alpha = 3e-4

    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    actor_mu_params = dict(
        network=ActorNetwork,
        n_features=n_features,
        input_shape=actor_input_shape,
        output_shape=mdp.info.action_space.shape,
    )
    actor_sigma_params = dict(
        network=ActorNetwork,
        n_features=n_features,
        input_shape=actor_input_shape,
        output_shape=mdp.info.action_space.shape,
    )

    actor_optimizer = {"class": optim.Adam, "params": {"lr": 1e-4}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(
        network=CriticNetwork,
        optimizer={"class": optim.Adam, "params": {"lr": 3e-4}},
        loss=F.mse_loss,
        n_features=n_features,
        input_shape=critic_input_shape,
        output_shape=(1,),
    )

    # Agent
    agent = alg(
        mdp.info,
        actor_mu_params,
        actor_sigma_params,
        actor_optimizer,
        critic_params,
        batch_size,
        initial_replay_size,
        max_replay_size,
        warmup_transitions,
        tau,
        lr_alpha,
        critic_fit_params=None,
    )

    # Algorithm
    core = Core(agent, mdp)

    # RUN
    dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

    J = np.mean(dataset.discounted_return)
    R = np.mean(dataset.undiscounted_return)
    E = agent.policy.entropy(dataset.state)

    for key, value in dataset.info.items():
        print(key, np.mean(value))

    logger.epoch_info(0, J=J, R=R, entropy=E)

    core.learn(
        n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size, quiet=True
    )

    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=1, quiet=True)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=True, quiet=True)

        J = np.mean(dataset.discounted_return)
        R = np.mean(dataset.undiscounted_return)
        E = agent.policy.entropy(dataset.state)

        for key, value in dataset.info.items():
            print(key, np.mean(value))

        logger.epoch_info(n + 1, J=J, R=R, entropy=E)

    logger.info("Press a button to visualize")
    input()
    core.evaluate(n_episodes=5, render=True)


if __name__ == "__main__":
    algs = [SAC]

    for alg in algs:
        experiment(alg=alg, n_epochs=50, n_steps=30000, n_episodes_test=10)
