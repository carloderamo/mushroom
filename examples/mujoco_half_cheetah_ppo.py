import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mushroom_rl.algorithms.actor_critic import PPO
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import HalfCheetah
from mushroom_rl.policy import GaussianTorchPolicy

from tqdm import trange


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Network, self).__init__()

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

    def forward(self, state, **kwargs):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a


def experiment(n_epochs, n_steps, n_episodes_test, seed=0):

    np.random.seed(seed)
    torch.manual_seed(seed)

    logger = Logger(PPO.__name__, results_dir=None)
    logger.strong_line()
    logger.info("Experiment Algorithm: " + PPO.__name__)

    mdp = HalfCheetah()

    actor_lr = 3e-4
    critic_lr = 3e-4
    n_features = 32
    batch_size = 64
    n_epochs_policy = 10
    eps = 0.2
    lam = 0.95
    std_0 = 1.0
    n_steps_per_fit = 2000

    critic_params = dict(
        network=Network,
        optimizer={"class": optim.Adam, "params": {"lr": critic_lr}},
        loss=F.mse_loss,
        n_features=n_features,
        batch_size=batch_size,
        input_shape=mdp.info.observation_space.shape,
        output_shape=(1,),
    )

    alg_params = dict(
        actor_optimizer={"class": optim.Adam, "params": {"lr": actor_lr}},
        n_epochs_policy=n_epochs_policy,
        batch_size=batch_size,
        eps_ppo=eps,
        lam=lam,
        critic_params=critic_params,
    )

    policy_params = dict(std_0=std_0, n_features=n_features)

    policy = GaussianTorchPolicy(
        Network,
        mdp.info.observation_space.shape,
        mdp.info.action_space.shape,
        **policy_params,
    )

    agent = PPO(mdp.info, policy, **alg_params)

    core = Core(agent, mdp)

    dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

    J = np.mean(dataset.discounted_return)
    R = np.mean(dataset.undiscounted_return)
    E = agent.policy.entropy()

    logger.epoch_info(0, J=J, R=R, entropy=E)

    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

        J = np.mean(dataset.discounted_return)
        R = np.mean(dataset.undiscounted_return)
        E = agent.policy.entropy()

        logger.epoch_info(it + 1, J=J, R=R, entropy=E)

    logger.info("Press a button to visualize")
    input()
    core.evaluate(n_episodes=5, render=True)


if __name__ == "__main__":
    experiment(n_epochs=50, n_steps=30000, n_episodes_test=10)
