import numpy as np

from mushroom_rl.core import Agent
from mushroom_rl.policy import VectorPolicy


class BlackBoxOptimization(Agent):
    """
    Base class for black box optimization algorithms.
    These algorithms work on a distribution of policy parameters and often they
    do not rely on stochastic and differentiable policies.

    """
    def __init__(self, mdp_info, distribution, policy, backend='numpy'):
        """
        Constructor.

        Args:
            distribution (Distribution): the distribution of policy parameters;
            policy (ParametricPolicy): the policy to use.

        """
        self.distribution = distribution

        self._add_save_attr(distribution='mushroom')

        super().__init__(mdp_info, policy, is_episodic=True, backend=backend)

    def episode_start(self, initial_state, episode_info):
        if isinstance(self.policy, VectorPolicy):
            self.policy = self.policy.get_flat_policy()

        theta = self.distribution.sample()
        self.policy.set_weights(theta)

        policy_state, _ = super().episode_start(initial_state, episode_info)

        return policy_state, theta

    def episode_start_vectorized(self, initial_states, episode_info, start_mask):
        n_envs = len(start_mask)
        if not isinstance(self.policy, VectorPolicy):
            self.policy = VectorPolicy(self.policy, n_envs)
        elif len(self.policy) != n_envs:
            self.policy.set_n(n_envs)

        theta = self.policy.get_weights()
        if start_mask.any():
            theta[start_mask] = self._agent_backend.from_list(
                [self.distribution.sample() for _ in range(start_mask.sum())])  # TODO change it
            self.policy.set_weights(theta)

        policy_states = self.policy.reset()

        return policy_states, theta

    def fit(self, dataset):
        Jep = dataset.discounted_return
        theta = self._agent_backend.from_list(dataset.theta_list)

        if self.distribution.is_contextual:
            initial_states = dataset.get_init_states()
            episode_info = dataset.episode_info
        else:
            initial_states = None
            episode_info = {}

        self._update(Jep, theta, initial_states, episode_info)

    def _update(self, Jep, theta, initial_states, episode_info):
        """
        Function that implements the update routine of distribution parameters.
        Every black box algorithms should implement this function with the
        proper update.

        Args:
            Jep (np.ndarray): a vector containing the J of the considered
                trajectories;
            theta (np.ndarray): a matrix of policy parameters of the considered
                trajectories.

        """
        raise NotImplementedError('BlackBoxOptimization is an abstract class')
