import numpy as np
import torch

from mushroom_rl.core import DatasetInfo, Dataset, Serializable
from mushroom_rl.rl_utils.parameters import to_parameter


class ReplayMemory(Serializable):
    """
    This class implements function to manage a replay memory as the one used in
    "Human-Level Control Through Deep Reinforcement Learning" by Mnih V. et al..

    """
    def __init__(self, mdp_info, agent_info, initial_size, max_size):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            agent_info (AgentInfo): information about the agent;
            initial_size (int): initial size of the replay buffer;
            max_size (int): maximum size of the replay buffer;

        """

        self._initial_size = initial_size
        self._max_size = max_size
        self._idx = 0
        self._full = False
        self._mdp_info = mdp_info
        self._agent_info = agent_info

        assert agent_info.backend in ["numpy", "torch"], f"{agent_info.backend} backend currently not supported in " \
                                                         f"the replay memory class."

        self._dataset = None
        self.reset()

        self._add_save_attr(
            _initial_size='primitive',
            _max_size='primitive',
            _mdp_info='mushroom',
            _agent_info='mushroom',
            _idx='primitive!',
            _full='primitive!',
            _dataset='mushroom!',
        )

    def add(self, dataset, n_steps_return=1, gamma=1.):
        """
        Add elements to the replay memory.

        Args:
            dataset (Dataset): dataset class elements to add to the replay memory;
            n_steps_return (int, 1): number of steps to consider for computing n-step return;
            gamma (float, 1.): discount factor for n-step return.

        """

        assert n_steps_return > 0
        assert self._dataset.is_stateful == dataset.is_stateful

        state, action, reward, next_state, absorbing, last = dataset.parse(to=self._agent_info.backend)

        if self._dataset.is_stateful:
            policy_state, policy_next_state = dataset.parse_policy_state(to=self._agent_info.backend)
        else:
            policy_state, policy_next_state = (None, None)

        # TODO: implement vectorized n_step_return to avoid loop
        i = 0
        while i < len(dataset) - n_steps_return + 1:
            j = 0
            while j < n_steps_return - 1:
                if last[i + j]:
                    i += j + 1
                    break
                j += 1
                reward[i] += gamma ** j * reward[i + j]
            else:

                if self._full:
                    self._dataset.state[self._idx] = state[i]
                    self._dataset.action[self._idx] = action[i]
                    self._dataset.reward[self._idx] = reward[i]

                    self._dataset.next_state[self._idx] = next_state[i + j]
                    self._dataset.absorbing[self._idx] = absorbing[i + j]
                    self._dataset.last[self._idx] = last[i + j]

                    if self._dataset.is_stateful:
                        self._dataset.policy_state[self._idx] = policy_state[i]
                        self._dataset.policy_next_state[self._idx] = policy_next_state[i + j]

                else:

                    sample = [state[i], action[i], reward[i], next_state[i + j], absorbing[i + j], last[i + j]]

                    if self._dataset.is_stateful:
                        sample += [policy_state[i], policy_next_state[i+j]]

                    self._dataset.append(sample, {})

                self._idx += 1
                if self._idx == self._max_size:
                    self._full = True
                    self._idx = 0

                i += 1

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.

        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """
        idxs = self._dataset.array_backend.randint(0, len(self._dataset), (n_samples,))

        dataset_batch = self._dataset[idxs]

        if self._dataset.is_stateful:
            return *dataset_batch.parse(), *dataset_batch.parse_policy_state()
        else:
            return dataset_batch.parse()

    def reset(self):
        """
        Reset the replay memory.

        """
        self._idx = 0
        self._full = False
        dataset_info = DatasetInfo.create_replay_memory_info(self._mdp_info, self._agent_info)
        self._dataset = Dataset(dataset_info, n_steps=self._max_size)

    @property
    def initialized(self):
        """
        Returns:
            Whether the replay memory has reached the number of elements that
            allows it to be used.

        """
        return self.size >= self._initial_size

    @property
    def size(self):
        """
        Returns:
            The number of elements contained in the replay memory.

        """
        return self._idx if not self._full else self._max_size

    @property
    def _backend(self):
        return self._dataset.array_backend

    def _post_load(self):
        if self._full is None:
            self.reset()


class SequenceReplayMemory(ReplayMemory):
    """
    This class extend the base replay memory to allow sampling sequences of a certain length. This is useful for
    training recurrent agents or agents operating on a window of states etc.

    """
    def __init__(self, mdp_info, agent_info, initial_size, max_size, truncation_length):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            agent_info (AgentInfo): information about the agent;
            initial_size (int): initial size of the replay buffer;
            max_size (int): maximum size of the replay buffer;
            truncation_length (int): truncation length to be sampled;
        """
        self._truncation_length = truncation_length
        self._action_space_shape = mdp_info.action_space.shape

        super(SequenceReplayMemory, self).__init__(mdp_info, agent_info, initial_size, max_size)

        self._add_save_attr(
            _truncation_length='primitive',
            _action_space_shape='primitive'
        )

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """

        s = self._backend.zeros(n_samples, self._truncation_length, *self._mdp_info.observation_space.shape,
                                dtype=self._mdp_info.observation_space.data_type)
        a = self._backend.zeros(n_samples, self._truncation_length, *self._mdp_info.action_space.shape,
                                dtype=self._mdp_info.action_space.data_type)
        r = self._backend.zeros(n_samples, 1)
        ss = self._backend.zeros(n_samples, self._truncation_length, *self._mdp_info.observation_space.shape,
                                 dtype=self._mdp_info.observation_space.data_type)
        ab = self._backend.zeros(n_samples, 1, dtype=int)
        last = self._backend.zeros(n_samples, dtype=int)
        ps = self._backend.zeros(n_samples, self._truncation_length, *self._agent_info.policy_state_shape)
        nps = self._backend.zeros(n_samples, self._truncation_length, *self._agent_info.policy_state_shape)
        pa = self._backend.zeros(n_samples, self._truncation_length, *self._mdp_info.action_space.shape,
                                 dtype=self._mdp_info.action_space.data_type)
        lengths = list()

        for num, i in enumerate(np.random.randint(self.size, size=n_samples)):

            with torch.no_grad():

                # determine the begin of a sequence
                begin_seq = np.maximum(i - self._truncation_length + 1, 0)
                end_seq = i + 1

                # the sequence can contain more than one trajectory, check to only include one
                lasts_absorbing = self._dataset.last[begin_seq: i]
                begin_traj = self._backend.where(lasts_absorbing > 0)
                more_than_one_traj = len(*begin_traj) > 0
                if more_than_one_traj:
                    # take the beginning of the last trajectory
                    begin_seq = begin_seq + begin_traj[0][-1] + 1

                # determine prev action
                if more_than_one_traj or begin_seq == 0 or self._dataset.last[begin_seq-1]:
                    prev_actions = self._dataset.action[begin_seq:end_seq - 1]
                    init_prev_action = self._backend.zeros(1, *self._action_space_shape)
                    if len(prev_actions) == 0:
                        prev_actions = init_prev_action
                    else:
                        prev_actions = self._backend.concatenate([init_prev_action, prev_actions])
                else:
                    prev_actions = self._dataset.action[begin_seq - 1:end_seq - 1]

                # write data
                s[num, :end_seq-begin_seq] = self._dataset.state[begin_seq:end_seq]
                ss[num, :end_seq-begin_seq] = self._dataset.next_state[begin_seq:end_seq]
                a[num, :end_seq-begin_seq] = self._dataset.action[begin_seq:end_seq]
                ps[num, :end_seq-begin_seq] = self._dataset.policy_state[begin_seq:end_seq]
                nps[num, :end_seq-begin_seq] = self._dataset.policy_next_state[begin_seq:end_seq]
                pa[num, :end_seq-begin_seq] = prev_actions
                r[num] = self._dataset.reward[i]
                ab[num] = self._dataset.absorbing[i]
                last[num] = self._dataset.last[i]

                lengths.append(end_seq - begin_seq)

        if self._dataset.is_stateful:
            return s, a, r, ss, ab, last, ps, nps, pa, lengths
        else:
            return s, a, r, ss, ab, last, pa, lengths


class SumTree(Serializable):
    """
    This class implements a sum tree data structure.
    This is used, for instance, by ``PrioritizedReplayMemory``.

    """
    def __init__(self, mdp_info, agent_info, max_size):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            agent_info (AgentInfo): information about the agent;
            max_size (int): maximum size of the tree;

        """
        self._max_size = max_size
        self._tree = np.zeros(2 * max_size - 1)
        self._mdp_info = mdp_info
        self._agent_info = agent_info
        dataset_info = DatasetInfo.create_replay_memory_info(self._mdp_info, self._agent_info)
        self.dataset = Dataset(dataset_info, n_steps=max_size)
        self._idx = 0
        self._full = False

        super().__init__()
        self._add_save_attr(
            _max_size="primitive",
            _idx="primitive",
            _full="primitive",
            _tree="numpy",
            _mdp_info="mushroom",
            _agent_info="mushroom",
            dataset="mushroom!")

    def add(self, dataset, priority, n_steps_return, gamma):
        """
        Add elements to the tree.

        Args:
            dataset (Dataset): dataset class elements to add to the replay memory;
            priority (np.ndarray): priority of each sample in the dataset;
            n_steps_return (int): number of steps to consider for computing n-step return;
            gamma (float): discount factor for n-step return.

        """

        assert np.all(np.array(priority) > 0.0)

        state, action, reward, next_state, absorbing, last = dataset.parse(to=self._agent_info.backend)

        if self.dataset.is_stateful:
            policy_state, policy_next_state = dataset.parse_policy_state(to=self._agent_info.backend)
        else:
            policy_state, policy_next_state = (None, None)

        # TODO: implement vectorized n_step_return to avoid loop
        i = 0
        while i < len(dataset) - n_steps_return + 1:
            j = 0
            while j < n_steps_return - 1:
                if last[i + j]:
                    i += j + 1
                    break
                j += 1
                reward[i] += gamma ** j * reward[i + j]
            else:

                if self._full:
                    self.dataset.state[self._idx] = state[i]
                    self.dataset.action[self._idx] = action[i]
                    self.dataset.reward[self._idx] = reward[i]

                    self.dataset.next_state[self._idx] = next_state[i + j]
                    self.dataset.absorbing[self._idx] = absorbing[i + j]
                    self.dataset.last[self._idx] = last[i + j]

                    if self.dataset.is_stateful:
                        self.dataset.policy_state[self._idx] = policy_state[i]
                        self.dataset.policy_next_state[self._idx] = policy_next_state[i + j]

                else:

                    sample = [state[i], action[i], reward[i], next_state[i + j], absorbing[i + j], last[i + j]]

                    if self.dataset.is_stateful:
                        sample += [policy_state[i], policy_next_state[i+j]]

                    self.dataset.append(sample, {})

                idx = self._idx + self._max_size - 1

                self.update([idx], [priority[i]])

                self._idx += 1
                if self._idx == self._max_size:
                    self._idx = 0
                    self._full = True

                i += 1

    def get(self, s):
        """
        Returns the provided number of states from the replay memory.

        Args:
            s (float): the value of the samples to return.

        Returns:
            The requested sample.

        """
        idx = self._retrieve(s, 0)
        data_idx = idx - self._max_size + 1

        return idx, self._tree[idx], self.dataset[data_idx]

    def get_ind(self, s):
        """
        Returns the provided number of states from the replay memory.

        Args:
            s (float): the value of the samples to return.

        Returns:
            The requested sample.

        """
        idx = self._retrieve(s, 0)
        data_idx = idx - self._max_size + 1

        return idx, self._tree[idx], data_idx

    def update(self, idx, priorities):
        """
        Update the priority of the sample at the provided index in the dataset.

        Args:
            idx (np.ndarray): indexes of the transitions in the dataset;
            priorities (np.ndarray): priorities of the transitions.

        """
        for i, p in zip(idx, priorities):
            delta = p - self._tree[i]

            self._tree[i] = p
            self._propagate(delta, i)

    def _propagate(self, delta, idx):
        parent_idx = (idx - 1) // 2

        self._tree[parent_idx] += delta

        if parent_idx != 0:
            self._propagate(delta, parent_idx)

    def _retrieve(self, s, idx):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self._tree):
            return idx

        if self._tree[left] == self._tree[right]:
            return self._retrieve(s, np.random.choice([left, right]))

        if s <= self._tree[left]:
            return self._retrieve(s, left)
        else:
            return self._retrieve(s - self._tree[left], right)

    @property
    def size(self):
        """
        Returns:
            The current size of the tree.

        """
        return self._idx if not self._full else self._max_size

    @property
    def max_p(self):
        """
        Returns:
            The maximum priority among the ones in the tree.

        """
        return self._tree[-self._max_size:].max()

    @property
    def total_p(self):
        """
        Returns:
            The sum of the priorities in the tree, i.e. the value of the root
            node.

        """
        return self._tree[0]

    def _post_load(self):
        if self.dataset is None:
            dataset_info = DatasetInfo.create_replay_memory_info(self._mdp_info, self._agent_info)
            self.dataset = Dataset(dataset_info, n_steps=self._max_size)


class PrioritizedReplayMemory(Serializable):
    """
    This class implements function to manage a prioritized replay memory as the
    one used in "Prioritized Experience Replay" by Schaul et al., 2015.

    """
    def __init__(self, mdp_info, agent_info, initial_size, max_size, alpha, beta, epsilon=.01):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information about the MDP;
            agent_info (AgentInfo): information about the agent;
            initial_size (int): initial number of elements in the replay
                memory;
            max_size (int): maximum number of elements that the replay memory
                can contain;
            alpha (float): prioritization coefficient;
            beta ([float, Parameter]): importance sampling coefficient;
            epsilon (float, .01): small value to avoid zero probabilities.

        """
        self._initial_size = initial_size
        self._max_size = max_size
        self._alpha = alpha
        self._beta = to_parameter(beta)
        self._epsilon = epsilon

        self._tree = SumTree(mdp_info, agent_info, max_size)

        self._add_save_attr(
            _initial_size='primitive',
            _max_size='primitive',
            _alpha='primitive',
            _beta='primitive',
            _epsilon='primitive',
            _tree='mushroom'
        )

    def add(self, dataset, p, n_steps_return=1, gamma=1.):
        """
        Add elements to the replay memory.

        Args:
            dataset (Dataset): list of elements to add to the replay memory;
            p (np.ndarray): priority of each sample in the dataset.
            n_steps_return (int, 1): number of steps to consider for computing n-step return;
            gamma (float, 1.): discount factor for n-step return.

        """
        assert n_steps_return > 0

        self._tree.add(dataset, p, n_steps_return, gamma)

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.

        Args:
            n_samples (int): the number of samples to return.

        Returns:
            The requested number of samples.

        """

        idxs = self._backend.zeros(n_samples, dtype=int)
        priorities = self._backend.zeros(n_samples, dtype=float)
        data_idxs = self._backend.zeros(n_samples, dtype=int)

        total_p = self._tree.total_p
        segment = total_p / n_samples

        a = self._backend.arange(0, n_samples) * segment
        b = self._backend.arange(1, n_samples + 1) * segment
        samples = np.random.uniform(a, b)

        for i, s in enumerate(samples):
            idx, p, data_idx = self._tree.get_ind(s)

            idxs[i] = idx
            priorities[i] = p
            data_idxs[i] = data_idx

        sampling_probabilities = priorities / self._tree.total_p
        is_weight = (self._tree.size * sampling_probabilities) ** -self._beta()
        is_weight /= is_weight.max()

        if self._tree.dataset.is_stateful:
            return *self._tree.dataset[data_idxs].parse(), \
                   *self._tree.dataset[data_idxs].parse_policy_state(), idxs, is_weight
        else:
            return *self._tree.dataset[data_idxs].parse(), idxs, is_weight

    def update(self, error, idx):
        """
        Update the priority of the sample at the provided index in the dataset.

        Args:
            error (np.ndarray): errors to consider to compute the priorities;
            idx (np.ndarray): indexes of the transitions in the dataset.

        """
        p = self._get_priority(error)
        self._tree.update(idx, p)

    def _get_priority(self, error):
        return (np.abs(error) + self._epsilon) ** self._alpha

    @property
    def initialized(self):
        """
        Returns:
            Whether the replay memory has reached the number of elements that
            allows it to be used.

        """
        return self._tree.size > self._initial_size

    @property
    def max_priority(self):
        """
        Returns:
            The maximum value of priority inside the replay memory.

        """
        return self._tree.max_p if self.initialized else 1.

    @property
    def _backend(self):
        return self._tree.dataset.array_backend
