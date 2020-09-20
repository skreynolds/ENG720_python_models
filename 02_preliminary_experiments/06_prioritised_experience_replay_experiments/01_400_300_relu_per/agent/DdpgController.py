import numpy as np
import random
import copy
from collections import namedtuple, deque

from agent.Model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(2e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
PRIORITY_ALPHA = 0.8    # priority exponent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Experience = namedtuple("Experience",
                        field_names=["state", "action", "reward", "next_state",
                                     "done"])
"""Experience tuple."""

class DdpgController():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed,
                 initial_beta=0.0, delta_beta=0.005, epsilon=0.05):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Prioritized replay
        self.initial_beta = initial_beta
        self.delta_beta = delta_beta
        self.beta = initial_beta
        self.epsilon = epsilon

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.p_max = 1.0
        self.memory = PrioritizedExperienceReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device)
    
    def step(self, state, action, reward, next_state, done, do_train):
        """
        Save experience in replay memory, and use random sample from buffer to learn.
        """
        
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done, self.p_max)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and do_train:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """
        Returns actions for given state as per current policy.
        """
        state = torch.from_numpy(state).float().to(device)
        
        # Temporarily set to evaluation mode & turn off autograd
        self.actor_local.eval()
        
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        
        # Resume training mode
        self.actor_local.train()
        
        # Add noise if exploring
        if add_noise:
            action += self.noise.sample()
            # Noise might take us out of range so clip
            action = np.clip(action, -1, 1)

        return action

    def reset(self):
        self.noise.reset()
        self.beta = min(1.0, self.beta + self.delta_beta)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        indices, states, actions, rewards, next_states, dones, priorities = experiences

        # Calculate importance sampling weights
        probs = priorities / self.memory.priority_sum()
        weights = (BATCH_SIZE * probs)**(-self.beta)
        weights /= torch.max(weights)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        
        # Update priorities
        td_error = Q_targets - Q_expected
        updated_priorities = abs(td_error) + self.epsilon
        self.memory.set_priorities(indices, updated_priorities**PRIORITY_ALPHA)
        self.p_max = max(self.p_max, torch.max(updated_priorities))

        #critic_loss = F.mse_loss(Q_expected, Q_targets)
        critic_loss = torch.mean(weights * td_error**2)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -(weights * self.critic_local(states, actions_pred)).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def noise_decay(self, decay):
        self.theta *= decay
        self.sigma *= decay

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class SumTree(object):
    def __init__(self, capacity):
        self.capacity = capacity

        # Round capacity to 2^n (could be done using lb instead)
        self.tree_depth = 1
        self.actual_capacity = 1
        while self.actual_capacity < capacity:
            self.actual_capacity *= 2
            self.tree_depth += 1

        self.tree_nodes = [np.zeros(2 ** i) for i in range(self.tree_depth)]
        self.start_index = -1

    def append(self, p):
        self.start_index = (self.start_index + 1) % self.capacity
        self.set(self.start_index, p)

    def get(self, i):
        return self.tree_nodes[-1][i]

    def set(self, i, p):
        self.tree_nodes[-1][i] = p

        # Update sums
        for j in range(self.tree_depth - 2, -1, -1):
            i //= 2
            self.tree_nodes[j][i] = (self.tree_nodes[j + 1][2 * i] +
                                     self.tree_nodes[j + 1][2 * i + 1])

    def set_multiple(self, indices, ps):
        # TODO: Smarter update which sets all and recalculates range as needed
        for i, p in zip(indices, ps):
            self.set(i, p)

    def total_sum(self):
        return self.tree_nodes[0][0]

    def index(self, p):
        i = 0
        for j in range(self.tree_depth - 1):
            left = self.tree_nodes[j + 1][2 * i]
            if p < left:
                i = 2 * i
            else:
                p = p - left
                i = 2 * i + 1
        return i

    def sample(self, size):
        indices = []
        bins = np.linspace(0, self.total_sum(), size + 1)
        for a, b in zip(bins, bins[1:]):
            # There's a chance we'll sample the same index more than once
            indices.append(self.index(np.random.uniform(a, b)))

        return indices

class PrioritizedExperienceReplayBuffer(object):
    """Fixed-size buffer for storing experience tuples."""

    def __init__(self, buffer_size, batch_size, device):
        """Initialize the replay buffer.

        Args:
            buffer_size (int): Max number of stored experiences
            batch_size (int): Size of training batches
            device (torch.device): Device for tensors
        """
        self.batch_size = batch_size
        """Size of training batches"""

        self.memory = deque(maxlen=buffer_size)
        """Stored experiences."""

        self.priorities = SumTree(capacity=buffer_size)
        """Stored priorities"""

        self.device = device
        """Device to be used for tensors."""

    def add(self, state, action, reward, next_state, done, priority):
        """Add an experience to memory.

        Args:
            state (Tensor): Current state
            action (int): Chosen action
            reward (float): Resulting reward
            next_state (Tensor): State after action
            done (bool): True if terminal state
            priority (float): Priority of experience (abs TD-error)
        """
        self.memory.append(Experience(state, action, reward, next_state, done))
        self.priorities.append(priority)

    def sample(self):
        """Returns a sample batch of experiences from memory.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: SARS'+done tuple"""
        indices = self.priorities.sample(self.batch_size)

        # TODO: Make this whole indexing and unpacking thing more efficient
        experiences = [self.memory[i] for i in indices]

        state_list = [e.state for e in experiences if e is not None]
        action_list = [e.action for e in experiences if e is not None]
        reward_list = [e.reward for e in experiences if e is not None]
        next_state_list = [e.next_state for e in experiences if e is not None]
        done_list = [e.done for e in experiences if e is not None]
        # TODO: Add `__getitem__` to SumTree
        priorities_list = [self.priorities.get(i) for i in indices]

        states = torch.from_numpy(np.vstack(state_list)).float().to(device)
        actions = torch.from_numpy(np.vstack(action_list)).float().to(device)
        rewards = torch.from_numpy(np.vstack(reward_list)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_state_list)).float().to(device)
        dones = torch.from_numpy(np.vstack(done_list).astype(np.uint8)).float().to(device)
        priorities = torch.from_numpy(np.vstack(priorities_list)).float().to(device)

        return (indices, states, actions, rewards, next_states, dones, priorities)

    def priority_sum(self):
        return self.priorities.total_sum()

    def set_priorities(self, i, p):
        # NB. Works with multiple indices and priorities
        self.priorities.set_multiple(i, p)

    def __len__(self):
        """Returns the current number of stored experiences.

        Returns:
            int: Number of stored experiences"""
        return len(self.memory)