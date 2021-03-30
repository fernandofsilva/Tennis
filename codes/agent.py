import random
import numpy as np
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

# Determine if the GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""
    def __init__(self,
                 state_size,
                 action_size,
                 num_agents,
                 buffer_size,
                 batch_size,
                 gamma,
                 tau,
                 lr_actor,
                 lr_critic,
                 weight_decay,
                 update_every,
                 num_updates,
                 seed):
        """Initialize an Agent object.

        Args:
            state_size: Integer. Dimension of each state
            action_size: Integer. Dimension of each action
            num_agents: Integer. Number of agents
            buffer_size: Integer. Replay buffer size
            batch_size: Integer. Mini-batch size
            gamma: Float. Discount factor
            tau: Float. For soft update of target parameters
            lr_actor: Float. Learning rate for actor local model
            lr_actor: Float. Learning rate for critic local model
            weight_decay: Float. L2 weight decay
            update_every: Integer. How often to update the network
            num_updates: Integer. The number of updates
            seed: Integer. Random seed
        """
        # Environment parameters
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents

        # Reward discount
        self.gamma = gamma

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size*num_agents, action_size*num_agents, seed).to(device)
        self.critic_target = Critic(state_size*num_agents, action_size*num_agents, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)

        # Model parameters
        self.loss_fn = F.mse_loss
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every
        self.num_updates = num_updates

        # Initialize time step (for updating every update_every steps)
        self.t_step = 0

        # Update weights
        self.hard_update(self.actor_local, self.actor_target)
        self.hard_update(self.critic_local, self.critic_target)

        # Noise process
        self.noise = OUNoise((num_agents, action_size), seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)

        # Set seed
        self.seed = random.seed(seed)

    def __str__(self):
        return 'RL_Agent_class'

    def __repr__(self):
        return 'RL_Agent_class'

    @staticmethod
    def env_step(env, actions, brain_name):
        """Apply an action and return the state, reward and done.

        Args:
            env: unity environment
            actions: Integer. Action to be done in the environment
            brain_name: String. Name of the agent of the unity environment

        Returns:
            A tuple of three items with
            next_states: List. Contains the next state returned,
            rewards: Float. Number of the reward returned.
            dones: Boolean. Indication if the episode ends.
        """

        # send the action to the environment
        env_info = env.step(actions)[brain_name]

        # get the next state
        next_states = env_info.vector_observations

        # Get the reward
        rewards = env_info.rewards

        # is it the episode ended?
        dones = env_info.local_done

        return next_states, rewards, dones

    def step(self, state, action, reward, next_state, done):
        """Save state on buffer and trigger learn according to update_every

        Args:
            state: The previous state of the environment
            action: Integer. Previous action selected by the agent
            reward: Float. Reward value
            next_state: The current state of the environment
            done: Boolean. Whether the episode is complete
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:

            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                for _ in range(self.num_updates):
                    experiences = self.memory.sample()
                    self.learn(experiences)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy.

        Args:
            state: A array like object or list with states
            add_noise: Boolean.

        Returns:
            An action selected by the network or by the epsilon-greedy method
        """
        # Reshape state
        state = torch.from_numpy(state).float().to(device)

        # Set model to prediction
        self.actor_local.eval()

        # Predict action
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()

        # Set model to training
        self.actor_local.train()

        # Add noise to the action
        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1)

    def reset(self):
        """Reset noise values."""
        self.noise.reset()

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Args:
            experiences: Tuple. Content of tuple (s, a, r, s', done)
        """
        # Unpack experiences
        states, actions, rewards, next_states, dones = experiences

        # -------------------------- update critic -------------------------- #
        # Get predicted next-state and Q values from target model
        actions_next = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states.view(-1, 48), actions_next.view(-1, 4))

        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.critic_local(states.view(-1, 48), actions.view(-1, 4))

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        # -------------------------- update actor -------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states.view(-1, 48), actions_pred.view(-1, 4)).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.actor_optimizer.step()

        # ---------------------- update target networks ---------------------- #
        # Update target network
        self.soft_update(self.actor_local, self.actor_target)
        self.soft_update(self.critic_local, self.critic_target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.

        The model is update using:
            θ_target = τ * θ_local + (1 - τ) * θ_target

        Args:
            local_model: PyTorch model. Weights will be copied from)
            target_model: PyTorch model. Weights will be copied to)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    @staticmethod
    def hard_update(local_model, target_model):
        """ copy weights from source to target network (part of initialization).

        Args:
            local_model: PyTorch model. Weights will be copied from)
            target_model: PyTorch model. Weights will be copied to)
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Args:
            action_size: Integer. Dimension of each action
            buffer_size: Integer. Maximum size of buffer
            batch_size: Integer. Size of each training batch
            seed: Integer. Random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def __str__(self):
        return 'ReplayBuffer_class'

    def __repr__(self):
        return 'ReplayBuffer_class'

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory.

        Args:
            state: The previous state of the environment
            action: Integer. Previous action selected by the agent
            reward: Float. Reward value
            next_state: The current state of the environment
            done: Boolean. Whether the episode is complete
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(device)

        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).float().to(device)

        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(device)

        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(device)

        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float().to(device)

        return states, actions, rewards, next_states, dones


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, scale=0.1, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process.

        :param size: Integer. Dimension of each state
        :param seed: Integer. Random seed
        :param scale: Float. Scale of the distribution
        :param mu: Float. Mean of the distribution
        :param theta: Float. Rate of the mean reversion of the distribution
        :param sigma: Float. Volatility of the distribution
        """
        self.size = size
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()
        random.seed(seed)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float()
