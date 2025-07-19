"""
Deep Q-Learning (DQN) Agent for reinforcement learning environments.

This module implements a DQN agent that can learn to play environments like
LunarLander using deep reinforcement learning. The agent uses an epsilon-greedy
exploration strategy, experience replay, and a target network for stable training.

The agent maintains two neural networks: a main network for action selection and
learning, and a target network that provides stable Q-value targets during training.
Experience replay is implemented using a deque buffer to store and sample past
experiences for training.

Key components:
- Experience replay buffer for storing transitions
- Epsilon-greedy action selection with decaying exploration
- Double DQN architecture with target network updates
- Configurable hyperparameters via DQNConfig

Author: vnniciusg
"""

import random
from collections import deque

import numpy as np
import torch

from config import DQNConfig
from dqn import DeepQNetwork


class Agent:
    """
    Deep Q-Learning (DQN) Agent for reinforcement learning.

    This agent implements the DQN algorithm with experience replay and target networks
    for stable learning in reinforcement learning environments. It uses an epsilon-greedy
    exploration strategy that decays over time to balance exploration and exploitation.

    The agent maintains two identical neural networks:
    - q_eval_network: Used for action selection and learning (updated frequently)
    - q_target_network: Used for computing target Q-values (updated periodically)

    Attributes:
        q_eval_network (DeepQNetwork): Main network for action selection and learning
        q_target_network (DeepQNetwork): Target network for stable Q-value computation
        __dqn_config (DQNConfig): Configuration object containing hyperparameters
        __learn_step_counter (int): Counter for tracking learning steps
        __memory (deque): Experience replay buffer storing transitions
        __n_actions (int): Number of possible actions in the environment
    """

    def __init__(
        self, input_dims: tuple[int, ...], n_actions: int, epsilon: float = 1.0
    ) -> None:
        """
        Initialize the DQN Agent.

        Sets up the neural networks, experience replay buffer, and configuration.
        The target network is initialized with the same weights as the main network
        and set to evaluation mode.

        Args:
            input_dims (tuple[int, ...]): Dimensions of the input state space
            n_actions (int): Number of possible actions in the environment
            epsilon (float): Initial exploration rate for epsilon-greedy policy default 1.0
        """
        self.__dqn_config = DQNConfig()

        self.__learn_step_counter = 0
        self.__memory: deque = deque(maxlen=self.__dqn_config.mem_size)

        _network_params = {
            "lr": self.__dqn_config.lr,
            "input_dims": input_dims,
            "fc1_dims": 128,
            "fc2_dims": 128,
            "n_actions": n_actions,
        }

        self.q_eval_network = DeepQNetwork(**_network_params)
        self.q_target_network = DeepQNetwork(**_network_params)

        self.q_target_network.load_state_dict(self.q_eval_network.state_dict())
        self.q_target_network.eval()

        self.__n_actions = n_actions
        self.epsilon = epsilon

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a transition in the experience replay buffer.

        Adds a single experience tuple (state, action, reward, next_state, done)
        to the replay buffer. The buffer automatically handles overflow by
        removing the oldest experiences when the maximum size is reached.

        Args:
            state (np.ndarray): Current state observation
            action (int): Action taken in the current state
            reward (float): Reward received after taking the action
            next_state (np.ndarray): Next state observation after taking the action
            done (bool): Whether the episode ended after this transition
        """
        self.__memory.append((state, action, reward, next_state, done))

    def choose_action(self, state: np.ndarray) -> int:
        """
        Choose an action using epsilon-greedy policy.

        With probability epsilon, chooses a random action for exploration.
        Otherwise, chooses the action with the highest Q-value according to
        the main network (exploitation). This balances exploration of new
        actions with exploitation of learned knowledge.

        Args:
            state (np.ndarray): Current state observation from the environment

        Returns:
            int: Selected action index (0 to n_actions-1)
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(self.__n_actions)

        state_tensor = torch.tensor([state], dtype=torch.float).to(
            self.q_eval_network.device
        )
        actions = self.q_eval_network.forward(state_tensor)
        return torch.argmax(actions).item()

    def learn(self) -> None:
        """
        Train the neural network using experience replay and target network.

        This method implements the core DQN learning algorithm:
        1. Samples a batch of experiences from the replay buffer
        2. Computes Q-values for current states using the main network
        3. Computes target Q-values using the target network and Bellman equation
        4. Updates the main network using the temporal difference error
        5. Periodically updates the target network with main network weights
        6. Decays epsilon for exploration-exploitation balance

        The method only trains if there are enough experiences in the buffer
        (at least batch_size transitions).

        Note:
            This method modifies the internal state of the agent by updating
            network weights, epsilon value, and learn step counter.
        """
        if len(self.__memory) < self.__dqn_config.batch_size:
            return

        batch = random.sample(self.__memory, self.__dqn_config.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)
        states_tensor = torch.tensor(states, dtype=torch.float32).to(
            self.q_eval_network.device
        )
        actions_tensor = (
            torch.tensor(actions, dtype=torch.int64)
            .unsqueeze(1)
            .to(self.q_eval_network.device)
        )
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(
            self.q_eval_network.device
        )
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(
            self.q_eval_network.device
        )
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(
            self.q_eval_network.device
        )

        q_pred = self.q_eval_network(states_tensor).gather(1, actions_tensor).squeeze()
        q_next = self.q_target_network(next_states_tensor).max(dim=1)[0]
        q_target = rewards_tensor + self.__dqn_config.gamma * q_next * (
            1 - dones_tensor
        )

        loss = self.q_eval_network.loss(q_pred, q_target)

        self.q_eval_network.optimizer.zero_grad()
        loss.backward()
        self.q_eval_network.optimizer.step()

        self.__learn_step_counter += 1

        if self.__learn_step_counter % self.__dqn_config.target_update == 0:
            self.q_target_network.load_state_dict(self.q_eval_network.state_dict())

        self.epsilon = max(
            self.__dqn_config.eps_min,
            self.epsilon * self.__dqn_config.eps_dec,
        )
