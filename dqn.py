"""
Deep Q-Network (DQN) implementation for reinforcement learning.

This module implements a neural network for Deep Q-Learning, a reinforcement learning
algorithm that learns to estimate Q-values (action-value function) for given states.
The network is designed to work with environments like LunarLander where an agent
needs to learn optimal actions based on observed states.

The DQN uses a fully connected neural network architecture with ReLU activations
and is optimized using the Adam optimizer with Mean Squared Error loss.

Author: vnniciusg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeeQNetwork(nn.Module):
    """
    Deep Q-Network (DQN) for reinforcement learning.

    A fully connected neural network that approximates the Q-value function,
    mapping states to action values. The network consists of three fully
    connected layers with ReLU activations between them.

    The network automatically moves to GPU if available, and uses Adam optimizer
    with Mean Squared Error loss for training.

    Attributes:
        lr (float): Learning rate for the optimizer
        input_dims (tuple[int]): Dimensions of the input state space
        fc1 (nn.Linear): First fully connected layer
        fc2 (nn.Linear): Second fully connected layer
        fc3 (nn.Linear): Output layer (maps to action space)
        optimizer (optim.Adam): Adam optimizer for training
        loss (nn.MSELoss): Mean Squared Error loss function
    """

    def __init__(
        self,
        lr: float,
        input_dims: tuple[int],
        fc1_dims: int,
        fc2_dims: int,
        n_actions: int,
    ) -> None:
        """
        Initialize the Deep Q-Network.

        Args:
            lr (float): Learning rate for the Adam optimizer
            input_dims (tuple[int]): Dimensions of the input state space
            fc1_dims (int): Number of neurons in the first hidden layer
            fc2_dims (int): Number of neurons in the second hidden layer
            n_actions (int): Number of possible actions (output layer size)
        """
        super().__init__()

        self.lr = lr
        self.input_dims = input_dims

        # network layers
        self.fc1 = nn.Linear(*self.input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, state):
        """
        Forward pass through the network.

        Computes Q-values for all possible actions given the current state.
        The state is passed through two hidden layers with ReLU activations,
        then through the output layer to produce Q-values.

        Args:
            state (torch.Tensor): Input state tensor of shape (batch_size, *input_dims)

        Returns:
            torch.Tensor: Q-values for all actions, shape (batch_size, n_actions)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return self.fc3(x)
