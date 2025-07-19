"""
Configuration settings for Deep Q-Learning (DQN) agent using Pydantic.

This module contains the configuration class that defines all hyperparameters
and settings used by the DQN agent for training on the LunarLander environment.

Author: vnniciusg
"""

from pydantic import BaseModel, Field


class DQNConfig(BaseModel):
    """
    Configuration class for Deep Q-Network (DQN) hyperparameters using Pydantic.

    This class contains all the hyperparameters needed for DQN training,
    including learning parameters, exploration strategy, memory settings,
    and training configuration. Pydantic provides automatic validation
    and serialization capabilities.

    Attributes:
        lr (float): Learning rate for the neural network optimizer
        gamma (float): Discount factor for future rewards (0 < gamma <= 1)
        eps_min (float): Minimum exploration rate (epsilon floor)
        eps_dec (float): Epsilon decay factor applied each step
        mem_size (int): Maximum size of the replay memory buffer
        batch_size (int): Number of samples per training batch
        target_update (int): Frequency (in steps) to update the target network
    """

    lr: float = Field(
        default=1e-3, description="Learning rate for the neural network optimizer"
    )
    gamma: float = Field(default=0.99, description="Discount factor for future rewards")
    eps_min: float = Field(
        default=0.01, description="Minimum exploration rate (epsilon floor)"
    )
    eps_dec: float = Field(
        default=0.995, description="Epsilon decay factor applied each step"
    )
    mem_size: int = Field(
        default=1000, description="Maximum size of the replay memory buffer"
    )
    batch_size: int = Field(
        default=64, description="Number of samples per training batch"
    )
    target_update: int = Field(
        default=1000, description="Frequency (in steps) to update the target network"
    )
