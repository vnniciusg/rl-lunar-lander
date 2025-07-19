import os
import warnings
from typing import List, Tuple

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
from gymnasium.utils.save_video import save_video
from loguru import logger

from agent import Agent

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

VIDEO_RECORD_INTERVAL = 100
RECORD_LAST_N_EPISODES = 5
RECORD_BEST_EPISODES = True


def setup_environment():
    """
    Set up the training environment and create necessary directories.

    Creates the videos directory for saving episode recordings and initializes
    the LunarLander-v3 environment with RGB array rendering mode. Also creates
    and configures the DQN agent with the appropriate input and action dimensions.

    Returns:
        tuple: A tuple containing:
            - env (gym.Env): The configured LunarLander environment
            - agent (Agent): The initialized DQN agent
    """
    os.makedirs("videos", exist_ok=True)
    env = gym.make("LunarLander-v3", render_mode="rgb_array_list")
    agent = Agent(input_dims=env.observation_space.shape, n_actions=env.action_space.n)
    return env, agent


def should_record_video(
    episode_idx: int, score: float, best_score: float, n_episodes: int
) -> Tuple[bool, str]:
    """
    Determine whether to record a video for the current episode and the reason.

    This function implements a video recording strategy based on multiple criteria:
    - Records at regular intervals (every VIDEO_RECORD_INTERVAL episodes)
    - Records the last few episodes of training
    - Records episodes that achieve new best scores (if enabled)

    Args:
        episode_idx (int): Current episode index (0-based)
        score (float): Score achieved in the current episode
        best_score (float): Best score achieved so far in training
        n_episodes (int): Total number of episodes planned for training

    Returns:
        tuple: A tuple containing:
            - bool: True if video should be recorded, False otherwise
            - str: Reason for recording ("interval", "final", "best", or None)
    """
    if episode_idx % VIDEO_RECORD_INTERVAL == 0:
        return True, "interval"

    if episode_idx >= n_episodes - RECORD_LAST_N_EPISODES:
        return True, "final"

    if RECORD_BEST_EPISODES and episode_idx > 50 and score > best_score:
        return True, "best"

    return False, None


def run_episode(env, agent) -> float:
    """
    Execute a complete training episode and return the total reward.

    Runs one full episode of the environment from start to finish, where the agent
    interacts with the environment by choosing actions, storing experiences in the
    replay buffer, and learning from those experiences. The episode continues until
    the environment signals termination or truncation.

    Args:
        env (gym.Env): The gymnasium environment to run the episode in
        agent (Agent): The DQN agent that will interact with the environment

    Returns:
        float: Total cumulative reward achieved during the episode

    Note:
        This function modifies the agent's internal state by:
        - Adding experiences to the replay buffer
        - Updating the neural network weights through learning
        - Decaying the exploration rate (epsilon)
    """
    done = False
    state, _ = env.reset()
    total_reward = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store_transition(state, action, reward, next_state, done)
        agent.learn()

        state = next_state
        total_reward += reward

    return total_reward


def save_episode_video(env, episode_idx: int, record_reason: str, total_reward: float):
    """
    Save a video recording of the episode to disk.

    Creates a video file from the rendered frames collected during the episode.
    The video is saved with a descriptive filename that includes the episode number,
    recording reason, and the total reward achieved. Handles errors gracefully
    and logs the outcome.

    Args:
        env (gym.Env): The environment containing the rendered frames
        episode_idx (int): Index of the episode being recorded
        record_reason (str): Reason for recording ("interval", "final", "best")
        total_reward (float): Total reward achieved in the episode

    Note:
        Videos are saved in the 'videos/' directory with filename format:
        'episode_XXXX_reason_reward_YYY.mp4'

    Raises:
        Logs error message if video saving fails, but doesn't raise exceptions
        to avoid interrupting the training process.
    """
    try:
        video_name = (
            f"episode_{episode_idx:04d}_{record_reason}_reward_{total_reward:.0f}"
        )
        save_video(
            frames=env.render(),
            video_folder="videos",
            name_prefix=video_name,
            fps=env.metadata["render_fps"],
        )
        logger.info(f"Video saved: {video_name} (reason: {record_reason})")
    except Exception as e:
        logger.error(f"Failed to save video for episode {episode_idx}: {e}")


def calculate_moving_average(scores: List[float], window_size: int) -> List[float]:
    """
    Calculate the moving average of scores over a specified window size.

    Computes a rolling average to smooth out the score data and reveal trends
    in the agent's performance over time. For episodes with insufficient history,
    it calculates the average of all available episodes up to that point.

    Args:
        scores (List[float]): List of episode scores/rewards
        window_size (int): Number of episodes to include in each average calculation

    Returns:
        List[float]: List of moving averages. Length equals len(scores) if
                    scores has at least one element, empty list otherwise.

    Example:
        >>> scores = [10, 20, 30, 40, 50]
        >>> calculate_moving_average(scores, 3)
        [10.0, 15.0, 20.0, 30.0, 40.0]

    Note:
        - For the first (window_size - 1) episodes, uses all available scores
        - Starting from episode window_size, uses exactly window_size scores
    """
    if len(scores) < window_size:
        return []

    moving_avg = []
    for i in range(len(scores)):
        if i >= window_size - 1:
            moving_avg.append(sum(scores[i - window_size + 1 : i + 1]) / window_size)
        else:
            moving_avg.append(sum(scores[: i + 1]) / (i + 1))
    return moving_avg


def plot_training_results(scores: List[float], best_score: float):
    """
    Generate and save a comprehensive plot of training results.

    Creates a detailed visualization showing the agent's learning progress over time.
    The plot includes individual episode scores, moving average trend line, and
    highlights the best score achieved. The plot is saved as a high-resolution PNG file.

    Args:
        scores (List[float]): List of all episode scores/rewards achieved during training
        best_score (float): The highest score achieved during training

    Side Effects:
        - Creates and saves 'training_results.png' in the current directory
        - Uses matplotlib's 'Agg' backend to avoid display issues in headless environments
        - Closes the plot after saving to free memory

    Plot Features:
        - Line plot of all episode scores with transparency
        - Horizontal line indicating the best score achieved
        - Moving average line (50-episode window) to show trends
        - Grid for better readability
        - Comprehensive legend and labels
        - High DPI (300) for publication-quality output
    """
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(scores, alpha=0.7, label="Episode Reward")
    plt.axhline(
        y=best_score, color="r", linestyle="--", label=f"Best Score: {best_score:.1f}"
    )

    window_size = 50
    moving_avg = calculate_moving_average(scores, window_size)
    if moving_avg:
        plt.plot(
            moving_avg,
            color="red",
            linewidth=2,
            label=f"Moving Average ({window_size} episodes)",
        )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN LunarLander Performance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_results.png", dpi=300, bbox_inches="tight")
    plt.close()


def log_video_summary():
    """
    Display a summary of all recorded videos in the videos directory.

    Scans the 'videos' directory for MP4 files and logs information about
    all recorded episodes. Provides feedback to the user about what videos
    were successfully created during training.

    Side Effects:
        - Logs the total number of videos found
        - Lists each video file name in sorted order
        - Warns if no videos were recorded

    Note:
        Only counts files with '.mp4' extension as valid video files.
        Video files are sorted alphabetically for consistent output.
    """
    video_files = [f for f in os.listdir("videos") if f.endswith(".mp4")]
    if video_files:
        logger.info(f"Recorded {len(video_files)} videos:")
        for video in sorted(video_files):
            logger.info(f"  - {video}")
    else:
        logger.warning("No videos were recorded!")


def train_agent(n_episodes: int = 1000):
    """
    Main training function that orchestrates the entire DQN training process.

    Executes the complete training loop for the specified number of episodes,
    managing video recording, performance tracking, and result visualization.
    The function handles the entire training workflow from environment setup
    to final result reporting.

    Args:
        n_episodes (int, optional): Total number of episodes to train for.
                                   Defaults to 1000.

    Returns:
        tuple: A tuple containing:
            - scores (List[float]): List of all episode scores achieved
            - best_score (float): The highest score achieved during training

    Training Process:
        1. Sets up environment and initializes agent
        2. For each episode:
           - Determines if video recording is needed
           - Runs complete episode with agent-environment interaction
           - Updates best score tracking
           - Records video if criteria are met
           - Logs progress information
        3. Generates training results plot
        4. Displays final performance summary
        5. Shows video recording summary

    Side Effects:
        - Creates videos in 'videos/' directory
        - Saves training plot as 'training_results.png'
        - Logs detailed progress information throughout training
        - Modifies agent's internal state (weights, replay buffer, epsilon)

    Example:
        >>> scores, best = train_agent(500)  # Train for 500 episodes
        >>> print(f"Training completed. Best score: {best}")
    """
    env, agent = setup_environment()

    scores = []
    best_score = -float("inf")

    for idx in range(n_episodes):
        record_this_episode, record_reason = should_record_video(
            idx, best_score, best_score, n_episodes
        )

        total_reward = run_episode(env, agent)

        if total_reward > best_score:
            best_score = total_reward
            if not record_this_episode:
                record_this_episode, record_reason = should_record_video(
                    idx, total_reward, best_score - 1, n_episodes
                )

        if record_this_episode:
            save_episode_video(env, idx, record_reason, total_reward)

        scores.append(total_reward)

        if idx % 10 == 0 or record_this_episode:
            logger.info(
                f"EPISODE {idx}, Reward: {total_reward}, Best: {best_score}, Epsilon: {agent.epsilon:.3f}"
            )

    plot_training_results(scores, best_score)

    logger.success("Training completed!")
    logger.info(f"Final performance: Best reward = {best_score}")
    logger.info("Graph saved as 'training_results.png'")
    logger.info("Videos saved in 'videos/' directory")

    log_video_summary()

    return scores, best_score


if __name__ == "__main__":
    train_agent(n_episodes=1000)
