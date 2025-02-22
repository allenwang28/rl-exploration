"""Implements REINFORCE as outlined in Sutton's 2020 RL book."""
import gymnasium as gym
import wandb
import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical


logger = logging.getLogger(__file__)


class Policy(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs)
        return self.model(obs)

    def get_action(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs)
        logits = self.model(obs)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action


def calculate_grad_norm(model: nn.Module):
    """Calculate the L2 grad norm."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def run_policy(policy, visualize=False):
    if visualize:
        env = gym.make("CartPole-v1", render_mode="human")
    else:
        env = gym.make("CartPole-v1")

    obs, _ = env.reset()
    total_reward = 0

    episode_over = False
    while not episode_over:
        action = policy.get_action(obs)
        action = env.action_space.sample()  # agent policy that uses the observation and info
        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
        total_reward += reward
        obs = next_obs

    env.close()
    return total_reward


def evaluate(num_eval_episodes, policy):
    """Return the average total return of a model on num_eval_episodes."""
    episodic_rewards = []
    for _ in range(num_eval_episodes):
        episodic_rewards.append(run_policy(policy, visualize=False))
    return np.mean(episodic_rewards)



def main():
    config = dict(
        use_wandb=True,
        env="CartPole-v1",
        seed=42,
        n_episodes=100,
        step_size=0.01,
        discount=0.9,
        learning_rate=2.5e-4,
        num_eval_episodes=10,
    )
    logger.info(f"config: {config}")
    wandb.init(
        mode="online" if config["use_wandb"] else "disabled",
        project=f"{config['env']}-reinforce",
        config=config,
    )
    torch.manual_seed(config["seed"])
    logger.info(f"creating training env {config['env']}")
    train_env = gym.make(config["env"])

    action_shape = train_env.action_space.n
    state_shape = train_env.observation_space.shape or train_env.observation_space.n
    logger.info(f"Creating policy with action_shape {action_shape} and state_shape {state_shape}")
    policy = Policy(action_shape=action_shape, state_shape=state_shape)
    logger.info(f"Creating Adam optimizer with learning rate {config['learning_rate']}")
    optim = torch.optim.Adam(policy.parameters(), lr=config["learning_rate"], eps=1e-5)

    global_step = 0
    
    for episode_num in range(config["n_episodes"]):
        total_return = 0
        trajectory_len = 0
        obs, info = train_env.reset()
        episode_over = False
        episodic_return = 0  # Initialize return at start of episode
        
        while not episode_over:
            # Get action and its log probability
            logits = policy(obs)
            probs = Categorical(logits=logits)
            action = probs.sample()
            log_prob = probs.log_prob(action)
            
            # Take action in environment
            next_obs, reward, term, trunc, info = train_env.step(action.item())
            episode_over = term or trunc

            # Calculate return (G)
            episodic_return = reward + config["discount"] * episodic_return
            total_return += reward

            # Calculate loss and update policy
            loss = -log_prob * episodic_return
            optim.zero_grad()
            loss.backward()
            optim.step()

            obs = next_obs  # Update observation

            wandb.log(dict(
                episodic_return=episodic_return,
                grad_norm=calculate_grad_norm(policy),
            ), step=global_step)

            trajectory_len += 1
            global_step += 1

        eval_returns = evaluate(config["num_eval_episodes"], policy)

        episodic_metrics = dict(
            trajectory_len=trajectory_len,
            episodic_return=episodic_return,
            total_return=total_return,
            eval_returns=eval_returns)

        logger.info(f"[ep_step={episode_num}] {episodic_metrics}")
        wandb.log(episodic_metrics, step=episode_num)

    run_policy(policy, visualize=True)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    main()
