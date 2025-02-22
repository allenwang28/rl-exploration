"""Implements REINFORCE as outlined in Sutton's 2020 RL book."""
import gymnasium as gym
import wandb
import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Policy(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_size=128):
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), hidden_size), nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True),
            nn.Linear(hidden_size, np.prod(action_shape)),
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
        action = policy.get_action(obs).item()
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
        hidden_size=256,
        n_episodes=1000,
        discount=0.99,
        learning_rate=2.5e-4,
        num_eval_episodes=10,
    )
    logger.info(f"config: {config}")
    wandb.init(
        mode="online" if config["use_wandb"] else "disabled",
        project=f"{config['env']}-reinforce",
        config=config,
        resume="allow",
    )
    torch.manual_seed(config["seed"])
    logger.info(f"creating training env {config['env']}")
    train_env = gym.make(config["env"])

    action_shape = train_env.action_space.n
    state_shape = train_env.observation_space.shape or train_env.observation_space.n
    logger.info(f"Creating policy with action_shape {action_shape} and state_shape {state_shape}")
    policy = Policy(action_shape=action_shape, state_shape=state_shape, hidden_size=config["hidden_size"])
    logger.info(f"Creating Adam optimizer with learning rate {config['learning_rate']}")
    optim = torch.optim.Adam(policy.parameters(), lr=config["learning_rate"], eps=1e-5)

    
    for episode_num in range(config["n_episodes"]):
        total_return = 0
        obs, info = train_env.reset()
        episode_over = False

        rewards = []
        log_probs = []
        
        # generate trajectory
        while not episode_over:
            # Get action and its log probability
            logits = policy(obs)
            probs = Categorical(logits=logits)
            action = probs.sample()
            log_prob = probs.log_prob(action)
            
            # Take action in environment
            next_obs, reward, term, trunc, info = train_env.step(action.item())
            episode_over = term or trunc
            rewards.append(reward)
            log_probs.append(log_prob)
            obs = next_obs  # Update observation

        episodic_return = 0
        returns = []
        for reward in reversed(rewards):
            # Calculate return (G)
            episodic_return = reward + config["discount"] * episodic_return
            total_return += reward
            returns.append(episodic_return)
        returns = torch.tensor(list(reversed(returns)))
        # Calculate loss and update policy
        policy_loss = []
        for log_prob, episodic_return in zip(log_probs, returns):
            policy_loss.append(-log_prob * episodic_return)
        loss = torch.stack(policy_loss).sum()
        optim.zero_grad()
        loss.backward()
        optim.step()
        eval_returns = evaluate(config["num_eval_episodes"], policy)
        episodic_metrics = dict(
            trajectory_len=len(returns),
            total_return=total_return,
            grad_norm=calculate_grad_norm(policy),
            eval_returns=eval_returns)
        logger.info(f"[ep_step={episode_num}] {episodic_metrics}")
        wandb.log(episodic_metrics, step=episode_num)
    
    logging.info("Evaluating the final policy...")
    ret = run_policy(policy, visualize=True)
    logging.info(f"Return: {ret}")


if __name__ == "__main__":
    main()
