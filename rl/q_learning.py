"""Implements Q-learning."""

from gridworld.gridworld import GridWorld, Action, Location
import logging
import numpy as np
from typing import Callable, Mapping


def evaluate(
    env: GridWorld,
    policy: Callable[[Location], Action],
    max_steps: int = 50,
    render: bool = False,
) -> int:
    """Evaluates a policy on GridWorld."""
    env.reset()
    done = False
    num_steps = 0
    total_reward = 0
    state = env.get_state()

    while num_steps < max_steps and not done:
        action = policy(state)
        state, reward, done = env.step(action)
        total_reward += reward
        num_steps += 1
        if done:
            break
    if render:
        env.render()
    return total_reward


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info("Initializing mdp policy iteration.")

    # Constants
    seed = 42
    n = 10
    m = 10
    num_holes = 5
    num_obstacles = 10
    step_size = 0.8
    gamma = 0.9
    num_episodes = 100
    max_steps_per_episode = 100
    epsilon = 0.1
    decay_rate = 0.0
    env = GridWorld(
        n=n,
        m=m,
        num_obstacles=num_obstacles,
        num_holes=num_holes,
        seed=seed,
        verbose=False,
        win_reward=10,
        step_reward=-1,
        hole_reward=-1000,
        wall_reward=-5,
    )
    logging.info("Initial state:")
    env.render()

    states = env.get_possible_states()
    actions = env.get_possible_actions()

    # Initializations
    np.random.seed(seed)
    q = {state: {action: np.random.normal() for action in actions} for state in states}
    for state in env.get_terminal_states():
        for a in actions:
            q[state][a] = 0

    def policy(state):
        return max(actions, key=lambda a: q[state][a])

    num_iterations = 0

    eval_history = [evaluate(env, policy, max_steps=max_steps_per_episode, render=True)]
    logging.info("Reward for randomly initialized policy: %d", eval_history[0])
    drawn_states = set()

    while num_iterations < num_episodes:
        env.reset()
        state = np.random.choice(states)
        drawn_states.add(state)

        done = False
        while not done:
            if np.random.uniform(0.0, 1.0) < epsilon / (
                1 + decay_rate * num_iterations
            ):
                action = np.random.choice(actions)
            else:
                action = policy(state)

            next_state, reward, done = env.step(action, state=state)

            q[state][action] += (
                step_size
                / (1 + decay_rate * num_iterations)
                * (
                    reward
                    + gamma * max(q[next_state][a] for a in actions)
                    - q[state][action]
                )
            )
            state = next_state

        if num_iterations % 100 == 0:
            eval_reward = evaluate(
                env, policy, max_steps=max_steps_per_episode, render=True
            )
            eval_history.append(eval_reward)
            logging.info(f"Iteration {num_iterations}, Reward: {eval_reward}")
        else:
            eval_history.append(
                evaluate(env, policy, max_steps=max_steps_per_episode, render=False)
            )
        num_iterations += 1

    # Final evaluation
    final_reward = evaluate(env, policy, max_steps=max_steps_per_episode, render=True)
    eval_history.append(final_reward)

    logging.info(
        "Reward for original policy was %d, for optimal policy: %d",
        eval_history[0],
        eval_history[-1],
    )
    logging.info("Converged in %d iterations", num_iterations)
    logging.info("Eval history: %s", eval_history)
    logging.info(
        "Num initial states drawn / total possible: %d / %d",
        len(drawn_states),
        len(states),
    )
