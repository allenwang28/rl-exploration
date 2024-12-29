"""Implements MDP value iteration on GridWorld."""

from gridworld.gridworld import GridWorld, Action, Location
import logging
import numpy as np
from typing import Mapping


def evaluate(
    env: GridWorld,
    policy: Mapping[Location, Action],
    max_steps: int = 100,
    render: bool = False,
) -> int:
    """Evaluates a policy on GridWorld."""
    env.reset()
    done = False
    num_steps = 0
    total_reward = 0
    state = env.get_state()

    while num_steps < max_steps and not done:
        action = policy[state]
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
    logging.info("Initializing MDP value iteration")

    # Constants
    seed = 42
    n = 5
    m = 5
    num_holes = 5
    num_obstacles = 5
    theta = 0.001  # the stopping condition
    gamma = 0.9  # discount factor
    env = GridWorld(
        n=n,
        m=m,
        num_obstacles=num_obstacles,
        num_holes=num_holes,
        seed=seed,
        verbose=False,
    )
    logging.info("Initial state:")
    env.render()

    states = env.get_possible_states()
    actions = env.get_possible_actions()

    # Inits
    np.random.seed(seed)
    values = {state: np.random.normal() for state in states}
    policy = {state: np.random.choice(actions) for state in states}

    eval_history = [evaluate(env, policy, render=True)]
    logging.info("Reward for randomly initialized policy: %d", eval_history[0])

    for state in env.get_terminal_states():
        values[state] = 0

    num_iterations = 0

    while True:
        delta = 0
        for state in states:
            v = values[state]

            optimal_value = None
            optimal_action = None
            for action in actions:
                next_state, reward, _ = env.step(action, state=state)
                current_value = reward + gamma * values[next_state]
                if optimal_value is None or current_value > optimal_value:
                    optimal_value = current_value
                    optimal_action = action
            values[state] = optimal_value
            delta = max(delta, abs(v - optimal_value))
            policy[state] = optimal_action
        if delta < theta:
            eval_history.append(evaluate(env, policy, render=True))
            break
        num_iterations += 1
        eval_history.append(evaluate(env, policy, render=False))

    logging.info(
        "Reward for original policy was %d, for optimal policy: %d",
        eval_history[0],
        eval_history[-1],
    )
    logging.info("Converged in %d iterations", num_iterations)
    logging.info("Eval history: %s", eval_history)
