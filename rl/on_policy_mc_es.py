"""Implements on-policy Monte Carlo with exploring starts on GridWorld."""

from gridworld.gridworld import GridWorld, Action, Location
import logging
import numpy as np
import random
from typing import List, Mapping


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


def reformat_mat(mat: Mapping[str, Mapping[str, int]], states: List[Location], actions: List[Action]):
    m = []
    for state in states:
        l = []
        for action in actions:
            l.append(mat[state][action])
        m.append(l)
    return m


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info("Initializing mdp policy iteration.")

    # Constants
    seed = 42
    n = 5
    m = 5
    num_holes = 2
    num_obstacles = 1
    gamma = 0.9  # discount factor
    mc_max_steps = 100
    min_mc_iters = 5000
    epsilon = 0.2
    min_returns = 5
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

    # Initializations
    np.random.seed(seed)
    q = {state: {action: np.random.normal() for action in actions} for state in states}
    returns = {state: {action: [] for action in actions} for state in states}
    policy = {state: np.random.choice(actions) for state in states}
    start_pairs = set()

    eval_history = [evaluate(env, policy, render=True)]
    num_iterations = 0

    while num_iterations < min_mc_iters:
        env.reset()
        # random draw
        state = random.choice(states)
        action = random.choice(actions)
        start_pairs.add((state, action))
        env.reset()
        #logging.info("[Iteration %d] state %s action %s", num_iterations, state, action)

        # generate an episode with the policy
        #logging.info("[Iteration %d] Generating policy...", num_iterations)
        num_steps = 0
        done = False
        next_state, reward, done = env.step(action, state=state)
        # Trajectory consists of S_t, A_t, R_{t+1}
        trajectory = [(state, action, reward)]
        state = next_state
        while num_steps < mc_max_steps and not done:
            # epsilon greedy
            if np.random.uniform(0., 1.) < epsilon:
                action = random.choice(actions)
            else:
                action = policy[state]

            next_state, reward, done = env.step(action, state=state)
            trajectory.append((state, action, reward))
            state = next_state
            num_steps += 1

        g = 0
        for i, step in enumerate(trajectory[::-1]):
            state, action, reward = step
            g = gamma * g + reward
            prior_pairs = set(map(lambda x: (x[0], x[1]), trajectory[::-1][i+1:]))
            if (state, action) not in prior_pairs:
                returns[state][action].append(g)

                if len(returns[state][action]) >= min_returns:
                    q[state][action] = np.mean(returns[state][action])
                    best_action = max(actions, key=lambda a: q[state][a])
                    policy[state] = best_action

        num_iterations += 1
        if num_iterations >= min_mc_iters:
            eval_history.append(evaluate(env, policy, render=True))
        else:
            eval_history.append(evaluate(env, policy, render=False))

    logging.info(
        "Reward for original policy was %d, for optimal policy: %d",
        eval_history[0],
        eval_history[-1],
    )
    logging.info("Converged in %d iterations", num_iterations)
    logging.info("Eval history: %s", eval_history)
    logging.info("Num initial SA pairs / total possible: %d / %d", len(start_pairs), len(states) * len(actions))
