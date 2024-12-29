"""Implements on-policy Monte Carlo with exploring starts on GridWorld.

Key findings:
- Convergence is very finicky
- Got major breakthroughs once we messed with the reward signals

"""

from gridworld.gridworld import GridWorld, Action, Location
import logging
import numpy as np
import random
from typing import List, Mapping


def evaluate(
    env: GridWorld,
    policy: Mapping[Location, Action],
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
        action = policy[state]
        state, reward, done = env.step(action)
        total_reward += reward
        num_steps += 1
        if done:
            break
    if render:
        env.render()
    return total_reward


def reformat_mat(
    mat: Mapping[str, Mapping[str, int]], states: List[Location], actions: List[Action]
):
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
    n = 20
    m = 20
    num_holes = 20
    num_obstacles = 20
    gamma = 0.99  # discount factor
    mc_max_steps = 200
    min_mc_iters = 1000
    epsilon = 0.2
    min_returns = 5
    env = GridWorld(
        n=n,
        m=m,
        num_obstacles=num_obstacles,
        num_holes=num_holes,
        seed=seed,
        verbose=False,
        win_reward=100,
        step_reward=-1,
        hole_reward=-100,
        wall_reward=-5,
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
    eval_history = [evaluate(env, policy, max_steps=mc_max_steps, render=True)]
    num_iterations = 0

    while num_iterations < min_mc_iters:
        # Generate episode
        env.reset()
        state = random.choice(states)
        action = random.choice(actions)
        start_pairs.add((state, action))

        # Generate episode with exploring starts and epsilon-greedy policy
        num_steps = 0
        done = False
        next_state, reward, done = env.step(action, state=state)
        trajectory = [(state, action, reward)]
        state = next_state

        while num_steps < mc_max_steps and not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = random.choice(actions)
            else:
                action = policy[state]

            next_state, reward, done = env.step(action)
            trajectory.append((state, action, reward))
            state = next_state
            if done:
                break
            num_steps += 1

        # Update Q-values and policy
        g = 0
        for i, step in enumerate(trajectory[::-1]):
            state, action, reward = step
            g = gamma * g + reward
            prior_pairs = set(map(lambda x: (x[0], x[1]), trajectory[::-1][i + 1 :]))

            if (state, action) not in prior_pairs:  # First-visit MC
                returns[state][action].append(g)

                # Only update policy if we have enough samples
                if len(returns[state][action]) >= min_returns:
                    q[state][action] = np.mean(returns[state][action])

                    # Find best action based on current Q-values
                    best_action = max(actions, key=lambda a: q[state][a])
                    policy[state] = best_action

        num_iterations += 1

        # Evaluate periodically
        if num_iterations % 100 == 0:
            eval_reward = evaluate(env, policy, max_steps=mc_max_steps, render=True)
            eval_history.append(eval_reward)
            logging.info(f"Iteration {num_iterations}, Reward: {eval_reward}")
        else:
            eval_history.append(
                evaluate(env, policy, max_steps=mc_max_steps, render=False)
            )

    # Final evaluation
    final_reward = evaluate(env, policy, max_steps=mc_max_steps, render=True)
    eval_history.append(final_reward)

    logging.info(
        "Reward for original policy was %d, for optimal policy: %d",
        eval_history[0],
        eval_history[-1],
    )
    logging.info("Converged in %d iterations", num_iterations)
    logging.info("Eval history: %s", eval_history)
    logging.info(
        "Num initial SA pairs / total possible: %d / %d",
        len(start_pairs),
        len(states) * len(actions),
    )
