"""Implements MDP policy iteration on GridWorld."""

from gridworld.gridworld import GridWorld, Action, Location
import logging
import numpy as np
from typing import Mapping


def evaluate(
    env: GridWorld,
    policy: Mapping[Location, Action],
    max_steps: int = 1000,
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
    logging.info("Initializing mdp policy iteration.")

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

    # Initializations
    np.random.seed(seed)
    values = np.random.normal(size=(len(states), 1))
    policy = {state: np.random.choice(actions) for state in states}

    eval_history = [evaluate(env, policy, render=True)]
    policy_evaluation_iters_history = []
    logging.info("Reward for randomly initialized policy: %d", eval_history[0])

    num_iterations = 0
    policy_stable = False

    while not policy_stable:
        pe_i = 0
        # Policy evaluation
        while True:
            delta = 0
            for i, state in enumerate(states):
                v = values[i]
                new_value = 0
                action = policy[state]

                next_state, reward, _ = env.step(action, state=state)
                next_state_index = states.index(next_state)
                new_value = reward + gamma * values[next_state_index]
                delta = max(delta, abs(v - new_value))
                values[i] = new_value
            pe_i += 1
            if delta < theta:
                # logging.info("[Iteration %d] Policy evaluated completed in %d steps", num_iterations, pe_i)
                policy_evaluation_iters_history.append(pe_i)
                break

        # Policy improvement
        policy_stable = True
        for i, state in enumerate(states):
            old_action = policy[state]
            optimal_action = None
            optimal_value = None
            for action in actions:
                env.set_state(state)
                next_state, reward, _ = env.step(action)
                next_state_index = states.index(next_state)
                projected_value = reward + gamma * values[next_state_index]
                if optimal_value is None or projected_value > optimal_value:
                    optimal_value = projected_value
                    optimal_action = action
            if optimal_action != old_action:
                # not policy stable!
                policy[state] = optimal_action
                policy_stable = False

        num_iterations += 1

        if policy_stable:
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
    logging.info(
        "Policy evaluation iterations history: %s", policy_evaluation_iters_history
    )
