"""Implements MDP policy iteration on GridWorld."""

from gridworld.gridworld import GridWorld, Action, Location
import logging
import numpy as np
from typing import Mapping


def evaluate(
    env: GridWorld, policy: Mapping[Location, Action], max_steps: int = 1000
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

    env.render()
    env.reset()
    return total_reward


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info("Initializing mdp policy iteration.")
    # Constants
    seed = 42
    n = 15
    m = 10
    num_holes = 10
    num_obstacles = 5
    theta = 0.001  # the stopping condition
    gamma = 0.01  # discount factor
    env = GridWorld(
        n=n,
        m=m,
        num_obstacles=num_obstacles,
        num_holes=num_holes,
        seed=seed,
        verbose=False,
    )
    states = env.get_states()
    actions = env.get_actions()

    # Initializations
    np.random.seed(seed)
    values = np.random.normal(size=(len(states), 1))
    policy = {state: np.random.choice(actions) for state in states}

    naive_reward = evaluate(env, policy)
    logging.info("Reward for randomly initialized policy: %d", naive_reward)

    num_iterations = 0
    policy_stable = False

    while not policy_stable:
        pe_i = 0
        # Policy evaluation
        while True:
            delta = 0
            for i, state in enumerate(states):
                env.reset()
                v = values[i]
                new_value = 0
                action = policy[state]

                env.set_state(state)
                next_state, reward, _ = env.step(action)
                next_state_index = states.index(next_state)
                new_value = reward + gamma * values[next_state_index]
                delta = max(delta, abs(v - new_value))
                values[i] = new_value
            pe_i += 1
            if delta < theta:
                # logging.info("[Iteration %d] Policy evaluated completed in %d steps", num_iterations, pe_i)
                break

        # Policy improvement
        policy_stable = True
        for i, state in enumerate(states):
            old_action = policy[state]
            optimal_action = None
            optimal_value = None
            for action in actions:
                env.reset()
                env.set_state(state)
                next_state, reward, _ = env.step(action)
                next_state_index = states.index(next_state)
                projected_value = reward + gamma * values[next_state_index]
                if optimal_value is None or projected_value > optimal_value:
                    optimal_value = projected_value
                    optimal_action = action
            # logging.info("[Iteration: %d] For state %s, original action was %s, optimal action is %s with value of %.2f",
            #             num_iterations, state, old_action, optimal_action, optimal_value)
            if optimal_action != old_action:
                # not policy stable!
                policy[state] = optimal_action
                policy_stable = False
                # logging.info("[Iteration: %d] Not policy stable", num_iterations)

        num_iterations += 1

    reward = evaluate(env, policy)
    logging.info(
        "Reward for original policy was %d, for optimal policy: %d",
        naive_reward,
        reward,
    )
