"""Implements n-step Q.
"""

from gridworld.gridworld import GridWorld, Action, Location
import logging
import numpy as np
import math
from typing import Callable, List, Mapping


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


def visualize_q(
    q: Mapping[Location, Mapping[Action, float]],
    states: List[Location],
    actions: List[Action],
):
    """Visualizes Q-values as a heatmap using colored grid cells.

    For each cell, shows the maximum Q-value and its corresponding action direction.
    Brighter colors indicate higher Q-values.
    """
    from colorama import Fore, Back, Style

    # Find min and max Q-values for normalization
    all_q_values = [q[s][a] for s in states for a in actions]
    q_min, q_max = min(all_q_values), max(all_q_values)

    # Unicode arrows for actions
    ARROWS = {Action.RIGHT: "→", Action.LEFT: "←", Action.UP: "↑", Action.DOWN: "↓"}

    # Find grid dimensions
    max_x = max(s.x for s in states)
    max_y = max(s.y for s in states)

    # Create the grid
    for y in range(max_y + 1):
        row = []
        for x in range(max_x + 1):
            current_loc = Location(x=x, y=y)
            if current_loc in q:
                # Get max Q-value and best action for this state
                best_action = max(actions, key=lambda a: q[current_loc][a])
                max_q = q[current_loc][best_action]

                # Normalize Q-value to [0, 1]
                if q_max != q_min:
                    normalized_q = (max_q - q_min) / (q_max - q_min)
                else:
                    normalized_q = 0.5

                # Choose background color intensity based on Q-value
                if normalized_q < 0.2:
                    bg = Back.RED
                elif normalized_q < 0.4:
                    bg = Back.YELLOW
                elif normalized_q < 0.6:
                    bg = Back.WHITE
                elif normalized_q < 0.8:
                    bg = Back.CYAN
                else:
                    bg = Back.GREEN

                # Format cell with arrow and Q-value
                cell = f"{bg}{Fore.BLACK}{ARROWS[best_action]}{max_q:3.0f}{Style.RESET_ALL}"
            else:
                cell = f"{Back.WHITE}{Fore.BLACK} · {Style.RESET_ALL}"
            row.append(cell)
        print(" ".join(row))
    print(Style.RESET_ALL)

    # Print legend
    print("\nColor legend (Q-value ranges):")
    print(
        f"{Back.RED}{Fore.BLACK} Lowest {Style.RESET_ALL} < "
        f"{Back.YELLOW}{Fore.BLACK} Low {Style.RESET_ALL} < "
        f"{Back.WHITE}{Fore.BLACK} Medium {Style.RESET_ALL} < "
        f"{Back.CYAN}{Fore.BLACK} High {Style.RESET_ALL} < "
        f"{Back.GREEN}{Fore.BLACK} Highest {Style.RESET_ALL}"
    )


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info("Initializing mdp policy iteration.")

    # Constants
    seed = 42
    n = 8
    m = 4
    num_holes = 5
    num_obstacles = 3
    step_size = 0.9
    gamma = 0.9
    num_episodes = 500
    max_steps_per_episode = 100
    epsilon = 0.3
    n_step = 5
    decay_rate = 0.1
    env = GridWorld(
        n=n,
        m=m,
        num_obstacles=num_obstacles,
        num_holes=num_holes,
        seed=seed,
        verbose=False,
        win_reward=1000,
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

    def policy(state):
        return max(actions, key=lambda a: q[state][a])

    num_iterations = 0

    eval_history = [evaluate(env, policy, max_steps=max_steps_per_episode, render=True)]
    logging.info("Reward for randomly initialized policy: %d", eval_history[0])
    drawn_states = set()

    while num_iterations < num_episodes:
        env.reset()
        state = np.random.choice(env.get_non_terminal_states())
        drawn_states.add(state)

        if np.random.uniform(0.0, 1.0) < epsilon / (1 + decay_rate * num_iterations):
            action = np.random.choice(actions)
        else:
            action = policy(state)

        recorded_states = [state]
        recorded_actions = [action]
        recorded_rewards = [None]
        end_time_step = math.inf  # T
        episode_completed = False  # tau = T - 1
        terminal_state_seen = False
        current_ts = 0  # t

        while not episode_completed:
            if not terminal_state_seen:  # t < T
                state, reward, terminal_state_seen = env.step(action, state=state)
                recorded_rewards.append(reward)
                recorded_states.append(state)

                if terminal_state_seen:
                    end_time_step = current_ts
                else:
                    if np.random.uniform(0.0, 1.0) < epsilon / (
                        1 + decay_rate * num_iterations
                    ):
                        action = np.random.choice(actions)
                    else:
                        action = policy(state)
                    recorded_actions.append(action)

            tau = current_ts - n_step
            if tau >= 0:
                g = 0
                for i in range(tau + 1, min(tau + n_step + 1, end_time_step + 1) + 1):
                    g += gamma ** (i - tau - 1) * recorded_rewards[i]
                if tau + n_step < end_time_step:
                    g = (
                        g
                        + (gamma**n_step)
                        * q[recorded_states[tau + n_step + 1]][
                            recorded_actions[tau + n_step + 1]
                        ]
                    )

                s_tau, a_tau = recorded_states[tau], recorded_actions[tau]
                q[s_tau][a_tau] += step_size * (g - q[s_tau][a_tau])

            if tau >= end_time_step:
                episode_completed = True
                break
            current_ts += 1

        if num_iterations % 100 == 0:
            visualize_q(q, states, actions)
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
    visualize_q(q, states, actions)
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
        len(env.get_non_terminal_states()),
    )
