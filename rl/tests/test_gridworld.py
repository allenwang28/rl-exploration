"""
To run this:

pytest tests/test_gridworld.py
"""

import pytest
import os
import logging
from gridworld.gridworld import GridWorld, Action, Location
import copy

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def test_init():
    # Test with default parameters
    # Test with obstacles and holes
    env = GridWorld(n=3, m=3, num_obstacles=2, num_holes=1)
    assert len(env.obstacles) == 2
    assert len(env.holes) == 1
    # Ensure obstacles and holes don't overlap with start and target
    assert (env.start.x, env.start.y) not in env.obstacles
    assert (env.target.x, env.target.y) not in env.obstacles
    assert (env.start.x, env.start.y) not in env.holes
    assert (env.target.x, env.target.y) not in env.holes


def test_reset():
    env = GridWorld(n=2, m=3, seed=42)
    initial_state = env.state
    initial_start = env.start
    initial_target = env.target

    env.reset()
    assert env.state == initial_state
    assert env.start == initial_start
    assert env.target == initial_target
    assert not env.done


def test_obstacle_interaction():
    env = GridWorld(n=2, m=2, num_obstacles=1, seed=42)
    initial_state = copy.copy(env.state)

    # Force an obstacle right next to the agent
    obstacle_x = min(initial_state.x + 1, env.m)
    obstacle_y = initial_state.y
    env.obstacles = {Location(obstacle_x, obstacle_y)}

    # Try to move into obstacle
    state, reward, done = env.step(Action.RIGHT)
    assert state == initial_state  # Position shouldn't change
    assert reward == env.lose_reward  # Should get obstacle penalty
    assert not done  # Shouldn't end episode


def test_hole_interaction():
    env = GridWorld(n=2, m=2, num_holes=1, seed=42)
    initial_state = copy.copy(env.state)

    # Force a hole right next to the agent
    hole_x = min(initial_state.x + 1, env.m)
    hole_y = initial_state.y
    env.holes = {Location(hole_x, hole_y)}

    # Try to move into hole
    state, reward, done = env.step(Action.RIGHT)
    assert reward == env.hole_reward  # Should get hole penalty
    assert done  # Should end episode


def test_win_condition():
    env = GridWorld(n=2, m=2, seed=42)
    # Force target next to agent
    env.set_state(Location(0, 0))
    env.target = Location(1, 0)

    # Move to target
    state, reward, done = env.step(Action.RIGHT)
    assert reward == env.win_reward
    assert done
    assert state == env.target


def test_done_state():
    env = GridWorld(n=2, m=2, seed=42)
    # Force target next to agent
    env.state = Location(0, 0)
    env.target = Location(1, 0)

    # Move to target
    _, _, done = env.step(Action.RIGHT)
    assert done

    # Try to move after episode is done
    state, reward, still_done = env.step(Action.UP)
    assert still_done
    assert reward == 0  # No reward when episode is done


def test_reset_with_random_start():
    env = GridWorld(n=2, m=3, seed=42, random_start=True)
    initial_state = env.state
    initial_start = env.start
    initial_target = env.target

    env.reset()
    assert env.state != initial_state
    assert env.start == initial_start
    assert env.target == initial_target
    assert not env.done


def test_step_movement():
    env = GridWorld(n=2, m=3, seed=42)
    initial_state = copy.copy(env.state)

    # Test moving RIGHT
    env.step(Action.RIGHT)
    assert env.state.x == initial_state.x + 1
    assert env.state.y == initial_state.y

    # Test moving LEFT
    env.step(Action.LEFT)
    assert env.state.x == initial_state.x
    assert env.state.y == initial_state.y


def test_boundary_conditions():
    env = GridWorld(n=2, m=2)

    # Force state to (0,0)
    env.state = Location(0, 0)

    # Should not move when at boundaries
    env.step(Action.LEFT)
    assert env.state == Location(0, 0)

    env.step(Action.UP)
    assert env.state == Location(0, 0)


def test_rewards():
    env = GridWorld(n=2, m=2, win_reward=10, lose_reward=-1)

    # Force state and target to be different
    env.state = Location(0, 0)
    env.target = Location(1, 1)

    # Should get lose_reward when not reaching target
    _, reward, _ = env.step(Action.RIGHT)
    assert reward == -1

    # Force state and target to be same
    env.state = Location(1, 0)

    # Should get win_reward when reaching target
    _, reward, _ = env.step(Action.DOWN)  # Any action when already at target
    assert reward == 10


if __name__ == "__main__":
    pytest.main(args=["-s", os.path.abspath(__file__)])
