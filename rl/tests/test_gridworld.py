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
    env = GridWorld(n=2, m=3)
    assert env.n == 2
    assert env.m == 3
    assert isinstance(env.state, Location)
    assert isinstance(env.target, Location)

def test_reset():
    env = GridWorld(n=2, m=3, seed=42)
    initial_state = env.state
    initial_start = env.start
    initial_target = env.target
    
    env.reset()
    assert env.state == initial_state
    assert env.start == initial_start
    assert env.target == initial_target

def test_reset_with_random_start():
    env = GridWorld(n=2, m=3, seed=42, random_start=True)
    initial_state = env.state
    initial_start = env.start
    initial_target = env.target
    
    env.reset()
    assert env.state != initial_state
    assert env.start == initial_start
    assert env.target == initial_target

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
    env = GridWorld(n=1, m=1)
    
    # Force state to (0,0)
    env.state = Location(0, 0)
    
    # Should not move when at boundaries
    env.step(Action.LEFT)
    assert env.state == Location(0, 0)
    
    env.step(Action.DOWN)
    assert env.state == Location(0, 0)

def test_rewards():
    env = GridWorld(n=1, m=1, win_reward=10, lose_reward=-1)
    
    # Force state and target to be different
    env.state = Location(0, 0)
    env.target = Location(1, 1)
    
    # Should get lose_reward when not reaching target
    _, reward = env.step(Action.RIGHT)
    assert reward == -1
    
    # Force state and target to be same
    env.state = Location(1, 1)
    
    # Should get win_reward when reaching target
    _, reward = env.step(Action.UP)  # Any action when already at target
    assert reward == 10


if __name__ == "__main__":
    pytest.main(args=["-s", os.path.abspath(__file__)])
