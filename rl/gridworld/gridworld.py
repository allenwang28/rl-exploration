"""
n=2, m=3

s_00 s_01 s_02
s_10 s_01 s_02

APIs to support:

env = gym.make("gridworld")

env.reset()
env.step(action)
env.

"""
from enum import Enum
import logging
import random
from dataclasses import dataclass
import copy


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


@dataclass
class Location:
    x: int
    y: int

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class GridWorld:

    def __init__(self, n: int, m: int,
                 seed: int = 0, verbose: bool = True,
                 random_start: bool = False,
                 win_reward: int = 10, lose_reward: int = -10):
        self.seed = seed
        self.n = n
        self.m = m
        self.random_start = random_start
        random.seed(seed)
        self.verbose = verbose
        self.log(f"Setting seed to {seed}.")
        self.target = Location(x=random.randint(0, self.n), y=random.randint(0, self.m))
        self.start = Location(x=random.randint(0, self.n), y=random.randint(0, self.m))
        self.state = None
        self.obstacles = []
        self.reset()
        self.win_reward = win_reward
        self.lose_reward = lose_reward
    
    def log(self, s: str):
        if self.verbose:
            logging.info("%s", s)

    def reset(self):
        if self.random_start:
            self.state = Location(x=random.randint(0, self.m), y=random.randint(0, self.m))
        else:
            self.state = copy.copy(self.start)
        self.log(f"Starting at {self.state}")
        self.log(f"Target is {self.target}")

    def step(self, action: Action) -> int:
        self.log(f"From {self.state}, taking action {action}.")
        new_state = copy.copy(self.state)
        if action == Action.UP and new_state.y < self.n:
            new_state.y += 1
        elif action == Action.DOWN and new_state.y > 0:
            new_state.y -= 1
        elif action == Action.LEFT and new_state.x > 0:
            new_state.x -= 1
        elif action == Action.RIGHT and new_state.x < self.m:
            new_state.x += 1

        self.state = new_state
        self.log(f"New location: {self.state}")
        if new_state == self.target:
            reward = self.win_reward
        else:
            reward = self.lose_reward
        self.log(f"Returning reward {reward}.")
        return reward
