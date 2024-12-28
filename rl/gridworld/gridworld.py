"""
x corresponds to m
y corresponds to n

n=3, m=2

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
from typing import List, Tuple
import copy
from colorama import init, Fore, Back, Style


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


@dataclass(frozen=True)
class Location:
    x: int
    y: int

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def as_tuple(self) -> Tuple[int, int]:
        return self.x, self.y


class GridWorld:
    def __init__(
        self,
        n: int,
        m: int,
        seed: int = 0,
        verbose: bool = True,
        random_start: bool = False,
        num_obstacles: int = 0,
        num_holes: int = 0,
        win_reward: int = 10,
        lose_reward: int = -10,
        hole_reward: int = -100,
    ):
        if m == n == 1:
            raise AssertionError("world size cannot be 1.")
        self.seed = seed
        self.n = n
        self.m = m
        self.random_start = random_start
        random.seed(seed)
        self.verbose = verbose
        self.log(f"Setting seed to {seed}.")
        self.target = Location(
            x=random.randint(0, self.m - 1), y=random.randint(0, self.n - 1)
        )
        self.start = copy.copy(self.target)
        while self.start == self.target:
            self.start = Location(
                x=random.randint(0, self.m - 1), y=random.randint(0, self.n - 1)
            )
        self.state = None
        self.history = []
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.hole_reward = hole_reward

        self.obstacles = set()
        self.holes = set()
        self.done = False

        available_spots = self.get_states()
        available_spots.remove(self.start)
        available_spots.remove(self.target)

        for _ in range(num_obstacles):
            if available_spots:
                loc = random.choice(available_spots)
                self.obstacles.add(loc)
                available_spots.remove(loc)
        for _ in range(num_holes):
            if available_spots:
                loc = random.choice(available_spots)
                self.holes.add(loc)
                available_spots.remove(loc)
        self.reset()

    def get_states(self) -> List[Location]:
        return [Location(x=x, y=y) for x in range(self.m) for y in range(self.n)]

    def get_actions(self) -> List[Action]:
        return (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)

    def get_state(self):
        return self.state

    def set_state(self, state: Location):
        self.state = state

    def log(self, s: str):
        if self.verbose:
            logging.info("%s", s)

    def reset(self) -> Location:
        self.done = False
        self.history = []
        if self.random_start:
            self.state = Location(
                x=random.randint(0, self.m - 1), y=random.randint(0, self.m - 1)
            )
            while self.state in self.holes or self.state in self.obstacles:
                self.state = Location(
                    x=random.randint(0, self.m - 1), y=random.randint(0, self.m - 1)
                )
        else:
            self.state = copy.copy(self.start)
        self.log(f"Starting at {self.state}")
        self.log(f"Target is {self.target}")
        return self.state

    def step(self, action: Action) -> Tuple[Location, int, bool]:
        if self.done:
            self.log("Episode is already done!")
            return copy.copy(self.state), 0, True

        self.log(f"From {self.state}, taking action {action}.")
        x, y = self.state.as_tuple()
        if action == Action.UP and y > 0:
            y -= 1
        elif action == Action.DOWN and y < self.n - 1:
            y += 1
        elif action == Action.LEFT and x > 0:
            x -= 1
        elif action == Action.RIGHT and x < self.m - 1:
            x += 1
        new_state = Location(x=x, y=y)

        self.log(f"New location: {self.state}")
        if new_state == self.target:
            reward = self.win_reward
            self.done = True
        elif new_state in self.obstacles:
            reward = self.lose_reward
            new_state = self.state
        elif new_state in self.holes:
            reward = self.hole_reward
            self.done = True
        else:
            reward = self.lose_reward

        self.history.append((copy.copy(self.state), action, reward))
        self.log(f"Returning reward {reward}.")
        self.state = new_state
        return copy.copy(self.state), reward, self.done

    def render(self, show_history: bool = True) -> str:
        """Renders the current state of the grid world with colors and box drawings.

        Args:
            show_history: If True, shows the path taken by the agent

        Returns:
            str: Pretty representation of the grid world
        """
        # Unicode characters
        TOP_LEFT = "┌"
        TOP_RIGHT = "┐"
        BOTTOM_LEFT = "└"
        BOTTOM_RIGHT = "┘"
        HORIZONTAL = "─"
        VERTICAL = "│"
        ARROWS = {Action.RIGHT: "→", Action.LEFT: "←", Action.UP: "↑", Action.DOWN: "↓"}

        # Create a dictionary to store movement history for each location
        history_map = {}
        if show_history:
            for idx, (loc, action, _) in enumerate(self.history, 1):
                history_map[f"{loc.x},{loc.y}"] = (idx, ARROWS[action])

        # Build the grid content
        grid = []
        # Add top border
        width = (self.m + 1) * 4 + 3  # Adjust width for 0 to m inclusive
        grid.append(TOP_LEFT + HORIZONTAL * width + TOP_RIGHT)

        # Add grid content
        for y in range(self.n + 1):  # Include n
            row = [VERTICAL + " "]  # Left border
            for x in range(self.m + 1):  # Include m
                current_loc = Location(x=x, y=y)
                loc_key = f"{x},{y}"

                if current_loc == self.state:
                    cell = f"{Back.BLUE}{Fore.WHITE} A {Style.RESET_ALL}"
                elif current_loc == self.target:
                    cell = f"{Back.GREEN}{Fore.WHITE} T {Style.RESET_ALL}"
                elif current_loc in self.obstacles:
                    cell = f"{Back.CYAN}{Fore.WHITE} # {Style.RESET_ALL}"
                elif current_loc in self.holes:
                    cell = f"{Back.BLACK}{Fore.WHITE} O {Style.RESET_ALL}"
                elif loc_key in history_map:
                    step_num, arrow = history_map[loc_key]
                    if step_num == 1:
                        cell = f"{Back.RED}{Fore.BLACK}{step_num:1}{arrow}{' ' if step_num < 10 else ''}{Style.RESET_ALL}"
                    else:
                        cell = f"{Back.YELLOW}{Fore.BLACK}{step_num:1}{arrow}{' ' if step_num < 10 else ''}{Style.RESET_ALL}"
                else:
                    cell = f"{Back.WHITE}{Fore.BLACK} · {Style.RESET_ALL}"
                row.append(cell)
            row.append(f" {VERTICAL}")  # Right border
            grid.append("".join(row))

        # Add bottom border
        grid.append(BOTTOM_LEFT + HORIZONTAL * width + BOTTOM_RIGHT)

        # Add legend
        if show_history and self.history:
            grid.append("\nPath taken:")
            path_str = "  ".join(
                [
                    f"{idx}:{ARROWS[action]}"
                    for idx, (_, action, _) in enumerate(self.history, 1)
                ]
            )
            grid.append(path_str)

        rendered = "\n".join(grid)
        print("\n" + rendered + "\n")
        return rendered


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info("Testing Gridworld.")

    # Test the rendering with history
    env = GridWorld(n=5, m=5, num_holes=2, num_obstacles=3)
    env.render()

    # Make some moves
    actions = [Action.RIGHT, Action.UP, Action.RIGHT, Action.DOWN]
    for action in actions:
        env.step(action)
        env.render()
