from enum import Enum
import logging
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple
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
        num_targets: int = 1,
        num_obstacles: int = 0,
        num_holes: int = 0,
        win_reward: int = 10,
        lose_reward: int = -10,
        hole_reward: int = -100000,
    ):
        self.seed = seed
        self.n = n
        self.m = m
        random.seed(seed)
        self.verbose = verbose
        self.log(f"Setting seed to {seed}.")
        self.state = None
        self.history = []
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.hole_reward = hole_reward

        self.available_states = self.get_possible_states()
        self.targets = []
        for _ in range(num_targets):
            self.targets.append(self.get_and_assign_random_state())
        self.start = self.get_and_assign_random_state()
        self.obstacles = []
        self.holes = []
        self.done = False
        for _ in range(num_obstacles):
            self.obstacles.append(self.get_and_assign_random_state())
        for _ in range(num_holes):
            self.holes.append(self.get_and_assign_random_state())
        self.reset()

    def get_possible_states(self) -> List[Location]:
        return [Location(x=x, y=y) for x in range(self.m) for y in range(self.n)]

    def get_available_states(self) -> List[Location]:
        return self.available_states

    def get_and_assign_random_state(self):
        """Assigns a random state from the available states."""
        if not self.available_states:
            raise AssertionError(
                "Could not assign a random state, implying that the space is not large enough."
            )
        state = random.choice(self.get_available_states())
        self.available_states.remove(state)
        return state

    def get_possible_actions(self) -> List[Action]:
        return (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)

    def get_state(self):
        return self.state

    def get_start(self):
        return self.start

    def set_state(self, state: Location):
        self.state = state

    def is_terminal(self, state: Location):
        terminal_states = self.targets + self.obstacles + self.holes
        return state in terminal_states

    def log(self, s: str):
        if self.verbose:
            logging.info("%s", s)

    def reset(self, random_start: bool = False) -> Location:
        self.history = []
        if random_start:
            self.state = random.choice(self.get_available_states())
        else:
            self.state = copy.copy(self.start)
        self.log(f"Starting at {self.state}")
        self.log(f"Target is {self.targets}")
        return self.state

    def step(
        self, action: Action, state: Optional[Location] = None
    ) -> Tuple[Location, int, bool]:
        if not state:
            state = self.state

        if self.is_terminal(state):
            self.log("State is terminal")
            return copy.copy(self.state), 0, True

        self.log(f"From {state}, taking action {action}.")
        x, y = state.as_tuple()
        if action == Action.UP and y > 0:
            y -= 1
        elif action == Action.DOWN and y < self.n - 1:
            y += 1
        elif action == Action.LEFT and x > 0:
            x -= 1
        elif action == Action.RIGHT and x < self.m - 1:
            x += 1
        new_state = Location(x=x, y=y)
        done = False
        self.log(f"New location: {new_state}")
        if new_state in self.targets:
            reward = self.win_reward
            done = True
        elif new_state in self.obstacles:
            reward = self.lose_reward
            new_state = self.state
        elif new_state in self.holes:
            reward = self.hole_reward
            done = True
        else:
            reward = self.lose_reward

        self.history.append((copy.copy(self.state), action, reward))
        self.log(f"Returning reward {reward}.")
        self.state = new_state
        return copy.copy(self.state), reward, done

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
                elif current_loc in self.targets:
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
