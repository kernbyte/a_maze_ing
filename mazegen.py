"""Maze generator module.

Generates a maze using recursive backtracking and can place a '42' pattern.
Exports the maze in hexadecimal format via hexa_writer.convert_to_hex.
"""


import random
import sys
from typing import Optional

from hexa_writer import convert_to_hex

coord = tuple[int, int]


class Cell:
    """Represents a single cell in the maze."""

    def __init__(self, x: int, y: int) -> None:
        """Initialize a cell at coordinates (x, y) with all walls intact."""
        self.x: int = x
        self.y: int = y
        self.walls: dict[str, bool] = {"N": True, "E": True,
                                       "W": True, "S": True}
        self.visited: bool = False


class MazeGenerator:
    """Generates a maze and optionally places a '42' pattern."""

    def __init__(
        self,
        width: int,
        height: int,
        entry: Optional[coord],
        exit_: Optional[coord],
        seed: Optional[int],
        perfect: bool,
    ) -> None:
        """Initialize the maze generator and build the maze."""
        self.count = 0
        self.width = width
        self.height = height
        self.entry = entry
        self.exit_ = exit_
        self.grid: list[list[Cell]] = []
        self.seed = seed
        self.perfect = perfect
        self.pattern_cells: set[coord] = set()
        sys.setrecursionlimit(150000000)
        for x in range(self.height):
            row = []
            for y in range(self.width):
                cell_object = Cell(x, y)
                row.append(cell_object)
            self.grid.append(row)
        if seed is not None:
            random.seed(seed)
        try:
            self.pattern_cells = self.place_42_pattern(entry, exit_)
            for r, c in self.pattern_cells:
                self.grid[r][c].visited = True
        except Exception as error:
            print(error)
        for r in range(self.height):
            for c in range(self.width):
                if not self.grid[r][c].visited:
                    self.maze_gen(r, c)
                    break
            else:
                continue
            break

        self.maze_gen(0, 0)
        convert_to_hex(self.grid, "maze.txt")

    def the_wall(self, x: int, y: int, x1: int, y1: int) -> str:
        """Return the wall direction between two adjacent cells."""
        if x1 > x:
            return "S"
        if x1 < x:
            return "N"
        if y1 > y:
            return "E"
        if y1 < y:
            return "W"
        raise ValueError(f"No valid wall between ({x},{y}) and ({x1},{y1})")

    def neighbors_check(self, x: int, y: int, neighbors: list[coord]) -> None:
        """Append unvisited neighbors of cell (x, y) to the neighbors list."""
        if y + 1 < self.width and not self.grid[x][y + 1].visited:
            neighbors.append((x, y + 1))
        if y - 1 >= 0 and not self.grid[x][y - 1].visited:
            neighbors.append((x, y - 1))
        if x + 1 < self.height and not self.grid[x + 1][y].visited:
            neighbors.append((x + 1, y))
        if x - 1 >= 0 and not self.grid[x - 1][y].visited:
            neighbors.append((x - 1, y))

    def maze_gen(self, x: int, y: int) -> None:
        """Recursive backtracking maze generation starting from cell (x, y)."""
        if self.grid[x][y].visited:
            return
        self.grid[x][y].visited = True
        neighbors: list[coord] = []
        self.neighbors_check(x, y, neighbors)
        neighbors_count = len(neighbors)
        i = 0
        while i < neighbors_count:
            neighbors = []
            self.neighbors_check(x, y, neighbors)
            neighbors_count = len(neighbors)
            if neighbors:
                chosen = random.choice(neighbors)
                x1, y1 = chosen
                wall = self.the_wall(x, y, x1, y1)
                opposite = {"N": "S", "S": "N", "W": "E", "E": "W"}
                self.grid[x][y].walls[wall] = False
                self.grid[x1][y1].walls[opposite[wall]] = False
                self.maze_gen(x1, y1)
            i += 1

    def place_42_pattern(
        self, entry: Optional[coord], exit_: Optional[coord]
    ) -> set[coord]:
        """Place the '42' pattern in the maze, avoiding entry and exit."""
        pattern_4: list[list[int]] = [
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
        ]
        pattern_2: list[list[int]] = [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ]
        pattern_height = 9
        pattern_width = 11

        if self.height <= pattern_height or self.width <= pattern_width:
            raise ValueError("the maze too small for pattern")
        center_r = (self.height - pattern_height) // 2
        center_c = (self.width - pattern_width) // 2
        starts: list[coord] = []
        for start_row in range(1, self.height - pattern_height):
            for start_col in range(1, self.width - pattern_width):
                starts.append((start_row, start_col))

        starts_with_dist: list[tuple[int, int, int]] = []
        for start_row, start_col in starts:
            dist = abs(start_row - center_r) + abs(start_col - center_c)
            starts_with_dist.append((start_row, start_col, dist))

        starts_with_dist.sort(key=lambda x: x[2])
        sorted_starts = [
            (start_r, start_c) for start_r, start_c, dist in starts_with_dist
        ]
        start_row = sorted_starts[0][0]
        start_col = sorted_starts[0][1]
        forbidden: set[coord] = set()
        if entry is not None:
            forbidden.add(entry)
        if exit_ is not None:
            forbidden.add(exit_)
        for start_row, start_col, _ in starts_with_dist:
            coords: set[coord] = set()
            for r in range(pattern_height):
                for c in range(5):
                    if pattern_4[r][c] == 1:
                        coords.add((r + start_row, c + start_col))
            for r in range(pattern_height):
                for c in range(5):
                    if pattern_2[r][c] == 1:
                        coords.add((r + start_row, c + 6 + start_col))
            if not coords & forbidden:
                return coords
        else:
            raise ValueError("42 pattern skipped: entry/exit conflict")
