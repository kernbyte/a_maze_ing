"""Reusable maze generator module.

This file is designed to be reused in future projects.

Basic usage:

    from mazegen import MazeGenerator

    gen = MazeGenerator(width=20, height=15, seed=42)
    grid = gen.grid
    path = gen.solve_shortest_path((0, 0), (14, 19))

The maze is stored as a 2D list of `Cell` instances (`grid[row][col]`). Each `Cell`
has a `walls` mapping with keys: "N", "E", "S", "W" where True means closed.
"""

from collections import deque
from dataclasses import dataclass
import random
from typing import Callable, Deque, Dict, List, Mapping, Optional, Set, Tuple


Coord = Tuple[int, int]  # (row, col)


OPPOSITE: Mapping[str, str] = {"N": "S", "S": "N", "W": "E", "E": "W"}


@dataclass
class Cell:
    """A single maze cell."""

    row: int
    col: int
    walls: Dict[str, bool]
    visited: bool

    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col
        self.walls = {"N": True, "E": True, "S": True, "W": True}
        self.visited = False

    def is_fully_closed(self) -> bool:
        """Return True if all four walls are closed."""

        return all(self.walls.values())


class MazeGenerator:
    """Generate a maze grid, optionally embedding a visible "42" pattern.

    The generated maze is perfect over all non-pattern cells (i.e., exactly one path
    between any two reachable non-pattern cells).
    """

    width: int
    height: int
    seed: int
    grid: List[List[Cell]]
    pattern_cells: Set[Coord]
    pattern_omitted_reason: Optional[str]
    _on_step: Optional[Callable[["MazeGenerator"], None]]

    def __init__(
        self,
        width: int,
        height: int,
        *,
        seed: int,
        perfect: bool = True,
        embed_42: bool = True,
        entry: Optional[Coord] = None,
        exit_: Optional[Coord] = None,
        on_step: Optional[Callable[["MazeGenerator"], None]] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.seed = seed

        self.grid = [[Cell(r, c) for c in range(width)] for r in range(height)]
        self.pattern_cells = set()
        self.pattern_omitted_reason = None
        self._on_step = on_step

        if embed_42:
            try:
                self.pattern_cells = self._place_42_pattern(entry=entry, exit_=exit_)
            except ValueError as exc:
                self.pattern_cells = set()
                self.pattern_omitted_reason = str(exc)

            for r, c in self.pattern_cells:
                self.grid[r][c].visited = True

        rng = random.Random(seed)
        self._generate_perfect_maze(rng)

        if not perfect:
            self._add_cycles(rng)

        if self._on_step is not None:
            self._on_step(self)

    def _add_cycles(self, rng: random.Random) -> None:
        """Optionally add extra openings to create multiple paths.

        This is only used when PERFECT=False.
        """

        candidates: List[Tuple[Coord, Coord]] = []
        for r in range(self.height):
            for c in range(self.width):
                if (r, c) in self.pattern_cells:
                    continue
                if r + 1 < self.height and (r + 1, c) not in self.pattern_cells:
                    if self.grid[r][c].walls["S"]:
                        candidates.append(((r, c), (r + 1, c)))
                if c + 1 < self.width and (r, c + 1) not in self.pattern_cells:
                    if self.grid[r][c].walls["E"]:
                        candidates.append(((r, c), (r, c + 1)))

        rng.shuffle(candidates)
        target = max(1, (self.width * self.height) // 25)
        opened = 0
        for a, b in candidates:
            if opened >= target:
                break
            self._carve_between(a, b)
            opened += 1

    def _neighbors_unvisited(self, r: int, c: int) -> List[Coord]:
        candidates: List[Coord] = []
        if r > 0 and not self.grid[r - 1][c].visited:
            candidates.append((r - 1, c))
        if r + 1 < self.height and not self.grid[r + 1][c].visited:
            candidates.append((r + 1, c))
        if c > 0 and not self.grid[r][c - 1].visited:
            candidates.append((r, c - 1))
        if c + 1 < self.width and not self.grid[r][c + 1].visited:
            candidates.append((r, c + 1))
        return candidates

    def _carve_between(self, a: Coord, b: Coord) -> None:
        ar, ac = a
        br, bc = b
        if br == ar - 1 and bc == ac:
            self.grid[ar][ac].walls["N"] = False
            self.grid[br][bc].walls["S"] = False
        elif br == ar + 1 and bc == ac:
            self.grid[ar][ac].walls["S"] = False
            self.grid[br][bc].walls["N"] = False
        elif br == ar and bc == ac - 1:
            self.grid[ar][ac].walls["W"] = False
            self.grid[br][bc].walls["E"] = False
        elif br == ar and bc == ac + 1:
            self.grid[ar][ac].walls["E"] = False
            self.grid[br][bc].walls["W"] = False
        else:
            raise ValueError("Cells are not adjacent")

        if self._on_step is not None:
            self._on_step(self)

    def _find_start_cell(self) -> Optional[Coord]:
        for r in range(self.height):
            for c in range(self.width):
                if not self.grid[r][c].visited:
                    return (r, c)
        return None

    def _generate_perfect_maze(self, rng: random.Random) -> None:
        start = self._find_start_cell()
        if start is None:
            return

        stack: List[Coord] = [start]
        self.grid[start[0]][start[1]].visited = True

        while stack:
            r, c = stack[-1]
            unvisited = self._neighbors_unvisited(r, c)
            if not unvisited:
                stack.pop()
                continue
            nxt = rng.choice(unvisited)
            self._carve_between((r, c), nxt)
            self.grid[nxt[0]][nxt[1]].visited = True
            stack.append(nxt)

        for r in range(self.height):
            for c in range(self.width):
                if (r, c) not in self.pattern_cells and not self.grid[r][c].visited:
                    raise RuntimeError("Maze generation produced disconnected cells")

    def _place_42_pattern(self, *, entry: Optional[Coord], exit_: Optional[Coord]) -> Set[Coord]:
        # Blocky 5x9 digits with a 1-column gap: total 11x9.
        # This matches the usual subject screenshot style.
        pattern_4 = [
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
        pattern_2 = [
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

        ph = 9
        pw = 11

        if self.width < pw or self.height < ph:
            raise ValueError("Maze too small for 42 pattern")

        center_r = (self.height - ph) // 2
        center_c = (self.width - pw) // 2

        starts: List[Coord] = []
        for r0 in range(1, self.height - ph):
            for c0 in range(1, self.width - pw):
                starts.append((r0, c0))
        starts.sort(key=lambda rc: abs(rc[0] - center_r) + abs(rc[1] - center_c))

        if not starts:
            raise ValueError("Maze too small for 42 pattern")

        forbidden: Set[Coord] = set()
        if entry is not None:
            forbidden.add(entry)
        if exit_ is not None:
            forbidden.add(exit_)

        for r0, c0 in starts:
            coords: Set[Coord] = set()
            ok = True
            for pr in range(ph):
                for pc in range(5):
                    if pattern_4[pr][pc] == 1:
                        coords.add((r0 + pr, c0 + pc))
            for pr in range(ph):
                for pc in range(5):
                    if pattern_2[pr][pc] == 1:
                        coords.add((r0 + pr, c0 + 6 + pc))

            if coords & forbidden:
                ok = False

            if ok:
                return coords

        raise ValueError("Could not place 42 pattern without overlapping entry/exit")

    def solve_shortest_path(self, entry: Coord, exit_: Coord) -> Optional[List[Coord]]:
        """Return the shortest path from entry to exit using BFS.

        Returns a list of coordinates including both endpoints, or None if no path exists.
        """

        if entry == exit_:
            return [entry]

        q: Deque[Coord] = deque([entry])
        prev: Dict[Coord, Optional[Coord]] = {entry: None}

        while q:
            r, c = q.popleft()
            if (r, c) == exit_:
                break

            cell = self.grid[r][c]
            neighbors: List[Coord] = []
            if not cell.walls["N"] and r > 0:
                neighbors.append((r - 1, c))
            if not cell.walls["S"] and r + 1 < self.height:
                neighbors.append((r + 1, c))
            if not cell.walls["W"] and c > 0:
                neighbors.append((r, c - 1))
            if not cell.walls["E"] and c + 1 < self.width:
                neighbors.append((r, c + 1))

            for nxt in neighbors:
                if nxt in prev:
                    continue
                if nxt in self.pattern_cells:
                    continue
                prev[nxt] = (r, c)
                q.append(nxt)

        if exit_ not in prev:
            return None

        out: List[Coord] = []
        cur: Optional[Coord] = exit_
        while cur is not None:
            out.append(cur)
            cur = prev[cur]
        out.reverse()
        return out
