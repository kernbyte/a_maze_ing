*This project has been created as part of the 42 curriculum by <hchiar>[, <aminel-h>].*

## Description
A-Maze-ing is a Python maze generator that:
- Reads a `KEY=VALUE` configuration file
- Generates a random maze (optionally perfect)
- Embeds a visible “42” pattern using fully closed cells (omitted with an error message if the maze is too small)
- Writes the maze to a file using the subject hexadecimal wall encoding
- Displays the maze in the terminal using `curses` with interactive controls

The maze generation logic is reusable via the single-file module `mazegen.py`.

## Instructions
- Run: `python3 a_maze_ing.py config.txt`
- Make targets (see subject requirements):
  - `make install`
  - `make run`
  - `make debug`
  - `make clean`
  - `make lint`
  - `make lint-strict`

## Configuration
The config file is plain text with one `KEY=VALUE` per line.

Mandatory keys:
- `WIDTH`
- `HEIGHT`
- `ENTRY` (x,y) where x is column, y is row
- `EXIT` (x,y) where x is column, y is row
- `OUTPUT_FILE`
- `PERFECT`

Supported additional keys:
- `SEED` (integer) for reproducible generation

## Maze generation algorithm
This project uses a randomized depth-first search (recursive backtracker style) which produces a perfect maze (spanning tree) on the grid (excluding the fully-closed cells used to draw the visible “42” pattern).

Why this algorithm:
- Simple implementation
- Produces perfect mazes efficiently
- Easy to make deterministic via a seed

## Reusable module (`mazegen.py`)
Instantiate and use:

```python
from mazegen import MazeGenerator

gen = MazeGenerator(width=20, height=15, seed=42)
path = gen.solve_shortest_path((0, 0), (14, 19))
```

Access the structure:
- `gen.grid[row][col].walls["N"|"E"|"S"|"W"]` is `True` if the wall is closed.
- `gen.pattern_cells` contains the coordinates of the fully-closed “42” pattern cells.

Building the reusable package:
- `python3 -m pip install build`
- `python3 -m build`

## Resources
- Python `curses` module documentation
- Graph theory / BFS (shortest path)
- Maze generation: recursive backtracker
