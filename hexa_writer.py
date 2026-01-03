"""Hexadecimal writer for maze grids.

Converts a maze grid of Cell objects into hexadecimal representation
and writes it to an output file.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mazegen import Cell


def convert_to_hex(grid: list[list[Cell]], output_file: str) -> None:
    """Convert a maze grid to hexadecimal and save it to a file.

    Each cell's walls are encoded as a 4-bit value:
        N=1, E=2, S=4, W=8
    The sum is converted to a single hex character per cell.

    Args:
        grid: 2D list of Cell objects representing the maze.
        output_file: Path to the file where the hex maze will be saved.
    """
    output = []
    for r in range(len(grid)):
        row = []
        for c in range(len(grid[0])):
            value = 0
            if grid[r][c].walls["N"]:
                value += 1
            if grid[r][c].walls["E"]:
                value += 2
            if grid[r][c].walls["S"]:
                value += 4
            if grid[r][c].walls["W"]:
                value += 8
            row.append(f"{value:X}")
        output.append("".join(row))
    try:
        with open(output_file, "w") as file:
            for line in output:
                file.write(line + "\n")
    except Exception as error:
        print(error)
