from collections import deque
import curses
import sys
from dataclasses import dataclass
from pathlib import Path
from mazegen import MazeGenerator
from typing import (
    Any,
    Deque,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    cast,
)


Coord = Tuple[int, int]  # (row, col)


WALL_CHAR = "█"


@dataclass(frozen=True)
class Config:
    """Parsed configuration for maze generation."""

    width: int
    height: int
    entry: Coord
    exit: Coord
    output_file: Path
    perfect: bool
    seed: int


class ConfigError(ValueError):
    """Configuration and validation error."""

    pass


def parse_bool(value: str) -> bool:
    """Parse a boolean from a config value."""

    v = value.strip().lower()
    if v in {"true", "1", "yes", "y", "on"}:
        return True
    if v in {"false", "0", "no", "n", "off"}:
        return False
    raise ConfigError(f"Invalid boolean: {value!r}")


def parse_int(value: str, *, key: str) -> int:
    """Parse an integer from a config value."""

    try:
        return int(value.strip())
    except ValueError as exc:
        msg = f"Invalid integer for {key}: {value!r}"
        raise ConfigError(msg) from exc


def parse_coord(value: str, *, key: str) -> Coord:
    """Parse coordinates as (x,y) and return internal (row,col).

    The subject uses coordinates (x,y) where x is a column (0..WIDTH-1)
    and y is a row (0..HEIGHT-1). Internally, this project uses (row,col).
    """

    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise ConfigError(
            f"Invalid coordinate for {key}: {value!r} (expected 'x,y')"
        )
    x = parse_int(parts[0], key=key)
    y = parse_int(parts[1], key=key)
    return (y, x)


def read_config(path: Path) -> Config:
    """Read and validate the configuration file."""

    raw: Dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" not in stripped:
                    raise ConfigError(
                        f"Bad syntax at\n"
                        f"line {line_no}: {line!r} (expected KEY=VALUE)"
                    )
                k, v = stripped.split("=", 1)
                key = k.strip().upper()
                raw[key] = v.strip()
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read"
                          f" config file: {path}: {exc}") from exc

    required = {"WIDTH", "HEIGHT", "ENTRY", "EXIT", "OUTPUT_FILE", "PERFECT"}
    missing = sorted(required - set(raw.keys()))
    if missing:
        raise ConfigError(
            f"Missing required config keys: {', '.join(missing)}"
        )

    width = parse_int(raw["WIDTH"], key="WIDTH")
    height = parse_int(raw["HEIGHT"], key="HEIGHT")
    entry = parse_coord(raw["ENTRY"], key="ENTRY")
    exit_ = parse_coord(raw["EXIT"], key="EXIT")
    output_file = Path(raw["OUTPUT_FILE"]).expanduser()
    perfect = parse_bool(raw["PERFECT"])

    seed = parse_int(raw.get("SEED", "0"), key="SEED")

    if width <= 0 or height <= 0:
        raise ConfigError("WIDTH and HEIGHT must be > 0")

    if entry == exit_:
        raise ConfigError("ENTRY and EXIT must be different")

    ex, ey = entry
    gx, gy = exit_
    if not (0 <= ex < height and 0 <= ey < width):
        raise ConfigError("ENTRY is outside maze bounds")
    if not (0 <= gx < height and 0 <= gy < width):
        raise ConfigError("EXIT is outside maze bounds")

    return Config(
        width=width,
        height=height,
        entry=entry,
        exit=exit_,
        output_file=output_file,
        perfect=perfect,
        seed=seed,
    )


class _HasWalls(Protocol):
    """Structural type for cells that expose a `walls` mapping."""

    walls: Mapping[str, bool]


Grid = Sequence[Sequence[_HasWalls]]


class _CursesWindow(Protocol):
    def getmaxyx(self) -> Tuple[int, int]:
        ...

    def clear(self) -> None:
        ...

    def refresh(self) -> None:
        ...

    def addstr(self, y: int, x: int, s: str) -> Any:
        ...

    def addch(self, y: int, x: int, ch: str, attr: int = 0) -> Any:
        ...

    def bkgd(self, ch: str, attr: int = 0) -> Any:
        ...

    def getch(self) -> int:
        ...


def validate_maze(
    grid: Grid,
    *,
    entry: Coord,
    exit_: Coord,
    pattern: Optional[Set[Coord]] = None,
    perfect: bool,
) -> None:
    """Validate the maze structure against subject rules.

    Checks:
    - Border walls remain closed to the outside.
    - Neighboring cells have coherent walls.
    - Full connectivity for non-pattern cells.
    - If perfect=True: exactly one unique path between entry and exit.
    """

    if not grid or not grid[0]:
        raise RuntimeError("Invalid maze: empty grid")

    h = len(grid)
    w = len(grid[0])
    blocked: Set[Coord] = set(pattern or set())

    if entry in blocked or exit_ in blocked:
        raise RuntimeError("Invalid maze: ENTRY/EXIT overlaps the 42 pattern")

    # Border walls must remain closed.
    for r in range(h):
        if not grid[r][0].walls["W"]:
            raise RuntimeError("Invalid maze: west border has an opening")
        if not grid[r][w - 1].walls["E"]:
            raise RuntimeError("Invalid maze: east border has an opening")
    for c in range(w):
        if not grid[0][c].walls["N"]:
            raise RuntimeError("Invalid maze: north border has an opening")
        if not grid[h - 1][c].walls["S"]:
            raise RuntimeError("Invalid maze: south border has an opening")

    # Wall coherence.
    for r in range(h):
        for c in range(w):
            cell = grid[r][c]
            if r + 1 < h:
                south = grid[r + 1][c]
                if cell.walls["S"] != south.walls["N"]:
                    raise RuntimeError("Invalid maze: incoherent S/N walls")
            if c + 1 < w:
                east = grid[r][c + 1]
                if cell.walls["E"] != east.walls["W"]:
                    raise RuntimeError("Invalid maze: incoherent E/W walls")

    # Connectivity: all non-pattern cells must be reachable from entry.
    if solve_bfs(grid, entry, exit_) is None:
        raise RuntimeError("Invalid maze: no path from ENTRY to EXIT")

    reachable: Set[Coord] = {entry}
    q: Deque[Coord] = deque([entry])
    while q:
        cur = q.popleft()
        for nxt in neighbors_open(grid, cur[0], cur[1]):
            if nxt in blocked or nxt in reachable:
                continue
            reachable.add(nxt)
            q.append(nxt)

    total_open = 0
    for r in range(h):
        for c in range(w):
            if (r, c) in blocked:
                continue
            total_open += 1

    if len(reachable) != total_open:
        raise RuntimeError("Invalid maze: disconnected cells exist")

    # No fully open 3x3 areas.
    # Corridors can't be wider than 2 cells.
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            # If the 42 pattern intersects the block, it cannot be
            # a fully open room.
            skip = False
            for rr in range(r0, r0 + 3):
                for cc in range(c0, c0 + 3):
                    if (rr, cc) in blocked:
                        skip = True
                        break
                if skip:
                    break
            if skip:
                continue

            open_room = True
            for rr in range(r0, r0 + 3):
                for cc in range(c0, c0 + 2):
                    if grid[rr][cc].walls["E"]:
                        open_room = False
                        break
                if not open_room:
                    break
            if open_room:
                for rr in range(r0, r0 + 2):
                    for cc in range(c0, c0 + 3):
                        if grid[rr][cc].walls["S"]:
                            open_room = False
                            break
                    if not open_room:
                        break
            if open_room:
                raise RuntimeError("Invalid maze: contains a 3x3 open area")

    if perfect:
        # A perfect maze over reachable cells is a tree: edges == nodes - 1.
        edges = 0
        for r in range(h):
            for c in range(w):
                if (r, c) in blocked:
                    continue
                cell = grid[r][c]
                if (
                    not cell.walls["E"]
                    and c + 1 < w
                    and (r, c + 1) not in blocked
                ):
                    edges += 1
                if (
                    not cell.walls["S"]
                    and r + 1 < h
                    and (r + 1, c) not in blocked
                ):
                    edges += 1
        if edges != total_open - 1:
            raise RuntimeError(
                "Invalid maze: PERFECT=True but "
                "maze contains loops"
            )


def neighbors_open(grid: Grid, x: int, y: int) -> Iterable[Coord]:
    """Yield neighboring coordinates reachable from (row,col).

    Uses open walls only.
    """
    cell = grid[x][y]

    if not cell.walls["N"] and x - 1 >= 0:
        yield (x - 1, y)
    if not cell.walls["S"] and x + 1 < len(grid):
        yield (x + 1, y)
    if not cell.walls["W"] and y - 1 >= 0:
        yield (x, y - 1)
    if not cell.walls["E"] and y + 1 < len(grid[0]):
        yield (x, y + 1)


def solve_bfs(grid: Grid, start: Coord, goal: Coord) -> Optional[List[Coord]]:
    """Compute a shortest path using BFS (returns list of coords or None)."""
    if start == goal:
        return [start]

    q: Deque[Coord] = deque([start])
    came_from: Dict[Coord, Optional[Coord]] = {start: None}

    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nxt in neighbors_open(grid, cur[0], cur[1]):
            if nxt in came_from:
                continue
            came_from[nxt] = cur
            q.append(nxt)

    if goal not in came_from:
        return None

    path: List[Coord] = [goal]
    while path[-1] != start:
        prev = came_from[path[-1]]
        if prev is None:
            return None
        path.append(prev)
    path.reverse()
    return path


def path_to_directions(path: Sequence[Coord]) -> str:
    """Convert a coordinate path into N/E/S/W directions."""

    if len(path) < 2:
        return ""

    out: List[str] = []
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        dx = x1 - x0
        dy = y1 - y0
        if dx == -1 and dy == 0:
            out.append("N")
        elif dx == 1 and dy == 0:
            out.append("S")
        elif dx == 0 and dy == 1:
            out.append("E")
        elif dx == 0 and dy == -1:
            out.append("W")
        else:
            raise ValueError("Non-adjacent steps in path")
    return "".join(out)


def cell_to_hex(cell: _HasWalls) -> str:
    """Encode a cell's closed walls as a single hexadecimal digit."""

    val = 0
    if cell.walls["N"]:
        val |= 1
    if cell.walls["E"]:
        val |= 2
    if cell.walls["S"]:
        val |= 4
    if cell.walls["W"]:
        val |= 8
    return format(val, "X")


def write_output_file(
    path: Path,
    grid: Grid,
    entry: Coord,
    exit_: Coord,
    directions: str,
) -> None:
    """Write the maze output file following the subject format."""

    lines: List[str] = []
    for row in grid:
        lines.append("".join(cell_to_hex(c) for c in row))

    lines.append("")
    # Subject format expects (x,y) == (col,row)
    lines.append(f"{entry[1]},{entry[0]}")
    lines.append(f"{exit_[1]},{exit_[0]}")
    lines.append(directions)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def render_block(
    grid: Grid,
    *,
    marks: Optional[Dict[Coord, str]] = None,
    pattern: Optional[Set[Coord]] = None,
) -> List[str]:
    """Render the maze as block characters for terminal/curses display."""

    marks = marks or {}
    h = len(grid)
    w = len(grid[0]) if h else 0

    out_h = 2 * h + 1
    out_w = 2 * w + 1
    canvas: List[List[str]] = [
        [WALL_CHAR for _ in range(out_w)]
        for _ in range(out_h)
    ]

    def set_at(r: int, c: int, ch: str) -> None:
        if 0 <= r < out_h and 0 <= c < out_w:
            canvas[r][c] = ch

    for x in range(h):
        for y in range(w):
            cx = 2 * x + 1
            cy = 2 * y + 1

            # cell center
            if pattern is not None and (x, y) in pattern:
                set_at(cx, cy, WALL_CHAR)
            else:
                set_at(cx, cy, " ")

            cell = grid[x][y]
            # carve passages
            if not cell.walls["N"]:
                set_at(cx - 1, cy, " ")
            if not cell.walls["S"]:
                set_at(cx + 1, cy, " ")
            if not cell.walls["W"]:
                set_at(cx, cy - 1, " ")
            if not cell.walls["E"]:
                set_at(cx, cy + 1, " ")

            # overlay marks at center
            m = marks.get((x, y))
            if m is not None and len(m) == 1:
                set_at(cx, cy, m)

    return ["".join(row) for row in canvas]


def curses_view(
    stdscr: _CursesWindow,
    *,
    config: Config,
) -> None:
    """Interactive terminal (curses) view for the maze."""

    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()

    wall_colors = [
        curses.COLOR_WHITE,
        curses.COLOR_YELLOW,
        curses.COLOR_GREEN,
        curses.COLOR_CYAN,
        curses.COLOR_MAGENTA,
    ]
    wall_color_idx: int = 0

    show_path = False
    seed: int = config.seed
    last_error: Optional[str] = None

    def safe_draw_maze(
        *,
        maze: MazeGenerator,
        wall_attr: int,
        pattern_attr: int,
        bg_attr: int,
    ) -> None:
        lines = render_block(
            cast(Grid, maze.grid),
            pattern=getattr(maze, "pattern_cells", None),
        )
        max_y, max_x = stdscr.getmaxyx()
        needed_y = len(lines)
        needed_x = max((len(line) for line in lines), default=0)

        if needed_y > max_y or needed_x > max_x:
            return

        try:
            stdscr.bkgd(" ", bg_attr)
        except curses.error:
            pass

        pattern_tiles: Set[Tuple[int, int]] = set()
        pattern_cells = getattr(maze, "pattern_cells", None)
        if pattern_cells is not None:
            for cell in pattern_cells:
                pattern_tiles.add((2 * cell[0] + 1, 2 * cell[1] + 1))

        for i, line in enumerate(lines):
            for j, ch_char in enumerate(line):
                if i >= max_y or j >= max_x:
                    continue
                try:
                    if (i, j) in pattern_tiles:
                        stdscr.addch(i, j, " ", pattern_attr)
                    elif ch_char == WALL_CHAR:
                        stdscr.addch(i, j, ch_char, wall_attr)
                    else:
                        stdscr.addch(i, j, ch_char)
                except curses.error:
                    continue

        stdscr.refresh()

    def init_pairs() -> Tuple[int, int, int, int]:
        wall_color_idx
        wall_foreground = wall_colors[wall_color_idx % len(wall_colors)]
        try:
            # 1: walls
            curses.init_pair(1, wall_foreground, curses.COLOR_BLACK)
            # 2: path
            curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_CYAN)
            # 3: entry
            curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_MAGENTA)
            # 4: exit
            curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_RED)
            pattern_bg = curses.COLOR_WHITE
            if getattr(curses, "COLORS", 0) >= 16:
                # In most 16/256-color terminals, color 8 is
                # a gray (bright black).
                pattern_bg = 8
            # 5: 42 pattern
            curses.init_pair(5, curses.COLOR_BLACK, pattern_bg)
            # 6: background/text
            curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLACK)
        except curses.error:
            pass
        return (
            curses.color_pair(1),
            curses.color_pair(2),
            curses.color_pair(3),
            curses.color_pair(4),
        )

    def generate_new_maze() -> Tuple[
        MazeGenerator,
        Optional[List[Coord]],
        str,
    ]:
        nonlocal last_error
        last_error = None
        cell_count = config.width * config.height
        draw_every = max(1, cell_count // 250)
        delay_ms = 5 if cell_count <= 900 else 1
        step_count = 0

        if curses.has_colors():
            wall_attr, _path_attr, _entry_attr, _exit_attr = init_pairs()
            pattern_attr = curses.color_pair(5)
            bg_attr = curses.color_pair(6)
        else:
            wall_attr = 0
            pattern_attr = 0
            bg_attr = 0

        def on_step(m: MazeGenerator) -> None:
            nonlocal step_count
            step_count += 1
            if step_count % draw_every != 0:
                return
            safe_draw_maze(
                maze=m,
                wall_attr=wall_attr,
                pattern_attr=pattern_attr,
                bg_attr=bg_attr,
            )
            try:
                curses.napms(delay_ms)
            except curses.error:
                return

        try:
            maze: MazeGenerator = MazeGenerator(
                config.width,
                config.height,
                seed=seed,
                perfect=config.perfect,
                embed_42=True,
                entry=config.entry,
                exit_=config.exit,
                on_step=on_step,
            )
            if getattr(maze, "pattern_omitted_reason", None) is not None:
                print(f"Error: {maze.pattern_omitted_reason}", file=sys.stderr)
            grid = cast(Grid, maze.grid)
            validate_maze(
                grid,
                entry=config.entry,
                exit_=config.exit,
                pattern=getattr(maze, "pattern_cells", None),
                perfect=config.perfect,
            )
            path = solve_bfs(grid, config.entry, config.exit)
            if path is None:
                directions = ""
            else:
                directions = path_to_directions(path)
            write_output_file(
                config.output_file,
                grid,
                config.entry,
                config.exit,
                directions,
            )
            return maze, path, directions
        except Exception as exc:
            # Keep the UI alive on regeneration; show a message instead.
            last_error = f"{type(exc).__name__}: {exc}"
            raise

    try:
        maze, path, _directions = generate_new_maze()
    except Exception as exc:
        stdscr.clear()
        max_y, max_x = stdscr.getmaxyx()
        msg = f"Error: {type(exc).__name__}: {exc}"
        try:
            stdscr.addstr(0, 0, msg[: max(0, max_x - 1)])
            stdscr.addstr(2, 0, "Press any key to quit."[: max(0, max_x - 1)])
        except curses.error:
            pass
        stdscr.refresh()
        stdscr.getch()
        return

    def to_canvas(c: Coord) -> Tuple[int, int]:
        return (2 * c[0] + 1, 2 * c[1] + 1)

    entry_rc = to_canvas(config.entry)
    exit_rc = to_canvas(config.exit)

    while True:
        pattern_tiles: Set[Tuple[int, int]] = set()
        pattern_cells = getattr(maze, "pattern_cells", None)
        if pattern_cells is not None:
            for cell in pattern_cells:
                pattern_tiles.add(to_canvas(cell))

        path_tiles: Set[Tuple[int, int]] = set()
        if show_path and path is not None:
            for a, b in zip(path, path[1:]):
                ar, ac = to_canvas(a)
                br, bc = to_canvas(b)
                path_tiles.add((ar, ac))
                path_tiles.add((br, bc))
                path_tiles.add(((ar + br) // 2, (ac + bc) // 2))

        lines = render_block(
            cast(Grid, maze.grid),
            pattern=getattr(maze, "pattern_cells", None),
        )
        max_y, max_x = stdscr.getmaxyx()

        menu_lines = 6
        needed_y = len(lines) + menu_lines
        needed_x = max((len(line) for line in lines), default=0)

        stdscr.clear()

        def safe_addstr(y: int, x: int, s: str) -> None:
            if y < 0 or x < 0:
                return
            if y >= max_y or x >= max_x:
                return
            try:
                stdscr.addstr(y, x, s[: max(0, max_x - x - 1)])
            except curses.error:
                return

        def safe_addch(y: int, x: int, ch: str, attr: int = 0) -> None:
            if y < 0 or x < 0:
                return
            if y >= max_y or x >= max_x:
                return
            try:
                stdscr.addch(y, x, ch, attr)
            except curses.error:
                return

        if max_y < needed_y or max_x < needed_x:
            safe_addstr(0, 0, "Terminal too small.")
            need_msg = (
                f"Need {needed_x}x{needed_y}, have {max_x}x{max_y}."
            )
            safe_addstr(1, 0, need_msg)
            safe_addstr(3, 0, "Resize or press 3 to quit.")
            stdscr.refresh()
            keycode = stdscr.getch()
            if keycode == ord("3"):
                return
            continue

        if curses.has_colors():
            wall_attr, path_attr, entry_attr, exit_attr = init_pairs()
            pattern_attr = curses.color_pair(5)
            bg_attr = curses.color_pair(6)
        else:
            wall_attr, path_attr, entry_attr, exit_attr = (0, 0, 0, 0)
            pattern_attr = 0
            bg_attr = 0

        try:
            stdscr.bkgd(" ", bg_attr)
        except curses.error:
            pass

        # Draw maze
        for i, line in enumerate(lines):
            for j, ch_char in enumerate(line):
                if (i, j) in pattern_tiles:
                    safe_addch(i, j, " ", pattern_attr)
                    continue

                if ch_char == WALL_CHAR:
                    safe_addch(i, j, ch_char, wall_attr)
                else:
                    if (i, j) == entry_rc:
                        safe_addch(i, j, " ", entry_attr)
                    elif (i, j) == exit_rc:
                        safe_addch(i, j, " ", exit_attr)
                    elif (i, j) in path_tiles:
                        safe_addch(i, j, " ", path_attr)
                    else:
                        safe_addch(i, j, ch_char)
        # Draw menu
        base = len(lines)
        safe_addstr(base + 0, 0, "=== A-Maze-ing ===")
        safe_addstr(base + 1, 0, "1. Re-generate a new maze")
        safe_addstr(base + 2, 0, "2. Show/hide path from entry to exit")
        safe_addstr(base + 3, 0, "3. Rotate maze colors")
        safe_addstr(base + 4, 0, "4. Quit")
        if last_error is not None:
            safe_addstr(base + 5, 0, f"Last error: {last_error}")
            safe_addstr(base + 6, 0, "Choice? (1-4): ")
        else:
            safe_addstr(base + 5, 0, "Choice? (1-4): ")
        stdscr.refresh()

        key = stdscr.getch()
        if key == ord("1"):
            seed += 1
            try:
                maze, path, _directions = generate_new_maze()
            except Exception:
                # Error message already stored in last_error.
                continue
        elif key == ord("2"):
            show_path = not show_path
        elif key == ord("4"):
            return
        elif key == ord("3"):
            wall_color_idx = (wall_color_idx + 1) % len(wall_colors)


def run(config: Config) -> int:
    """Generate the maze, write the output file,
    then show the interactive view."""

    maze: MazeGenerator = MazeGenerator(
        config.width,
        config.height,
        seed=config.seed,
        perfect=config.perfect,
        embed_42=True,
        entry=config.entry,
        exit_=config.exit,
    )
    if getattr(maze, "pattern_omitted_reason", None) is not None:
        print(f"Error: {maze.pattern_omitted_reason}", file=sys.stderr)

    grid = cast(Grid, maze.grid)
    validate_maze(
        grid,
        entry=config.entry,
        exit_=config.exit,
        pattern=getattr(maze, "pattern_cells", None),
        perfect=config.perfect,
    )
    path = solve_bfs(grid, config.entry, config.exit)
    if path is None:
        raise RuntimeError("No valid path from ENTRY to EXIT")

    directions = path_to_directions(path)
    write_output_file(
        config.output_file,
        grid,
        config.entry,
        config.exit,
        directions,
    )

    # Display the maze in curses.
    try:
        curses.wrapper(curses_view, config=config)
    except curses.error as exc:
        print(f"Error: curses: {exc}", file=sys.stderr)
        # Output file has already been written; still exit cleanly.
        return 0
    return 0


def main(argv: Sequence[str]) -> int:
    """CLI entrypoint."""

    if len(argv) != 2:
        print("Usage: python3 a_maze_ing.py config.txt", file=sys.stderr)
        return 2

    try:
        config = read_config(Path(argv[1]))
        return run(config)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (ConfigError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Avoid tracebacks during review; keep output concise.
        print(f"Unexpected error: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
