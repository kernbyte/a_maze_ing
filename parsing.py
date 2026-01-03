"""Parsing module for maze configuration files.

This module validates and parses a maze config file into a dictionary.
It checks WIDTH, HEIGHT, ENTRY, EXIT, PERFECT, SEED, and OUTPUT_FILE keys.
"""


from typing import Any

coord = tuple[int, int]


def value_supposed(key: str, value: str, line: str, line_number: int) -> None:
    """Check if a config value is valid for its key."""
    if key == "WIDTH":
        try:
            w: int = int(value)
        except ValueError:
            raise ValueError(
                f"Line {line_number}: WIDTH must be an integer\n" f"→ {line}"
            )
        if w <= 0:
            raise ValueError(
                f"Line {line_number}: WIDTH must "
                "be greater than 0\n" f"→ {line}"
            )
    elif key == "HEIGHT":
        try:
            h: int = int(value)
        except ValueError:
            raise ValueError(
                f"Line {line_number}: HEIGHT must be an integer\n" f"→ {line}"
            )
        if h <= 0:
            raise ValueError(
                f"Line {line_number}: HEIGHT must be "
                "greater than 0\n" f"→ {line}"
            )
    elif key == "ENTRY" or key == "EXIT":
        try:
            tuple(map(int, value.split(",")))
        except ValueError:
            raise ValueError(
                f"Line {line_number}: {key} must be in format"
                " x,y with integers\n"
                f"→ {line}"
            )
    elif key == "PERFECT" and (not value == "True" and not value == "False"):
        raise ValueError(
            f"Line {line_number}: PERFECT must be"
            " 'True' or 'False'\n" f"→ {line}"
        )
    elif key == "SEED":
        try:
            int(value)
        except ValueError:
            raise ValueError(
                f"Line {line_number}: SEED must be an integer\n" f"→ {line}"
            )


def parse_config(path: str) -> dict[str, Any]:
    """Parse a maze configuration file and
    return a dictionary of config values.

    Raises ValueError if any config key is missing, invalid, or out of bounds.
    """
    try:
        with open(path, "r") as file:
            lines: list[str] = file.readlines()
            keys_required = [
                "WIDTH",
                "HEIGHT",
                "ENTRY",
                "EXIT",
                "OUTPUT_FILE",
                "PERFECT",
            ]
            acceptable_keys = [
                "WIDTH",
                "HEIGHT",
                "ENTRY",
                "EXIT",
                "OUTPUT_FILE",
                "PERFECT",
                "SEED",
            ]
            config: dict[str, Any] = {}
            for line_number, line in enumerate(lines, 1):
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                if line.startswith("#"):
                    continue
                if "=" not in line:
                    raise ValueError(
                        f"Line {line_number}: Invalid syntax"
                        " (expected KEY=VALUE)\n"
                        f"→ {line}"
                    )
                key, value = line.split("=", 1)
                key = key.strip()
                if key not in acceptable_keys:
                    raise ValueError(
                        f"Line {line_number}: Unknown configuration "
                        f"key '{key}'\n"
                        f"→ {line}"
                    )
                value = value.strip()
                value_supposed(key, value, line, line_number)
                if key in ("WIDTH", "HEIGHT", "SEED"):
                    config[key] = int(value)
                elif key in ("ENTRY", "EXIT"):
                    coordinates = tuple(map(int, value.split(",")))
                    config[key] = coordinates
                else:
                    config[key] = value
            missing_keys = []
            missing_keys = [key for key in keys_required if key not in config]
            if missing_keys:
                raise ValueError("Missing required"
                                 " key(s): " + ", ".join(missing_keys))
            for point_key in ("ENTRY", "EXIT"):
                x, y = config[point_key]
                if x < 0 or x >= config["WIDTH"] \
                        or y < 0 or y >= config["HEIGHT"]:
                    raise ValueError(
                        f"{point_key} coordinates out of bounds "
                        f"(0 ≤ x < WIDTH, 0 ≤ y < HEIGHT)"
                    )
            return config

    except Exception as error:
        print(f"ERROR: {error}")
        return {}
