import math
from typing import List, Tuple, Set

Coord = Tuple[int, int]


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def in_bounds(pos: Coord, size: int) -> bool:
    x, y = pos
    return 0 <= x < size and 0 <= y < size


def neighbors4(pos: Coord, size: int):
    x, y = pos
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if in_bounds((nx, ny), size):
            yield (nx, ny)


def to_index(pos: Coord, size: int) -> int:
    return pos[0] * size + pos[1]


def from_index(idx: int, size: int) -> Coord:
    return (idx // size, idx % size)


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def fov_cells(center: Coord, radius: int, size: int) -> List[Coord]:
    cx, cy = center
    cells = []
    for x in range(cx - radius, cx + radius + 1):
        for y in range(cy - radius, cy + radius + 1):
            if in_bounds((x, y), size) and manhattan((cx, cy), (x, y)) <= radius:
                cells.append((x, y))
    return cells
