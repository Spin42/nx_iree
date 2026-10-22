"""Independent scalar SGM oracle for the small correctness fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

DIRECTIONS: Final = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (-1, 1),
    (1, -1),
    (-1, -1),
)


def costs(
    pair: tuple[NDArray[np.int32], NDArray[np.int32]], disparities: int
) -> NDArray[np.int32]:
    """Compute independent 3x3 SAD volumes for left and mirrored-right views."""
    left, right = pair
    height, width = left.shape
    result = np.empty((height, width, disparities), dtype=np.int32)
    for y in range(height):
        for x in range(width):
            for d in range(disparities):
                if x < d:
                    result[y, x, d] = 1_000_000
                    continue
                result[y, x, d] = sum(
                    abs(
                        int(
                            left[
                                min(max(y + dy, 0), height - 1),
                                min(max(x + dx, 0), width - 1),
                            ]
                        )
                        - int(
                            right[
                                min(max(y + dy, 0), height - 1),
                                min(max(x - d + dx, 0), width - 1),
                            ]
                        )
                    )
                    for dy in (-1, 0, 1)
                    for dx in (-1, 0, 1)
                )
    return result


def aggregate(
    cost: NDArray[np.int32],
    *,
    penalties: tuple[int, int] = (72, 288),
    directions: tuple[tuple[int, int], ...] = DIRECTIONS,
) -> NDArray[np.int32]:
    """Scan every path using the direct, unnormalized recurrence candidates."""
    height, width, disparities = cost.shape
    total = np.zeros_like(cost)
    p1, p2 = penalties
    for dx, dy in directions:
        for start_y in range(height):
            for start_x in range(width):
                if 0 <= start_x - dx < width and 0 <= start_y - dy < height:
                    continue
                x, y = start_x, start_y
                previous = np.zeros(disparities, dtype=np.int32)
                while 0 <= x < width and 0 <= y < height:
                    minimum = int(previous.min())
                    current = np.array(
                        [
                            int(cost[y, x, d])
                            + min(
                                int(previous[d]),
                                int(previous[d - 1]) + p1 if d > 0 else 2_000_000_000,
                                int(previous[d + 1]) + p1
                                if d + 1 < disparities
                                else 2_000_000_000,
                                minimum + p2,
                            )
                            - minimum
                            for d in range(disparities)
                        ],
                        dtype=np.int32,
                    )
                    total[y, x] += current
                    previous = current
                    x, y = x + dx, y + dy
    return total


def select(total: NDArray[np.int32], *, uniqueness: float = 1.1) -> NDArray[np.float32]:
    """Select unique minima, refine parabolically and preserve invalid pixels."""
    height, width, disparities = total.shape
    result = np.full((height, width), -1, dtype=np.float32)
    for y in range(height):
        for x in range(width):
            scores = total[y, x]
            winner = int(scores.argmin())
            best = int(scores[winner])
            second = min(
                (int(scores[d]) for d in range(disparities) if abs(d - winner) > 1),
                default=2_147_483_647,
            )
            if second <= np.float32(best) * np.float32(uniqueness):
                continue
            value = float(winner)
            if 0 < winner < disparities - 1:
                lower, upper = int(scores[winner - 1]), int(scores[winner + 1])
                denominator = lower + upper - 2 * best
                if denominator > 0:
                    value += min(max((lower - upper) / (2 * denominator), -0.5), 0.5)
            result[y, x] = value
    return result


def consistent(
    pair: tuple[NDArray[np.float32], NDArray[np.float32]],
) -> NDArray[np.float32]:
    """Reject pixels without a unique right-view match within one pixel."""
    left, right = pair
    height, width = left.shape
    result = np.full_like(left, -1)
    for y in range(height):
        for x in range(width):
            d = float(left[y, x])
            other_x = int(np.floor(x - d + 0.5))
            if d >= 0 and 0 <= other_x < width:
                rd = float(right[y, other_x])
                if rd >= 0 and abs(d - rd) <= 1:
                    result[y, x] = d
    return result
