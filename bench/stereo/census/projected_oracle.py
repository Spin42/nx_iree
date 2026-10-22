#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "typer==0.16.0", "opencv-python-headless==4.12.0.88"]
# ///
# ─── How to run ───
# After fixtures.py, run: uv run projected_oracle.py FIXTURE_DIRECTORY
# Add --disparities 128 for that fixture shape. This modifies only oracles.
# ──────────────────
"""Adapt scalar fixtures to OpenCV's projected-winner consistency pipeline."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 -- Typer resolves annotations at runtime.
from typing import TYPE_CHECKING, Final

import cv2
import numpy as np
import typer

if TYPE_CHECKING:
    from numpy.typing import NDArray

FIXTURE_HEIGHT: Final = 17
FIXTURE_EXTRA_WIDTH: Final = 9
INVALID_WINNER: Final = 0xFFFF_FFFF


def project(
    total: NDArray[np.int32], left: NDArray[np.float32]
) -> tuple[NDArray[np.uint32], NDArray[np.float32]]:
    """Scatter unique left winners, keeping largest disparity at equal cost."""
    height, width, disparities = total.shape
    winners = np.full(left.shape, INVALID_WINNER, dtype=np.uint32)
    right = np.full_like(left, -1)
    right_cost = np.full(left.shape, 2_147_483_647, dtype=np.int32)
    for y in range(height):
        for x in range(width):
            if left[y, x] < 0:
                continue
            d = int(total[y, x, : min(x + 1, disparities)].argmin())
            cost = int(total[y, x, d])
            winners[y, x] = cost * 65536 + d
            if cost <= right_cost[y, x - d]:
                right_cost[y, x - d] = cost
                right[y, x - d] = d
    return winners, right


def consistent(
    pair: tuple[NDArray[np.float32], NDArray[np.float32]],
) -> NDArray[np.float32]:
    """Reject only when both defined integer correspondences disagree."""
    left, right = pair
    height, width = left.shape
    result = np.full_like(left, -1)
    for y in range(height):
        for x in range(width):
            value = float(left[y, x])
            if value < 0:
                continue
            rejected = all(
                0 <= (rx := x - d) < width
                and right[y, rx] >= 0
                and abs(float(right[y, rx]) - d) > 1
                for d in (int(np.floor(value)), int(np.ceil(value)))
            )
            if not rejected:
                result[y, x] = value
    return result


def write_consistency_cases(data: Path, disparities: int) -> None:
    """Write native IREE CLI cases for rejection, unknown matches and ties."""
    shape = (FIXTURE_HEIGHT, disparities + FIXTURE_EXTRA_WIDTH)
    left = np.full(shape, -1, dtype=np.float32)
    total = np.full((*shape, disparities), 1_000_000, dtype=np.int32)
    records = (
        (2, 20, 4, 4.25, 40),
        (2, 23, 7, 7.0, 10),
        (2, 22, 7, 7.0, 10),
        (4, 20, 4, 4.25, 40),
        (4, 23, 7, 7.0, 10),
        (6, 20, 4, 4.25, 40),
        (6, 23, 7, 7.0, 40),
        (6, 22, 7, 7.0, 40),
        (8, 20, 4, 4.0, 40),
        (8, 23, 7, 7.0, 10),
        (12, 0, 0, 0.0, 20),
        (14, disparities + 6, disparities - 1, disparities - 0.75, 20),
    )
    for y, x, d, value, cost in records:
        left[y, x] = value
        total[y, x, d] = cost
    winners, right = project(total, left)
    checked = consistent((left, right))
    np.testing.assert_array_equal(checked[[2, 6, 8], 20], -1)
    np.testing.assert_equal(checked[4, 20], 4.25)
    np.testing.assert_equal(checked[12, 0], 0)
    for name, array in (("left", left), ("winners", winners), ("expected", checked)):
        np.save(data / f"consistency_edge_{name}.npy", array, allow_pickle=False)


def main(data: Path, disparities: int = 64) -> None:
    """Reuse existing scalar costs and left winners; write projection oracles."""
    if disparities not in (64, 128):
        raise typer.BadParameter("use disparities=64|128")
    files = sorted(data.glob("*.left_disparity.oracle.bin"))
    if not files:
        raise typer.BadParameter("generate base fixtures before projection oracles")
    shape = (FIXTURE_HEIGHT, disparities + FIXTURE_EXTRA_WIDTH)
    for path in files:
        prefix = str(path).removesuffix(".left_disparity.oracle.bin")
        left = np.fromfile(path, dtype="<f4").reshape(shape)
        paths = np.fromfile(prefix + ".paths.oracle.bin", dtype=np.uint8)
        total = paths.reshape(-1, *shape, disparities).sum(axis=0, dtype=np.int32)
        winners, right = project(total, left)
        checked = consistent((left, right))
        result = cv2.medianBlur(checked, 3)
        for name, array in (
            ("winners", winners.astype("<u4")),
            ("right_disparity", right.astype("<f4")),
            ("consistent", checked.astype("<f4")),
            ("output", result.astype("<f4")),
        ):
            _ = array.tofile(f"{prefix}.{name}.oracle.bin")
    write_consistency_cases(data, disparities)
    print(f"WROTE {len(files)} projected-winner fixture oracles to {data}")


if __name__ == "__main__":
    typer.run(main)
