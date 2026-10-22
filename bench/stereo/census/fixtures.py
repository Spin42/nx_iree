#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "typer==0.16.0", "opencv-python-headless==4.12.0.88"]
# ///
"""Independent CPU fixtures for the Census Vulkan port, never a target runtime."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 -- Typer inspects annotations at runtime.
from typing import TYPE_CHECKING, Final

import cv2
import numpy as np
import typer
from reference import DIRECTIONS, aggregate, select

if TYPE_CHECKING:
    from numpy.typing import NDArray

FIRST_AMBIGUOUS: Final = 3


def census(image: NDArray[np.int32], *, classic: bool = False) -> NDArray[np.uint64]:
    """Build comparison bits with a scalar definition of each window."""
    height, width = image.shape
    result = np.zeros(image.shape, dtype=np.uint64)
    if classic:
        offsets = [
            (dy, dx)
            for dy in range(-3, 4)
            for dx in range(-4, 5)
            if dy != 0 and dx != 0
        ]
    else:
        offsets = [(dy, dx) for dy in range(-3, 0) for dx in range(-4, 5)]
        offsets.extend((0, dx) for dx in range(-4, 0))
    for y in range(height):
        for x in range(width):
            feature = 0
            for dy, dx in offsets:
                a = (
                    image[y, x]
                    if classic
                    else image[
                        min(max(y + dy, 0), height - 1), min(max(x + dx, 0), width - 1)
                    ]
                )
                by, bx = (y + dy, x + dx) if classic else (y - dy, x - dx)
                b = image[min(max(by, 0), height - 1), min(max(bx, 0), width - 1)]
                feature = feature * 2 + int(a > b)
            result[y, x] = feature
    return result


def outputs(
    total: NDArray[np.int32], *, uniqueness: float = 1.1
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Select both views from one volume and apply library median filtering."""
    height, width, disparities = total.shape
    valid_left = total.copy()
    for x in range(width):
        valid_left[:, x, x + 1 :] = 1_000_000
    left = select(valid_left, uniqueness=uniqueness)
    # The original scalar selector also checks an unavailable adjacent minimum.
    # Interior subpixel refinement only uses geometrically valid neighbors.
    for y in range(height):
        for x in range(min(width, disparities - 1)):
            if left[y, x] >= 0 and int(valid_left[y, x].argmin()) == x:
                left[y, x] = x
    right = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            right[y, x] = min(
                range(min(disparities, width - x)),
                key=lambda d: int(total[y, x + d, d]),
            )
    return left, right, postprocess((left, right), disparities)


def postprocess(
    pair: tuple[NDArray[np.float32], NDArray[np.float32]], disparities: int
) -> NDArray[np.float32]:
    """Use OpenCV's median implementation and independent scalar consistency."""
    left, right = pair
    height, width = left.shape
    filtered_left = cv2.medianBlur(
        np.where(left < 0, 65535, left).astype(np.float32), 3
    )
    filtered_right = cv2.medianBlur(right, 3)
    result = np.full_like(left, -1)
    for y in range(height):
        for x in range(width):
            d = float(filtered_left[y, x])
            for candidate in (int(np.floor(d)), int(np.ceil(d))):
                rx = x - candidate
                if (
                    0 <= d < disparities
                    and 0 <= rx < width
                    and abs(float(filtered_right[y, rx]) - candidate) <= 1
                ):
                    result[y, x] = d
    return result


def main(
    output: Path,
    disparities: int = 64,
    paths: int = 8,
    *,
    classic: bool = False,
    matcher: tuple[int, int, float] = (10, 120, 1.1),
) -> None:
    """Write textures, ambiguity and boundary fixtures plus all stage oracles."""
    if disparities not in (64, 128) or paths not in (3, 4, 8):
        raise typer.BadParameter("use D=64|128 and paths=3|4|8")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261022)
    left = rng.integers(1, 256, (17, disparities + 9), dtype=np.int32)
    shifted = np.concatenate((left[:, 4:], left[:, :4]), axis=1)
    pairs = (
        (left, shifted),
        (left, rng.integers(0, 256, left.shape, dtype=np.int32)),
        (np.full_like(left, 100), np.full_like(left, 100)),
        (np.zeros_like(left), np.full_like(left, 255)),
    )
    for index, pair in enumerate(pairs, 1):
        prefix = output / f"frame_{index:04d}"
        lf, rf = (census(image, classic=classic) for image in pair)
        cost = np.empty((*left.shape, disparities), dtype=np.int32)
        for d in range(disparities):
            other = np.zeros_like(rf)
            if d < left.shape[1]:
                other[:, d:] = rf[:, : left.shape[1] - d]
            cost[:, :, d] = np.bitwise_count(lf ^ other)
        directional = np.stack(
            [
                aggregate(
                    cost, penalties=(matcher[0], matcher[1]), directions=(direction,)
                )
                for direction in DIRECTIONS[:paths]
            ]
        )
        total = directional.sum(axis=0, dtype=np.int32)
        ld, rd, result = outputs(total, uniqueness=matcher[2])
        for name, array in (
            ("left", pair[0].astype("<i4")),
            ("right", pair[1].astype("<i4")),
            ("left_features.oracle", lf.astype("<u8" if classic else "<u4")),
            ("right_features.oracle", rf.astype("<u8" if classic else "<u4")),
            ("paths.oracle", directional.astype("u1")),
            ("sum.oracle", total.astype("<u2")),
            ("left_disparity.oracle", ld.astype("<f4")),
            ("right_disparity.oracle", rd.astype("<f4")),
            ("output.oracle", result.astype("<f4")),
        ):
            _ = array.tofile(f"{prefix}.{name}.bin")
        if index == 1:
            np.testing.assert_allclose(result[5:-5, 16:-12], 4, atol=0.35)
        if index >= FIRST_AMBIGUOUS:
            np.testing.assert_array_equal(result[4:-4, 12:-12], -1)
    print(f"WROTE {len(pairs)} Census fixtures to {output}")


if __name__ == "__main__":
    typer.run(main)
