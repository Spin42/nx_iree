#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "typer==0.16.0"]
# ///
# ─── How to run ───
# Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run: uv run fixtures.py OUTPUT_DIRECTORY
# ──────────────────
"""Produce small full-pipeline fixtures from an independent CPU oracle."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 -- Typer resolves annotations at runtime.
from typing import Final

import numpy as np
import typer

from reference import aggregate, consistent, costs, select

MIN_DISPARITIES: Final = 3
MAX_DISPARITIES: Final = 128
DEFAULT_DISPARITIES: Final = 33


def main(output: Path, disparities: int = DEFAULT_DISPARITIES) -> None:
    """Write shift, random, ambiguity and maximum-SAD intermediate oracles."""
    if not MIN_DISPARITIES <= disparities <= MAX_DISPARITIES:
        raise typer.BadParameter("disparities must be in 3..128")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261022)
    width = 37 if disparities <= DEFAULT_DISPARITIES else disparities + 5
    left = rng.integers(0, 256, (13, width), dtype=np.int32)
    shift = min(4, disparities - 1)
    shifted = np.concatenate(
        (left[:, shift:], rng.integers(0, 256, (13, shift), dtype=np.int32)), axis=1
    )
    pairs = (
        (left, shifted),
        (left, rng.integers(0, 256, left.shape, dtype=np.int32)),
        (np.full_like(left, 100), np.full_like(left, 100)),
        (np.zeros_like(left), np.full_like(left, 255)),
    )
    for index, (left_image, right_image) in enumerate(pairs, 1):
        prefix = output / f"frame_{index:04d}"
        left_image.astype("<i4").tofile(f"{prefix}.left.bin")
        right_image.astype("<i4").tofile(f"{prefix}.right.bin")
        left_cost = costs((left_image, right_image), disparities)
        right_cost = costs((right_image[:, ::-1], left_image[:, ::-1]), disparities)[
            :, ::-1
        ].copy()
        left_total, right_total = aggregate(left_cost), aggregate(right_cost)
        left_disparity, right_disparity = select(left_total), select(right_total)
        left_cost.astype("<i4").tofile(f"{prefix}.cost_left.bin")
        right_cost.astype("<i4").tofile(f"{prefix}.cost_right.bin")
        left_total.astype("<i4").tofile(f"{prefix}.aggregate_left.bin")
        right_total.astype("<i4").tofile(f"{prefix}.aggregate_right.bin")
        result = consistent((left_disparity, right_disparity))
        result.astype("<f4").tofile(f"{prefix}.expected.bin")
        if index == 1:
            np.testing.assert_allclose(result[2:-2, shift + 3 : -4], shift, atol=0.25)
    print(f"WROTE {len(pairs)} fixtures to {output}")


if __name__ == "__main__":
    typer.run(main)
