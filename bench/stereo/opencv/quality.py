#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "numpy==2.2.6", "pillow==11.3.0", "typer==0.16.0",
#   "opencv-python-headless==4.12.0.88",
# ]
# ///
# ─── How to run ───
# Run from the repository root:
# PYTHONPATH=bench/stereo/video uv run bench/stereo/opencv/quality.py \
#   DATA RESULTS JSON_OUTPUT
# ──────────────────
"""Evaluate prepared stereo runs using the shared accuracy accumulator."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 -- Typer resolves annotations at runtime.
from typing import Final

import numpy as np
import typer
from evaluate import Accuracy, summary
from PIL import Image

MAX_DISPARITY: Final = 63


def main(data: Path, results: Path, output: Path) -> None:
    """Measure 50 outputs per run on full and common x>=64 regions."""
    report = {}
    directories = (
        [results]
        if any(results.glob("*.disparity.bin"))
        else sorted(path for path in results.iterdir() if path.is_dir())
    )
    if not directories:
        raise typer.BadParameter(f"No results in {results}")
    for directory in directories:
        mode = directory.name
        predictions = sorted(directory.glob("*.disparity.bin"))
        if len(predictions) != 50:  # noqa: PLR2004 -- This benchmark is the fixed 50-pair sequence.
            raise typer.BadParameter(f"Expected 50 predictions in {directory}")
        full: list[Accuracy] = []
        common: list[Accuracy] = []
        in_range: list[Accuracy] = []
        common_in_range: list[Accuracy] = []
        outside_search = 0
        maximum_disparity = 0.0
        for path in predictions:
            frame = path.name.removesuffix(".disparity.bin")
            with Image.open(data / f"{frame}.left.png") as image:
                shape = (image.height, image.width)
            truth = np.fromfile(data / f"{frame}.gt.bin", dtype="<f4").reshape(shape)
            prediction = np.fromfile(path, dtype="<f4").reshape(truth.shape)
            occluded = np.fromfile(
                data / f"{frame}.occlusions.bin", dtype=np.uint8
            ).reshape(truth.shape)
            outside = np.fromfile(
                data / f"{frame}.outofframe.bin", dtype=np.uint8
            ).reshape(truth.shape)
            visible = (
                (occluded == 0) & (outside == 0) & np.isfinite(truth) & (truth >= 0)
            )
            outside_search += int((visible & (truth > MAX_DISPARITY)).sum())
            maximum_disparity = max(
                maximum_disparity, float(truth[visible].max(initial=0))
            )
            full.append(Accuracy.measure(prediction, truth, visible))
            common.append(
                Accuracy.measure(prediction[:, 64:], truth[:, 64:], visible[:, 64:])
            )
            searchable = visible & (truth <= MAX_DISPARITY)
            in_range.append(Accuracy.measure(prediction, truth, searchable))
            common_in_range.append(
                Accuracy.measure(prediction[:, 64:], truth[:, 64:], searchable[:, 64:])
            )
        fps: float | None = None
        timing_file = directory / "timings_ms.txt"
        if timing_file.is_file():
            times = np.loadtxt(timing_file, ndmin=1)
            if times.size != len(predictions):
                raise typer.BadParameter("Timing count differs from prediction count")
            fps = float(1000 * times.size / times.sum())
        report[mode] = {
            "fps": fps,
            "full_visible": summary(full),
            "common_x_ge_64": summary(common),
            "visible_in_range": summary(in_range),
            "common_x_ge_64_in_range": summary(common_in_range),
            "visible_gt_outside_search_percent": 100
            * outside_search
            / sum(c.population for c in full),
            "max_visible_gt_disparity_px": maximum_disparity,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    _ = output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    typer.run(main)
