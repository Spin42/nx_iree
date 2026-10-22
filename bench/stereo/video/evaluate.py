#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "numpy==2.2.6", "pillow==11.3.0", "typer==0.16.0",
#   "opencv-python-headless==4.12.0.88",
# ]
# ///
# ─── How to run ───
# Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run: uv run evaluate.py DATA_DIRECTORY RESULT_DIRECTORY REPORT_DIRECTORY
# ──────────────────
"""Evaluate visible-pixel accuracy and render a stereo sequence contact sheet."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 -- Typer resolves annotations at runtime.
from typing import TYPE_CHECKING, Final, TypedDict

import cv2
import numpy as np
import typer
from PIL import Image

from preview import color, save_panel

if TYPE_CHECKING:
    from numpy.typing import NDArray

MAX_DISPARITY: Final = 63
COARSE_ERROR: Final = 3


class AccuracySummary(TypedDict):
    """Serializable accuracy measurements in pixels and percentages."""

    evaluated_pixels: float
    coverage_percent: float
    mae_accepted_px: float
    bad1_accepted_percent: float
    bad3_accepted_percent: float
    bad1_or_invalid_percent: float
    bad3_or_invalid_percent: float


@dataclass(frozen=True, slots=True)
class Accuracy:
    """Pixel-weighted counts; invalid output counts as failure, never zero error."""

    population: int
    accepted: int
    absolute_error: float
    bad_one: int
    bad_three: int

    @classmethod
    def measure(
        cls,
        prediction: NDArray[np.float32],
        truth: NDArray[np.float32],
        mask: NDArray[np.bool_],
    ) -> Accuracy:
        """Measure accepted accuracy separately from coverage of the population."""
        accepted = mask & np.isfinite(prediction) & (prediction >= 0)
        error = np.abs(prediction[accepted] - truth[accepted])
        return cls(
            int(mask.sum()),
            int(accepted.sum()),
            float(error.sum(dtype=np.float64)),
            int((error > 1).sum()),
            int((error > COARSE_ERROR).sum()),
        )


def summary(counts: list[Accuracy]) -> AccuracySummary:
    """Reduce counts across all pixels, keeping rejection visible in the report."""
    population = sum(c.population for c in counts)
    accepted = sum(c.accepted for c in counts)
    rejected = population - accepted
    return {
        "evaluated_pixels": float(population),
        "coverage_percent": 100 * accepted / population,
        "mae_accepted_px": sum(c.absolute_error for c in counts) / accepted,
        "bad1_accepted_percent": 100 * sum(c.bad_one for c in counts) / accepted,
        "bad3_accepted_percent": 100 * sum(c.bad_three for c in counts) / accepted,
        "bad1_or_invalid_percent": 100
        * (sum(c.bad_one for c in counts) + rejected)
        / population,
        "bad3_or_invalid_percent": 100
        * (sum(c.bad_three for c in counts) + rejected)
        / population,
    }


def main(data: Path, results: Path, report: Path) -> None:
    """Compare Vulkan and an OpenCV quality baseline on the same prepared pairs."""
    report.mkdir(parents=True, exist_ok=True)
    panels = report / "preview_frames"
    panels.mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    baseline = cv2.StereoSGBM.create(
        minDisparity=0,
        numDisparities=64,
        blockSize=3,
        P1=72,
        P2=288,
        disp12MaxDiff=1,
        preFilterCap=31,
        uniquenessRatio=10,
        speckleWindowSize=0,
        speckleRange=0,
        mode=cv2.STEREO_SGBM_MODE_HH,
    )
    gpu_scores: list[Accuracy] = []
    cpu_scores: list[Accuracy] = []
    common_gpu_scores: list[Accuracy] = []
    common_cpu_scores: list[Accuracy] = []
    in_range_scores: list[Accuracy] = []
    covered_pixels = 0
    outside_range = 0
    maximum_disparity = 0.0
    files = sorted(data.glob("*.left.png"))
    if not files:
        raise typer.BadParameter(f"No prepared stereo frames in {data}")
    for index, filename in enumerate(files, 1):
        frame = filename.name.removesuffix(".left.png")
        with Image.open(filename) as source:
            left = source.convert("RGB")
        with Image.open(data / f"{frame}.right.png") as source:
            right = source.convert("RGB")
        width, height = left.size
        shape = (height, width)
        predicted = np.fromfile(
            results / f"{frame}.disparity.bin", dtype="<f4"
        ).reshape(shape)
        truth = np.fromfile(data / f"{frame}.gt.bin", dtype="<f4").reshape(shape)
        occlusion = np.fromfile(
            data / f"{frame}.occlusions.bin", dtype=np.uint8
        ).reshape(shape)
        out_of_frame = np.fromfile(
            data / f"{frame}.outofframe.bin", dtype=np.uint8
        ).reshape(shape)
        visible = (
            (occlusion == 0) & (out_of_frame == 0) & np.isfinite(truth) & (truth >= 0)
        )
        covered_pixels += int(visible.sum())
        outside_range += int((visible & (truth > MAX_DISPARITY)).sum())
        maximum_disparity = max(maximum_disparity, float(truth[visible].max()))
        cpu = (
            baseline.compute(
                np.asarray(left.convert("L")), np.asarray(right.convert("L"))
            ).astype(np.float32)
            / 16
        ).astype(np.float32)
        gpu_scores.append(Accuracy.measure(predicted, truth, visible))
        cpu_scores.append(Accuracy.measure(cpu, truth, visible))
        common = visible & (np.arange(width)[None, :] > MAX_DISPARITY)
        common_gpu_scores.append(Accuracy.measure(predicted, truth, common))
        common_cpu_scores.append(Accuracy.measure(cpu, truth, common))
        in_range_scores.append(
            Accuracy.measure(predicted, truth, visible & (truth <= MAX_DISPARITY))
        )
        error = np.where(
            visible & (predicted >= 0), np.abs(predicted - truth), -1
        ).astype(np.float32)
        tiles = (
            (left, "Left image"),
            (right, "Right image"),
            (color(truth, 32), "Ground truth (0-32 px)"),
            (color(predicted, 32), "Vulkan SGM (black = rejected)"),
            (color(error, 3), "Absolute error (0-3 px)"),
            (color(cpu, 32), "OpenCV SGBM quality baseline"),
        )
        save_panel(tiles, panels / f"{index:04d}.png")
    times = np.loadtxt(results / "timings_ms.txt", ndmin=1)
    if len(times) != len(files):
        raise typer.BadParameter("Timing count differs from the sequence length")
    metrics = {
        "frames": len(files),
        "fps": float(1000 / times.mean()),
        "p50_ms": float(np.percentile(times, 50, method="inverted_cdf")),
        "p95_ms": float(np.percentile(times, 95, method="inverted_cdf")),
        "p99_ms": float(np.percentile(times, 99, method="inverted_cdf")),
        "vulkan_visible": summary(gpu_scores),
        "vulkan_visible_in_range": summary(in_range_scores),
        "opencv_visible": summary(cpu_scores),
        "common_roi_vulkan": summary(common_gpu_scores),
        "common_roi_opencv": summary(common_cpu_scores),
        "visible_gt_outside_search_percent": 100 * outside_range / covered_pixels,
        "max_visible_gt_disparity_px": maximum_disparity,
        "evaluation": (
            "resized clean pass; non-occluded, in-frame GT; pixel-weighted; "
            "not official benchmark score"
        ),
    }
    _ = (report / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    typer.run(main)
