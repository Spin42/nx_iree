#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "pillow==11.3.0", "typer==0.16.0"]
# ///
# ─── How to run ───
# Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run: uv run prepare.py SINTEL_ZIP OUTPUT_DIRECTORY --sequence alley_1
# ──────────────────
"""Prepare a complete Sintel clean-pass sequence at a chosen resolution."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import typer
from PIL import Image


def main(
    archive: Path,
    output: Path,
    sequence: str = "alley_1",
    width: int = 320,
    height: int = 136,
) -> None:
    """Decode pairs, disparity magnitudes and visibility masks outside timing."""
    if width <= 0 or height <= 0:
        raise typer.BadParameter("Image dimensions must be positive")
    output.mkdir(parents=True, exist_ok=True)
    size = (width, height)
    with ZipFile(archive) as source:
        names = sorted(
            name
            for name in source.namelist()
            if name.startswith(f"training/clean_left/{sequence}/")
            and name.endswith(".png")
        )
        if not names:
            raise typer.BadParameter(f"No clean stereo frames for {sequence}")
        for name in names:
            frame = Path(name).stem
            for side in ("left", "right"):
                with (
                    source.open(
                        f"training/clean_{side}/{sequence}/{frame}.png"
                    ) as entry,
                    Image.open(entry) as image,
                ):
                    resized = image.convert("RGB").resize(
                        size, Image.Resampling.LANCZOS
                    )
                    resized.save(output / f"{frame}.{side}.png")
                    np.asarray(resized.convert("L"), dtype="<i4").tofile(
                        output / f"{frame}.{side}.bin"
                    )
            with (
                source.open(f"training/disparities/{sequence}/{frame}.png") as entry,
                Image.open(entry) as image,
            ):
                channels = np.asarray(image, dtype=np.float32)
                # The official SDK decodes a positive magnitude: x_right = x_left - d.
                disparity = (
                    channels[:, :, 0] * 4
                    + channels[:, :, 1] / 64
                    + channels[:, :, 2] / 16384
                )
                scaled = Image.fromarray(disparity).resize(
                    size, Image.Resampling.NEAREST
                )
                (np.asarray(scaled, dtype=np.float32) * (size[0] / image.width)).astype(
                    "<f4"
                ).tofile(output / f"{frame}.gt.bin")
            for kind in ("occlusions", "outofframe"):
                with (
                    source.open(f"training/{kind}/{sequence}/{frame}.png") as entry,
                    Image.open(entry) as image,
                ):
                    mask = image.convert("L").resize(size, Image.Resampling.NEAREST)
                    (np.asarray(mask) > 0).astype(np.uint8).tofile(
                        output / f"{frame}.{kind}.bin"
                    )
    metadata = {
        "source": "https://sintel.is.tuebingen.mpg.de/stereo",
        "archive": archive.name,
        "sequence": sequence,
        "pass": "clean",
        "width": size[0],
        "height": size[1],
        "frames": len(names),
        "image_resampling": "Lanczos",
        "ground_truth_resampling": "nearest",
        "disparity_scale": size[0] / 1024,
        "official_leaderboard_comparable": False,
    }
    _ = (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    typer.run(main)
