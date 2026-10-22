"""Render comparable stereo outputs with fixed color scales."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def color(values: NDArray[np.float32], maximum: float) -> Image.Image:
    """Use a fixed disparity/error scale across the entire sequence."""
    normalized = np.clip(values / maximum * 255, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(
        cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB
    )
    rgb[values < 0] = 0
    return Image.fromarray(rgb)


def save_panel(tiles: tuple[tuple[Image.Image, str], ...], output: Path) -> None:
    """Label six same-sized images in a three-column grid."""
    width, height = tiles[0][0].size
    panel = Image.new("RGB", (width * 3, (height + 24) * 2))
    draw = ImageDraw.Draw(panel)
    for tile, (picture, title) in enumerate(tiles):
        x, y = (tile % 3) * width, (tile // 3) * (height + 24)
        draw.text((x + 5, y + 5), title, fill="white")
        panel.paste(picture, (x, y + 24))
    panel.save(output)
