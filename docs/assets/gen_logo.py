from pathlib import Path
import io

import matplotlib.pyplot as plt
import numpy as np

import xml.etree.ElementTree as ET

if __name__ == "__main__":
    output_path = Path(__file__).with_name("logo.svg")

    x = np.linspace(-3.5, 3.5, 400)
    y_line_left = 1.2 + 0.9 * x + 1.0 * x ** 3
    y_line_right = 1.2 - 0.9 * x - 1.0 * x ** 3
    y_parabola = 0.2 * x ** 2 + 0.2

    fig, ax = plt.subplots(figsize=(6.0, 6.0 / 1.414))

    ax.plot(x, y_line_left, color="k", linewidth=2.0)
    ax.plot(x, y_line_right, color="k", linewidth=2.0)
    ax.plot(x, y_parabola, color="k", linewidth=2.0)

    upper_envelope = np.minimum(y_line_left, y_line_right)
    region_mask = y_parabola <= upper_envelope
    x_region = x[region_mask]
    if x_region.size:
        ax.fill_between(
            x_region,
            y_parabola[region_mask],
            upper_envelope[region_mask],
            color="k",
            alpha=1,
        )

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-.25, 1.75)
    # ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    # ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Save base figure
    fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True, format="svg")

    # After saving, manually modify base figure to auto invert for dark/light mode
    # by adding this snip in the <defs> block.
    # The saved SVG is not quite proper XML, otherwise this would be easy to automate
    style = """
    <style type="text/css">
    @media (prefers-color-scheme: light) {
        svg { filter: invert(93%) hue-rotate(180deg); background-color: transparent !important; }
        image { filter: invert(100%) hue-rotate(180deg) saturate(1.25); }
    }
    </style>"""
