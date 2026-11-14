#!/usr/bin/env python3
"""Make one synthetic pipeline diagnostic, never a benchmark or shared pilot case."""
import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    image = Image.new("RGB", (768, 432), "#d6e3ed")
    drawing = ImageDraw.Draw(image)
    drawing.rectangle((0, 270, 767, 431), fill="#a3a5a8")
    drawing.ellipse((175, 330, 355, 362), fill="#77797c")
    drawing.ellipse((190, 200, 330, 340), fill="#bc292d", outline="#702328", width=3)
    drawing.ellipse((216, 217, 244, 240), fill="#e89285")
    image.save(args.out / "synthetic_ball.png")
    with (args.out / "cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case_id", "category", "image_path", "prompt", "seed"])
        writer.writerow([
            "smoke_synthetic_001", "diagnostic", "synthetic_ball.png",
            "A red ball rolls slowly from left to right across a flat gray floor. The camera stays still.",
            6132,
        ])
    (args.out / "NOT_BENCHMARK.txt").write_text(
        "Synthetic environment check only. Exclude from shared pilot, formal evaluation, "
        "model comparison and human scoring.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
