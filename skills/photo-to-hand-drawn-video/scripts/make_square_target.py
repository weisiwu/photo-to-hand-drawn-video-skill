#!/usr/bin/env python3
"""Build a 1:1 target that fills the canvas from a portrait reference.

The naive letterbox target leaves large side margins, so the painted figure
looks small and loses detail. This tool instead crops the portrait to a
square centered on the face (slightly above the vertical center), keeping the
face at ~0.22 of the target height so the semantic preflight topology checks
(face center distance, face scale ratio, subject IoU) still pass.

Usage:
    make_square_target.py --reference ref.png --target out.png \
        [--face-bounds 107,61,213,177] [--size 960]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

TARGET_FACE_FRACTION = 0.22  # face center lands at 22% of target height


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument(
        "--face-bounds",
        type=str,
        help="Optional face bounds 'left,top,right,bottom' in reference pixels; "
        "auto-detected from a semantic-preflight.json next to the reference "
        "if omitted.",
    )
    parser.add_argument("--size", type=int, default=960)
    return parser.parse_args()


def load_face_bounds(reference: Path, explicit: str | None) -> tuple[int, int, int, int]:
    if explicit:
        values = [int(part) for part in explicit.split(",")]
        if len(values) != 4:
            raise SystemExit("--face-bounds must be 'left,top,right,bottom'")
        return tuple(values)  # type: ignore[return-value]
    report = reference.parent / "semantic-preflight.json"
    if report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        face = (data.get("reference") or {}).get("bounds", {}).get("face")
        if face and face.get("pixels"):
            pixels = face["pixels"]
            return pixels["left"], pixels["top"], pixels["right"], pixels["bottom"]
    raise SystemExit(
        "No face bounds found; pass --face-bounds 'left,top,right,bottom' explicitly."
    )


def main() -> int:
    arguments = parse_arguments()
    image = Image.open(arguments.reference).convert("RGB")
    width, height = image.size
    face_left, face_top, face_right, face_bottom = load_face_bounds(
        arguments.reference, arguments.face_bounds
    )
    face_cx = (face_left + face_right) / 2.0
    face_cy = (face_top + face_bottom) / 2.0

    # Upscale so the square window (size x size) fits inside the image, then
    # crop in upscaled coordinates. Keeps the face near the top of the frame.
    scale = max(1.0, arguments.size / min(width, height)) * 2.0
    big_w, big_h = int(round(width * scale)), int(round(height * scale))
    big_image = image.resize((big_w, big_h), Image.Resampling.LANCZOS)
    big_cx, big_cy = face_cx * scale, face_cy * scale
    side = arguments.size

    # Face center should land near 22% of the square, but never crop the top
    # of the head: clamp the window so the full face stays inside.
    window_top = min(
        big_cy - TARGET_FACE_FRACTION * side,
        face_top * scale - 20.0,
    )
    window_left = big_cx - side / 2.0
    window_top = max(0.0, min(float(big_h - side), window_top))
    window_left = max(0.0, min(float(big_w - side), window_left))

    crop = big_image.crop(
        (
            int(round(window_left)),
            int(round(window_top)),
            int(round(window_left + side)),
            int(round(window_top + side)),
        )
    )
    crop = crop.resize((arguments.size, arguments.size), Image.Resampling.LANCZOS)
    crop.save(arguments.target)
    transform = {
        "scale": scale,
        "windowLeft": window_left,
        "windowTop": window_top,
        "windowSide": side,
        "targetSize": arguments.size,
    }
    transform_path = arguments.target.with_name("target-transform.json")
    transform_path.write_text(
        json.dumps(transform, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "target": str(arguments.target.resolve()),
                "transform": str(transform_path.resolve()),
                "size": arguments.size,
                "scale": scale,
                "cropWindowPx": [window_left, window_top, window_left + side, window_top + side],
                "faceCenterInTarget": [
                    round((big_cx - window_left) / side, 4),
                    round((big_cy - window_top) / side, 4),
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
