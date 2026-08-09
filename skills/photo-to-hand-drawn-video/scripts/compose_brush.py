#!/usr/bin/env python3
"""Two-layer compositing for the hand-drawn video pipeline.

Pipeline (user-approved): render the brush-free fill video first, then
derive the brush rhythm FROM THE VIDEO (frame-to-frame changes inside the
paper region = where the hand is actually drawing), then draw a smooth
continuous cursor and composite. The cursor follows the real drawing pace:
it only moves when the picture changes.

Modes:
- analyze (default): derive cursor positions from frame diffs of the base
  video. No plan/trace needed — works with any fixed-camera base video.
- trace: use an exported brush-trace.json instead (older mode).

Output is H.264 via ffmpeg pipe (cv2 mp4v is not playable on macOS).

Usage:
  compose_brush.py --base base.mp4 --output final.mp4 [--paper "0,518,1080,1080"]
"""
import argparse
import json
import subprocess

import cv2
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Brush-free base video.")
    parser.add_argument("--output", required=True, help="Composited video path.")
    parser.add_argument("--trace", default=None,
                        help="Optional brush-trace.json (older mode; default is frame-diff analysis).")
    parser.add_argument("--paper", default="0,518,1080,1080",
                        help="Paper region 'left,top,width,height' in video px "
                             "(where drawing changes are detected).")
    parser.add_argument("--smooth", type=int, default=3,
                        help="Moving-average window in frames.")
    parser.add_argument("--diff-threshold", type=int, default=16,
                        help="Per-pixel abs-diff sum threshold for 'changed'.")
    parser.add_argument("--min-changed", type=int, default=40,
                        help="Minimum changed pixels to count as an active stroke.")
    parser.add_argument("--fps", type=float, default=None,
                        help="Video fps (auto-detected when omitted).")
    return parser.parse_args()


def draw_brush(frame: np.ndarray, x: float, y: float) -> None:
    """Round brush nib: dark tip, amber body, white outline."""
    center = (int(round(x)), int(round(y)))
    cv2.circle(frame, center, 24, (245, 245, 245), -1)
    cv2.circle(frame, center, 20, (7, 119, 217), -1)
    cv2.circle(frame, center, 8, (18, 45, 124), -1)
    cv2.circle(frame, center, 24, (90, 90, 90), 2)


def smooth_jumps(xs: np.ndarray, ys: np.ndarray, threshold: float = 60.0,
                 transition: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Replace stroke-to-stroke teleports with short linear glides."""
    n = len(xs)
    for i in range(1, n):
        distance = float(np.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))
        if distance > threshold:
            start = max(0, i - transition)
            end = min(n - 1, i + transition)
            if end - start >= 2:
                for j in range(start + 1, end):
                    fraction = (j - start) / (end - start)
                    xs[j] = xs[start] + (xs[end] - xs[start]) * fraction
                    ys[j] = ys[start] + (ys[end] - ys[start]) * fraction
    return xs, ys


def analyze_brush_rhythm(base_path: str, paper: tuple[int, int, int, int],
                         diff_threshold: int, min_changed: int) -> list:
    """Frame-to-frame diff inside the paper region -> cursor positions."""
    left, top, width, height = paper
    cap = cv2.VideoCapture(base_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    positions: list[tuple[float, float] | None] = []
    previous = None
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if previous is not None:
            current_crop = frame[top:top + height, left:left + width].astype(np.int16)
            previous_crop = previous[top:top + height, left:left + width].astype(np.int16)
            diff = np.abs(current_crop - previous_crop).sum(axis=2)
            mask = diff > diff_threshold
            changed = int(mask.sum())
            if changed >= min_changed:
                weights = diff[mask].astype(np.float64)
                ys, xs = np.where(mask)
                centroid_x = float((xs * weights).sum() / weights.sum()) + left
                centroid_y = float((ys * weights).sum() / weights.sum()) + top
                positions.append((centroid_x, centroid_y))
            else:
                positions.append(None)
        else:
            positions.append(None)
        previous = frame
        frame_index += 1
        if frame_index % 128 == 0:
            print(f"  analyze {frame_index}/{total}")
    cap.release()
    return positions, fps


def main() -> int:
    arguments = parse_arguments()
    paper = tuple(int(v) for v in arguments.paper.split(","))

    if arguments.trace:
        trace = json.load(open(arguments.trace, encoding="utf-8"))
        trace_t = np.array([entry["t"] for entry in trace], dtype=np.float64)
        raw_x = np.array([entry["x"] for entry in trace], dtype=np.float64)
        raw_y = np.array([entry["y"] for entry in trace], dtype=np.float64)
        valid = raw_x >= 0
        indices = np.arange(len(trace))
        x = np.interp(indices, indices[valid], raw_x[valid]) if valid.any() else raw_x
        y = np.interp(indices, indices[valid], raw_y[valid]) if valid.any() else raw_y
        fps = arguments.fps or (round(1.0 / np.median(np.diff(trace_t))) if len(trace_t) > 2 else 8.0)
    else:
        # Rhythm from the video itself: where the picture changed, the hand was.
        positions, fps = analyze_brush_rhythm(
            arguments.base, paper, arguments.diff_threshold, arguments.min_changed
        )
        n = len(positions)
        trace_t = np.arange(n) / fps
        x = np.full(n, np.nan, dtype=np.float64)
        y = np.full(n, np.nan, dtype=np.float64)
        for i, position in enumerate(positions):
            if position is not None:
                x[i], y[i] = position
        valid = ~np.isnan(x)
        if not valid.any():
            print("no drawing changes detected — is --paper correct?")
            return 1
        indices = np.arange(n)
        x = np.interp(indices, indices[valid], x[valid])
        y = np.interp(indices, indices[valid], y[valid])
        fps = arguments.fps or fps

    print(f"fps={fps:.2f} points={len(x)}")

    x, y = smooth_jumps(x, y)
    window = max(1, arguments.smooth)
    kernel = np.ones(window) / window
    x = np.convolve(x, kernel, mode="same")
    y = np.convolve(y, kernel, mode="same")

    cap = cv2.VideoCapture(arguments.base)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps:.3f}", "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-preset", "medium",
            arguments.output,
        ],
        stdin=subprocess.PIPE,
    )

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        time_seconds = frame_index / fps
        # Time-accurate lookup (interpolate), never an index arithmetic guess:
        # trace timestamps may mix rAF and seek samples.
        position_x = float(np.interp(time_seconds, trace_t, x))
        position_y = float(np.interp(time_seconds, trace_t, y))
        draw_brush(frame, position_x, position_y)
        encoder.stdin.write(frame.tobytes())
        frame_index += 1
        if frame_index % 64 == 0:
            print(f"  {frame_index}/{total}")

    cap.release()
    encoder.stdin.close()
    encoder.wait()
    if encoder.returncode != 0:
        print(f"ffmpeg exited with {encoder.returncode}")
        return 1
    print(f"done: {arguments.output} ({frame_index} frames, H.264)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
