#!/usr/bin/env python3
"""Two-layer compositing: draw a smooth continuous brush cursor over the
brush-free base video using the exported brush trajectory.

The trajectory recorded per video frame jumps between strokes (the tip snaps
to the next stroke start). A moving-average smoothing window turns those
jumps into slow, continuous glides — the cursor is always visible, never
blinks, never shakes.

Usage:
  compose_brush.py --base base.mp4 --trace brush-trace.json --output out.mp4
"""
import argparse
import json
import subprocess

import cv2
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Brush-free base video (1080x1920).")
    parser.add_argument("--trace", required=True, help="brush-trace.json from the renderer.")
    parser.add_argument("--output", required=True, help="Composited video path.")
    parser.add_argument("--smooth", type=int, default=7,
                        help="Moving-average window in frames (higher = slower, "
                             "smoother cursor motion).")
    parser.add_argument("--fps", type=float, default=None,
                        help="Video fps (auto-detected from the trace when omitted).")
    return parser.parse_args()


def draw_brush(frame: np.ndarray, x: float, y: float) -> None:
    """Round brush nib: dark tip, amber body, white outline."""
    center = (int(round(x)), int(round(y)))
    cv2.circle(frame, center, 24, (245, 245, 245), -1)      # white base
    cv2.circle(frame, center, 20, (7, 119, 217), -1)        # amber body (#d97706)
    cv2.circle(frame, center, 8, (18, 45, 124), -1)         # dark nib (#7c2d12)
    cv2.circle(frame, center, 24, (90, 90, 90), 2)          # soft outline


def main() -> int:
    arguments = parse_arguments()
    trace = json.load(open(arguments.trace, encoding="utf-8"))
    if not trace:
        print("empty trace")
        return 1

    frame_times = [entry["t"] for entry in trace]
    if arguments.fps is None:
        gaps = [b - a for a, b in zip(frame_times, frame_times[1:]) if b > a]
        fps = round(1.0 / (sum(gaps) / len(gaps))) if gaps else 8.0
    else:
        fps = arguments.fps
    print(f"fps={fps} trace_points={len(trace)}")

    raw_x = np.array([entry["x"] for entry in trace], dtype=np.float64)
    raw_y = np.array([entry["y"] for entry in trace], dtype=np.float64)
    valid = raw_x >= 0

    # Interpolate missing points (x=-1 gaps) from neighbours so the cursor
    # glides instead of disappearing between strokes.
    indices = np.arange(len(trace))
    x = np.interp(indices, indices[valid], raw_x[valid]) if valid.any() else raw_x
    y = np.interp(indices, indices[valid], raw_y[valid]) if valid.any() else raw_y

    # Stroke-to-stroke snaps: the raw tip jumps to the next stroke start.
    # Replace each jump with a short linear glide (transition frames) so the
    # cursor travels smoothly instead of teleporting — low-frequency motion.
    def smooth_jumps(xs, ys, threshold=40.0, transition=5):
        n = len(xs)
        for i in range(1, n):
            if i - 1 < 0:
                continue
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

    x, y = smooth_jumps(x, y)

    # Light moving-average smoothing (keeps the remaining micro-jitter out
    # without lagging the hand noticeably).
    window = max(1, arguments.smooth)
    kernel = np.ones(window) / window
    x = np.convolve(x, kernel, mode="same")
    y = np.convolve(y, kernel, mode="same")

    cap = cv2.VideoCapture(arguments.base)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Encode with ffmpeg/libx264 (H.264) — cv2's mp4v (MPEG-4 Part 2) is not
    # playable by QuickTime/browsers on macOS.
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "medium",
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
        trace_position = int(round(time_seconds / max(1e-6, frame_times[1] - frame_times[0]))) \
            if len(frame_times) > 1 else frame_index
        trace_position = min(trace_position, len(trace) - 1)
        if valid[trace_position]:
            draw_brush(frame, x[trace_position], y[trace_position])
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
