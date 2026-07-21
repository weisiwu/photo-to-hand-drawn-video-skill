#!/usr/bin/env python3
"""Measure reveal fingerprints, overpainting, and final-frame similarity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity


ANALYSIS_SIZE_PX = 240
PAPER_CHANGE_THRESHOLD = 12.0
FRAME_CHANGE_THRESHOLD = 5.5
FIRST_FINAL_MATCH_THRESHOLD = 10.0


def parse_crop(crop_text: str) -> tuple[int, int, int, int]:
    crop_values = tuple(int(value) for value in crop_text.split(","))
    if len(crop_values) != 4:
        raise argparse.ArgumentTypeError("crop must be x,y,width,height")
    return crop_values


def read_video_frame(video_capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame_bgr = video_capture.read()
    if not success:
        raise RuntimeError(f"unable to read frame {frame_index}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def crop_and_resize(frame_rgb: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    crop_x, crop_y, crop_width, crop_height = crop
    cropped = frame_rgb[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
    return cv2.resize(cropped, (ANALYSIS_SIZE_PX, ANALYSIS_SIZE_PX), interpolation=cv2.INTER_AREA)


def color_distance(first_rgb: np.ndarray, second_rgb: np.ndarray) -> np.ndarray:
    difference = first_rgb.astype(np.float32) - second_rgb.astype(np.float32)
    return np.sqrt(np.mean(difference * difference, axis=2))


def analyze_video(
    video_path: Path,
    crop: tuple[int, int, int, int],
    target_path: Path,
    frame_step: int,
) -> dict:
    video_capture = cv2.VideoCapture(str(video_path))
    if not video_capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(video_capture.get(cv2.CAP_PROP_FPS))
    baseline_frame = crop_and_resize(read_video_frame(video_capture, 0), crop)
    final_frame = crop_and_resize(read_video_frame(video_capture, max(0, frame_count - 2)), crop)

    first_visible_color = np.zeros_like(final_frame)
    has_appeared = np.zeros((ANALYSIS_SIZE_PX, ANALYSIS_SIZE_PX), dtype=bool)
    significant_change_count = np.zeros((ANALYSIS_SIZE_PX, ANALYSIS_SIZE_PX), dtype=np.uint16)
    previous_frame = baseline_frame

    sampled_frame_indices = list(range(0, frame_count, max(1, frame_step)))
    if sampled_frame_indices[-1] != frame_count - 2:
        sampled_frame_indices.append(frame_count - 2)

    for frame_index in sampled_frame_indices[1:]:
        current_frame = crop_and_resize(read_video_frame(video_capture, frame_index), crop)
        distance_from_paper = color_distance(current_frame, baseline_frame)
        newly_visible = (~has_appeared) & (distance_from_paper > PAPER_CHANGE_THRESHOLD)
        first_visible_color[newly_visible] = current_frame[newly_visible]
        has_appeared[newly_visible] = True

        changed_since_previous = color_distance(current_frame, previous_frame) > FRAME_CHANGE_THRESHOLD
        significant_change_count[changed_since_previous & has_appeared] += 1
        previous_frame = current_frame
    video_capture.release()

    final_visible = color_distance(final_frame, baseline_frame) > PAPER_CHANGE_THRESHOLD
    analyzable_pixels = final_visible & has_appeared
    analyzable_count = int(np.count_nonzero(analyzable_pixels))
    if analyzable_count == 0:
        raise RuntimeError("no painted pixels detected")

    first_final_distance = color_distance(first_visible_color, final_frame)
    first_value_matches_final_ratio = float(
        np.mean(first_final_distance[analyzable_pixels] <= FIRST_FINAL_MATCH_THRESHOLD)
    )
    repeatedly_modified_ratio = float(
        np.mean(significant_change_count[analyzable_pixels] >= 2)
    )

    target_bgr = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    if target_bgr is None:
        raise RuntimeError(f"unable to read target: {target_path}")
    target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
    target_rgb = cv2.resize(target_rgb, (ANALYSIS_SIZE_PX, ANALYSIS_SIZE_PX), interpolation=cv2.INTER_AREA)
    final_ssim = float(
        structural_similarity(target_rgb, final_frame, channel_axis=2, data_range=255)
    )

    passed = (
        first_value_matches_final_ratio < 0.92
        and repeatedly_modified_ratio > 0.08
        and 0.40 <= final_ssim <= 0.90
    )
    return {
        "passed": passed,
        "video": str(video_path),
        "frameCount": frame_count,
        "fps": fps,
        "sampledFrameCount": len(sampled_frame_indices),
        "analyzablePixelCount": analyzable_count,
        "firstValueMatchesFinalRatio": round(first_value_matches_final_ratio, 5),
        "repeatedlyModifiedRatio": round(repeatedly_modified_ratio, 5),
        "finalSsim": round(final_ssim, 5),
        "thresholds": {
            "maxFirstValueMatchesFinalRatio": 0.92,
            "minRepeatedlyModifiedRatio": 0.08,
            "finalSsimRange": [0.40, 0.90],
        },
    }


def main() -> int:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--video", type=Path, required=True)
    argument_parser.add_argument("--crop", type=parse_crop, required=True)
    argument_parser.add_argument("--target", type=Path, required=True)
    argument_parser.add_argument("--frame-step", type=int, default=1)
    argument_parser.add_argument("--report", type=Path)
    arguments = argument_parser.parse_args()

    report = analyze_video(arguments.video, arguments.crop, arguments.target, arguments.frame_step)
    report_text = json.dumps(report, ensure_ascii=False, indent=2)
    print(report_text)
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(report_text, encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
