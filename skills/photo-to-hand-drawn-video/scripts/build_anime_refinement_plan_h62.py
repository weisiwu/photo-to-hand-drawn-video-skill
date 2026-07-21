#!/usr/bin/env python3
"""Build a high-resolution, stroke-only anime paint plan from a square target.

H6.2 preserves the approved H6.1 construction process and adds dedicated
subject and identity passes. The browser renders the normalized plan at 2x
resolution and downsamples it; no runtime target-image compositing is used.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage import measure, morphology
from skimage.metrics import structural_similarity

from build_flat_marker_plan_v7 import (
    ARTBOARD_SIZE_PX,
    build_region_hatch_strokes,
    calculate_polyline_length,
    collect_fill_regions,
    merge_collinear_paths,
    normalize_points,
    quantize_to_palette,
    resample_points,
    to_hex_color,
    trace_skeleton_paths,
)


PROJECT_DIRECTORY = Path(__file__).resolve().parent
TARGET_IMAGE_PATH = PROJECT_DIRECTORY / "assets" / "marker-target-h6-seaside.png"
PLAN_JSON_PATH = PROJECT_DIRECTORY / "assets" / "marker-paint-plan-h62.json"
PLAN_JAVASCRIPT_PATH = PROJECT_DIRECTORY / "assets" / "marker-paint-plan-h62.js"
PREVIEW_PATH = PROJECT_DIRECTORY / "verification-v7" / "h62-plan-preview.png"
METRICS_PATH = PROJECT_DIRECTORY / "verification-v7" / "h62-plan-metrics.json"

RANDOM_SEED = 20260722
MULTISCALE_PASSES = (
    {
        "phase": "anime_refine_broad",
        "widthPx": 14.0,
        "blurSigma": 3.2,
        "errorThreshold": 13.0,
        "colorTolerance": 34.0,
        "maxStrokes": 640,
        "lengthWidths": 5.8,
    },
    {
        "phase": "anime_refine_medium",
        "widthPx": 8.0,
        "blurSigma": 1.8,
        "errorThreshold": 9.0,
        "colorTolerance": 25.0,
        "maxStrokes": 1050,
        "lengthWidths": 5.2,
    },
    {
        "phase": "anime_refine_detail",
        "widthPx": 4.2,
        "blurSigma": 0.8,
        "errorThreshold": 6.0,
        "colorTolerance": 18.0,
        "maxStrokes": 1800,
        "lengthWidths": 4.8,
    },
    {
        "phase": "anime_refine_micro",
        "widthPx": 2.2,
        "blurSigma": 0.0,
        "errorThreshold": 4.2,
        "colorTolerance": 13.0,
        "maxStrokes": 2500,
        "lengthWidths": 4.2,
    },
    {
        "phase": "anime_refine_ultra",
        "widthPx": 1.5,
        "blurSigma": 0.0,
        "errorThreshold": 2.8,
        "colorTolerance": 9.0,
        "maxStrokes": 3200,
        "lengthWidths": 3.6,
    },
)

SUBJECT_BOUNDS_NORMALIZED = (0.24, 0.04, 0.88, 1.0)
SUBJECT_ERROR_WEIGHT = 1.7
EDGE_ERROR_WEIGHT = 1.25

FINAL_INK_LUMINANCE_MAX = 48.0
FINAL_INK_MAX_HALF_WIDTH_PX = 5.0
FINAL_INK_MIN_PATH_LENGTH_PX = 2.5


def hex_to_rgb(color_hex: str) -> np.ndarray:
    return np.array([int(color_hex[offset : offset + 2], 16) for offset in (1, 3, 5)], dtype=np.float32)


def draw_stroke_on_simulation(simulation_rgb: np.ndarray, stroke: dict[str, Any]) -> None:
    points_px = np.clip(
        np.rint(np.asarray(stroke["points"], dtype=np.float32) * ARTBOARD_SIZE_PX).astype(np.int32),
        0,
        ARTBOARD_SIZE_PX - 1,
    )
    if len(points_px) < 2:
        return
    width_px = max(1, int(round(float(stroke["width"]) * ARTBOARD_SIZE_PX)))
    padding_px = max(2, width_px + 2)
    left = max(0, int(points_px[:, 0].min()) - padding_px)
    top = max(0, int(points_px[:, 1].min()) - padding_px)
    right = min(ARTBOARD_SIZE_PX, int(points_px[:, 0].max()) + padding_px + 1)
    bottom = min(ARTBOARD_SIZE_PX, int(points_px[:, 1].max()) + padding_px + 1)
    local_points_px = points_px - np.array([left, top], dtype=np.int32)
    stroke_mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
    cv2.polylines(
        stroke_mask,
        [local_points_px.reshape(-1, 1, 2)],
        False,
        255,
        width_px,
        cv2.LINE_AA,
    )
    alpha = stroke_mask.astype(np.float32)[:, :, None] / 255.0 * float(stroke.get("alpha", 1.0))
    stroke_rgb = hex_to_rgb(stroke["color"])
    simulation_patch = simulation_rgb[top:bottom, left:right]
    if stroke.get("blend") == "multiply":
        painted_rgb = simulation_patch * (stroke_rgb[None, None, :] / 255.0)
    else:
        painted_rgb = np.broadcast_to(stroke_rgb, simulation_patch.shape)
    simulation_patch[:] = simulation_patch * (1.0 - alpha) + painted_rgb * alpha


def simulate_plan(plan: dict[str, Any]) -> np.ndarray:
    paper_rgb = hex_to_rgb(plan["paperColor"])
    simulation_rgb = np.full((ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX, 3), paper_rgb, dtype=np.float32)
    for stroke in plan["strokes"]:
        if stroke.get("mode"):
            continue
        draw_stroke_on_simulation(simulation_rgb, stroke)
    return simulation_rgb


def build_tangent_field(target_rgb: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    grayscale = cv2.cvtColor(target_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gradient_x = cv2.Sobel(grayscale, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(grayscale, cv2.CV_32F, 0, 1, ksize=3)
    if sigma > 0:
        gradient_x = cv2.GaussianBlur(gradient_x, (0, 0), sigma)
        gradient_y = cv2.GaussianBlur(gradient_y, (0, 0), sigma)
    tangent_x = -gradient_y
    tangent_y = gradient_x
    tangent_norm = np.hypot(tangent_x, tangent_y)
    tangent_x /= np.maximum(tangent_norm, 1e-5)
    tangent_y /= np.maximum(tangent_norm, 1e-5)
    return tangent_x, tangent_y


def build_flow_path(
    seed_x: int,
    seed_y: int,
    target_rgb: np.ndarray,
    tangent_x: np.ndarray,
    tangent_y: np.ndarray,
    width_px: float,
    color_tolerance: float,
    length_widths: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    seed_color_rgb = target_rgb[seed_y, seed_x].astype(np.float32)
    seed_tangent = np.array([tangent_x[seed_y, seed_x], tangent_y[seed_y, seed_x]], dtype=np.float32)
    if float(np.linalg.norm(seed_tangent)) < 0.25:
        fallback_angle = float(random_generator.uniform(0, math.tau))
        seed_tangent = np.array([math.cos(fallback_angle), math.sin(fallback_angle)], dtype=np.float32)

    step_px = max(1.25, width_px * 0.62)
    maximum_steps = max(2, int(math.ceil(length_widths * width_px / step_px / 2.0)))
    forward_points: list[np.ndarray] = []
    backward_points: list[np.ndarray] = []
    for direction_sign, destination in ((1.0, forward_points), (-1.0, backward_points)):
        point_xy = np.array([seed_x, seed_y], dtype=np.float32)
        direction_xy = seed_tangent.copy() * direction_sign
        for _ in range(maximum_steps):
            point_xy = point_xy + direction_xy * step_px
            point_x = int(round(float(point_xy[0])))
            point_y = int(round(float(point_xy[1])))
            if not (0 <= point_x < ARTBOARD_SIZE_PX and 0 <= point_y < ARTBOARD_SIZE_PX):
                break
            if float(np.linalg.norm(target_rgb[point_y, point_x].astype(np.float32) - seed_color_rgb)) > color_tolerance:
                break
            local_tangent = np.array([tangent_x[point_y, point_x], tangent_y[point_y, point_x]], dtype=np.float32)
            if float(np.linalg.norm(local_tangent)) >= 0.25:
                if float(np.dot(local_tangent, direction_xy)) < 0:
                    local_tangent *= -1
                direction_xy = direction_xy * 0.55 + local_tangent * 0.45
                direction_xy /= max(float(np.linalg.norm(direction_xy)), 1e-5)
            destination.append(point_xy.copy())

    ordered_points = [*reversed(backward_points), np.array([seed_x, seed_y], dtype=np.float32), *forward_points]
    if len(ordered_points) < 2:
        perpendicular = np.array([-seed_tangent[1], seed_tangent[0]], dtype=np.float32)
        half_length_px = max(1.5, width_px * 0.7)
        ordered_points = [
            np.array([seed_x, seed_y], dtype=np.float32) - perpendicular * half_length_px,
            np.array([seed_x, seed_y], dtype=np.float32) + perpendicular * half_length_px,
        ]
    return np.asarray(ordered_points, dtype=np.float32)


def build_multiscale_refinement_strokes(
    simulation_rgb: np.ndarray,
    target_rgb: np.ndarray,
    subject_mask: np.ndarray | None = None,
    budget_scale: float = 1.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    random_generator = np.random.default_rng(RANDOM_SEED)
    refinement_strokes: list[dict[str, Any]] = []
    pass_metrics: list[dict[str, Any]] = []
    subject_left = int(round(SUBJECT_BOUNDS_NORMALIZED[0] * ARTBOARD_SIZE_PX))
    subject_top = int(round(SUBJECT_BOUNDS_NORMALIZED[1] * ARTBOARD_SIZE_PX))
    subject_right = int(round(SUBJECT_BOUNDS_NORMALIZED[2] * ARTBOARD_SIZE_PX))
    subject_bottom = int(round(SUBJECT_BOUNDS_NORMALIZED[3] * ARTBOARD_SIZE_PX))
    subject_weight_map = np.ones((ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX), dtype=np.float32)
    if subject_mask is None:
        subject_weight_map[subject_top:subject_bottom, subject_left:subject_right] = SUBJECT_ERROR_WEIGHT
    else:
        resized_subject_mask = cv2.resize(
            subject_mask.astype(np.uint8),
            (ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        subject_weight_map[resized_subject_mask] = SUBJECT_ERROR_WEIGHT

    for pass_options in MULTISCALE_PASSES:
        blur_sigma = float(pass_options["blurSigma"])
        pass_target_rgb = (
            cv2.GaussianBlur(target_rgb.astype(np.float32), (0, 0), blur_sigma)
            if blur_sigma > 0
            else target_rgb.astype(np.float32)
        )
        tangent_x, tangent_y = build_tangent_field(pass_target_rgb, max(0.6, blur_sigma))
        target_grayscale = cv2.cvtColor(pass_target_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        target_gradient_x = cv2.Sobel(target_grayscale, cv2.CV_32F, 1, 0, ksize=3)
        target_gradient_y = cv2.Sobel(target_grayscale, cv2.CV_32F, 0, 1, ksize=3)
        target_edge_strength = np.hypot(target_gradient_x, target_gradient_y)
        edge_scale = float(np.percentile(target_edge_strength, 98.0))
        normalized_edge_strength = np.clip(target_edge_strength / max(edge_scale, 1e-5), 0.0, 1.0)
        selection_weight_map = subject_weight_map * (1.0 + EDGE_ERROR_WEIGHT * normalized_edge_strength)
        error_map = np.mean(np.abs(pass_target_rgb - simulation_rgb), axis=2)
        error_map = cv2.GaussianBlur(error_map.astype(np.float32), (0, 0), max(0.5, pass_options["widthPx"] * 0.12))
        error_map[error_map < pass_options["errorThreshold"]] = 0.0
        selection_map = error_map * selection_weight_map
        pass_stroke_count = 0

        maximum_strokes = max(1, int(round(pass_options["maxStrokes"] * budget_scale)))
        while pass_stroke_count < maximum_strokes:
            seed_flat_index = int(np.argmax(selection_map))
            seed_y, seed_x = np.unravel_index(seed_flat_index, selection_map.shape)
            if error_map[seed_y, seed_x] <= 0:
                break
            path_xy = build_flow_path(
                seed_x,
                seed_y,
                pass_target_rgb,
                tangent_x,
                tangent_y,
                float(pass_options["widthPx"]),
                float(pass_options["colorTolerance"]),
                float(pass_options["lengthWidths"]),
                random_generator,
            )
            path_xy[:, 0] = np.clip(path_xy[:, 0], 0, ARTBOARD_SIZE_PX - 1)
            path_xy[:, 1] = np.clip(path_xy[:, 1], 0, ARTBOARD_SIZE_PX - 1)
            stroke_color_rgb = pass_target_rgb[seed_y, seed_x]
            stroke = {
                "phase": pass_options["phase"],
                "points": normalize_points(resample_points(path_xy)),
                "color": to_hex_color(stroke_color_rgb),
                "width": round(float(pass_options["widthPx"]) / ARTBOARD_SIZE_PX, 6),
                "alpha": 0.94 if pass_options["phase"] != "anime_refine_micro" else 0.97,
                "blend": "source-over",
            }
            refinement_strokes.append(stroke)
            draw_stroke_on_simulation(simulation_rgb, stroke)

            suppression_mask = np.zeros(error_map.shape, dtype=np.uint8)
            cv2.polylines(
                suppression_mask,
                [np.rint(path_xy).astype(np.int32).reshape(-1, 1, 2)],
                False,
                255,
                max(2, int(round(float(pass_options["widthPx"]) * 1.15))),
                cv2.LINE_AA,
            )
            error_map[suppression_mask > 0] = 0.0
            selection_map[suppression_mask > 0] = 0.0
            pass_stroke_count += 1

        pass_ssim = float(
            structural_similarity(
                target_rgb.astype(np.uint8),
                np.clip(simulation_rgb, 0, 255).astype(np.uint8),
                channel_axis=2,
                data_range=255,
            )
        )
        pass_metrics.append(
            {
                "phase": pass_options["phase"],
                "strokeCount": pass_stroke_count,
                "previewSsim": round(pass_ssim, 6),
            }
        )
    return refinement_strokes, pass_metrics


def build_final_ink_strokes(target_rgb: np.ndarray) -> list[dict[str, Any]]:
    target_lab = cv2.cvtColor(target_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    luminance = target_lab[:, :, 0] * 100.0 / 255.0
    dark_mask = luminance < FINAL_INK_LUMINANCE_MAX
    dark_half_width = distance_transform_edt(dark_mask)
    ink_mask = dark_mask & (dark_half_width <= FINAL_INK_MAX_HALF_WIDTH_PX)
    ink_mask = morphology.binary_closing(ink_mask, morphology.disk(1))
    ink_mask = morphology.remove_small_objects(ink_mask, 3)
    ink_skeleton = morphology.skeletonize(ink_mask)

    final_ink_strokes: list[dict[str, Any]] = []
    raw_ink_paths = trace_skeleton_paths(ink_skeleton)
    for path_xy in merge_collinear_paths(raw_ink_paths):
        simplified = measure.approximate_polygon(path_xy[:, ::-1], tolerance=0.45)[:, ::-1]
        path_length_px = calculate_polyline_length(simplified)
        if len(simplified) < 2 or path_length_px < FINAL_INK_MIN_PATH_LENGTH_PX:
            continue
        sample_indices = np.clip(np.rint(path_xy).astype(int), 0, ARTBOARD_SIZE_PX - 1)
        median_half_width_px = float(
            np.median(dark_half_width[sample_indices[:, 1], sample_indices[:, 0]])
        )
        width_px = float(np.clip(median_half_width_px * 2.0, 1.4, 6.0))
        sampled_rgb = target_rgb[sample_indices[:, 1], sample_indices[:, 0]].mean(axis=0)
        final_ink_strokes.append(
            {
                "phase": "anime_final_ink",
                "points": normalize_points(resample_points(simplified)),
                "color": to_hex_color(sampled_rgb * 0.78),
                "width": round(width_px / ARTBOARD_SIZE_PX, 6),
                "alpha": 0.94,
                "blend": "source-over",
                "lengthPx": path_length_px,
            }
        )
    final_ink_strokes.sort(key=lambda stroke: (-stroke["lengthPx"], stroke["points"][0][1]))
    for stroke in final_ink_strokes:
        stroke.pop("lengthPx")
    return final_ink_strokes


def build_scene_palette(target_rgb: np.ndarray, color_count: int = 7) -> list[str]:
    """Extract a stable small palette for the on-screen swatches."""
    sample_rgb = cv2.resize(target_rgb.astype(np.uint8), (120, 120), interpolation=cv2.INTER_AREA)
    sample_pixels = sample_rgb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.25)
    _compactness, labels, centers = cv2.kmeans(
        sample_pixels,
        color_count,
        None,
        criteria,
        8,
        cv2.KMEANS_PP_CENTERS,
    )
    counts = np.bincount(labels.ravel(), minlength=color_count)
    ordered_centers = centers[np.argsort(-counts)]
    return [to_hex_color(center) for center in ordered_centers]


def build_shape_following_fill_strokes(
    target_rgb: np.ndarray,
    paper_rgb: np.ndarray,
    focus_regions: list[dict[str, float]] | None = None,
) -> tuple[list[dict[str, Any]], list[str], int]:
    """Hatch coherent color shapes instead of scanning across the canvas."""
    target_bgr = cv2.cvtColor(target_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    flattened_bgr = cv2.pyrMeanShiftFiltering(target_bgr, sp=8, sr=20, maxLevel=1)
    flattened_rgb = cv2.cvtColor(flattened_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    palette_rgb, label_map = quantize_to_palette(flattened_rgb)
    if focus_regions is None:
        focus_regions = [{"left": 0.42, "top": 0.13, "width": 0.27, "height": 0.30}]
    fill_regions, _small_dark_features = collect_fill_regions(
        palette_rgb,
        label_map,
        paper_rgb,
        focus_regions,
    )
    random_generator = np.random.default_rng(RANDOM_SEED)
    fill_strokes: list[dict[str, Any]] = []
    for region in fill_regions:
        region_color_hex = to_hex_color(palette_rgb[region["colorIndex"]])
        fill_strokes.extend(
            build_region_hatch_strokes(
                region,
                region_color_hex,
                random_generator,
                width_scale=0.82,
                spacing_factor=0.57,
                residual_depth=1,
            )
        )
    palette_hex = [to_hex_color(color_rgb) for color_rgb in palette_rgb]
    return fill_strokes, palette_hex, len(fill_regions)


def build_structure_strokes(final_ink_strokes: list[dict[str, Any]], maximum_count: int = 360) -> list[dict[str, Any]]:
    """Reuse the strongest target contours as the visible construction drawing."""
    structure_strokes: list[dict[str, Any]] = []
    for ink_stroke in final_ink_strokes[:maximum_count]:
        width_px = float(ink_stroke["width"]) * ARTBOARD_SIZE_PX
        structure_strokes.append(
            {
                "phase": "structure_line",
                "points": ink_stroke["points"],
                "color": ink_stroke["color"],
                "width": round(min(8.0, max(2.2, width_px * 1.35)) / ARTBOARD_SIZE_PX, 6),
                "alpha": 0.58,
                "blend": "multiply",
            }
        )
    return structure_strokes


def build_subject_detail_mask() -> np.ndarray:
    """Select the person, clothing, hair, and wrist accessory for extra detail."""
    mask = np.zeros((ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX), dtype=np.uint8)

    def point(normalized_x: float, normalized_y: float) -> tuple[int, int]:
        return (
            int(round(normalized_x * ARTBOARD_SIZE_PX)),
            int(round(normalized_y * ARTBOARD_SIZE_PX)),
        )

    cv2.ellipse(mask, point(0.56, 0.28), point(0.19, 0.23), 0, 0, 360, 255, -1)
    arm_width_px = int(round(0.13 * ARTBOARD_SIZE_PX))
    cv2.line(mask, point(0.31, 0.18), point(0.45, 0.39), 255, arm_width_px)
    cv2.line(mask, point(0.82, 0.25), point(0.72, 0.48), 255, arm_width_px)
    dress_polygon = np.asarray(
        [point(0.38, 0.38), point(0.73, 0.39), point(0.87, 1.0), point(0.27, 1.0)],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [dress_polygon], 255)
    cv2.circle(mask, point(0.80, 0.31), int(round(0.09 * ARTBOARD_SIZE_PX)), 255, -1)
    return mask.astype(bool)


def build_identity_detail_mask() -> np.ndarray:
    """Select the face, fringe, and adjacent hair where likeness is decided."""
    mask = np.zeros((ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX), dtype=np.uint8)
    center = (int(round(0.555 * ARTBOARD_SIZE_PX)), int(round(0.285 * ARTBOARD_SIZE_PX)))
    axes = (int(round(0.125 * ARTBOARD_SIZE_PX)), int(round(0.165 * ARTBOARD_SIZE_PX)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask.astype(bool)


def build_local_detail_strokes(
    simulation_rgb: np.ndarray,
    target_rgb: np.ndarray,
    detail_mask: np.ndarray,
    phase: str,
    width_px: float,
    maximum_strokes: int,
    error_threshold: float,
    color_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Spend a bounded fine-stroke budget only inside a selected semantic area."""
    tangent_x, tangent_y = build_tangent_field(target_rgb, 0.55)
    target_grayscale = cv2.cvtColor(target_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gradient_x = cv2.Sobel(target_grayscale, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(target_grayscale, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength = np.hypot(gradient_x, gradient_y)
    edge_scale = max(float(np.percentile(edge_strength[detail_mask], 98.0)), 1e-5)
    edge_weight = 1.0 + np.clip(edge_strength / edge_scale, 0.0, 1.0) * 1.8
    error_map = np.mean(np.abs(target_rgb - simulation_rgb), axis=2)
    selection_score = np.where(
        detail_mask & (error_map >= error_threshold),
        error_map * edge_weight,
        0.0,
    )
    ordered_indices = np.argsort(selection_score.reshape(-1))[::-1]
    available_mask = selection_score > 0
    random_generator = np.random.default_rng(RANDOM_SEED + int(round(width_px * 100)))
    detail_strokes: list[dict[str, Any]] = []

    for flat_index in ordered_indices:
        if len(detail_strokes) >= maximum_strokes:
            break
        seed_y, seed_x = np.unravel_index(int(flat_index), selection_score.shape)
        if not available_mask[seed_y, seed_x] or selection_score[seed_y, seed_x] <= 0:
            continue
        path_xy = build_flow_path(
            seed_x,
            seed_y,
            target_rgb,
            tangent_x,
            tangent_y,
            width_px,
            color_tolerance,
            3.8,
            random_generator,
        )
        path_xy[:, 0] = np.clip(path_xy[:, 0], 0, ARTBOARD_SIZE_PX - 1)
        path_xy[:, 1] = np.clip(path_xy[:, 1], 0, ARTBOARD_SIZE_PX - 1)
        stroke = {
            "phase": phase,
            "points": normalize_points(resample_points(path_xy)),
            "color": to_hex_color(target_rgb[seed_y, seed_x]),
            "width": round(width_px / ARTBOARD_SIZE_PX, 6),
            "alpha": 0.97,
            "blend": "source-over",
        }
        detail_strokes.append(stroke)
        draw_stroke_on_simulation(simulation_rgb, stroke)

        suppression_mask = np.zeros(available_mask.shape, dtype=np.uint8)
        cv2.polylines(
            suppression_mask,
            [np.rint(path_xy).astype(np.int32).reshape(-1, 1, 2)],
            False,
            255,
            max(2, int(round(width_px * 1.35))),
            cv2.LINE_AA,
        )
        available_mask[suppression_mask > 0] = False

    pass_ssim = float(
        structural_similarity(
            target_rgb.astype(np.uint8),
            np.clip(simulation_rgb, 0, 255).astype(np.uint8),
            channel_axis=2,
            data_range=255,
        )
    )
    metrics = {
        "phase": phase,
        "strokeCount": len(detail_strokes),
        "previewSsim": round(pass_ssim, 6),
    }
    return detail_strokes, metrics


def build_identity_ink_strokes(
    final_ink_strokes: list[dict[str, Any]],
    identity_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Redraw decisive facial, hair, wrist, and collar lines with a finer nib."""
    identity_ink_strokes: list[dict[str, Any]] = []
    for ink_stroke in final_ink_strokes:
        points = np.asarray(ink_stroke["points"], dtype=np.float32)
        center_x, center_y = points.mean(axis=0)
        if identity_mask is None:
            in_face = 0.41 <= center_x <= 0.70 and 0.10 <= center_y <= 0.46
            in_scrunchie = 0.70 <= center_x <= 0.91 and 0.20 <= center_y <= 0.43
            in_collar = 0.32 <= center_x <= 0.80 and 0.36 <= center_y <= 0.62
            if not (in_face or in_scrunchie or in_collar):
                continue
        else:
            center_column = min(ARTBOARD_SIZE_PX - 1, max(0, round(center_x * ARTBOARD_SIZE_PX)))
            center_row = min(ARTBOARD_SIZE_PX - 1, max(0, round(center_y * ARTBOARD_SIZE_PX)))
            if not identity_mask[center_row, center_column]:
                continue
        identity_ink_strokes.append(
            {
                "phase": "anime_identity_ink",
                "points": ink_stroke["points"],
                "color": ink_stroke["color"],
                "width": round(max(0.00072, float(ink_stroke["width"]) * 0.72), 6),
                "alpha": 0.88,
                "blend": "source-over",
            }
        )
    return identity_ink_strokes


def main() -> None:
    target_image = Image.open(TARGET_IMAGE_PATH).convert("RGB")
    target_image = target_image.resize((ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX), Image.Resampling.LANCZOS)
    target_rgb = np.asarray(target_image, dtype=np.float32)
    final_ink_strokes = build_final_ink_strokes(target_rgb)
    structure_strokes = build_structure_strokes(final_ink_strokes)
    paper_rgb = np.array([247, 241, 227], dtype=np.float32)
    fill_strokes, palette_hex, fill_region_count = build_shape_following_fill_strokes(
        target_rgb,
        paper_rgb,
    )
    base_plan = {
        "paperColor": to_hex_color(paper_rgb),
        "palette": palette_hex,
        "focusRegions": {
            "face": {"left": 0.42, "top": 0.13, "width": 0.27, "height": 0.30},
            "upperBody": {"left": 0.24, "top": 0.05, "width": 0.64, "height": 0.57},
            "dress": {"left": 0.29, "top": 0.48, "width": 0.57, "height": 0.52},
        },
        "strokes": [*structure_strokes, *fill_strokes],
    }
    simulation_rgb = simulate_plan(base_plan)

    refinement_strokes, pass_metrics = build_multiscale_refinement_strokes(simulation_rgb, target_rgb)
    subject_detail_strokes, subject_detail_metrics = build_local_detail_strokes(
        simulation_rgb,
        target_rgb,
        build_subject_detail_mask(),
        "anime_subject_detail",
        width_px=1.35,
        maximum_strokes=5200,
        error_threshold=2.2,
        color_tolerance=10.0,
    )
    identity_detail_strokes, identity_detail_metrics = build_local_detail_strokes(
        simulation_rgb,
        target_rgb,
        build_identity_detail_mask(),
        "anime_identity_detail",
        width_px=0.90,
        maximum_strokes=4800,
        error_threshold=1.35,
        color_tolerance=7.0,
    )
    for stroke in final_ink_strokes:
        draw_stroke_on_simulation(simulation_rgb, stroke)
    identity_ink_strokes = build_identity_ink_strokes(final_ink_strokes)
    for stroke in identity_ink_strokes:
        draw_stroke_on_simulation(simulation_rgb, stroke)

    all_strokes = [dict(stroke) for stroke in base_plan["strokes"]]
    all_strokes.extend(refinement_strokes)
    all_strokes.extend(subject_detail_strokes)
    all_strokes.extend(identity_detail_strokes)
    all_strokes.extend(final_ink_strokes)
    all_strokes.extend(identity_ink_strokes)
    final_preview_rgb = np.clip(simulation_rgb, 0, 255).astype(np.uint8)
    final_ssim = float(
        structural_similarity(
            target_rgb.astype(np.uint8),
            final_preview_rgb,
            channel_axis=2,
            data_range=255,
        )
    )
    normalized_rmse = float(np.sqrt(np.mean((target_rgb - final_preview_rgb.astype(np.float32)) ** 2))) / 255.0

    plan = {
        "version": 62,
        "outputArtboardSizePx": ARTBOARD_SIZE_PX,
        "paperColor": base_plan["paperColor"],
        "palette": base_plan["palette"],
        "focusRegions": base_plan["focusRegions"],
        "strokes": all_strokes,
        "stats": {
            "total": len(all_strokes),
            "phaseCounts": dict(Counter(stroke["phase"] for stroke in all_strokes)),
            "baseStrokeCount": len(base_plan["strokes"]),
            "refinementStrokeCount": len(refinement_strokes),
            "subjectDetailStrokeCount": len(subject_detail_strokes),
            "identityDetailStrokeCount": len(identity_detail_strokes),
            "finalInkStrokeCount": len(final_ink_strokes),
            "identityInkStrokeCount": len(identity_ink_strokes),
            "fillRegionCount": fill_region_count,
            "previewSsim": round(final_ssim, 6),
            "previewNormalizedRmse": round(normalized_rmse, 6),
        },
    }
    compact_plan_json = json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
    PLAN_JSON_PATH.write_text(compact_plan_json, encoding="utf-8")
    PLAN_JAVASCRIPT_PATH.write_text(f"window.MARKER_PAINT_PLAN_H62={compact_plan_json};\n", encoding="utf-8")
    Image.fromarray(final_preview_rgb).save(PREVIEW_PATH)

    metrics = {
        "planVersion": 62,
        "strokeCount": len(all_strokes),
        "phaseCounts": plan["stats"]["phaseCounts"],
        "passMetrics": [*pass_metrics, subject_detail_metrics, identity_detail_metrics],
        "finalPreviewSsim": round(final_ssim, 6),
        "finalPreviewNormalizedRmse": round(normalized_rmse, 6),
        "runtimeTargetCompositing": False,
        "wholeCanvasScanStrokes": 0,
        "fillRegionCount": fill_region_count,
        "rendererInternalScale": 2,
    }
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
