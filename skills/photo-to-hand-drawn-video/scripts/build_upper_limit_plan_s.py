#!/usr/bin/env python3
"""Build the S upper-bound plan with stroke-only residual optimization.

The H6.2 plan supplies structurally coherent strokes. S keeps every one of
those strokes, then searches continuous curve variations that reduce the
remaining image-space error. Every accepted correction remains a finite-width
polyline rendered by the same browser brush engine; no raster target layer is
embedded in the plan or composited at runtime.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from build_anime_refinement_plan_h62 import draw_stroke_on_simulation, simulate_plan
from build_flat_marker_plan_v7 import to_hex_color


PROJECT_DIRECTORY = Path(__file__).resolve().parent
TARGET_IMAGE_PATH = PROJECT_DIRECTORY / "assets" / "marker-target-h6-seaside.png"
BASE_PLAN_PATH = PROJECT_DIRECTORY / "assets" / "marker-paint-plan-h62.json"
BASE_PREVIEW_PATH = PROJECT_DIRECTORY / "verification-v7" / "h62-plan-preview.png"
PLAN_JSON_PATH = PROJECT_DIRECTORY / "assets" / "marker-paint-plan-s.json"
PLAN_JAVASCRIPT_PATH = PROJECT_DIRECTORY / "assets" / "marker-paint-plan-s.js"
VERIFICATION_DIRECTORY = PROJECT_DIRECTORY / "verification-s"
PREVIEW_PATH = VERIFICATION_DIRECTORY / "s-plan-preview.png"
METRICS_PATH = VERIFICATION_DIRECTORY / "s-plan-metrics.json"

ARTBOARD_SIZE_PX = 960
RANDOM_SEED = 20260719
MINIMUM_STROKE_WIDTH_PX = 0.55
MINIMUM_STROKE_LENGTH_PX = 4.5
FACE_CORE_DILATION_SIZE_PX = 25
PRIMARY_SCENE_CONTEXT_PADDING_PX = 96

DRAWING_STAGE_ORDER = (
    "subject_structure",
    "subject_blocking",
    "primary_scene_structure",
    "primary_scene_blocking",
    "primary_scene_refinement",
    "subject_refinement",
    "subject_identity",
    "background_structure",
    "background_blocking",
    "background_suggestion",
    "final_subject_correction",
    "person_clothes_polish",
    "person_skin_polish",
    "person_contour_reinforcement",
    "person_hair_polish",
    "face_skin_polish",
    "face_feature_structure",
)

FACE_FEATURE_PATHS = {
    "leftEyebrow": (70, 63, 105, 66, 107),
    "rightEyebrow": (336, 296, 334, 293, 300),
    "leftEye": (33, 160, 158, 133, 153, 144, 33),
    "rightEye": (362, 385, 387, 263, 373, 380, 362),
    "noseBridge": (168, 6, 197, 195, 5, 4),
    "noseBase": (98, 2, 327),
    "outerLips": (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291),
}

BACKGROUND_PHASE_RETENTION = {
    "structure_line": 0.42,
    "region_fill": 0.82,
    "region_glaze": 0.65,
    "refine_coat": 0.42,
    "hair_detail": 0.24,
    "detail_line": 0.22,
    "focus_detail": 0.18,
    "face_detail": 0.10,
    "anime_refine_broad": 0.62,
    "anime_refine_medium": 0.42,
    "anime_refine_detail": 0.20,
    "anime_refine_micro": 0.08,
    "anime_refine_ultra": 0.04,
    "anime_final_ink": 0.28,
    "s_structure_optimization": 0.24,
    "s_surface_optimization": 0.14,
}

FACE_BOUNDS = (0.405, 0.095, 0.710, 0.470)
SUBJECT_BOUNDS = (0.235, 0.035, 0.900, 1.000)


@dataclass(frozen=True)
class OptimizationStage:
    phase: str
    width_px: float
    length_px: float
    maximum_strokes: int
    minimum_error: float
    edge_weight: float
    region: str
    alpha: float
    batch_size: int = 240
    minimum_stage_gain: float = 0.00018


OPTIMIZATION_STAGES = (
    OptimizationStage(
        phase="s_structure_optimization",
        width_px=1.20,
        length_px=13.0,
        maximum_strokes=10000,
        minimum_error=3.8,
        edge_weight=1.35,
        region="whole",
        alpha=0.94,
    ),
    OptimizationStage(
        phase="s_surface_optimization",
        width_px=0.90,
        length_px=9.5,
        maximum_strokes=5200,
        minimum_error=2.6,
        edge_weight=0.65,
        region="whole",
        alpha=0.92,
    ),
    OptimizationStage(
        phase="s_subject_optimization",
        width_px=0.70,
        length_px=7.2,
        maximum_strokes=6200,
        minimum_error=1.8,
        edge_weight=1.15,
        region="subject",
        alpha=0.90,
    ),
    OptimizationStage(
        phase="s_visual_anchor_optimization",
        width_px=0.65,
        length_px=6.2,
        maximum_strokes=7000,
        minimum_error=1.45,
        edge_weight=1.45,
        region="visual_anchor",
        alpha=0.90,
        batch_size=210,
        minimum_stage_gain=0.00012,
    ),
    OptimizationStage(
        phase="s_identity_optimization",
        width_px=0.55,
        length_px=5.2,
        maximum_strokes=4600,
        minimum_error=1.15,
        edge_weight=1.80,
        region="identity",
        alpha=0.88,
        batch_size=180,
        minimum_stage_gain=0.00010,
    ),
    OptimizationStage(
        phase="s_face_core_optimization",
        width_px=0.72,
        length_px=5.2,
        maximum_strokes=1600,
        minimum_error=1.10,
        edge_weight=2.30,
        region="face_core",
        alpha=0.90,
        batch_size=160,
        minimum_stage_gain=0.00007,
    ),
)

FACE_SKIN_POLISH_STAGE = OptimizationStage(
    phase="face_skin_polish",
    width_px=1.45,
    length_px=7.0,
    maximum_strokes=1800,
    minimum_error=1.65,
    edge_weight=0.35,
    region="face_skin",
    alpha=0.94,
    batch_size=150,
    minimum_stage_gain=0.00008,
)

PERSON_POLISH_STAGES = (
    OptimizationStage(
        phase="person_clothes_polish",
        width_px=0.90,
        length_px=8.0,
        maximum_strokes=3600,
        minimum_error=1.35,
        edge_weight=0.95,
        region="clothes",
        alpha=0.92,
        batch_size=180,
        minimum_stage_gain=0.00008,
    ),
    OptimizationStage(
        phase="person_skin_polish",
        width_px=1.25,
        length_px=7.4,
        maximum_strokes=2200,
        minimum_error=1.40,
        edge_weight=0.45,
        region="body_skin_without_face",
        alpha=0.93,
        batch_size=160,
        minimum_stage_gain=0.00007,
    ),
    OptimizationStage(
        phase="person_contour_reinforcement",
        width_px=0.76,
        length_px=9.0,
        maximum_strokes=2200,
        minimum_error=1.25,
        edge_weight=2.00,
        region="subject_contour",
        alpha=0.91,
        batch_size=160,
        minimum_stage_gain=0.00007,
    ),
    OptimizationStage(
        phase="person_hair_polish",
        width_px=0.72,
        length_px=8.8,
        maximum_strokes=2800,
        minimum_error=1.20,
        edge_weight=1.65,
        region="hair",
        alpha=0.90,
        batch_size=170,
        minimum_stage_gain=0.00007,
    ),
)

PERSON_POLISH_TARGET_SIGMA = {
    "person_clothes_polish": 0.45,
    "person_skin_polish": 0.95,
    "person_contour_reinforcement": 0.0,
    "person_hair_polish": 0.25,
}
PERSON_POLISH_PHASES = frozenset(stage.phase for stage in PERSON_POLISH_STAGES)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add continuous stroke-only residual optimization to an S base plan."
    )
    parser.add_argument("--target", type=Path, default=TARGET_IMAGE_PATH)
    parser.add_argument("--base-plan", type=Path, default=BASE_PLAN_PATH)
    parser.add_argument("--base-preview", type=Path, default=BASE_PREVIEW_PATH)
    parser.add_argument("--plan-json", type=Path, default=PLAN_JSON_PATH)
    parser.add_argument("--plan-js", type=Path, default=PLAN_JAVASCRIPT_PATH)
    parser.add_argument("--preview", type=Path, default=PREVIEW_PATH)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    parser.add_argument("--semantic-report", type=Path)
    parser.add_argument("--budget-scale", type=float, default=1.0)
    parser.add_argument("--javascript-global", default="")
    return parser.parse_args()


def normalized_bounds_to_slices(bounds: tuple[float, float, float, float]) -> tuple[slice, slice]:
    left, top, right, bottom = bounds
    return (
        slice(int(round(top * ARTBOARD_SIZE_PX)), int(round(bottom * ARTBOARD_SIZE_PX))),
        slice(int(round(left * ARTBOARD_SIZE_PX)), int(round(right * ARTBOARD_SIZE_PX))),
    )


def calculate_ssim(target_rgb: np.ndarray, canvas_rgb: np.ndarray) -> float:
    return float(
        structural_similarity(
            target_rgb.astype(np.uint8),
            np.clip(canvas_rgb, 0, 255).astype(np.uint8),
            channel_axis=2,
            data_range=255,
        )
    )


def calculate_region_ssim(
    target_rgb: np.ndarray,
    canvas_rgb: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> float:
    row_slice, column_slice = normalized_bounds_to_slices(bounds)
    return calculate_ssim(target_rgb[row_slice, column_slice], canvas_rgb[row_slice, column_slice])


def calculate_masked_normalized_rmse(
    target_rgb: np.ndarray,
    canvas_rgb: np.ndarray,
    region_mask: np.ndarray,
) -> float:
    if not np.any(region_mask):
        return 0.0
    region_difference = target_rgb[region_mask] - canvas_rgb[region_mask]
    return float(np.sqrt(np.mean(region_difference**2)) / 255.0)


def build_region_mask(
    region: str,
    semantic_region_masks: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    if semantic_region_masks and region in semantic_region_masks:
        return semantic_region_masks[region]
    mask = np.zeros((ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX), dtype=np.uint8)
    if region == "whole":
        mask.fill(255)
    elif region == "subject":
        row_slice, column_slice = normalized_bounds_to_slices(SUBJECT_BOUNDS)
        mask[row_slice, column_slice] = 255
    elif region in {"face", "identity", "face_core"}:
        row_slice, column_slice = normalized_bounds_to_slices(FACE_BOUNDS)
        mask[row_slice, column_slice] = 255
    elif region == "visual_anchor":
        row_slice, column_slice = normalized_bounds_to_slices(SUBJECT_BOUNDS)
        mask[row_slice, column_slice] = 255
    else:
        raise ValueError(f"unsupported optimization region: {region}")
    return mask.astype(bool)


def build_orientation_fields(target_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grayscale = cv2.cvtColor(target_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    smoothed = cv2.GaussianBlur(grayscale, (0, 0), 0.75)
    gradient_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength = np.hypot(gradient_x, gradient_y)
    tangent_x = -gradient_y
    tangent_y = gradient_x
    tangent_norm = np.maximum(np.hypot(tangent_x, tangent_y), 1e-5)
    return tangent_x / tangent_norm, tangent_y / tangent_norm, edge_strength


def build_curve_points(
    seed_x: int,
    seed_y: int,
    tangent_x: float,
    tangent_y: float,
    length_px: float,
    angle_offset_radians: float,
    length_scale: float,
    curvature_sign: float,
) -> np.ndarray:
    cosine = math.cos(angle_offset_radians)
    sine = math.sin(angle_offset_radians)
    direction_x = tangent_x * cosine - tangent_y * sine
    direction_y = tangent_x * sine + tangent_y * cosine
    if abs(direction_x) + abs(direction_y) < 1e-5:
        direction_x, direction_y = 1.0, 0.0
    normal_x, normal_y = -direction_y, direction_x
    half_length = max(MINIMUM_STROKE_LENGTH_PX, length_px * length_scale) * 0.5
    curvature_px = curvature_sign * min(1.4, half_length * 0.16)
    start = np.array([seed_x - direction_x * half_length, seed_y - direction_y * half_length])
    control = np.array([seed_x + normal_x * curvature_px, seed_y + normal_y * curvature_px])
    end = np.array([seed_x + direction_x * half_length, seed_y + direction_y * half_length])
    samples = np.linspace(0.0, 1.0, 7, dtype=np.float32)[:, None]
    points = (1 - samples) ** 2 * start + 2 * (1 - samples) * samples * control + samples**2 * end
    points[:, 0] = np.clip(points[:, 0], 0, ARTBOARD_SIZE_PX - 1)
    points[:, 1] = np.clip(points[:, 1], 0, ARTBOARD_SIZE_PX - 1)
    return points.astype(np.float32)


def sample_target_color(target_rgb: np.ndarray, curve_points: np.ndarray) -> np.ndarray:
    rounded_points = np.rint(curve_points).astype(np.int32)
    sampled_colors = target_rgb[rounded_points[:, 1], rounded_points[:, 0]]
    return np.median(sampled_colors, axis=0).astype(np.float32)


def curve_patch_bounds(curve_points: np.ndarray, width_px: float) -> tuple[int, int, int, int]:
    padding_px = max(3, int(math.ceil(width_px * 2.5)))
    left = max(0, int(math.floor(float(curve_points[:, 0].min()))) - padding_px)
    top = max(0, int(math.floor(float(curve_points[:, 1].min()))) - padding_px)
    right = min(ARTBOARD_SIZE_PX, int(math.ceil(float(curve_points[:, 0].max()))) + padding_px + 1)
    bottom = min(ARTBOARD_SIZE_PX, int(math.ceil(float(curve_points[:, 1].max()))) + padding_px + 1)
    return left, top, right, bottom


def render_curve_candidate(
    canvas_rgb: np.ndarray,
    target_rgb: np.ndarray,
    curve_points: np.ndarray,
    stroke_color_rgb: np.ndarray,
    width_px: float,
    alpha: float,
) -> tuple[float, tuple[int, int, int, int], np.ndarray]:
    left, top, right, bottom = curve_patch_bounds(curve_points, width_px)
    local_points = np.rint(curve_points - np.array([left, top], dtype=np.float32)).astype(np.int32)
    mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
    cv2.polylines(
        mask,
        [local_points.reshape(-1, 1, 2)],
        False,
        255,
        max(1, int(round(width_px))),
        cv2.LINE_AA,
    )
    alpha_mask = mask.astype(np.float32)[:, :, None] / 255.0 * alpha
    canvas_patch = canvas_rgb[top:bottom, left:right]
    target_patch = target_rgb[top:bottom, left:right]
    candidate_patch = canvas_patch * (1.0 - alpha_mask) + stroke_color_rgb[None, None, :] * alpha_mask
    influence = np.maximum(alpha_mask, 1e-4)
    before_loss = float(np.sum(np.abs(target_patch - canvas_patch) * influence))
    after_loss = float(np.sum(np.abs(target_patch - candidate_patch) * influence))
    return before_loss - after_loss, (left, top, right, bottom), candidate_patch


def optimize_curve_at_seed(
    canvas_rgb: np.ndarray,
    target_rgb: np.ndarray,
    seed_x: int,
    seed_y: int,
    tangent_x: float,
    tangent_y: float,
    stage: OptimizationStage,
    random_generator: np.random.Generator,
) -> tuple[dict[str, Any] | None, tuple[int, int, int, int] | None, np.ndarray | None, np.ndarray | None]:
    best_improvement = 0.0
    best_curve_points = None
    best_patch_bounds = None
    best_candidate_patch = None
    best_color_rgb = None
    angle_offsets = np.deg2rad((-24.0, -12.0, 0.0, 12.0, 24.0))
    length_scales = (0.78, 1.0, 1.24)
    curvature_signs = (-1.0, 0.0, 1.0)

    for angle_offset in angle_offsets:
        for length_scale in length_scales:
            curvature_sign = curvature_signs[int(random_generator.integers(0, len(curvature_signs)))]
            curve_points = build_curve_points(
                seed_x,
                seed_y,
                tangent_x,
                tangent_y,
                stage.length_px,
                float(angle_offset),
                length_scale,
                curvature_sign,
            )
            stroke_color_rgb = sample_target_color(target_rgb, curve_points)
            improvement, patch_bounds, candidate_patch = render_curve_candidate(
                canvas_rgb,
                target_rgb,
                curve_points,
                stroke_color_rgb,
                stage.width_px,
                stage.alpha,
            )
            if improvement > best_improvement:
                best_improvement = improvement
                best_curve_points = curve_points
                best_patch_bounds = patch_bounds
                best_candidate_patch = candidate_patch
                best_color_rgb = stroke_color_rgb

    if best_curve_points is None or best_improvement <= 0.25:
        return None, None, None, None

    normalized_points = np.round(best_curve_points / ARTBOARD_SIZE_PX, 6).tolist()
    color_hex = "#" + "".join(f"{int(round(channel)):02x}" for channel in np.clip(best_color_rgb, 0, 255))
    stroke = {
        "phase": stage.phase,
        "points": normalized_points,
        "color": color_hex,
        "width": round(max(MINIMUM_STROKE_WIDTH_PX, stage.width_px) / ARTBOARD_SIZE_PX, 7),
        "alpha": stage.alpha,
        "blend": "source-over",
    }
    return stroke, best_patch_bounds, best_candidate_patch, best_curve_points


def optimize_stage(
    canvas_rgb: np.ndarray,
    target_rgb: np.ndarray,
    tangent_x: np.ndarray,
    tangent_y: np.ndarray,
    edge_strength: np.ndarray,
    stage: OptimizationStage,
    random_generator: np.random.Generator,
    semantic_region_masks: dict[str, np.ndarray] | None = None,
    face_bounds: tuple[float, float, float, float] = FACE_BOUNDS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    region_mask = build_region_mask(stage.region, semantic_region_masks)
    stage_start_ssim = calculate_ssim(target_rgb, canvas_rgb)
    stage_start_face_ssim = calculate_region_ssim(target_rgb, canvas_rgb, face_bounds)
    if not np.any(region_mask):
        return [], {
            "phase": stage.phase,
            "region": stage.region,
            "strokeWidthPx": stage.width_px,
            "strokeLengthPx": stage.length_px,
            "strokeCount": 0,
            "startSsim": round(stage_start_ssim, 6),
            "endSsim": round(stage_start_ssim, 6),
            "ssimGain": 0.0,
            "startFaceSsim": round(stage_start_face_ssim, 6),
            "endFaceSsim": round(stage_start_face_ssim, 6),
            "faceSsimGain": 0.0,
            "startRegionNormalizedRmse": 0.0,
            "endRegionNormalizedRmse": 0.0,
            "regionNormalizedRmseReduction": 0.0,
            "lastThreeBatchGain": 0.0,
            "skippedEmptyRegion": True,
        }
    stage_start_region_rmse = calculate_masked_normalized_rmse(
        target_rgb,
        canvas_rgb,
        region_mask,
    )
    accepted_strokes: list[dict[str, Any]] = []
    batch_gains: list[float] = []
    edge_scale = max(float(np.percentile(edge_strength[region_mask], 98.0)), 1e-5)

    while len(accepted_strokes) < stage.maximum_strokes:
        error_map = np.mean(np.abs(target_rgb - canvas_rgb), axis=2)
        normalized_edge = np.clip(edge_strength / edge_scale, 0.0, 1.0)
        score_map = np.where(
            region_mask & (error_map >= stage.minimum_error),
            error_map * (1.0 + normalized_edge * stage.edge_weight),
            0.0,
        )
        local_maximum = score_map >= cv2.dilate(score_map, np.ones((3, 3), dtype=np.uint8))
        candidate_indices = np.flatnonzero(local_maximum & (score_map > 0))
        if candidate_indices.size == 0:
            break
        candidate_scores = score_map.reshape(-1)[candidate_indices]
        selected_count = min(stage.batch_size * 3, candidate_indices.size)
        selected_order = np.argpartition(candidate_scores, -selected_count)[-selected_count:]
        selected_indices = candidate_indices[selected_order]
        selected_indices = selected_indices[np.argsort(score_map.reshape(-1)[selected_indices])[::-1]]
        accepted_before_batch = len(accepted_strokes)
        batch_start_ssim = calculate_ssim(target_rgb, canvas_rgb)
        batch_suppression = np.zeros(score_map.shape, dtype=np.uint8)

        for flat_index in selected_indices:
            if len(accepted_strokes) >= stage.maximum_strokes:
                break
            seed_y, seed_x = np.unravel_index(int(flat_index), score_map.shape)
            if batch_suppression[seed_y, seed_x] or error_map[seed_y, seed_x] < stage.minimum_error:
                continue
            stroke, patch_bounds, candidate_patch, curve_points = optimize_curve_at_seed(
                canvas_rgb,
                target_rgb,
                seed_x,
                seed_y,
                float(tangent_x[seed_y, seed_x]),
                float(tangent_y[seed_y, seed_x]),
                stage,
                random_generator,
            )
            if stroke is None:
                continue
            left, top, right, bottom = patch_bounds
            canvas_rgb[top:bottom, left:right] = candidate_patch
            accepted_strokes.append(stroke)
            cv2.polylines(
                batch_suppression,
                [np.rint(curve_points).astype(np.int32).reshape(-1, 1, 2)],
                False,
                255,
                max(2, int(round(stage.width_px * 2.4))),
                cv2.LINE_AA,
            )
            if len(accepted_strokes) - accepted_before_batch >= stage.batch_size:
                break

        batch_end_ssim = calculate_ssim(target_rgb, canvas_rgb)
        batch_gain = batch_end_ssim - batch_start_ssim
        batch_gains.append(batch_gain)
        accepted_in_batch = len(accepted_strokes) - accepted_before_batch
        print(
            f"{stage.phase}: {len(accepted_strokes)}/{stage.maximum_strokes} strokes, "
            f"batch +{batch_gain:.6f}, SSIM {batch_end_ssim:.6f}",
            flush=True,
        )
        if accepted_in_batch == 0 or (len(batch_gains) >= 3 and sum(batch_gains[-3:]) < stage.minimum_stage_gain):
            break

    stage_end_ssim = calculate_ssim(target_rgb, canvas_rgb)
    stage_end_face_ssim = calculate_region_ssim(target_rgb, canvas_rgb, face_bounds)
    stage_end_region_rmse = calculate_masked_normalized_rmse(
        target_rgb,
        canvas_rgb,
        region_mask,
    )
    metrics = {
        "phase": stage.phase,
        "region": stage.region,
        "strokeWidthPx": stage.width_px,
        "strokeLengthPx": stage.length_px,
        "strokeCount": len(accepted_strokes),
        "startSsim": round(stage_start_ssim, 6),
        "endSsim": round(stage_end_ssim, 6),
        "ssimGain": round(stage_end_ssim - stage_start_ssim, 6),
        "startFaceSsim": round(stage_start_face_ssim, 6),
        "endFaceSsim": round(stage_end_face_ssim, 6),
        "faceSsimGain": round(stage_end_face_ssim - stage_start_face_ssim, 6),
        "startRegionNormalizedRmse": round(stage_start_region_rmse, 6),
        "endRegionNormalizedRmse": round(stage_end_region_rmse, 6),
        "regionNormalizedRmseReduction": round(
            stage_start_region_rmse - stage_end_region_rmse,
            6,
        ),
        "lastThreeBatchGain": round(sum(batch_gains[-3:]), 6),
    }
    return accepted_strokes, metrics


def resolve_project_path(path_text: str) -> Path:
    candidate = Path(path_text)
    return candidate if candidate.is_absolute() else PROJECT_DIRECTORY / candidate


def load_resized_mask(path: Path) -> np.ndarray:
    mask_image = Image.open(path).convert("L").resize(
        (ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX), Image.Resampling.NEAREST
    )
    return np.asarray(mask_image) >= 128


def build_primary_scene_mask(
    target_rgb: np.ndarray,
    subject_mask: np.ndarray,
    visual_anchor_mask: np.ndarray,
) -> np.ndarray:
    rows, columns = np.nonzero(subject_mask)
    if len(columns) == 0:
        return visual_anchor_mask.copy()
    context_window = np.zeros(subject_mask.shape, dtype=bool)
    left = max(0, int(columns.min()) - PRIMARY_SCENE_CONTEXT_PADDING_PX)
    top = max(0, int(rows.min()) - PRIMARY_SCENE_CONTEXT_PADDING_PX)
    right = min(ARTBOARD_SIZE_PX, int(columns.max()) + PRIMARY_SCENE_CONTEXT_PADDING_PX + 1)
    bottom = min(ARTBOARD_SIZE_PX, int(rows.max()) + PRIMARY_SCENE_CONTEXT_PADDING_PX + 1)
    context_window[top:bottom, left:right] = True

    grayscale = cv2.cvtColor(target_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    scene_edges = cv2.Canny(grayscale, 55, 135)
    connected_edges = cv2.dilate(
        scene_edges,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
        (connected_edges > 0).astype(np.uint8),
        connectivity=8,
    )
    subject_neighborhood = cv2.dilate(
        subject_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 91)),
    ).astype(bool)
    connected_scene_edges = np.zeros(subject_mask.shape, dtype=bool)
    for component_label in range(1, component_count):
        component_mask = component_labels == component_label
        component_area = int(component_stats[component_label, cv2.CC_STAT_AREA])
        if component_area < 18 or not np.any(component_mask & subject_neighborhood):
            continue
        connected_scene_edges |= component_mask & context_window
    connected_scene_context = cv2.dilate(
        connected_scene_edges.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
    ).astype(bool)
    return subject_mask | visual_anchor_mask | connected_scene_context


def mask_bounds_normalized(mask: np.ndarray) -> tuple[float, float, float, float]:
    rows, columns = np.nonzero(mask)
    if len(columns) == 0:
        raise ValueError("Semantic identity mask is empty.")
    left = max(0, int(columns.min()) - 4)
    top = max(0, int(rows.min()) - 4)
    right = min(ARTBOARD_SIZE_PX, int(columns.max()) + 5)
    bottom = min(ARTBOARD_SIZE_PX, int(rows.max()) + 5)
    return (
        left / ARTBOARD_SIZE_PX,
        top / ARTBOARD_SIZE_PX,
        right / ARTBOARD_SIZE_PX,
        bottom / ARTBOARD_SIZE_PX,
    )


def build_face_feature_protection_mask(
    target_rgb: np.ndarray,
    face_core_mask: np.ndarray,
    face_landmarks: list[dict[str, float]],
) -> np.ndarray:
    """Protect identity-bearing contours while polishing broad facial skin planes."""
    protection_mask = np.zeros(face_core_mask.shape, dtype=np.uint8)
    if len(face_landmarks) >= 468:
        face_rows, face_columns = np.nonzero(face_core_mask)
        face_width_px = (
            int(face_columns.max() - face_columns.min() + 1)
            if len(face_columns)
            else 96
        )
        feature_line_width_px = int(np.clip(round(face_width_px * 0.032), 5, 10))
        for landmark_indices in FACE_FEATURE_PATHS.values():
            points_px = np.asarray(
                [
                    [
                        face_landmarks[index]["x"] * ARTBOARD_SIZE_PX,
                        face_landmarks[index]["y"] * ARTBOARD_SIZE_PX,
                    ]
                    for index in landmark_indices
                ],
                dtype=np.float32,
            )
            points_px = np.clip(
                np.rint(points_px).astype(np.int32),
                0,
                ARTBOARD_SIZE_PX - 1,
            )
            cv2.polylines(
                protection_mask,
                [points_px.reshape((-1, 1, 2))],
                False,
                255,
                feature_line_width_px,
                cv2.LINE_AA,
            )

    luminance = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY)
    dark_identity_contours = face_core_mask & (luminance < 92)
    dark_identity_contours = cv2.dilate(
        dark_identity_contours.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    protection_mask = np.maximum(protection_mask, dark_identity_contours * 255)
    return (protection_mask > 0) & face_core_mask


def load_semantic_regions(
    semantic_report_path: Path | None,
    target_rgb: np.ndarray,
) -> tuple[
    dict[str, np.ndarray] | None,
    tuple[float, float, float, float],
    list[dict[str, float]],
]:
    if semantic_report_path is None:
        return None, FACE_BOUNDS, []
    semantic_report = json.loads(semantic_report_path.read_text(encoding="utf-8"))
    if semantic_report.get("status") != "PASS" or not semantic_report.get("target"):
        raise RuntimeError("S optimization requires a PASS semantic report with a target.")
    target_mask_files = semantic_report["target"]["maskFiles"]
    subject_mask = load_resized_mask(resolve_project_path(target_mask_files["subject"]))
    identity_mask = load_resized_mask(resolve_project_path(target_mask_files["identity"]))
    face_hull_mask = load_resized_mask(resolve_project_path(target_mask_files["face_hull"]))
    accessory_mask = load_resized_mask(resolve_project_path(target_mask_files["accessories"]))
    body_skin_mask = load_resized_mask(resolve_project_path(target_mask_files["body_skin"]))
    face_skin_mask = load_resized_mask(resolve_project_path(target_mask_files["face_skin"]))
    clothes_mask = load_resized_mask(resolve_project_path(target_mask_files["clothes"]))
    hair_mask = load_resized_mask(resolve_project_path(target_mask_files["hair"]))
    face_landmarks = semantic_report["target"].get("faceLandmarks", [])
    face_core_mask = cv2.dilate(
        face_hull_mask.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (FACE_CORE_DILATION_SIZE_PX, FACE_CORE_DILATION_SIZE_PX),
        ),
    ).astype(bool)
    visual_anchor_mask = identity_mask | accessory_mask | body_skin_mask | clothes_mask
    subject_priority_mask = cv2.dilate(
        (subject_mask | visual_anchor_mask).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
    ).astype(bool)
    primary_scene_mask = build_primary_scene_mask(
        target_rgb,
        subject_mask,
        visual_anchor_mask,
    )
    protected_primary_mask = cv2.dilate(
        primary_scene_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ).astype(bool)
    face_feature_protection_mask = build_face_feature_protection_mask(
        target_rgb,
        face_core_mask,
        face_landmarks,
    )
    subject_contour_mask = cv2.dilate(
        subject_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ).astype(bool) & ~cv2.erode(
        subject_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ).astype(bool)
    semantic_regions = {
        "whole": np.ones((ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX), dtype=bool),
        "subject": subject_mask,
        "subject_priority": subject_priority_mask,
        "face": identity_mask,
        "identity": identity_mask,
        "face_core": face_core_mask,
        "face_skin": face_skin_mask & face_core_mask & ~face_feature_protection_mask,
        "body_skin_without_face": body_skin_mask & ~face_core_mask,
        "clothes": clothes_mask,
        "hair": hair_mask,
        "subject_contour": subject_contour_mask,
        "visual_anchor": visual_anchor_mask,
        "primary_scene": primary_scene_mask,
        "background": ~protected_primary_mask,
    }
    return semantic_regions, mask_bounds_normalized(face_core_mask), face_landmarks


def sample_stroke_points_px(stroke: dict[str, Any]) -> np.ndarray:
    control_points = np.asarray(stroke["points"], dtype=np.float32) * ARTBOARD_SIZE_PX
    if len(control_points) < 2:
        return np.rint(control_points).astype(np.int32)
    sampled_points: list[np.ndarray] = []
    for start_point, end_point in zip(control_points[:-1], control_points[1:]):
        segment_length = float(np.linalg.norm(end_point - start_point))
        sample_count = max(2, min(10, int(math.ceil(segment_length / 6.0)) + 1))
        sampled_points.extend(
            start_point + (end_point - start_point) * interpolation
            for interpolation in np.linspace(0.0, 1.0, sample_count, endpoint=False)
        )
    sampled_points.append(control_points[-1])
    return np.clip(
        np.rint(np.asarray(sampled_points)).astype(np.int32),
        0,
        ARTBOARD_SIZE_PX - 1,
    )


def stroke_primary_overlap(stroke: dict[str, Any], primary_scene_mask: np.ndarray) -> float:
    sampled_points = sample_stroke_points_px(stroke)
    if sampled_points.size == 0:
        return 0.0
    return float(np.mean(primary_scene_mask[sampled_points[:, 1], sampled_points[:, 0]]))


def deterministic_retention_score(stroke_index: int, phase: str) -> float:
    phase_code = sum((character_index + 1) * ord(character) for character_index, character in enumerate(phase))
    bucket = ((stroke_index + 1) * 2654435761 + phase_code * 2246822519) % 10000
    return bucket / 10000.0


def drawing_stage_for_stroke(
    stroke: dict[str, Any],
    is_subject: bool,
    is_primary_scene: bool,
) -> str:
    phase = stroke["phase"]
    if phase in PERSON_POLISH_PHASES:
        return phase
    if phase == "face_skin_polish":
        return "face_skin_polish"
    if phase == "face_feature_structure":
        return "face_feature_structure"
    if is_subject:
        if phase == "structure_line":
            return "subject_structure"
        if phase in {
            "region_fill",
            "region_glaze",
            "refine_coat",
            "anime_refine_broad",
            "anime_refine_medium",
        }:
            return "subject_blocking"
        if phase in {
            "face_detail",
            "anime_identity_detail",
            "anime_identity_ink",
            "s_identity_optimization",
            "s_face_core_optimization",
        }:
            return "subject_identity"
        if phase in {"s_subject_optimization", "s_visual_anchor_optimization"}:
            return "final_subject_correction"
        return "subject_refinement"
    if is_primary_scene:
        if phase in {"structure_line", "anime_final_ink"}:
            return "primary_scene_structure"
        if phase in {
            "region_fill",
            "region_glaze",
            "refine_coat",
            "anime_refine_broad",
            "anime_refine_medium",
        }:
            return "primary_scene_blocking"
        return "primary_scene_refinement"
    if phase in {"structure_line", "anime_final_ink"}:
        return "background_structure"
    if phase in {
        "region_fill",
        "region_glaze",
        "refine_coat",
        "anime_refine_broad",
        "anime_refine_medium",
    }:
        return "background_blocking"
    return "background_suggestion"


def prepare_semantic_drawing_strokes(
    strokes: list[dict[str, Any]],
    subject_mask: np.ndarray,
    primary_scene_mask: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    staged_strokes: list[dict[str, Any]] = []
    discarded_background_strokes = 0
    for stroke_index, source_stroke in enumerate(strokes):
        stroke = dict(source_stroke)
        subject_overlap = stroke_primary_overlap(stroke, subject_mask)
        primary_scene_overlap = stroke_primary_overlap(stroke, primary_scene_mask)
        is_subject = subject_overlap >= 0.22
        is_primary_scene = primary_scene_overlap >= 0.22
        drawing_stage = drawing_stage_for_stroke(
            stroke,
            is_subject,
            is_primary_scene,
        )
        if drawing_stage.startswith("background_"):
            retention_rate = BACKGROUND_PHASE_RETENTION.get(stroke["phase"], 0.16)
            if deterministic_retention_score(stroke_index, stroke["phase"]) > retention_rate:
                discarded_background_strokes += 1
                continue
        stroke["drawingStage"] = drawing_stage
        staged_strokes.append(stroke)
    stage_index = {stage: index for index, stage in enumerate(DRAWING_STAGE_ORDER)}
    staged_strokes.sort(key=lambda stroke: stage_index[stroke["drawingStage"]])
    drawing_stage_counts: dict[str, int] = {}
    for stroke in staged_strokes:
        drawing_stage = stroke["drawingStage"]
        drawing_stage_counts[drawing_stage] = drawing_stage_counts.get(drawing_stage, 0) + 1
    return staged_strokes, drawing_stage_counts, discarded_background_strokes


def build_face_feature_strokes(
    target_rgb: np.ndarray,
    face_landmarks: list[dict[str, float]],
    face_bounds: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    if len(face_landmarks) < 468:
        return []
    face_width_px = (face_bounds[2] - face_bounds[0]) * ARTBOARD_SIZE_PX
    feature_strokes: list[dict[str, Any]] = []
    for feature_name, landmark_indices in FACE_FEATURE_PATHS.items():
        normalized_points = np.asarray(
            [
                [face_landmarks[index]["x"], face_landmarks[index]["y"]]
                for index in landmark_indices
            ],
            dtype=np.float32,
        )
        sampled_points_px = np.clip(
            np.rint(normalized_points * ARTBOARD_SIZE_PX).astype(np.int32),
            0,
            ARTBOARD_SIZE_PX - 1,
        )
        sampled_rgb = np.median(
            target_rgb[sampled_points_px[:, 1], sampled_points_px[:, 0]],
            axis=0,
        )
        if "Eyebrow" in feature_name:
            stroke_rgb = np.clip(sampled_rgb * 0.72, 24, 128)
            alpha = 0.70
            width_px = float(np.clip(face_width_px * 0.008, 0.68, 1.08))
        elif "Eye" in feature_name:
            stroke_rgb = np.clip(sampled_rgb * 0.72, 24, 132)
            alpha = 0.62
            width_px = float(np.clip(face_width_px * 0.007, 0.62, 0.96))
        elif feature_name == "outerLips":
            stroke_rgb = np.clip(sampled_rgb * np.array([0.86, 0.72, 0.72]), 38, 190)
            alpha = 0.66
            width_px = float(np.clip(face_width_px * 0.006, 0.58, 0.90))
        else:
            stroke_rgb = np.clip(sampled_rgb * 0.78, 48, 176)
            alpha = 0.36
            width_px = float(np.clip(face_width_px * 0.0045, 0.48, 0.72))
        feature_strokes.append(
            {
                "phase": "face_feature_structure",
                "drawingStage": "face_feature_structure",
                "feature": feature_name,
                "points": np.round(normalized_points, 6).tolist(),
                "color": to_hex_color(stroke_rgb),
                "width": round(width_px / ARTBOARD_SIZE_PX, 7),
                "alpha": alpha,
                "blend": "source-over",
            }
        )
    return feature_strokes


def main() -> None:
    arguments = parse_arguments()
    if not 0.001 <= arguments.budget_scale <= 2.0:
        raise ValueError("--budget-scale must be between 0.001 and 2.0")
    for output_path in (
        arguments.plan_json,
        arguments.plan_js,
        arguments.preview,
        arguments.metrics,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    target_rgb = np.asarray(
        Image.open(arguments.target).convert("RGB").resize(
            (ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX),
            Image.Resampling.LANCZOS,
        ),
        dtype=np.float32,
    )
    semantic_region_masks, face_bounds, face_landmarks = load_semantic_regions(
        arguments.semantic_report,
        target_rgb,
    )
    canvas_rgb = np.asarray(
        Image.open(arguments.base_preview).convert("RGB").resize(
            (ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX),
            Image.Resampling.LANCZOS,
        ),
        dtype=np.float32,
    ).copy()
    base_plan = json.loads(arguments.base_plan.read_text(encoding="utf-8"))
    tangent_x, tangent_y, edge_strength = build_orientation_fields(target_rgb)
    random_generator = np.random.default_rng(RANDOM_SEED)

    optimized_strokes: list[dict[str, Any]] = []
    stage_metrics: list[dict[str, Any]] = []
    optimization_stages = tuple(
        replace(
            stage,
            maximum_strokes=max(1, int(round(stage.maximum_strokes * arguments.budget_scale))),
            batch_size=max(8, int(round(stage.batch_size * arguments.budget_scale ** 0.5))),
        )
        for stage in OPTIMIZATION_STAGES
    )

    for stage in optimization_stages:
        stage_strokes, metrics = optimize_stage(
            canvas_rgb,
            target_rgb,
            tangent_x,
            tangent_y,
            edge_strength,
            stage,
            random_generator,
            semantic_region_masks=semantic_region_masks,
            face_bounds=face_bounds,
        )
        optimized_strokes.extend(stage_strokes)
        stage_metrics.append(metrics)

    person_polish_stroke_count = 0
    if semantic_region_masks is not None:
        for person_polish_stage_template in PERSON_POLISH_STAGES:
            target_sigma = PERSON_POLISH_TARGET_SIGMA[person_polish_stage_template.phase]
            person_polish_target_rgb = (
                cv2.GaussianBlur(target_rgb, (0, 0), target_sigma)
                if target_sigma > 0
                else target_rgb
            )
            (
                person_polish_tangent_x,
                person_polish_tangent_y,
                person_polish_edge_strength,
            ) = build_orientation_fields(person_polish_target_rgb)
            person_polish_stage = replace(
                person_polish_stage_template,
                maximum_strokes=max(
                    1,
                    int(round(person_polish_stage_template.maximum_strokes * arguments.budget_scale)),
                ),
                batch_size=max(
                    8,
                    int(
                        round(
                            person_polish_stage_template.batch_size
                            * arguments.budget_scale**0.5
                        )
                    ),
                ),
            )
            person_polish_strokes, person_polish_metrics = optimize_stage(
                canvas_rgb,
                person_polish_target_rgb,
                person_polish_tangent_x,
                person_polish_tangent_y,
                person_polish_edge_strength,
                person_polish_stage,
                random_generator,
                semantic_region_masks=semantic_region_masks,
                face_bounds=face_bounds,
            )
            person_polish_metrics["targetMode"] = (
                f"gaussian-sigma-{target_sigma:.2f}" if target_sigma > 0 else "original-target"
            )
            person_polish_stroke_count += len(person_polish_strokes)
            optimized_strokes.extend(person_polish_strokes)
            stage_metrics.append(person_polish_metrics)

    face_feature_strokes: list[dict[str, Any]] = []
    if semantic_region_masks is not None:
        smoothed_face_target_rgb = cv2.GaussianBlur(target_rgb, (0, 0), 1.35)
        polish_tangent_x, polish_tangent_y, polish_edge_strength = build_orientation_fields(
            smoothed_face_target_rgb
        )
        face_skin_polish_stage = replace(
            FACE_SKIN_POLISH_STAGE,
            maximum_strokes=max(
                1,
                int(round(FACE_SKIN_POLISH_STAGE.maximum_strokes * arguments.budget_scale)),
            ),
            batch_size=max(
                8,
                int(round(FACE_SKIN_POLISH_STAGE.batch_size * arguments.budget_scale**0.5)),
            ),
        )
        face_skin_polish_strokes, face_skin_polish_metrics = optimize_stage(
            canvas_rgb,
            smoothed_face_target_rgb,
            polish_tangent_x,
            polish_tangent_y,
            polish_edge_strength,
            face_skin_polish_stage,
            random_generator,
            semantic_region_masks=semantic_region_masks,
            face_bounds=face_bounds,
        )
        face_skin_polish_metrics["targetMode"] = "smoothed-face-skin"
        optimized_strokes.extend(face_skin_polish_strokes)
        stage_metrics.append(face_skin_polish_metrics)
        face_feature_strokes = build_face_feature_strokes(
            target_rgb,
            face_landmarks,
            face_bounds,
        )

    candidate_strokes = [*base_plan["strokes"], *optimized_strokes, *face_feature_strokes]
    discarded_background_strokes = 0
    drawing_stage_counts: dict[str, int] = {}
    retained_base_strokes = list(base_plan["strokes"])
    if semantic_region_masks is not None:
        retained_base_strokes, _, _ = prepare_semantic_drawing_strokes(
            list(base_plan["strokes"]),
            semantic_region_masks["subject_priority"],
            semantic_region_masks["primary_scene"],
        )
        all_strokes, drawing_stage_counts, discarded_background_strokes = (
            prepare_semantic_drawing_strokes(
                candidate_strokes,
                semantic_region_masks["subject_priority"],
                semantic_region_masks["primary_scene"],
            )
        )
    else:
        all_strokes = candidate_strokes
    retained_optimized_stroke_count = max(
        0,
        len(all_strokes) - len(retained_base_strokes) - len(face_feature_strokes),
    )

    baseline_canvas_rgb = simulate_plan(
        {
            "paperColor": base_plan["paperColor"],
            "strokes": retained_base_strokes,
        }
    )
    final_canvas_rgb = simulate_plan(
        {
            "paperColor": base_plan["paperColor"],
            "strokes": all_strokes,
        }
    )
    baseline_ssim = calculate_ssim(target_rgb, baseline_canvas_rgb)
    baseline_face_ssim = calculate_region_ssim(target_rgb, baseline_canvas_rgb, face_bounds)
    final_ssim = calculate_ssim(target_rgb, final_canvas_rgb)
    final_face_ssim = calculate_region_ssim(target_rgb, final_canvas_rgb, face_bounds)
    baseline_person_region_rmse: dict[str, float] = {}
    final_person_region_rmse: dict[str, float] = {}
    if semantic_region_masks is not None:
        for region_name in (
            "subject",
            "clothes",
            "body_skin_without_face",
            "hair",
            "subject_contour",
        ):
            region_mask = semantic_region_masks[region_name]
            baseline_person_region_rmse[region_name] = round(
                calculate_masked_normalized_rmse(target_rgb, baseline_canvas_rgb, region_mask),
                6,
            )
            final_person_region_rmse[region_name] = round(
                calculate_masked_normalized_rmse(target_rgb, final_canvas_rgb, region_mask),
                6,
            )
    final_normalized_rmse = (
        float(np.sqrt(np.mean((target_rgb - final_canvas_rgb) ** 2))) / 255.0
    )
    phase_counts: dict[str, int] = {}
    for stroke in all_strokes:
        phase = stroke["phase"]
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    plan_version = "S-generalized-4" if arguments.semantic_report else "S"
    plan = {
        "version": plan_version,
        "outputArtboardSizePx": ARTBOARD_SIZE_PX,
        "paperColor": base_plan["paperColor"],
        "palette": base_plan["palette"],
        "focusRegions": base_plan["focusRegions"],
        "strokes": all_strokes,
        "stats": {
            "total": len(all_strokes),
            "baseStrokeCount": len(retained_base_strokes),
            "optimizedStrokeCount": retained_optimized_stroke_count,
            "faceFeatureStrokeCount": len(face_feature_strokes),
            "personPolishStrokeCount": person_polish_stroke_count,
            "phaseCounts": phase_counts,
            "drawingStageCounts": drawing_stage_counts,
            "discardedBackgroundStrokeCount": discarded_background_strokes,
            "baselineSsim": round(baseline_ssim, 6),
            "finalPreviewSsim": round(final_ssim, 6),
            "baselineFaceSsim": round(baseline_face_ssim, 6),
            "finalPreviewFaceSsim": round(final_face_ssim, 6),
            "baselinePersonRegionNormalizedRmse": baseline_person_region_rmse,
            "finalPersonRegionNormalizedRmse": final_person_region_rmse,
            "previewNormalizedRmse": round(final_normalized_rmse, 6),
        },
    }
    compact_plan_json = json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
    arguments.plan_json.write_text(compact_plan_json, encoding="utf-8")
    javascript_global = arguments.javascript_global or (
        "MARKER_PAINT_PLAN_GENERALIZED" if arguments.semantic_report else "MARKER_PAINT_PLAN_S"
    )
    arguments.plan_js.write_text(
        f"window.{javascript_global}={compact_plan_json};\n", encoding="utf-8"
    )
    Image.fromarray(np.clip(final_canvas_rgb, 0, 255).astype(np.uint8)).save(arguments.preview)

    widest_normalized_span = 0.0
    for stroke in optimized_strokes:
        points = np.asarray(stroke["points"], dtype=np.float32)
        widest_normalized_span = max(widest_normalized_span, float(np.ptp(points[:, 0])))
    metrics = {
        "planVersion": plan_version,
        "basePlanVersion": base_plan.get("version", "unknown"),
        "baseStrokeCount": len(retained_base_strokes),
        "optimizedStrokeCount": retained_optimized_stroke_count,
        "faceFeatureStrokeCount": len(face_feature_strokes),
        "personPolishStrokeCount": person_polish_stroke_count,
        "totalStrokeCount": len(all_strokes),
        "baselineSsim": round(baseline_ssim, 6),
        "finalPreviewSsim": round(final_ssim, 6),
        "ssimGain": round(final_ssim - baseline_ssim, 6),
        "baselineFaceSsim": round(baseline_face_ssim, 6),
        "finalPreviewFaceSsim": round(final_face_ssim, 6),
        "faceSsimGain": round(final_face_ssim - baseline_face_ssim, 6),
        "baselinePersonRegionNormalizedRmse": baseline_person_region_rmse,
        "finalPersonRegionNormalizedRmse": final_person_region_rmse,
        "finalPreviewNormalizedRmse": round(final_normalized_rmse, 6),
        "stageMetrics": stage_metrics,
        "drawingStageCounts": drawing_stage_counts,
        "discardedBackgroundStrokeCount": discarded_background_strokes,
        "minimumStrokeWidthPx": MINIMUM_STROKE_WIDTH_PX,
        "minimumStrokeLengthPx": MINIMUM_STROKE_LENGTH_PX,
        "widestOptimizedStrokeHorizontalSpan": round(widest_normalized_span, 6),
        "runtimeTargetCompositing": False,
        "wholeCanvasScanStrokes": 0,
        "rendererInternalScale": 3,
        "stoppedByDiminishingReturns": all(
            metrics["strokeCount"] < stage.maximum_strokes
            or metrics["lastThreeBatchGain"] < stage.minimum_stage_gain
            for stage, metrics in zip(optimization_stages, stage_metrics)
        ),
        "semanticRegions": arguments.semantic_report is not None,
        "manualCoordinateOverrides": 0 if arguments.semantic_report else 2,
        "budgetScale": arguments.budget_scale,
    }
    arguments.metrics.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
