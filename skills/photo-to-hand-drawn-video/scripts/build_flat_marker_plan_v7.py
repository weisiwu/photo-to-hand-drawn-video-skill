#!/usr/bin/env python3
"""Build a trace-what-is-there marker plan from a flat cel-shaded marker target.

v7 assumes the static target is already physically achievable with markers:
dark ink outlines plus flat limited-palette fills. The planner therefore
1. vectorizes thin structures that contrast with their surroundings into long
   curves (2-4 px line strokes; darker-than-surroundings lines at any palette
   color, plus clearly lighter details like fringe or sign characters),
2. heals the line pixels back into their surrounding fill color and hatches
   every palette region with wide (>=13 px) single-color strokes whose
   centerlines are inset by half the pen width so paint stays inside the shape,
   with hand-drawn irregularities (overshoot, wobble, jitter, serpentine).
No pixel colors are copied per stroke; every stroke is palette-colored geometry.
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
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage import measure, morphology
from skimage.filters.rank import modal
from skimage.metrics import structural_similarity


PROJECT_DIRECTORY = Path(__file__).resolve().parent
SOURCE_IMAGE_PATH = PROJECT_DIRECTORY / "assets" / "marker-target-v7.png"
PLAN_JSON_PATH = PROJECT_DIRECTORY / "assets" / "marker-stroke-plan-v7.json"
PLAN_JAVASCRIPT_PATH = PROJECT_DIRECTORY / "assets" / "marker-stroke-plan-v7.js"
REVEAL_PLAN_JSON_PATH = PROJECT_DIRECTORY / "assets" / "marker-reveal-plan-v9.json"
REVEAL_PLAN_JAVASCRIPT_PATH = PROJECT_DIRECTORY / "assets" / "marker-reveal-plan-v9.js"
REVEAL_LAYER_BASE_PATH = PROJECT_DIRECTORY / "assets" / "reveal-layer-base-v9.png"
REVEAL_LAYER_SOFT_PATH = PROJECT_DIRECTORY / "assets" / "reveal-layer-soft-v9.png"
PAINT_PLAN_JSON_PATH = PROJECT_DIRECTORY / "assets" / "marker-paint-plan-v10.json"
PAINT_PLAN_JAVASCRIPT_PATH = PROJECT_DIRECTORY / "assets" / "marker-paint-plan-v10.js"
PAINT_PREVIEW_PATH = PROJECT_DIRECTORY / "verification-v7" / "v10-paint-preview.png"
SHADING_STROKE_ALPHAS = (0.60, 0.52)
COLOR_STOP_MAX_COUNT = 6
REVEAL_LINE_WIDTH_FACTOR = 1.6
REVEAL_LINE_WIDTH_PAD_PX = 2.0
SHADING_MIN_MEAN_DIFF = 6.0
RESIDUAL_MIN_DIFF = 20.0
RESIDUAL_MIN_AREA_PX = 12
RESIDUAL_MAX_COMPONENTS = 140
VERIFICATION_DIRECTORY = PROJECT_DIRECTORY / "verification-v7"
PREVIEW_IMAGE_PATH = VERIFICATION_DIRECTORY / "direct-stroke-preview.png"
LINE_PREVIEW_IMAGE_PATH = VERIFICATION_DIRECTORY / "structure-lines-only.png"
FILL_PREVIEW_IMAGE_PATH = VERIFICATION_DIRECTORY / "fills-only-preview.png"
INK_DENSITY_IMAGE_PATH = VERIFICATION_DIRECTORY / "ink-density-map.png"
METRICS_PATH = VERIFICATION_DIRECTORY / "planner-metrics.json"

ARTBOARD_SIZE_PX = 960
PALETTE_COLOR_COUNT = 24
PALETTE_MERGE_DELTA_E = 7.0
RANDOM_SEED = 20260717

THIN_MAX_HALF_WIDTH_PX = 4.6
DARK_LINE_MIN_CONTRAST = 13.0
DARK_LINE_MIN_PEAK_HALF_WIDTH_PX = 1.1
LIGHT_LINE_MIN_CONTRAST = 25.0
LIGHT_LINE_MIN_PEAK_HALF_WIDTH_PX = 2.0
PAPER_LINE_DELTA_E_REJECT = 10.0

MIN_LINE_LENGTH_PX = 30.0
FOCUS_MIN_LINE_LENGTH_PX = 17.0
LINE_WIDTH_RANGE_PX = (2.5, 3.3)
FOCUS_LINE_WIDTH_RANGE_PX = (2.0, 2.4)
LIGHT_LINE_WIDTH_RANGE_PX = (3.4, 4.0)
LINE_END_OVERSHOOT_RANGE_PX = (1.5, 4.0)
SKETCH_LINE_MIN_LENGTH_PX = 96.0
SKETCH_LINE_MAX_COUNT = 64

MIN_REGION_AREA_PX = 1000
MIN_RARE_REGION_AREA_PX = 140
MIN_LIGHT_REGION_AREA_PX = 1600
RARE_COLOR_MAX_FREQUENCY = 0.02
LIGHT_REGION_LUMINANCE = 80.0
PAPER_DELTA_E_SKIP = 3.0
FILL_ALPHA_RANGE = (0.93, 0.99)
FILL_SPACING_FACTOR = 0.58
FILL_INSET_FACTOR = 0.45
EDGING_MIN_REGION_AREA_PX = 1200
MAX_TOTAL_STROKES = 1750
MIN_TOTAL_STROKES = 500
MAX_POINTS_PER_STROKE = 96
GLAZE_REGION_COUNT = 10

FOCUS_REGION_COUNT = 2
FOCUS_BOX_NORMALIZED = 0.24

# h-round overrides: MyPaint dabs cover less than round-cap canvas strokes at
# equal width, so fills pack tighter and hug region borders closer.
LINE_OPENING_DISK_RADIUS = 4

import os as _os_h
if _os_h.environ.get("H_PLAN") == "1":
    FILL_INSET_FACTOR = 0.30
    FILL_SPACING_FACTOR = 0.50
    # h-round targets use bold 6-9 px feature lines: keep them in the line
    # channel instead of letting the mass-opening swallow them.
    LINE_OPENING_DISK_RADIUS = 6
    THIN_MAX_HALF_WIDTH_PX = 6.5
    LINE_WIDTH_RANGE_PX = (3.0, 4.6)
    FOCUS_LINE_WIDTH_RANGE_PX = (2.6, 3.6)


def to_hex_color(color_rgb: np.ndarray) -> str:
    red, green, blue = (int(round(float(channel))) for channel in np.clip(color_rgb, 0, 255))
    return f"#{red:02x}{green:02x}{blue:02x}"


def rgb_to_lab(colors_rgb: np.ndarray) -> np.ndarray:
    """Lab with L normalized to the standard 0-100 L* scale (OpenCV stores 0-255)."""
    shaped = np.uint8(np.clip(colors_rgb, 0, 255)).reshape(-1, 1, 3)
    lab = cv2.cvtColor(shaped, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    lab[:, 0] *= 100.0 / 255.0
    return lab


def load_source_image() -> np.ndarray:
    source_image = Image.open(SOURCE_IMAGE_PATH).convert("RGB")
    source_image = source_image.resize((ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX), Image.Resampling.LANCZOS)
    source_bgr = cv2.cvtColor(np.asarray(source_image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    # Edge-preserving flattening: the generated target keeps subtle gradients
    # (blazer shading, floor reflections) that shatter k-means regions into
    # noise; mean-shift posterization flattens them while thin dark ink lines
    # survive on contrast.
    flattened_bgr = cv2.pyrMeanShiftFiltering(source_bgr, sp=8, sr=20, maxLevel=1)
    return cv2.cvtColor(flattened_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)


def quantize_to_palette(source_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (palette_rgb[K,3], raw label_map[H,W]) using Lab-space k-means.

    No spatial smoothing here: thin lines must survive for vectorization.
    Near-duplicate palette entries are merged so flat areas stay one region.
    """
    lab_image = cv2.cvtColor(source_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    pixels_lab = lab_image.reshape(-1, 3)
    random_generator = np.random.default_rng(RANDOM_SEED)
    sample_indices = random_generator.choice(len(pixels_lab), 160_000, replace=False)
    cv2.setRNGSeed(RANDOM_SEED)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.25)
    _, _, centers_lab = cv2.kmeans(
        pixels_lab[sample_indices],
        PALETTE_COLOR_COUNT,
        None,
        criteria,
        8,
        cv2.KMEANS_PP_CENTERS,
    )

    assignment = np.argmin(
        np.linalg.norm(pixels_lab[sample_indices][:, None, :] - centers_lab[None, :, :], axis=2),
        axis=1,
    )
    center_weights = np.bincount(assignment, minlength=len(centers_lab)).astype(np.float32)
    merged_centers: list[np.ndarray] = []
    merged_weights: list[float] = []
    for center, weight in sorted(zip(centers_lab, center_weights), key=lambda item: -item[1]):
        for merged_index, merged_center in enumerate(merged_centers):
            if np.linalg.norm(center - merged_center) < PALETTE_MERGE_DELTA_E:
                total_weight = merged_weights[merged_index] + weight
                merged_centers[merged_index] = (
                    merged_center * (merged_weights[merged_index] / total_weight)
                    + center * (weight / total_weight)
                )
                merged_weights[merged_index] = total_weight
                break
        else:
            merged_centers.append(center.copy())
            merged_weights.append(float(weight))
    centers_lab = np.array(merged_centers, dtype=np.float32)

    distances = np.linalg.norm(pixels_lab[:, None, :] - centers_lab[None, :, :], axis=2)
    label_map = np.argmin(distances, axis=1).astype(np.uint8).reshape(source_rgb.shape[:2])
    centers_lab_uint8 = np.uint8(np.clip(centers_lab, 0, 255)).reshape(-1, 1, 3)
    palette_rgb = cv2.cvtColor(centers_lab_uint8, cv2.COLOR_LAB2RGB).reshape(-1, 3).astype(np.float32)
    return palette_rgb, label_map


def extract_small_dark_objects(
    palette_rgb: np.ndarray,
    label_map: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Whole small dark shapes (glasses, buttons, nostrils) taken as units.

    Extracted from the RAW quantized map before any healing so the shape stays
    intact; the skeleton of the full object is how an artist would pen it.
    Returns (object_mask, line entries).
    """
    palette_lab = rgb_to_lab(palette_rgb)
    object_mask = np.zeros(label_map.shape, dtype=bool)
    entries: list[dict[str, Any]] = []
    for color_index in range(len(palette_rgb)):
        if palette_lab[color_index, 0] > 52.0:
            continue
        color_mask = label_map == color_index
        if not color_mask.any():
            continue
        component_labels = measure.label(color_mask, connectivity=2)
        for component in measure.regionprops(component_labels):
            top, left, bottom, right = component.bbox
            if not 50 <= component.area <= 2600:
                continue
            if max(bottom - top, right - left) > 100:
                continue
            component_mask = component_labels == component.label
            mean_half_width = float(distance_transform_edt(component_mask)[component_mask].mean())
            if mean_half_width > 4.5:
                continue
            object_mask |= component_mask
            skeleton = morphology.skeletonize(component_mask)
            width_px = float(np.clip(1.8 * mean_half_width, 2.0, 4.0))
            for path_xy in merge_collinear_paths(
                [path for path in trace_skeleton_paths(skeleton) if calculate_polyline_length(path) >= 5]
            ):
                simplified = measure.approximate_polygon(path_xy[:, ::-1], tolerance=1.2)[:, ::-1]
                if len(simplified) < 2:
                    continue
                entries.append(
                    {
                        "colorIndex": color_index,
                        "isDark": True,
                        "isObject": True,
                        "widthPx": width_px,
                        "points": resample_points(simplified),
                    }
                )
    return object_mask, entries


def extract_thin_detail_components(
    palette_rgb: np.ndarray,
    label_map: np.ndarray,
    luminance_map: np.ndarray,
    paper_rgb: np.ndarray,
    exclude_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Find thin structures that contrast with their surroundings.

    Returns (all_detail_mask, components) where each component carries
    {"colorIndex", "isDark", "mask", "area"}.
    """
    palette_lab = rgb_to_lab(palette_rgb)
    paper_lab = rgb_to_lab(paper_rgb.reshape(1, 3))[0]
    all_detail_mask = np.zeros(label_map.shape, dtype=bool)
    components_out: list[dict[str, Any]] = []

    # --- Dark ink lines: treat ALL dark palette colors as one union, then
    # separate LINES from MASSES morphologically (top-hat). One physical ink
    # line often oscillates across several dark palette entries; per-color
    # extraction shreds it into fragments. Opening removes anything thinner
    # than the disk, so what survives is mass; union minus dilated mass = ink
    # lines, kept as whole connected curves. AA halos hug the mass and are
    # excluded with it.
    dark_union = palette_lab[label_map, 0] <= 50.0
    if exclude_mask is not None:
        dark_union &= ~exclude_mask
    if dark_union.any():
        opening_disk = morphology.disk(LINE_OPENING_DISK_RADIUS).astype(np.uint8)
        mass_mask = cv2.morphologyEx(dark_union.astype(np.uint8), cv2.MORPH_OPEN, opening_disk)
        mass_zone = cv2.dilate(mass_mask, morphology.disk(2).astype(np.uint8)).astype(bool)
        thin_dark = dark_union & ~mass_zone
        thin_dark = morphology.binary_closing(thin_dark, morphology.disk(1)) & dark_union
        thin_dark = morphology.remove_small_objects(thin_dark, 20)
        component_labels = measure.label(thin_dark, connectivity=2)
        for component in measure.regionprops(component_labels):
            component_mask = component_labels == component.label
            surrounding_ring = morphology.binary_dilation(component_mask, morphology.disk(4)) & ~component_mask
            if not surrounding_ring.any():
                continue
            component_luminance = float(luminance_map[component_mask].mean())
            ring_luminances = luminance_map[surrounding_ring]
            if float(np.percentile(ring_luminances, 75)) - component_luminance < DARK_LINE_MIN_CONTRAST:
                continue
            component_colors = label_map[component_mask]
            dominant_color = int(np.bincount(component_colors, minlength=len(palette_rgb)).argmax())
            all_detail_mask |= component_mask
            components_out.append(
                {
                    "colorIndex": dominant_color,
                    "isDark": True,
                    "mask": component_mask,
                    "area": int(component.area),
                }
            )

    # --- Light thin details (fringe, characters) stay per-color.
    for color_index in range(len(palette_rgb)):
        if palette_lab[color_index, 0] <= 50.0:
            continue
        color_mask = label_map == color_index
        if exclude_mask is not None:
            color_mask = color_mask & ~exclude_mask
        if not color_mask.any():
            continue
        half_width = distance_transform_edt(color_mask)
        thin_mask = color_mask & (half_width <= THIN_MAX_HALF_WIDTH_PX)
        thin_mask = morphology.binary_closing(thin_mask, morphology.disk(1))
        thin_mask &= color_mask
        thin_mask = morphology.remove_small_objects(thin_mask, 20)
        if not thin_mask.any():
            continue
        color_delta_to_paper = float(np.linalg.norm(palette_lab[color_index] - paper_lab))
        component_labels = measure.label(thin_mask, connectivity=2)
        for component in measure.regionprops(component_labels):
            component_mask = component_labels == component.label
            peak_half_width = float(half_width[component_mask].max())
            surrounding_ring = morphology.binary_dilation(component_mask, morphology.disk(4)) & ~component_mask
            if not surrounding_ring.any():
                continue
            component_luminance = float(luminance_map[component_mask].mean())
            ring_luminances = luminance_map[surrounding_ring]
            # Percentiles instead of the mean: a spectacle frame is surrounded
            # by bright skin AND dark hair; the bright side alone justifies it.
            dark_contrast = float(np.percentile(ring_luminances, 75)) - component_luminance
            light_contrast = component_luminance - float(np.percentile(ring_luminances, 25))
            if dark_contrast >= light_contrast:
                is_dark = True
                if dark_contrast < DARK_LINE_MIN_CONTRAST or peak_half_width < DARK_LINE_MIN_PEAK_HALF_WIDTH_PX:
                    continue
            else:
                is_dark = False
                if (
                    light_contrast < LIGHT_LINE_MIN_CONTRAST
                    or peak_half_width < LIGHT_LINE_MIN_PEAK_HALF_WIDTH_PX
                    or color_delta_to_paper < PAPER_LINE_DELTA_E_REJECT
                ):
                    continue
            all_detail_mask |= component_mask
            components_out.append(
                {
                    "colorIndex": color_index,
                    "isDark": is_dark,
                    "mask": component_mask,
                    "area": int(component.area),
                }
            )
    return all_detail_mask, components_out


def trace_skeleton_paths(edge_mask: np.ndarray) -> list[np.ndarray]:
    pixel_coordinates_yx = np.argwhere(edge_mask)
    edge_pixels = {(int(pixel_y), int(pixel_x)) for pixel_y, pixel_x in pixel_coordinates_yx}
    if not edge_pixels:
        return []

    neighbor_offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

    def neighbors(pixel: tuple[int, int]) -> list[tuple[int, int]]:
        pixel_y, pixel_x = pixel
        return [
            (pixel_y + offset_y, pixel_x + offset_x)
            for offset_y, offset_x in neighbor_offsets
            if (pixel_y + offset_y, pixel_x + offset_x) in edge_pixels
        ]

    neighbor_cache = {pixel: neighbors(pixel) for pixel in edge_pixels}
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def canonical_edge(first: tuple[int, int], second: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        return (first, second) if first <= second else (second, first)

    def follow_path(start_pixel: tuple[int, int], next_pixel: tuple[int, int]) -> np.ndarray:
        path_pixels = [start_pixel, next_pixel]
        previous_pixel = start_pixel
        current_pixel = next_pixel
        visited_edges.add(canonical_edge(start_pixel, next_pixel))
        while True:
            onward_pixels = [
                neighbor_pixel
                for neighbor_pixel in neighbor_cache[current_pixel]
                if neighbor_pixel != previous_pixel
                and canonical_edge(current_pixel, neighbor_pixel) not in visited_edges
            ]
            if len(neighbor_cache[current_pixel]) != 2 or not onward_pixels:
                break
            following_pixel = onward_pixels[0]
            visited_edges.add(canonical_edge(current_pixel, following_pixel))
            path_pixels.append(following_pixel)
            previous_pixel, current_pixel = current_pixel, following_pixel
        return np.array([[pixel_x, pixel_y] for pixel_y, pixel_x in path_pixels], dtype=np.float32)

    paths_xy: list[np.ndarray] = []
    junction_pixels = sorted(pixel for pixel, adjacent_pixels in neighbor_cache.items() if len(adjacent_pixels) != 2)
    for junction_pixel in junction_pixels:
        for adjacent_pixel in neighbor_cache[junction_pixel]:
            if canonical_edge(junction_pixel, adjacent_pixel) in visited_edges:
                continue
            paths_xy.append(follow_path(junction_pixel, adjacent_pixel))

    for pixel in sorted(edge_pixels):
        for adjacent_pixel in neighbor_cache[pixel]:
            if canonical_edge(pixel, adjacent_pixel) in visited_edges:
                continue
            paths_xy.append(follow_path(pixel, adjacent_pixel))
    return paths_xy


def merge_collinear_paths(paths_xy: list[np.ndarray]) -> list[np.ndarray]:
    """Join path fragments whose endpoints touch and continue smoothly."""
    merged_paths: list[np.ndarray | None] = [path.copy() for path in paths_xy]
    changed = True
    while changed:
        changed = False
        for first_index in range(len(merged_paths)):
            first_path = merged_paths[first_index]
            if first_path is None or len(first_path) < 2:
                continue
            for second_index in range(first_index + 1, len(merged_paths)):
                second_path = merged_paths[second_index]
                if second_path is None or len(second_path) < 2:
                    continue
                joined = try_join_paths(first_path, second_path)
                if joined is not None:
                    merged_paths[first_index] = joined
                    merged_paths[second_index] = None
                    first_path = joined
                    changed = True
    return [path for path in merged_paths if path is not None]


def try_join_paths(
    first_path: np.ndarray,
    second_path: np.ndarray,
    join_distance_px: float = 4.5,
    max_turn_degrees: float = 55.0,
) -> np.ndarray | None:
    def end_direction(path: np.ndarray, at_start: bool) -> np.ndarray:
        if at_start:
            vector = path[0] - path[min(3, len(path) - 1)]
        else:
            vector = path[-1] - path[max(-4, -len(path))]
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 1e-6 else vector

    combos = (
        (first_path[-1], second_path[0], False, True, lambda: np.vstack([first_path, second_path])),
        (first_path[-1], second_path[-1], False, False, lambda: np.vstack([first_path, second_path[::-1]])),
        (first_path[0], second_path[0], True, True, lambda: np.vstack([first_path[::-1], second_path])),
        (first_path[0], second_path[-1], True, False, lambda: np.vstack([first_path[::-1], second_path[::-1]])),
    )
    for first_end, second_end, first_at_start, second_at_start, builder in combos:
        if np.linalg.norm(first_end - second_end) > join_distance_px:
            continue
        outgoing = end_direction(first_path, first_at_start)
        incoming = -end_direction(second_path, second_at_start)
        cosine = float(np.clip(np.dot(outgoing, incoming), -1, 1))
        if math.degrees(math.acos(cosine)) <= max_turn_degrees:
            return builder()
    return None


def calculate_polyline_length(points_xy: np.ndarray) -> float:
    if len(points_xy) < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(points_xy[:, 0]), np.diff(points_xy[:, 1]))))


def resample_points(points_xy: np.ndarray, maximum_points: int = MAX_POINTS_PER_STROKE) -> np.ndarray:
    if len(points_xy) <= maximum_points:
        return points_xy
    selected_indices = np.linspace(0, len(points_xy) - 1, maximum_points).round().astype(int)
    return points_xy[selected_indices]


def normalize_points(points_xy: np.ndarray) -> list[list[float]]:
    normalized = np.clip(points_xy / ARTBOARD_SIZE_PX, -0.04, 1.04)
    return [[round(float(point_x), 5), round(float(point_y), 5)] for point_x, point_y in normalized]


def extend_path_to_length(path_xy: np.ndarray, target_length_px: float) -> np.ndarray:
    """Symmetrically extend a path along its end tangents to a target length.

    A 10 px eyebrow becomes an 18 px slightly-exaggerated eyebrow: still pure
    deterministic geometry, just a bolder cartoon mark that satisfies the
    line aspect-ratio gate.
    """
    current_length = calculate_polyline_length(path_xy)
    if current_length >= target_length_px or len(path_xy) < 2:
        return path_xy
    deficit = (target_length_px - current_length) / 2 + 0.4
    start_direction = path_xy[0] - path_xy[min(2, len(path_xy) - 1)]
    end_direction = path_xy[-1] - path_xy[max(-3, -len(path_xy))]
    extended = [path_xy]
    start_norm = np.linalg.norm(start_direction)
    end_norm = np.linalg.norm(end_direction)
    if start_norm > 1e-6:
        extended.insert(0, (path_xy[0] + start_direction / start_norm * deficit)[None, :])
    if end_norm > 1e-6:
        extended.append((path_xy[-1] + end_direction / end_norm * deficit)[None, :])
    return np.vstack(extended)


def overshoot_path(path_xy: np.ndarray, random_generator: np.random.Generator) -> np.ndarray:
    """Extend both ends along their tangents, like a hand overshooting a line."""
    if len(path_xy) < 2:
        return path_xy
    start_direction = path_xy[0] - path_xy[min(2, len(path_xy) - 1)]
    end_direction = path_xy[-1] - path_xy[max(-3, -len(path_xy))]
    directions = []
    for direction in (start_direction, end_direction):
        norm = np.linalg.norm(direction)
        directions.append(direction / norm if norm > 1e-6 else direction)
    start_extension = path_xy[0] + directions[0] * float(random_generator.uniform(*LINE_END_OVERSHOOT_RANGE_PX))
    end_extension = path_xy[-1] + directions[1] * float(random_generator.uniform(*LINE_END_OVERSHOOT_RANGE_PX))
    return np.vstack([start_extension, path_xy, end_extension])


def vectorize_thin_components(thin_components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Skeletonize accepted thin components into simplified paths per (color, darkness)."""
    grouped_masks: dict[tuple[int, bool], np.ndarray] = {}
    for component in thin_components:
        key = (component["colorIndex"], component["isDark"])
        if key not in grouped_masks:
            grouped_masks[key] = np.zeros(component["mask"].shape, dtype=bool)
        grouped_masks[key] |= component["mask"]

    line_entries: list[dict[str, Any]] = []
    for (color_index, is_dark), group_mask in grouped_masks.items():
        skeleton = morphology.skeletonize(group_mask)
        raw_paths = [path for path in trace_skeleton_paths(skeleton) if calculate_polyline_length(path) >= 7]
        merged_paths = merge_collinear_paths(raw_paths)
        for path_xy in merged_paths:
            simplified = measure.approximate_polygon(path_xy[:, ::-1], tolerance=1.5)[:, ::-1]
            if len(simplified) < 2:
                continue
            line_entries.append(
                {
                    "colorIndex": color_index,
                    "isDark": is_dark,
                    "points": resample_points(simplified),
                }
            )
    return line_entries


def component_orientation(component_mask: np.ndarray) -> float:
    """Principal orientation of a component in [0, pi)."""
    coordinates_yx = np.argwhere(component_mask).astype(np.float32)
    centered = coordinates_yx - coordinates_yx.mean(axis=0)
    covariance = centered.T @ centered / max(len(centered), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_axis_yx = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = math.atan2(float(major_axis_yx[0]), float(major_axis_yx[1]))
    return angle % math.pi


def build_detail_density_map(
    thin_components: list[dict[str, Any]],
    label_map: np.ndarray,
) -> np.ndarray:
    """Density of dark thin structure, weighted toward short fragments and
    modulated by local stroke-orientation entropy.

    Identity-bearing detail (faces, logos, animal features) is made of many
    short fragments pointing in DIVERSE directions; mechanical texture
    (grilles, parallel hatching) is fragment-dense but direction-uniform.
    Both cues are class-agnostic.
    """
    del label_map
    if not thin_components:
        return np.zeros((ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX), dtype=np.float32)
    orientation_bin_count = 8
    weighted = np.zeros((ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX), dtype=np.float32)
    orientation_bins = np.zeros(
        (orientation_bin_count, ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX), dtype=np.float32
    )
    for component in thin_components:
        if not component["isDark"]:
            continue
        weight = float(np.clip(600.0 / max(component["area"], 40), 0.5, 5.0))
        weighted[component["mask"]] += weight
        bin_index = int(component_orientation(component["mask"]) / math.pi * orientation_bin_count) % orientation_bin_count
        # One vote per fragment, spread over its pixels.
        orientation_bins[bin_index][component["mask"]] += 1.0 / max(component["area"], 1)

    blurred_bins = np.stack([gaussian_filter(bin_map, 20.0) for bin_map in orientation_bins])
    bin_totals = blurred_bins.sum(axis=0)
    probabilities = blurred_bins / np.maximum(bin_totals[None, :, :], 1e-9)
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-9)), axis=0)
    entropy_normalized = np.clip(entropy / math.log(orientation_bin_count), 0, 1)
    entropy_normalized[bin_totals < 1e-7] = 0.0

    density = gaussian_filter(weighted, 20.0) * (0.15 + 0.85 * entropy_normalized**2)
    coordinate_axis = np.linspace(-1, 1, ARTBOARD_SIZE_PX, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(coordinate_axis, coordinate_axis)
    center_prior = np.exp(-0.5 * ((grid_x / 1.1) ** 2 + (grid_y / 1.1) ** 2))
    density *= (0.72 + 0.28 * center_prior)
    maximum = float(density.max())
    return density / maximum if maximum > 1e-8 else density


FOCUS_REGIONS_CACHE_PATH = PROJECT_DIRECTORY / "assets" / "focus-regions-v7.json"


def load_vision_focus_regions() -> list[dict[str, float]]:
    """Focus regions pre-annotated by a general vision model (class-agnostic).

    Produced by get_focus_regions_v7.py; absent file -> density fallback.
    """
    if not FOCUS_REGIONS_CACHE_PATH.exists():
        return []
    try:
        cached = json.loads(FOCUS_REGIONS_CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    half_box = FOCUS_BOX_NORMALIZED / 2
    focus_regions: list[dict[str, float]] = []
    for region in cached.get("regions", [])[:FOCUS_REGION_COUNT]:
        center_x = min(max(float(region["centerX"]), half_box), 1 - half_box)
        center_y = min(max(float(region["centerY"]), half_box), 1 - half_box)
        focus_regions.append(
            {
                "left": round(center_x - half_box, 5),
                "top": round(center_y - half_box, 5),
                "width": round(2 * half_box, 5),
                "height": round(2 * half_box, 5),
            }
        )
    return focus_regions


def region_center(region: dict[str, float]) -> tuple[float, float]:
    return (region["left"] + region["width"] / 2, region["top"] + region["height"] / 2)


def select_focus_regions(detail_density_map: np.ndarray) -> list[dict[str, float]]:
    focus_regions: list[dict[str, float]] = []
    for vision_region in load_vision_focus_regions():
        vision_center = region_center(vision_region)
        if all(
            math.dist(vision_center, region_center(existing)) >= 0.15
            for existing in focus_regions
        ):
            focus_regions.append(vision_region)
    working_density = detail_density_map.copy()
    half_box_px = int(FOCUS_BOX_NORMALIZED * ARTBOARD_SIZE_PX / 2)
    for existing in focus_regions:
        existing_x, existing_y = region_center(existing)
        suppress_half = int(ARTBOARD_SIZE_PX * 0.17)
        peak_x = int(existing_x * ARTBOARD_SIZE_PX)
        peak_y = int(existing_y * ARTBOARD_SIZE_PX)
        working_density[
            max(0, peak_y - suppress_half) : peak_y + suppress_half,
            max(0, peak_x - suppress_half) : peak_x + suppress_half,
        ] = 0.0
    while len(focus_regions) < FOCUS_REGION_COUNT:
        peak_flat_index = int(np.argmax(working_density))
        peak_y, peak_x = np.unravel_index(peak_flat_index, working_density.shape)
        if working_density[peak_y, peak_x] <= 0.05:
            break
        center_x = min(max(peak_x, half_box_px), ARTBOARD_SIZE_PX - half_box_px)
        center_y = min(max(peak_y, half_box_px), ARTBOARD_SIZE_PX - half_box_px)
        focus_regions.append(
            {
                "left": round((center_x - half_box_px) / ARTBOARD_SIZE_PX, 5),
                "top": round((center_y - half_box_px) / ARTBOARD_SIZE_PX, 5),
                "width": round(2 * half_box_px / ARTBOARD_SIZE_PX, 5),
                "height": round(2 * half_box_px / ARTBOARD_SIZE_PX, 5),
            }
        )
        suppress_half = int(ARTBOARD_SIZE_PX * 0.17)
        working_density[
            max(0, peak_y - suppress_half) : peak_y + suppress_half,
            max(0, peak_x - suppress_half) : peak_x + suppress_half,
        ] = 0.0
    return focus_regions


def point_in_focus(point_xy: np.ndarray, focus_regions: list[dict[str, float]]) -> bool:
    normalized_x = point_xy[0] / ARTBOARD_SIZE_PX
    normalized_y = point_xy[1] / ARTBOARD_SIZE_PX
    for region in focus_regions:
        if (
            region["left"] <= normalized_x <= region["left"] + region["width"]
            and region["top"] <= normalized_y <= region["top"] + region["height"]
        ):
            return True
    return False


def jitter_path(path_xy: np.ndarray, random_generator: np.random.Generator, amplitude_px: float) -> np.ndarray:
    offsets = random_generator.normal(0, amplitude_px * 0.5, size=path_xy.shape)
    smooth_offsets = np.column_stack(
        [gaussian_filter(offsets[:, 0], 2.5), gaussian_filter(offsets[:, 1], 2.5)]
    )
    return path_xy + np.clip(smooth_offsets, -amplitude_px, amplitude_px)


def build_line_strokes(
    line_entries: list[dict[str, Any]],
    palette_rgb: np.ndarray,
    focus_regions: list[dict[str, float]],
    random_generator: np.random.Generator,
    ink_rgb: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sketch_strokes: list[dict[str, Any]] = []
    detail_strokes: list[dict[str, Any]] = []
    focus_strokes: list[dict[str, Any]] = []

    entries_by_length = sorted(
        line_entries, key=lambda entry: calculate_polyline_length(entry["points"]), reverse=True
    )
    sketch_budget = SKETCH_LINE_MAX_COUNT
    for entry in entries_by_length:
        if entry.get("isClosed"):
            path_xy = entry["points"]
        else:
            path_xy = overshoot_path(entry["points"], random_generator)
        length_px = calculate_polyline_length(path_xy)
        midpoint = path_xy[len(path_xy) // 2]
        in_focus = point_in_focus(midpoint, focus_regions)
        is_dark = entry["isDark"]

        force_ink_mix = False
        if entry.get("isObject"):
            width_px = float(entry["widthPx"])
            width_px = max(2.0, min(width_px, length_px / 8.8))
            minimum_length = 17.7
        elif not is_dark and in_focus:
            # Inside a focus area, a light thin sliver is the negative space
            # BETWEEN dark features (skin gaps around glasses, brows, mouth);
            # its midline is the feature contour — pen it in a dark mix.
            force_ink_mix = ink_rgb is not None
            width_px = float(random_generator.uniform(2.0, 2.6))
            minimum_length = max(FOCUS_MIN_LINE_LENGTH_PX, width_px * 8.6)
            is_dark = True
        elif not is_dark:
            width_px = float(random_generator.uniform(*LIGHT_LINE_WIDTH_RANGE_PX))
            minimum_length = max(30.0, width_px * 8.6)
        elif in_focus:
            width_px = float(random_generator.uniform(*FOCUS_LINE_WIDTH_RANGE_PX))
            minimum_length = max(FOCUS_MIN_LINE_LENGTH_PX, width_px * 8.6)
        else:
            width_px = float(random_generator.uniform(*LINE_WIDTH_RANGE_PX))
            minimum_length = max(MIN_LINE_LENGTH_PX, width_px * 8.6)
        if entry.get("isClosed"):
            minimum_length = max(width_px * 8.6, 15.0 if in_focus else 24.0)
        if length_px < minimum_length:
            if in_focus and is_dark and length_px >= 8.5 and not entry.get("isClosed"):
                path_xy = extend_path_to_length(path_xy, minimum_length)
                length_px = calculate_polyline_length(path_xy)
            else:
                continue
        if length_px < minimum_length:
            continue

        if force_ink_mix:
            stroke_color = mix_toward_ink(palette_rgb[entry["colorIndex"]], ink_rgb, 0.6)
        else:
            stroke_color = entry.get("colorHex") or to_hex_color(palette_rgb[entry["colorIndex"]])
        stroke = {
            "points": normalize_points(path_xy),
            "color": stroke_color,
            "width": round(width_px / ARTBOARD_SIZE_PX, 6),
            "alpha": round(float(random_generator.uniform(0.78, 0.88)), 3),
            "blend": "multiply",
        }
        if in_focus:
            focus_strokes.append({"phase": "focus_detail", **stroke})
        else:
            detail_strokes.append({"phase": "detail_line", **stroke})

        if is_dark and length_px >= SKETCH_LINE_MIN_LENGTH_PX and sketch_budget > 0:
            sketch_budget -= 1
            sketch_strokes.append(
                {
                    "phase": "structure_line",
                    "points": normalize_points(jitter_path(path_xy, random_generator, 2.2)),
                    "color": "#8d8676",
                    "width": round(float(random_generator.uniform(2.7, 3.4)) / ARTBOARD_SIZE_PX, 6),
                    "alpha": 0.34,
                    "blend": "multiply",
                }
            )
    return sketch_strokes, detail_strokes, focus_strokes


def mix_toward_ink(color_rgb: np.ndarray, ink_rgb: np.ndarray, ink_share: float = 0.62) -> str:
    return to_hex_color(color_rgb * (1 - ink_share) + ink_rgb * ink_share)


def build_boundary_line_entries(
    fill_regions: list[dict[str, Any]],
    palette_rgb: np.ndarray,
    fill_label_map: np.ndarray,
    ink_color_index: int,
) -> list[dict[str, Any]]:
    """Contour lines along region boundaries where this region is the darker side.

    A tracing artist inks the silhouette between a dark shape and a lighter
    neighbour (hairline, jaw against sign, person against car body). Boundaries
    between two similar-luminance regions (hair vs. engine bay) are skipped.
    Uses ordered findContours curves so lines stay smooth, and colors each line
    as the region color pulled toward ink (skin gets a warm dark outline, not
    pure black).
    """
    palette_lab = rgb_to_lab(palette_rgb)
    luminance_image = palette_lab[fill_label_map, 0]
    ink_rgb = palette_rgb[ink_color_index]
    entries: list[dict[str, Any]] = []
    for region in fill_regions:
        if region["area"] < 1800:
            continue
        region_luminance = region["luminance"]
        mask = region["mask"]
        lighter_outside = (
            morphology.binary_dilation(mask, morphology.disk(1))
            & ~mask
            & (luminance_image >= region_luminance + 14.0)
        )
        if not lighter_outside.any():
            continue
        near_lighter = morphology.binary_dilation(lighter_outside, morphology.disk(2))
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        line_color_hex = mix_toward_ink(palette_rgb[region["colorIndex"]], ink_rgb)
        for contour in contours:
            contour_points = contour.reshape(-1, 2).astype(np.float32)
            if len(contour_points) < 12:
                continue
            adjacent_flags = near_lighter[
                np.clip(contour_points[:, 1].astype(int), 0, ARTBOARD_SIZE_PX - 1),
                np.clip(contour_points[:, 0].astype(int), 0, ARTBOARD_SIZE_PX - 1),
            ]
            # Split the closed contour into consecutive runs adjacent to lighter areas.
            segments: list[np.ndarray] = []
            current: list[np.ndarray] = []
            for point, flagged in zip(contour_points, adjacent_flags):
                if flagged:
                    current.append(point)
                elif current:
                    segments.append(np.array(current))
                    current = []
            if current:
                if segments and adjacent_flags[0]:
                    segments[0] = np.vstack([np.array(current), segments[0]])
                else:
                    segments.append(np.array(current))
            for segment in segments:
                if len(segment) < 2:
                    continue
                simplified = measure.approximate_polygon(segment[:, ::-1], tolerance=2.2)[:, ::-1]
                if len(simplified) < 2 or calculate_polyline_length(simplified) < 48:
                    continue
                entries.append(
                    {
                        "colorIndex": region["colorIndex"],
                        "colorHex": line_color_hex,
                        "isDark": True,
                        "points": resample_points(simplified),
                    }
                )
    return entries


def build_small_dark_feature_entries(
    small_dark_features: list[dict[str, Any]],
    palette_rgb: np.ndarray,
) -> list[dict[str, Any]]:
    """Small dark shapes (glasses frames, nostrils, buttons) become pen outlines.

    Drawing the closed outer contour reproduces the shape the way an artist
    would ink it (a spectacle frame, a button circle) without skeleton spurs.
    """
    entries: list[dict[str, Any]] = []
    for feature in small_dark_features:
        contours, _ = cv2.findContours(
            feature["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        for contour in contours:
            contour_points = contour.reshape(-1, 2).astype(np.float32)
            if len(contour_points) < 8:
                continue
            closed = np.vstack([contour_points, contour_points[:1]])
            simplified = measure.approximate_polygon(closed[:, ::-1], tolerance=1.0)[:, ::-1]
            if len(simplified) < 3:
                continue
            entries.append(
                {
                    "colorIndex": feature["colorIndex"],
                    "isDark": True,
                    "isClosed": True,
                    "points": resample_points(simplified),
                }
            )
    return entries


def build_fill_label_map(label_map: np.ndarray, detail_mask: np.ndarray) -> np.ndarray:
    """Replace line pixels with their nearest fill label, then despeckle."""
    _, nearest_indices = distance_transform_edt(detail_mask, return_indices=True)
    healed = label_map[nearest_indices[0], nearest_indices[1]]
    healed = modal(healed.astype(np.uint8), morphology.disk(3))
    return healed


def collect_fill_regions(
    palette_rgb: np.ndarray,
    fill_label_map: np.ndarray,
    paper_rgb: np.ndarray,
    focus_regions: list[dict[str, float]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the healed label map into hatchable regions and small dark features.

    Regions below the fill-size threshold are usually simplification noise and
    are dropped — EXCEPT small dark thin shapes (glasses frames, buttons),
    which an artist would draw with the pen; those are returned separately.
    """
    palette_lab = rgb_to_lab(palette_rgb)
    paper_lab = rgb_to_lab(paper_rgb.reshape(1, 3))[0]
    color_counts = np.bincount(fill_label_map.reshape(-1), minlength=len(palette_rgb))
    color_frequencies = color_counts / float(fill_label_map.size)
    fill_regions: list[dict[str, Any]] = []
    small_dark_features: list[dict[str, Any]] = []
    for color_index in range(len(palette_rgb)):
        delta_to_paper = float(np.linalg.norm(palette_lab[color_index] - paper_lab))
        if delta_to_paper < PAPER_DELTA_E_SKIP:
            continue
        luminance = float(palette_lab[color_index, 0])
        if float(color_frequencies[color_index]) < RARE_COLOR_MAX_FREQUENCY:
            minimum_area = MIN_RARE_REGION_AREA_PX
        elif luminance >= LIGHT_REGION_LUMINANCE:
            minimum_area = MIN_LIGHT_REGION_AREA_PX
        else:
            minimum_area = MIN_REGION_AREA_PX
        color_mask = fill_label_map == color_index
        if not color_mask.any():
            continue
        if luminance <= 52.0:
            # Bridge gaps between neighbouring dark patches so hatch runs
            # become strips instead of rows of round dabs; glossy near-black
            # masses (car paint) get a stronger bridge.
            closing_radius = 3 if luminance <= 30.0 else 2
            color_mask = morphology.binary_closing(color_mask, morphology.disk(closing_radius))
        component_labels = measure.label(color_mask, connectivity=2)
        for component in measure.regionprops(component_labels):
            if component.area < minimum_area:
                # Inside a focus region a painter works finer: keep small
                # patches (hair core, iris, badge) down to 150 px^2 there.
                top, left, bottom, right = component.bbox
                component_center = np.array([(left + right) / 2.0, (top + bottom) / 2.0])
                in_focus = focus_regions and point_in_focus(component_center, focus_regions[:1])
                if not (in_focus and component.area >= 150):
                    continue
            fill_regions.append(
                {
                    "colorIndex": color_index,
                    "luminance": luminance,
                    "mask": component_labels == component.label,
                    "area": int(component.area),
                    "bbox": component.bbox,
                    "orientation": float(component.orientation),
                    "eccentricity": float(component.eccentricity),
                }
            )
    fill_regions.sort(key=lambda region: (-region["luminance"], -region["area"]))
    return fill_regions, small_dark_features


def fill_width_for_area(area: int, width_scale: float, random_generator: np.random.Generator) -> float:
    if area < 900:
        base_width = random_generator.uniform(13.0, 15.0)
    elif area < 9000:
        base_width = random_generator.uniform(16.0, 20.0)
    elif area < 40000:
        base_width = random_generator.uniform(22.0, 30.0)
    else:
        base_width = random_generator.uniform(32.0, 42.0)
    return float(max(13.0, base_width * width_scale))


def build_region_hatch_strokes(
    region: dict[str, Any],
    color_hex: str,
    random_generator: np.random.Generator,
    width_scale: float,
    spacing_factor: float = FILL_SPACING_FACTOR,
    residual_depth: int = 1,
) -> list[dict[str, Any]]:
    top, left, bottom, right = region["bbox"]
    region_mask_crop = region["mask"][top:bottom, left:right]
    half_width_map = distance_transform_edt(region_mask_crop)
    region_thickness = float(half_width_map.max())

    width_px = fill_width_for_area(region["area"], width_scale, random_generator)
    width_px = float(min(width_px, max(13.0, 2.1 * region_thickness)))
    spacing_px = width_px * spacing_factor

    # Dark masses may bleed slightly past their border (glossy paint, hair);
    # light fills stay conservative so they never invade the line work.
    inset_factor = 0.32 if region["luminance"] <= 30.0 else FILL_INSET_FACTOR
    inset_px = width_px * inset_factor
    inset_mask_crop = half_width_map >= inset_px
    if not inset_mask_crop.any():
        # Region thinner than the pen: one dab along the major axis, mild bleed.
        center_yx = np.argwhere(region_mask_crop).mean(axis=0)
        angle = -region["orientation"]
        direction = np.array([math.cos(angle), math.sin(angle)])
        extent = max(4.0, min((right - left), (bottom - top)) * 0.4)
        center_xy = np.array([left + center_yx[1], top + center_yx[0]])
        dab = np.vstack([center_xy - direction * extent, center_xy + direction * extent])
        dab_alpha_scale = float(region.get("alphaScale", 1.0))
        return [
            {
                "phase": "region_fill",
                "points": normalize_points(dab),
                "color": color_hex,
                "width": round(max(13.0, min(width_px, 2.4 * region_thickness)) / ARTBOARD_SIZE_PX, 6),
                "alpha": round(float(random_generator.uniform(*FILL_ALPHA_RANGE)) * dab_alpha_scale, 3),
                "blend": "source-over" if dab_alpha_scale >= 1.0 else "multiply",
            }
        ]

    if region["eccentricity"] > 0.88:
        hatch_angle = -region["orientation"]
    else:
        hatch_angle = math.radians(-38.0)
    hatch_angle += float(random_generator.uniform(-0.16, 0.16))

    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    diagonal = math.hypot(right - left, bottom - top)
    direction = np.array([math.cos(hatch_angle), math.sin(hatch_angle)])
    normal = np.array([-direction[1], direction[0]])

    strokes_prefix: list[dict[str, Any]] = []
    if region["area"] >= EDGING_MIN_REGION_AREA_PX and spacing_factor <= 1.0:
        # Edging pass: run the marker once along the inside of the region
        # outline before filling — the way an artist closes the shape so the
        # fill never leaves a paper halo at the border.
        edge_width_px = float(min(width_px, 17.0))
        edge_inset = max(1, int(round(edge_width_px * 0.48)))
        edge_core = cv2.erode(
            region_mask_crop.astype(np.uint8), morphology.disk(edge_inset).astype(np.uint8)
        ).astype(bool)
        edge_contours, _ = cv2.findContours(
            edge_core.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        for contour in edge_contours:
            contour_points = contour.reshape(-1, 2).astype(np.float32)
            if len(contour_points) < 10:
                continue
            closed = np.vstack([contour_points, contour_points[:1]]) + np.array([left, top], np.float32)
            simplified = measure.approximate_polygon(closed[:, ::-1], tolerance=1.6)[:, ::-1]
            if len(simplified) < 3:
                continue
            strokes_prefix.append(
                {
                    "phase": "region_fill",
                    "points": normalize_points(resample_points(simplified)),
                    "color": color_hex,
                    "width": round(edge_width_px / ARTBOARD_SIZE_PX, 6),
                    "alpha": round(float(random_generator.uniform(*FILL_ALPHA_RANGE)), 3),
                    "blend": "source-over",
                }
            )

    sample_step_px = 6.0
    strokes: list[dict[str, Any]] = list(strokes_prefix)
    line_count = int(diagonal / spacing_px) + 2
    serpentine_forward = True
    crop_height, crop_width = inset_mask_crop.shape
    for line_index in range(-line_count // 2, line_count // 2 + 1):
        offset = line_index * spacing_px + float(random_generator.uniform(-0.14, 0.14) * spacing_px)
        line_origin = np.array([center_x, center_y]) + normal * offset
        sample_count = int(diagonal / sample_step_px) + 2
        run_points: list[np.ndarray] = []
        runs: list[list[np.ndarray]] = []
        gap_run = 0
        for sample_index in range(-sample_count // 2, sample_count // 2 + 1):
            sample_point = line_origin + direction * (sample_index * sample_step_px)
            crop_x = int(round(sample_point[0])) - left
            crop_y = int(round(sample_point[1])) - top
            inside = (
                0 <= crop_x < crop_width
                and 0 <= crop_y < crop_height
                and inset_mask_crop[crop_y, crop_x]
            )
            if inside:
                run_points.append(sample_point)
                gap_run = 0
            else:
                gap_run += 1
                if run_points and gap_run == 2:
                    runs.append(run_points)
                    run_points = []
        if run_points:
            runs.append(run_points)

        for run in runs:
            run_array = np.array(run)
            if len(run_array) == 1:
                run_array = np.vstack([run_array[0] - direction * 3.0, run_array[0] + direction * 3.0])
            overshoot_start = float(random_generator.uniform(0.5, 3.0))
            overshoot_end = float(random_generator.uniform(0.5, 3.0))
            run_array = np.vstack(
                [
                    run_array[0] - direction * overshoot_start,
                    run_array,
                    run_array[-1] + direction * overshoot_end,
                ]
            )
            wobble_phase = float(random_generator.uniform(0, math.tau))
            wobble_amplitude = float(random_generator.uniform(0.6, 1.4))
            indices = np.arange(len(run_array))
            wobble = np.sin(indices * 0.55 + wobble_phase) * wobble_amplitude
            run_array = run_array + normal[None, :] * wobble[:, None]
            simplified = measure.approximate_polygon(run_array[:, ::-1], tolerance=0.9)[:, ::-1]
            if len(simplified) < 2:
                simplified = run_array[[0, -1]]
            if not serpentine_forward:
                simplified = simplified[::-1]
            alpha_scale = float(region.get("alphaScale", 1.0))
            strokes.append(
                {
                    "phase": "region_fill",
                    "points": normalize_points(resample_points(simplified)),
                    "color": color_hex,
                    "width": round(width_px / ARTBOARD_SIZE_PX, 6),
                    "alpha": round(float(random_generator.uniform(*FILL_ALPHA_RANGE)) * alpha_scale, 3),
                    "blend": "source-over" if alpha_scale >= 1.0 else "multiply",
                }
            )
            serpentine_forward = not serpentine_forward

    if residual_depth > 0 and spacing_factor <= 1.0:
        # Thin lobes of a big shape (hair on the engine bay, tire slivers) sit
        # inside the inset margin of a wide pen; repaint what the wide pass
        # missed with a pen matched to the lobe thickness (never below 13 px).
        painted = np.zeros_like(region_mask_crop, dtype=np.uint8)
        for stroke in strokes:
            points = np.array(stroke["points"], dtype=np.float32) * ARTBOARD_SIZE_PX
            points -= np.array([left, top], dtype=np.float32)
            stroke_width = max(1, int(round(float(stroke["width"]) * ARTBOARD_SIZE_PX)))
            cv2.polylines(
                painted,
                [np.rint(points).astype(np.int32).reshape(-1, 1, 2)],
                False,
                255,
                stroke_width,
            )
        uncovered = region_mask_crop & (painted == 0)
        uncovered = morphology.remove_small_objects(uncovered, 150)
        if uncovered.any():
            component_labels = measure.label(uncovered, connectivity=2)
            for component in measure.regionprops(component_labels):
                if component.area < 150:
                    continue
                sub_top, sub_left, sub_bottom, sub_right = component.bbox
                full_mask = np.zeros((ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX), dtype=bool)
                full_mask[top:bottom, left:right] = component_labels == component.label
                sub_region = {
                    "colorIndex": region["colorIndex"],
                    "luminance": region["luminance"],
                    "mask": full_mask,
                    "area": int(component.area),
                    "bbox": (sub_top + top, sub_left + left, sub_bottom + top, sub_right + left),
                    "orientation": float(component.orientation),
                    "eccentricity": float(component.eccentricity),
                }
                strokes.extend(
                    build_region_hatch_strokes(
                        sub_region,
                        color_hex,
                        random_generator,
                        width_scale=1.0,
                        spacing_factor=spacing_factor,
                        residual_depth=residual_depth - 1,
                    )
                )
    return strokes


def build_focus_line_shading_strokes(
    region: dict[str, Any],
    color_hex: str,
    random_generator: np.random.Generator,
) -> list[dict[str, Any]]:
    """Shade a small dark region with fine pen hatching (hair, eyes, badges).

    Inside a focus area an artist switches to the liner instead of the wide
    marker; thin strokes must keep aspect ratio >= 8, so short runs are
    extended with overshoot or skipped.
    """
    top, left, bottom, right = region["bbox"]
    region_mask = region["mask"]
    hatch_angle = -region["orientation"] if region["eccentricity"] > 0.80 else math.radians(-24.0)
    hatch_angle += float(random_generator.uniform(-0.12, 0.12))
    direction = np.array([math.cos(hatch_angle), math.sin(hatch_angle)])
    normal = np.array([-direction[1], direction[0]])
    center = np.array([(left + right) / 2.0, (top + bottom) / 2.0])
    diagonal = math.hypot(right - left, bottom - top)
    is_true_black = region["luminance"] <= 25.0
    spacing_px = 3.6 if is_true_black else 5.2
    shading_alpha_range = (0.72, 0.85) if is_true_black else (0.42, 0.58)
    sample_step_px = 3.0
    strokes: list[dict[str, Any]] = []
    line_count = int(diagonal / spacing_px) + 2
    for line_index in range(-line_count // 2, line_count // 2 + 1):
        offset = line_index * spacing_px + float(random_generator.uniform(-0.6, 0.6))
        line_origin = center + normal * offset
        sample_count = int(diagonal / sample_step_px) + 2
        run: list[np.ndarray] = []
        runs: list[np.ndarray] = []
        for sample_index in range(-sample_count // 2, sample_count // 2 + 1):
            point = line_origin + direction * (sample_index * sample_step_px)
            x, y = int(round(point[0])), int(round(point[1]))
            if 0 <= x < ARTBOARD_SIZE_PX and 0 <= y < ARTBOARD_SIZE_PX and region_mask[y, x]:
                run.append(point)
            elif run:
                runs.append(np.array(run))
                run = []
        if run:
            runs.append(np.array(run))
        for run_array in runs:
            width_px = float(random_generator.uniform(2.0, 2.6))
            overshoot = float(random_generator.uniform(2.0, 3.6))
            if len(run_array) == 1:
                run_array = np.vstack([run_array[0] - direction * 2, run_array[0] + direction * 2])
            run_array = np.vstack(
                [run_array[0] - direction * overshoot, run_array, run_array[-1] + direction * overshoot]
            )
            if calculate_polyline_length(run_array) < 7.0:
                continue
            run_array = extend_path_to_length(run_array, width_px * 8.8 + 0.6)
            strokes.append(
                {
                    "phase": "focus_detail",
                    "points": normalize_points(resample_points(run_array)),
                    "color": color_hex,
                    "width": round(width_px / ARTBOARD_SIZE_PX, 6),
                    "alpha": round(float(random_generator.uniform(*shading_alpha_range)), 3),
                    "blend": "multiply",
                }
            )
    return strokes


def build_glaze_strokes(
    fill_regions: list[dict[str, Any]],
    palette_rgb: np.ndarray,
    random_generator: np.random.Generator,
    width_scale: float,
) -> list[dict[str, Any]]:
    glaze_strokes: list[dict[str, Any]] = []
    largest_regions = sorted(fill_regions, key=lambda region: -region["area"])[:GLAZE_REGION_COUNT]
    for region in largest_regions:
        color_hex = to_hex_color(palette_rgb[region["colorIndex"]])
        strokes = build_region_hatch_strokes(
            region, color_hex, random_generator, width_scale, spacing_factor=2.4
        )
        for stroke in strokes:
            stroke["phase"] = "region_glaze"
            stroke["blend"] = "multiply"
            stroke["alpha"] = round(float(random_generator.uniform(0.08, 0.15)), 3)
        glaze_strokes.extend(strokes)
    return glaze_strokes


def region_seal_contours(region_mask: np.ndarray) -> list[list[list[float]]]:
    """Closed contours (outer + holes) of a region for the v9 reveal seal.

    Rendered with the even-odd rule so holes stay unrevealed until their own
    region is painted.
    """
    contours, _ = cv2.findContours(
        region_mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    seal_contours: list[list[list[float]]] = []
    for contour in contours:
        contour_points = contour.reshape(-1, 2).astype(np.float32)
        if len(contour_points) < 6:
            continue
        closed = np.vstack([contour_points, contour_points[:1]])
        simplified = measure.approximate_polygon(closed[:, ::-1], tolerance=1.2)[:, ::-1]
        if len(simplified) < 3:
            continue
        seal_contours.append(normalize_points(resample_points(simplified)))
    return seal_contours


def estimate_paper_color(source_rgb: np.ndarray) -> np.ndarray:
    border = np.concatenate(
        [
            source_rgb[:14].reshape(-1, 3),
            source_rgb[-14:].reshape(-1, 3),
            source_rgb[:, :14].reshape(-1, 3),
            source_rgb[:, -14:].reshape(-1, 3),
        ]
    )
    median_rgb = np.median(border, axis=0)
    luminance = float(median_rgb @ np.array([0.299, 0.587, 0.114]))
    if luminance < 205:
        return np.array([248, 243, 231], dtype=np.float32)
    return median_rgb.astype(np.float32)


def render_stroke_plan(
    strokes: list[dict[str, Any]], paper_rgb: np.ndarray, phases: set[str] | None = None
) -> np.ndarray:
    canvas_rgb = np.full((ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX, 3), paper_rgb, dtype=np.float32)
    for stroke in strokes:
        if phases is not None and stroke["phase"] not in phases:
            continue
        points = np.array(stroke["points"], dtype=np.float32) * ARTBOARD_SIZE_PX
        points_int = np.rint(points).astype(np.int32).reshape(-1, 1, 2)
        stroke_mask = np.zeros((ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX), dtype=np.uint8)
        width_px = max(1, int(round(float(stroke["width"]) * ARTBOARD_SIZE_PX)))
        cv2.polylines(stroke_mask, [points_int], False, 255, width_px, cv2.LINE_AA)
        alpha_mask = (stroke_mask.astype(np.float32) / 255.0 * float(stroke["alpha"]))[:, :, None]
        color_rgb = np.array(
            [int(stroke["color"][offset : offset + 2], 16) for offset in (1, 3, 5)],
            dtype=np.float32,
        )
        if stroke["blend"] == "multiply":
            blended_rgb = canvas_rgb * (color_rgb[None, None, :] / 255.0)
        else:
            blended_rgb = np.broadcast_to(color_rgb, canvas_rgb.shape)
        canvas_rgb = canvas_rgb * (1 - alpha_mask) + blended_rgb * alpha_mask
    return np.clip(canvas_rgb, 0, 255).astype(np.uint8)


def main() -> None:
    VERIFICATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    random_generator = np.random.default_rng(RANDOM_SEED)
    source_rgb = load_source_image()
    paper_rgb = estimate_paper_color(source_rgb)
    palette_rgb, label_map = quantize_to_palette(source_rgb)
    luminance_map = (
        cv2.cvtColor(source_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
        * 100.0 / 255.0
    )

    detail_mask, thin_components = extract_thin_detail_components(
        palette_rgb, label_map, luminance_map, paper_rgb
    )
    detail_density_map = build_detail_density_map(thin_components, label_map)
    focus_regions = select_focus_regions(detail_density_map)

    fill_label_map = build_fill_label_map(label_map, detail_mask)
    fill_regions, _ = collect_fill_regions(palette_rgb, fill_label_map, paper_rgb, focus_regions)

    palette_lab_for_ink = rgb_to_lab(palette_rgb)
    ink_color_index = int(np.argmin(palette_lab_for_ink[:, 0]))
    line_entries = (
        vectorize_thin_components(thin_components)
        + build_boundary_line_entries(fill_regions, palette_rgb, fill_label_map, ink_color_index)
    )
    sketch_strokes, detail_line_strokes, focus_detail_strokes = build_line_strokes(
        line_entries, palette_rgb, focus_regions, random_generator,
        ink_rgb=palette_rgb[ink_color_index],
    )

    def region_in_focus(region: dict[str, Any]) -> bool:
        top, left, bottom, right = region["bbox"]
        # Pen shading is reserved for the identity-critical primary focus.
        return point_in_focus(
            np.array([(left + right) / 2.0, (top + bottom) / 2.0]), focus_regions[:1]
        )

    pen_shading_regions = [
        region
        for region in fill_regions
        if region["area"] < 2600 and region["luminance"] <= 46.0 and region_in_focus(region)
    ]
    pen_shading_region_ids = {id(region) for region in pen_shading_regions}
    wide_fill_regions = [region for region in fill_regions if id(region) not in pen_shading_region_ids]
    for region in wide_fill_regions:
        # Small mid-tone shadow patches at the focal point (lip/chin shadows)
        # read as heavy lumps at full opacity; glaze them lightly instead.
        if region_in_focus(region) and region["area"] < 3200 and 46.0 < region["luminance"] <= 72.0:
            region["alphaScale"] = 0.42

    width_scale = 1.0
    fill_strokes: list[dict[str, Any]] = []
    glaze_strokes: list[dict[str, Any]] = []
    pen_shading_strokes: list[dict[str, Any]] = []
    fill_groups: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    pen_shading_groups: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for _ in range(5):
        fill_random = np.random.default_rng(RANDOM_SEED + 11)
        fill_strokes = []
        fill_groups = []
        for region in wide_fill_regions:
            color_hex = to_hex_color(palette_rgb[region["colorIndex"]])
            region_strokes = build_region_hatch_strokes(region, color_hex, fill_random, width_scale)
            fill_groups.append((region, region_strokes))
            fill_strokes.extend(region_strokes)
        pen_shading_strokes = []
        pen_shading_groups = []
        for region in pen_shading_regions:
            color_hex = to_hex_color(palette_rgb[region["colorIndex"]])
            region_strokes = build_focus_line_shading_strokes(region, color_hex, fill_random)
            pen_shading_groups.append((region, region_strokes))
            pen_shading_strokes.extend(region_strokes)
        glaze_strokes = build_glaze_strokes(wide_fill_regions, palette_rgb, fill_random, width_scale)
        total = (
            len(sketch_strokes)
            + len(fill_strokes)
            + len(glaze_strokes)
            + len(pen_shading_strokes)
            + len(detail_line_strokes)
            + len(focus_detail_strokes)
        )
        if total <= MAX_TOTAL_STROKES:
            break
        width_scale *= 1.16

    strokes = (
        sketch_strokes
        + fill_strokes
        + glaze_strokes
        + detail_line_strokes
        + focus_detail_strokes
        + pen_shading_strokes
    )
    phase_counts = Counter(stroke["phase"] for stroke in strokes)

    plan = {
        "version": 7,
        "planningSizePx": ARTBOARD_SIZE_PX,
        "outputArtboardSizePx": ARTBOARD_SIZE_PX,
        "paperColor": to_hex_color(paper_rgb),
        "palette": [to_hex_color(color) for color in palette_rgb],
        "focusRegions": focus_regions,
        "strokes": strokes,
        "stats": {
            "phaseCounts": dict(sorted(phase_counts.items())),
            "fillRegionCount": len(fill_regions),
            "linePathCount": len(line_entries),
            "total": len(strokes),
        },
    }
    compact_json = json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
    PLAN_JSON_PATH.write_text(compact_json, encoding="utf-8")
    PLAN_JAVASCRIPT_PATH.write_text(f"window.MARKER_STROKE_PLAN_V7={compact_json};\n", encoding="utf-8")

    # ---- v9 layered reveal plan. Perceptual rule: one stroke may only carry
    # one color's worth of change. So the animation reveals three IMAGES in
    # painterly order instead of the final target directly:
    #   layer 0 = flat base colors (palette image of the healed label map),
    #   layer 1 = soft shading, lines inpainted away (second marker pass),
    #   layer 2 = full target, revealed ONLY through pen-width strokes and
    #             small highlight dabs (features, lashes, speculars).
    def seal_stroke(phase: str, layer: int, mask: np.ndarray, slow: bool = False) -> dict[str, Any]:
        stroke: dict[str, Any] = {
            "phase": phase,
            "mode": "seal",
            "layer": layer,
            "contours": region_seal_contours(mask),
            "width": 0.001,
            "points": [[0, 0], [0.001, 0.001]],
        }
        if slow:
            stroke["slow"] = True
        return stroke

    base_layer_rgb = palette_rgb[fill_label_map].astype(np.uint8)
    Image.fromarray(base_layer_rgb).save(REVEAL_LAYER_BASE_PATH)
    inpaint_mask = morphology.binary_dilation(detail_mask, morphology.disk(1)).astype(np.uint8) * 255
    soft_layer_rgb = cv2.inpaint(source_rgb.astype(np.uint8), inpaint_mask, 3, cv2.INPAINT_TELEA)
    Image.fromarray(soft_layer_rgb).save(REVEAL_LAYER_SOFT_PATH)

    reveal_strokes: list[dict[str, Any]] = []
    for stroke in sketch_strokes:
        reveal_strokes.append({**stroke, "mode": "paint", "layer": 0})

    # Base flats: wide sweeps + seal, revealing single-color regions.
    for region, region_strokes in fill_groups:
        for stroke in region_strokes:
            reveal_strokes.append(
                {
                    "phase": "region_fill",
                    "mode": "reveal",
                    "layer": 0,
                    "points": stroke["points"],
                    "width": stroke["width"],
                }
            )
        reveal_strokes.append(seal_stroke("region_fill", 0, region["mask"]))

    covered_mask = np.zeros((ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX), dtype=bool)
    for region, _ in fill_groups:
        covered_mask |= region["mask"]
    for region, _ in pen_shading_groups:
        covered_mask |= region["mask"]
    leftover_mask = ~covered_mask
    if leftover_mask.any():
        reveal_strokes.append(seal_stroke("region_fill", 0, leftover_mask, slow=True))

    # Shading pass: regions whose soft layer differs from the base get a
    # second sweep that reveals layer 1 (blush, folds, floor shadow).
    base_float = base_layer_rgb.astype(np.float32)
    soft_float = soft_layer_rgb.astype(np.float32)
    shading_diff = np.mean(np.abs(soft_float - base_float), axis=2)
    shading_random = np.random.default_rng(RANDOM_SEED + 23)
    shading_region_count = 0
    for region, _ in fill_groups:
        if float(shading_diff[region["mask"]].mean()) < SHADING_MIN_MEAN_DIFF:
            continue
        shading_region_count += 1
        sweeps = build_region_hatch_strokes(
            region,
            "#000000",
            shading_random,
            width_scale=1.0,
            spacing_factor=1.25,
            residual_depth=0,
        )
        for sweep in sweeps:
            reveal_strokes.append(
                {
                    "phase": "region_glaze",
                    "mode": "reveal",
                    "layer": 1,
                    "points": sweep["points"],
                    "width": sweep["width"],
                }
            )
        reveal_strokes.append(seal_stroke("region_glaze", 1, region["mask"]))
    # The soft layer equals the base everywhere else; one slow wash keeps the
    # composite consistent without any visible change.
    reveal_strokes.append(
        seal_stroke("region_glaze", 1, np.ones((ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX), dtype=bool), slow=True)
    )

    # Fine detail: pen-width reveals of the full target only.
    for stroke_list, default_phase in ((detail_line_strokes, "detail_line"), (focus_detail_strokes, "focus_detail")):
        for stroke in stroke_list:
            widened = min(0.14, stroke["width"] * REVEAL_LINE_WIDTH_FACTOR + REVEAL_LINE_WIDTH_PAD_PX / ARTBOARD_SIZE_PX)
            reveal_strokes.append(
                {
                    "phase": default_phase,
                    "mode": "reveal",
                    "layer": 2,
                    "points": stroke["points"],
                    "width": round(widened, 6),
                }
            )
    for region, region_strokes in pen_shading_groups:
        for stroke in region_strokes:
            widened = min(0.14, stroke["width"] * REVEAL_LINE_WIDTH_FACTOR + REVEAL_LINE_WIDTH_PAD_PX / ARTBOARD_SIZE_PX)
            reveal_strokes.append(
                {
                    "phase": "focus_detail",
                    "mode": "reveal",
                    "layer": 2,
                    "points": stroke["points"],
                    "width": round(widened, 6),
                }
            )

    # Highlight/residual dabs: wherever the target still differs strongly from
    # the soft layer (eye lights, speculars, stray strands), touch up with
    # small gel-pen style reveals.
    residual_map = np.mean(np.abs(source_rgb - soft_float), axis=2) > RESIDUAL_MIN_DIFF
    residual_map = morphology.remove_small_objects(residual_map, RESIDUAL_MIN_AREA_PX)
    residual_labels = measure.label(residual_map, connectivity=2)
    residual_components = sorted(
        measure.regionprops(residual_labels), key=lambda component: -component.area
    )[:RESIDUAL_MAX_COMPONENTS]
    residual_count = 0
    for component in residual_components:
        component_mask = residual_labels == component.label
        component_mask = morphology.binary_dilation(component_mask, morphology.disk(1))
        top, left, bottom, right = component.bbox
        center = np.array([(left + right) / 2.0, (top + bottom) / 2.0])
        phase = "focus_detail" if point_in_focus(center, focus_regions) else "detail_line"
        reveal_strokes.append(seal_stroke(phase, 2, component_mask))
        residual_count += 1

    reveal_plan = {
        "version": 9,
        "outputArtboardSizePx": ARTBOARD_SIZE_PX,
        "paperColor": to_hex_color(paper_rgb),
        "palette": [to_hex_color(color) for color in palette_rgb],
        "focusRegions": focus_regions,
        "strokes": reveal_strokes,
        "stats": {
            "total": len(reveal_strokes),
            "modes": dict(Counter(stroke["mode"] for stroke in reveal_strokes)),
            "shadingRegions": shading_region_count,
            "residualDabs": residual_count,
        },
    }
    reveal_json = json.dumps(reveal_plan, ensure_ascii=False, separators=(",", ":"))
    REVEAL_PLAN_JSON_PATH.write_text(reveal_json, encoding="utf-8")
    REVEAL_PLAN_JAVASCRIPT_PATH.write_text(
        f"window.MARKER_REVEAL_PLAN_V9={reveal_json};\n", encoding="utf-8"
    )

    # ---- v10 gradient-stroke PAINT plan: the finished frame is again a sum of
    # strokes (no target image at runtime), but shading strokes carry a slow
    # color gradient sampled from the soft layer — like marker pressure and
    # wetness — so the reconstruction approaches the target while every single
    # stroke still reads as one stroke of paint.
    def masked_region_blur(region: dict[str, Any]) -> np.ndarray:
        """Soft-layer colors blurred WITHIN the region only, so sweeps never
        pick up a neighbouring region's color (hair bleeding onto the wall)."""
        mask_float = region["mask"].astype(np.float32)
        weighted = cv2.GaussianBlur(soft_layer_rgb.astype(np.float32) * mask_float[:, :, None], (0, 0), 9)
        weight = cv2.GaussianBlur(mask_float, (0, 0), 9)
        return weighted / np.maximum(weight, 1e-4)[:, :, None]

    def sample_color_stops(points_normalized: list[list[float]], blurred_rgb: np.ndarray) -> list[str]:
        points_px = np.array(points_normalized, dtype=np.float32) * ARTBOARD_SIZE_PX
        stop_count = min(COLOR_STOP_MAX_COUNT, max(2, len(points_px)))
        stop_indices = np.linspace(0, len(points_px) - 1, stop_count).round().astype(int)
        stops = []
        for index in stop_indices:
            sample_x = int(np.clip(points_px[index][0], 0, ARTBOARD_SIZE_PX - 1))
            sample_y = int(np.clip(points_px[index][1], 0, ARTBOARD_SIZE_PX - 1))
            stops.append(to_hex_color(blurred_rgb[sample_y, sample_x]))
        return stops

    paint_strokes: list[dict[str, Any]] = []
    paint_strokes.extend(dict(stroke) for stroke in sketch_strokes)
    for region, region_strokes in fill_groups:
        paint_strokes.extend(dict(stroke) for stroke in region_strokes)
    shading_random_v10 = np.random.default_rng(RANDOM_SEED + 31)
    for pass_index, pass_alpha in enumerate(SHADING_STROKE_ALPHAS):
        for region, _ in fill_groups:
            if float(shading_diff[region["mask"]].mean()) < SHADING_MIN_MEAN_DIFF:
                continue
            sweeps = build_region_hatch_strokes(
                region,
                "#000000",
                shading_random_v10,
                width_scale=1.0,
                spacing_factor=1.05 + 0.25 * pass_index,
                residual_depth=0,
            )
            region_blur = masked_region_blur(region)
            for sweep in sweeps:
                stops = sample_color_stops(sweep["points"], region_blur)
                paint_strokes.append(
                    {
                        "phase": "region_glaze",
                        "points": sweep["points"],
                        "width": sweep["width"],
                        "alpha": round(pass_alpha * float(shading_random_v10.uniform(0.9, 1.05)), 3),
                        "blend": "source-over",
                        "color": stops[len(stops) // 2],
                        "colorStops": stops,
                    }
                )
    paint_strokes.extend(dict(stroke) for stroke in detail_line_strokes)
    for stroke in focus_detail_strokes:
        softened = dict(stroke)
        softened["alpha"] = round(float(stroke["alpha"]) * 0.78, 3)
        paint_strokes.append(softened)
    for region, region_strokes in pen_shading_groups:
        paint_strokes.extend(dict(stroke) for stroke in region_strokes)
    # Highlight dabs: gel-pen touch-ups where the target pops above the soft
    # layer (eye lights, speculars) — painted with the residual's own color.
    highlight_count = 0
    for component in residual_components:
        # Only compact spots become gel-pen dabs; long thin residuals are
        # edges that the line pass already handles.
        if component.major_axis_length > 44:
            continue
        component_mask = residual_labels == component.label
        mean_rgb = source_rgb[component_mask].mean(axis=0)
        mean_lab = rgb_to_lab(np.array(mean_rgb))[0]
        chroma = float(np.hypot(mean_lab[1] - 128, mean_lab[2] - 128))
        # Gel-pen dabs are for bright highlights or saturated accents; muddy
        # mid-tone residuals read as floating smudges — skip them.
        if mean_lab[0] < 78.0 and chroma < 28.0:
            continue
        center_xy = np.array([component.centroid[1], component.centroid[0]], dtype=np.float32)
        angle = -component.orientation
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        half_length = min(max(3.0, float(component.major_axis_length) * 0.5), 12.0)
        dab_points = np.vstack([center_xy - direction * half_length, center_xy + direction * half_length])
        dab_width = float(np.clip(component.minor_axis_length * 1.6, 2.5, 9.0))
        phase = "focus_detail" if point_in_focus(center_xy, focus_regions) else "detail_line"
        paint_strokes.append(
            {
                "phase": phase,
                "points": normalize_points(dab_points),
                "color": to_hex_color(mean_rgb),
                "width": round(dab_width / ARTBOARD_SIZE_PX, 6),
                "alpha": 0.8,
                "blend": "source-over",
            }
        )
        highlight_count += 1

    # Coat 3 — error-driven fine refinement, still pure paint. Wherever the
    # coat stack (≈ soft layer) misses the target most (faces, floral print,
    # hair strands), lay 7 px flow-following strokes whose colors are sampled
    # from a σ4 blur of the target: classic coarse-to-fine painterly
    # reconstruction, no reveal anywhere.
    fine_blur_rgb = cv2.GaussianBlur(source_rgb.astype(np.uint8), (0, 0), 4).astype(np.float32)
    coat_error = np.mean(np.abs(source_rgb - soft_float), axis=2)
    coat_error = cv2.GaussianBlur(coat_error, (0, 0), 2)
    # Thin features (facial lines, lashes, characters) belong to the pen
    # passes — a 7 px coat stroke would only smear them into mud.
    feature_zone = morphology.binary_dilation(detail_mask, morphology.disk(4))
    coat_error[feature_zone] = 0.0
    grayscale = cv2.cvtColor(source_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gradient_x = cv2.GaussianBlur(cv2.Sobel(grayscale, cv2.CV_32F, 1, 0, ksize=3), (0, 0), 6)
    gradient_y = cv2.GaussianBlur(cv2.Sobel(grayscale, cv2.CV_32F, 0, 1, ksize=3), (0, 0), 6)

    COAT3_ERROR_THRESHOLD = 8.0
    COAT3_MAX_STROKES = 460
    COAT3_WIDTH_PX = 7.0
    COAT3_STEP_PX = 4.0
    coat3_error_mask = coat_error.copy()
    coat3_error_mask[coat3_error_mask < COAT3_ERROR_THRESHOLD] = 0.0
    coat3_strokes: list[dict[str, Any]] = []
    coat3_random = np.random.default_rng(RANDOM_SEED + 47)
    while len(coat3_strokes) < COAT3_MAX_STROKES:
        seed_flat = int(np.argmax(coat3_error_mask))
        seed_y, seed_x = np.unravel_index(seed_flat, coat3_error_mask.shape)
        if coat3_error_mask[seed_y, seed_x] <= 0:
            break
        flow_x, flow_y = -float(gradient_y[seed_y, seed_x]), float(gradient_x[seed_y, seed_x])
        flow_norm = math.hypot(flow_x, flow_y)
        if flow_norm < 1e-4:
            angle = float(coat3_random.uniform(0, math.tau))
            flow_x, flow_y = math.cos(angle), math.sin(angle)
        else:
            flow_x, flow_y = flow_x / flow_norm, flow_y / flow_norm
        path_points = [np.array([seed_x, seed_y], dtype=np.float32)]
        for direction_sign in (1.0, -1.0):
            point = np.array([seed_x, seed_y], dtype=np.float32)
            for _ in range(11):
                point = point + np.array([flow_x, flow_y]) * COAT3_STEP_PX * direction_sign
                px, py = int(round(point[0])), int(round(point[1]))
                if not (0 <= px < ARTBOARD_SIZE_PX and 0 <= py < ARTBOARD_SIZE_PX):
                    break
                if coat_error[py, px] < COAT3_ERROR_THRESHOLD * 0.55:
                    break
                if direction_sign > 0:
                    path_points.append(point.copy())
                else:
                    path_points.insert(0, point.copy())
        path_array = np.array(path_points)
        if calculate_polyline_length(path_array) < 9.0:
            path_array = np.vstack(
                [
                    path_array[0] - np.array([flow_x, flow_y]) * 5.0,
                    path_array[-1] + np.array([flow_x, flow_y]) * 5.0,
                ]
            )
        cv2.circle(coat3_error_mask, (seed_x, seed_y), 6, 0.0, -1)
        cv2.polylines(
            coat3_error_mask,
            [np.rint(path_array).astype(np.int32).reshape(-1, 1, 2)],
            False,
            0.0,
            9,
        )
        normalized = normalize_points(path_array)
        coat3_strokes.append(
            {
                "phase": "refine_coat",
                "points": normalized,
                "width": round(COAT3_WIDTH_PX / ARTBOARD_SIZE_PX, 6),
                "alpha": round(float(coat3_random.uniform(0.88, 0.96)), 3),
                "blend": "source-over",
                "color": to_hex_color(fine_blur_rgb[seed_y, seed_x]),
                "colorStops": sample_color_stops(normalized, fine_blur_rgb),
            }
        )
    paint_strokes.extend(coat3_strokes)

    # v10-keep variant (EMIT_REFINE=1): the user chose to preserve the v10.1
    # look (face converges to the target under the pen) as an alternate line.
    import os as _os
    if _os.environ.get("EMIT_REFINE") == "1":
        for stroke in focus_detail_strokes:
            paint_strokes.append(
                {
                    "phase": "focus_detail",
                    "mode": "refine",
                    "points": stroke["points"],
                    "width": round(min(0.14, stroke["width"] * 1.5 + 2.4 / ARTBOARD_SIZE_PX), 6),
                    "color": "#000000",
                    "alpha": 1,
                    "blend": "source-over",
                }
            )
        if focus_regions:
            primary = focus_regions[0]
            inset = 0.012
            box = [
                [primary["left"] + inset, primary["top"] + inset],
                [primary["left"] + primary["width"] - inset, primary["top"] + inset],
                [primary["left"] + primary["width"] - inset, primary["top"] + primary["height"] - inset],
                [primary["left"] + inset, primary["top"] + primary["height"] - inset],
                [primary["left"] + inset, primary["top"] + inset],
            ]
            paint_strokes.append(
                {
                    "phase": "focus_detail",
                    "mode": "refine-seal",
                    "slow": True,
                    "contours": [box],
                    "points": [[0, 0], [0.001, 0.001]],
                    "width": 0.001,
                    "color": "#000000",
                    "alpha": 1,
                    "blend": "source-over",
                }
            )

    paint_plan = {
        "version": 10,
        "outputArtboardSizePx": ARTBOARD_SIZE_PX,
        "paperColor": to_hex_color(paper_rgb),
        "palette": [to_hex_color(color) for color in palette_rgb],
        "focusRegions": focus_regions,
        "strokes": paint_strokes,
        "stats": {
            "total": len(paint_strokes),
            "phaseCounts": dict(Counter(stroke["phase"] for stroke in paint_strokes)),
            "highlightDabs": highlight_count,
        },
    }
    paint_json = json.dumps(paint_plan, ensure_ascii=False, separators=(",", ":"))
    PAINT_PLAN_JSON_PATH.write_text(paint_json, encoding="utf-8")
    PAINT_PLAN_JAVASCRIPT_PATH.write_text(
        f"window.MARKER_PAINT_PLAN_V10={paint_json};\n", encoding="utf-8"
    )

    # Offline check: rasterize the v10 plan and measure against the target.
    v10_canvas = np.full((ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX, 3), paper_rgb, dtype=np.float32)
    v10_phase_order = {
        "structure_line": 0,
        "region_fill": 1,
        "region_glaze": 2,
        "refine_coat": 3,
        "detail_line": 4,
        "focus_detail": 5,
    }
    for stroke in sorted(paint_strokes, key=lambda s: v10_phase_order.get(s["phase"], 9)):
        if stroke.get("mode"):
            continue  # refine strokes composite the target at runtime only
        points = np.array(stroke["points"], dtype=np.float32) * ARTBOARD_SIZE_PX
        width_px = max(1, int(round(float(stroke["width"]) * ARTBOARD_SIZE_PX)))
        alpha = float(stroke["alpha"])
        stops = stroke.get("colorStops")
        if stops:
            stroke_buffer = np.zeros((ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX), dtype=np.uint8)
            segment_colors: list[np.ndarray] = []
            cumulative = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
            total_length = max(float(cumulative[-1]), 1e-6)
            stop_rgb = np.array(
                [[int(stop[offset : offset + 2], 16) for offset in (1, 3, 5)] for stop in stops],
                dtype=np.float32,
            )
            buffer_rgb = np.zeros((ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX, 3), dtype=np.float32)
            buffer_mask = np.zeros((ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX), dtype=np.float32)
            for segment_index in range(len(points) - 1):
                t = cumulative[segment_index] / total_length * (len(stops) - 1)
                lower = int(np.floor(t))
                fraction = t - lower
                color = stop_rgb[lower] * (1 - fraction) + stop_rgb[min(lower + 1, len(stops) - 1)] * fraction
                segment_mask = np.zeros((ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX), dtype=np.uint8)
                cv2.line(
                    segment_mask,
                    tuple(np.rint(points[segment_index]).astype(int)),
                    tuple(np.rint(points[segment_index + 1]).astype(int)),
                    255,
                    width_px,
                    cv2.LINE_AA,
                )
                seg = segment_mask.astype(np.float32) / 255.0
                new_area = np.maximum(seg - buffer_mask, 0)
                buffer_rgb += color[None, None, :] * new_area[:, :, None]
                buffer_mask = np.maximum(buffer_mask, seg)
            color_image = buffer_rgb / np.maximum(buffer_mask, 1e-6)[:, :, None]
            alpha_mask = (buffer_mask * alpha)[:, :, None]
            v10_canvas = np.clip(v10_canvas * (1 - alpha_mask) + color_image * alpha_mask, 0, 255)
        else:
            stroke_mask = np.zeros((ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX), dtype=np.uint8)
            cv2.polylines(
                stroke_mask,
                [np.rint(points).astype(np.int32).reshape(-1, 1, 2)],
                False,
                255,
                width_px,
                cv2.LINE_AA,
            )
            alpha_mask = (stroke_mask.astype(np.float32) / 255.0 * alpha)[:, :, None]
            color_rgb = np.array(
                [int(stroke["color"][offset : offset + 2], 16) for offset in (1, 3, 5)],
                dtype=np.float32,
            )
            if stroke.get("blend") == "multiply":
                blended = v10_canvas * (color_rgb[None, None, :] / 255.0)
            else:
                blended = np.broadcast_to(color_rgb, v10_canvas.shape)
            v10_canvas = v10_canvas * (1 - alpha_mask) + blended * alpha_mask
    v10_preview = np.clip(v10_canvas, 0, 255).astype(np.uint8)
    Image.fromarray(v10_preview).save(PAINT_PREVIEW_PATH)
    v10_rmse = float(np.sqrt(np.mean((v10_preview.astype(np.float32) - source_rgb) ** 2))) / 255.0
    print(json.dumps({"v10PaintStrokes": len(paint_strokes), "v10PreviewRmse": round(v10_rmse, 5)}))

    preview_rgb = render_stroke_plan(strokes, paper_rgb)
    Image.fromarray(preview_rgb).save(PREVIEW_IMAGE_PATH)
    Image.fromarray(
        render_stroke_plan(strokes, paper_rgb, {"structure_line", "detail_line", "focus_detail"})
    ).save(LINE_PREVIEW_IMAGE_PATH)
    Image.fromarray(render_stroke_plan(strokes, paper_rgb, {"region_fill", "region_glaze"})).save(
        FILL_PREVIEW_IMAGE_PATH
    )
    Image.fromarray(np.uint8(np.clip(detail_density_map * 255, 0, 255))).save(INK_DENSITY_IMAGE_PATH)

    preview_ssim = structural_similarity(
        source_rgb.astype(np.uint8), preview_rgb, channel_axis=2, data_range=255
    )
    metrics = {
        "strokeCount": len(strokes),
        "phaseCounts": dict(sorted(phase_counts.items())),
        "fillRegionCount": len(fill_regions),
        "linePathCount": len(line_entries),
        "paletteSizeAfterMerge": len(palette_rgb),
        "fillWidthScale": width_scale,
        "previewSsimAgainstFlatTarget": round(float(preview_ssim), 6),
        "focusRegions": focus_regions,
        "architecture": {
            "structure": "contrast-relative thin structures -> skeleton -> merged curves per palette color",
            "color": "healed palette regions hatched with inset-centerline 13-42 px serpentine strokes",
            "residualPolicy": "no residual micro-strokes; paper-similar regions stay unpainted",
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if not MIN_TOTAL_STROKES <= len(strokes) <= MAX_TOTAL_STROKES:
        raise SystemExit(f"stroke count {len(strokes)} outside {MIN_TOTAL_STROKES}..{MAX_TOTAL_STROKES}")


if __name__ == "__main__":
    main()
