#!/usr/bin/env python3
"""Build the semantic, coordinate-free base plan for a generalized S run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from build_anime_refinement_plan_h62 import (
    ARTBOARD_SIZE_PX,
    build_final_ink_strokes,
    build_identity_ink_strokes,
    build_local_detail_strokes,
    build_multiscale_refinement_strokes,
    build_shape_following_fill_strokes,
    build_structure_strokes,
    draw_stroke_on_simulation,
    simulate_plan,
)
from build_flat_marker_plan_v7 import to_hex_color


PROJECT_ROOT = Path(__file__).resolve().parent
PAPER_RGB = np.array([247, 241, 227], dtype=np.float32)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an S-compatible stroke plan from semantic masks, without manual coordinates."
    )
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--semantic-report", required=True, type=Path)
    parser.add_argument("--plan-json", required=True, type=Path)
    parser.add_argument("--plan-js", required=True, type=Path)
    parser.add_argument("--preview", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument(
        "--budget-scale",
        type=float,
        default=1.0,
        help="Scale expensive refinement budgets. Production default is 1.0.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_project_path(path_text: str) -> Path:
    candidate = Path(path_text)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_target_mask(report: dict[str, Any], name: str) -> np.ndarray:
    mask_path_text = report["target"]["maskFiles"][name]
    mask_image = Image.open(resolve_project_path(mask_path_text)).convert("L")
    resized_mask = mask_image.resize(
        (ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX), Image.Resampling.NEAREST
    )
    return np.asarray(resized_mask) >= 128


def padded_normalized_region(
    bounds: dict[str, Any],
    padding_x_ratio: float,
    padding_y_ratio: float,
) -> dict[str, float]:
    normalized = bounds["normalized"]
    width = normalized["right"] - normalized["left"]
    height = normalized["bottom"] - normalized["top"]
    left = max(0.0, normalized["left"] - width * padding_x_ratio)
    top = max(0.0, normalized["top"] - height * padding_y_ratio)
    right = min(1.0, normalized["right"] + width * padding_x_ratio)
    bottom = min(1.0, normalized["bottom"] + height * padding_y_ratio)
    return {
        "left": round(left, 6),
        "top": round(top, 6),
        "width": round(right - left, 6),
        "height": round(bottom - top, 6),
    }


def build_focus_regions(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    target_bounds = report["target"]["bounds"]
    face_region = padded_normalized_region(target_bounds["identity"], 0.18, 0.14)
    subject_region = padded_normalized_region(target_bounds["subject"], 0.04, 0.03)
    subject_top = subject_region["top"]
    upper_body_bottom = min(
        subject_top + subject_region["height"],
        max(
            face_region["top"] + face_region["height"],
            subject_top + subject_region["height"] * 0.62,
        ),
    )
    upper_body_region = {
        "left": subject_region["left"],
        "top": subject_top,
        "width": subject_region["width"],
        "height": round(max(0.08, upper_body_bottom - subject_top), 6),
    }
    return {
        "face": face_region,
        "upperBody": upper_body_region,
        "subject": subject_region,
    }


def scaled_budget(base_budget: int, coverage: float, reference_coverage: float, scale: float) -> int:
    coverage_scale = float(np.clip(coverage / max(reference_coverage, 1e-6), 0.55, 1.45))
    return max(1, int(round(base_budget * coverage_scale * scale)))


def main() -> int:
    arguments = parse_arguments()
    if not 0.001 <= arguments.budget_scale <= 2.0:
        raise ValueError("--budget-scale must be between 0.001 and 2.0")

    semantic_report = json.loads(arguments.semantic_report.read_text(encoding="utf-8"))
    if semantic_report.get("status") != "PASS" or not semantic_report.get("target"):
        raise RuntimeError("Generalized planning requires a PASS report with a target image.")

    for output_path in (
        arguments.plan_json,
        arguments.plan_js,
        arguments.preview,
        arguments.metrics,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    target_image = Image.open(arguments.target).convert("RGB").resize(
        (ARTBOARD_SIZE_PX, ARTBOARD_SIZE_PX), Image.Resampling.LANCZOS
    )
    target_rgb = np.asarray(target_image, dtype=np.float32)
    subject_mask = load_target_mask(semantic_report, "subject")
    identity_mask = load_target_mask(semantic_report, "identity")
    focus_regions = build_focus_regions(semantic_report)

    final_ink_strokes = build_final_ink_strokes(target_rgb)
    structure_count = max(80, int(round(360 * arguments.budget_scale ** 0.35)))
    structure_strokes = build_structure_strokes(final_ink_strokes, structure_count)
    fill_strokes, palette_hex, fill_region_count = build_shape_following_fill_strokes(
        target_rgb,
        PAPER_RGB,
        [focus_regions["face"]],
    )
    base_plan = {
        "paperColor": to_hex_color(PAPER_RGB),
        "palette": palette_hex,
        "focusRegions": focus_regions,
        "strokes": [*structure_strokes, *fill_strokes],
    }
    simulation_rgb = simulate_plan(base_plan)

    refinement_strokes, refinement_metrics = build_multiscale_refinement_strokes(
        simulation_rgb,
        target_rgb,
        subject_mask=subject_mask,
        budget_scale=arguments.budget_scale,
    )
    subject_budget = scaled_budget(
        5200,
        float(subject_mask.mean()),
        reference_coverage=0.385,
        scale=arguments.budget_scale,
    )
    identity_budget = scaled_budget(
        4800,
        float(identity_mask.mean()),
        reference_coverage=0.095,
        scale=arguments.budget_scale,
    )
    subject_detail_strokes, subject_metrics = build_local_detail_strokes(
        simulation_rgb,
        target_rgb,
        subject_mask,
        "anime_subject_detail",
        width_px=1.35,
        maximum_strokes=subject_budget,
        error_threshold=2.2,
        color_tolerance=10.0,
    )
    identity_detail_strokes, identity_metrics = build_local_detail_strokes(
        simulation_rgb,
        target_rgb,
        identity_mask,
        "anime_identity_detail",
        width_px=0.90,
        maximum_strokes=identity_budget,
        error_threshold=1.35,
        color_tolerance=7.0,
    )

    for stroke in final_ink_strokes:
        draw_stroke_on_simulation(simulation_rgb, stroke)
    identity_ink_strokes = build_identity_ink_strokes(final_ink_strokes, identity_mask)
    for stroke in identity_ink_strokes:
        draw_stroke_on_simulation(simulation_rgb, stroke)

    all_strokes = [
        *base_plan["strokes"],
        *refinement_strokes,
        *subject_detail_strokes,
        *identity_detail_strokes,
        *final_ink_strokes,
        *identity_ink_strokes,
    ]
    preview_rgb = np.clip(simulation_rgb, 0, 255).astype(np.uint8)
    preview_ssim = float(
        structural_similarity(
            target_rgb.astype(np.uint8),
            preview_rgb,
            channel_axis=2,
            data_range=255,
        )
    )
    normalized_rmse = float(np.sqrt(np.mean((target_rgb - preview_rgb) ** 2))) / 255.0
    phase_counts = dict(Counter(stroke["phase"] for stroke in all_strokes))

    plan = {
        "version": "S-generalized-base-1",
        "outputArtboardSizePx": ARTBOARD_SIZE_PX,
        "paperColor": base_plan["paperColor"],
        "palette": base_plan["palette"],
        "focusRegions": focus_regions,
        "strokes": all_strokes,
        "stats": {
            "total": len(all_strokes),
            "phaseCounts": phase_counts,
            "fillRegionCount": fill_region_count,
            "subjectCoverage": round(float(subject_mask.mean()), 6),
            "identityCoverage": round(float(identity_mask.mean()), 6),
            "subjectDetailBudget": subject_budget,
            "identityDetailBudget": identity_budget,
            "previewSsim": round(preview_ssim, 6),
            "previewNormalizedRmse": round(normalized_rmse, 6),
        },
    }
    compact_plan_json = json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
    arguments.plan_json.write_text(compact_plan_json, encoding="utf-8")
    arguments.plan_js.write_text(
        f"window.MARKER_PAINT_PLAN_GENERALIZED={compact_plan_json};\n",
        encoding="utf-8",
    )
    Image.fromarray(preview_rgb).save(arguments.preview)

    metrics = {
        "planVersion": plan["version"],
        "targetSha256": sha256_file(arguments.target),
        "semanticReportSha256": sha256_file(arguments.semantic_report),
        "budgetScale": arguments.budget_scale,
        "strokeCount": len(all_strokes),
        "phaseCounts": phase_counts,
        "passMetrics": [*refinement_metrics, subject_metrics, identity_metrics],
        "focusRegions": focus_regions,
        "subjectDetailBudget": subject_budget,
        "identityDetailBudget": identity_budget,
        "finalPreviewSsim": round(preview_ssim, 6),
        "finalPreviewNormalizedRmse": round(normalized_rmse, 6),
        "runtimeTargetCompositing": False,
        "wholeCanvasScanStrokes": 0,
        "manualCoordinateOverrides": 0,
    }
    arguments.metrics.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

