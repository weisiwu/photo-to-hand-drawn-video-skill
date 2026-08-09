#!/usr/bin/env python3
"""Offline semantic preflight for anime/illustration single-person runs.

The anime adapter drives the face region from the hair mask instead of
MediaPipe face detectors, which are trained on real faces and are unreliable
for stylized 2D illustrations. Everything else (subject segmentation, target
topology comparison, issue codes) shares the same contract as the human
preflight so the rest of the pipeline is unchanged.

Usage:
    semantic_preflight_anime.py --reference ref.png --target target.png \
        --output-dir runs/xxx --attempt 0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from semantic_preflight import (
    MODEL_DIRECTORY,
    SEGMENTATION_MODEL_PATH,
    SUBJECT_LABELS,
    RETRYABLE_TARGET_CODES,
    Bounds,
    analyze_image,
    compare_target_topology,
    derive_status,
    face_mask_from_bounds,
    save_mask,
    validate_model_files,
)

CONTRACT = "single-person-anime-photo-v1"

# MediaPipe face-mesh indices used by the S refinement stages (see
# build_upper_limit_plan_s.py FACE_FEATURE_PATHS). Positions below are
# relative to the face hull bounds (x: 0..1 left-to-right, y: 0..1 top-to-bottom).
ANIME_FEATURE_POSITIONS: dict[int, tuple[float, float]] = {
    # leftEyebrow
    70: (0.30, 0.20), 63: (0.36, 0.17), 105: (0.42, 0.15), 66: (0.48, 0.14), 107: (0.52, 0.15),
    # rightEyebrow
    336: (0.70, 0.20), 296: (0.64, 0.17), 334: (0.58, 0.15), 293: (0.52, 0.14), 300: (0.48, 0.15),
    # leftEye
    33: (0.30, 0.32), 160: (0.36, 0.30), 158: (0.42, 0.30), 133: (0.47, 0.32),
    153: (0.43, 0.35), 144: (0.36, 0.35),
    # rightEye
    362: (0.70, 0.32), 385: (0.64, 0.30), 387: (0.58, 0.30), 263: (0.53, 0.32),
    373: (0.57, 0.35), 380: (0.64, 0.35),
    # noseBridge
    168: (0.50, 0.40), 6: (0.50, 0.44), 197: (0.52, 0.48), 195: (0.48, 0.48),
    5: (0.50, 0.52), 4: (0.50, 0.56),
    # noseBase
    98: (0.46, 0.60), 2: (0.50, 0.62), 327: (0.54, 0.60),
    # outerLips
    61: (0.38, 0.70), 146: (0.42, 0.68), 91: (0.46, 0.67), 181: (0.50, 0.67),
    84: (0.54, 0.68), 17: (0.58, 0.70), 314: (0.56, 0.74), 405: (0.52, 0.75),
    321: (0.48, 0.75), 375: (0.44, 0.74), 291: (0.40, 0.72),
}


def generate_rule_based_landmarks(
    face_bounds_px: dict[str, int], width: int, height: int
) -> list[dict[str, float]]:
    """Synthesize 468 face-mesh landmarks for stylized 2D faces.

    Anime faces are not detected by MediaPipe, but the S refinement stages
    require 468 landmarks (FACE_FEATURE_PATHS indices) to emit facial-feature
    strokes. We lay out the feature indices from ANIME_FEATURE_POSITIONS and
    fill the remaining indices with an elliptical grid over the face hull.
    """
    left = float(face_bounds_px["left"])
    top = float(face_bounds_px["top"])
    face_w = float(face_bounds_px["width"])
    face_h = float(face_bounds_px["height"])
    center_x = left + face_w * 0.5
    center_y = top + face_h * 0.52  # features sit slightly above center
    landmarks: list[dict[str, float]] = []
    for index in range(468):
        if index in ANIME_FEATURE_POSITIONS:
            fx, fy = ANIME_FEATURE_POSITIONS[index]
            px = left + fx * face_w
            py = top + fy * face_h
        else:
            # Deterministic elliptical grid over the face hull.
            grid_position = index / 468.0
            angle = grid_position * math.tau
            radius_x = 0.5 * math.sqrt(max(0.0, 1.0 - ((index % 13) / 13.0) ** 2))
            radius_y = 0.5 * (1.0 - (index % 13) / 13.0 * 0.55)
            px = center_x + math.cos(angle) * radius_x * face_w
            py = center_y + math.sin(angle) * radius_y * face_h * 0.9
        landmarks.append(
            {
                "x": round(max(0.0, min(1.0, px / width)), 6),
                "y": round(max(0.0, min(1.0, py / height)), 6),
            }
        )
    return landmarks


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a single-person anime/illustration reference and target offline."
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--attempt",
        type=int,
        default=0,
        choices=(0, 1, 2),
        help="Target-generation attempt. Retryable target failures become UNSUPPORTED at 2.",
    )
    parser.add_argument("--report-name", default="semantic-preflight.json")
    parser.add_argument(
        "--target-transform",
        type=Path,
        help="Optional JSON written by make_square_target.py describing the "
        "crop/scale applied to build the target. When present, the face region "
        "is geometry-mapped from the reference (MediaPipe segmentation is "
        "unreliable on upscaled anime crops), and subject-drift checks are "
        "relaxed because the target is a controlled reframe.",
    )
    return parser.parse_args()


def map_face_from_reference(
    reference: dict[str, Any], transform: dict[str, Any]
) -> dict[str, Any] | None:
    """Map the reference face bounds through the target transform.

    Transform semantics (see make_square_target.py): the reference is scaled
    by `scale`, then a square window (windowSide x windowSide) at
    (windowLeft, windowTop) is cropped and resized to targetSize. Normalized
    coordinates survive the final resize, so target_norm = (ref_px * scale
    - window_origin) / window_side.
    """
    face_bounds = (reference.get("bounds") or {}).get("face")
    if not face_bounds or not face_bounds.get("pixels"):
        return None
    pixels = face_bounds["pixels"]
    scale = float(transform["scale"])
    window_left = float(transform["windowLeft"])
    window_top = float(transform["windowTop"])
    window_side = float(transform["windowSide"])

    def to_normalized(pixel_value: float, window_origin: float) -> float:
        return max(0.0, min(1.0, (pixel_value * scale - window_origin) / window_side))

    normalized = {
        "left": round(to_normalized(pixels["left"], window_left), 6),
        "top": round(to_normalized(pixels["top"], window_top), 6),
        "right": round(to_normalized(pixels["right"], window_left), 6),
        "bottom": round(to_normalized(pixels["bottom"], window_top), 6),
    }
    normalized["width"] = round(normalized["right"] - normalized["left"], 6)
    normalized["height"] = round(normalized["bottom"] - normalized["top"], 6)
    return normalized


def main() -> int:
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    validated_models = validate_model_files()

    # Anime mode only needs the segmentation model; face detection is driven
    # by the hair-mask estimator instead of MediaPipe face models.
    segmenter_options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(SEGMENTATION_MODEL_PATH),
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.IMAGE,
        output_category_mask=True,
        output_confidence_masks=False,
    )

    with vision.ImageSegmenter.create_from_options(segmenter_options) as segmenter:
        reference, reference_masks, reference_issues = analyze_image(
            arguments.reference,
            "reference",
            arguments.output_dir,
            None,
            segmenter,
            None,
            anime_mode=True,
        )
        target = None
        comparison = None
        issue_codes = list(reference_issues)
        if arguments.target:
            target, target_masks, target_issues = analyze_image(
                arguments.target,
                "target",
                arguments.output_dir,
                None,
                segmenter,
                None,
                anime_mode=True,
            )
            target_transform = None
            if arguments.target_transform and arguments.target_transform.exists():
                target_transform = json.loads(
                    arguments.target_transform.read_text(encoding="utf-8")
                )
            if target_transform:
                mapped_face = map_face_from_reference(reference, target_transform)
                if mapped_face:
                    target_width = target["width"]
                    target_height = target["height"]
                    target["faceLandmarkSource"] = "geometry-mapped"
                    target["bounds"]["face"] = {
                        "pixels": {
                            "left": round(mapped_face["left"] * target_width),
                            "top": round(mapped_face["top"] * target_height),
                            "right": round(mapped_face["right"] * target_width),
                            "bottom": round(mapped_face["bottom"] * target_height),
                            "width": round(mapped_face["width"] * target_width),
                            "height": round(mapped_face["height"] * target_height),
                        },
                        "normalized": mapped_face,
                    }
                    face_bounds_px = target["bounds"]["face"]["pixels"]
                    face_hull_mask = face_mask_from_bounds(
                        Bounds(
                            face_bounds_px["left"],
                            face_bounds_px["top"],
                            face_bounds_px["right"],
                            face_bounds_px["bottom"],
                        ),
                        target_width,
                        target_height,
                    )
                    save_mask(arguments.output_dir / "target-face_hull.png", face_hull_mask)
                    save_mask(arguments.output_dir / "target-identity.png", face_hull_mask)
                    # Rebuild a plausible hair region above/around the face so
                    # hair-detail planning has a region to work with.
                    hair_top = max(0, face_bounds_px["top"] - round(face_bounds_px["height"] * 0.9))
                    hair_mask = np.zeros((target_height, target_width), dtype=bool)
                    hair_mask[
                        hair_top : face_bounds_px["bottom"],
                        max(0, face_bounds_px["left"] - round(face_bounds_px["width"] * 0.5)) : min(
                            target_width,
                            face_bounds_px["right"] + round(face_bounds_px["width"] * 0.5),
                        ),
                    ] = True
                    save_mask(arguments.output_dir / "target-hair.png", hair_mask)
                    # Rebuild subject/clothes/skin masks from the mapped face so
                    # refinement and polish stages have regions to work with
                    # (MediaPipe segmentation fails on upscaled anime crops).
                    subject_mask = np.zeros((target_height, target_width), dtype=bool)
                    subject_top = max(0, face_bounds_px["top"] - round(face_bounds_px["height"] * 0.35))
                    subject_left = max(0, face_bounds_px["left"] - round(face_bounds_px["width"] * 0.9))
                    subject_right = min(target_width, face_bounds_px["right"] + round(face_bounds_px["width"] * 0.9))
                    subject_bottom = target_height
                    subject_mask[subject_top:subject_bottom, subject_left:subject_right] = True
                    # Shave the rectangle toward the painted subject: keep the
                    # central vertical band wide, narrow toward the bottom.
                    taper = np.ones(target_height, dtype=np.float32)
                    body_start = face_bounds_px["bottom"]
                    for y in range(body_start, target_height):
                        fraction = (y - body_start) / max(1, target_height - body_start)
                        taper[y] = max(0.30, 1.0 - fraction * 0.75)
                    for y in range(subject_top, target_height):
                        half_width = int((subject_right - subject_left) / 2 * taper[y])
                        center = (subject_left + subject_right) // 2
                        subject_mask[y, max(0, center - half_width):min(target_width, center + half_width)] = True
                    save_mask(arguments.output_dir / "target-subject.png", subject_mask)
                    clothes_top = min(target_height - 1, face_bounds_px["bottom"] + round(face_bounds_px["height"] * 0.1))
                    clothes_mask = np.zeros((target_height, target_width), dtype=bool)
                    clothes_mask[clothes_top:target_height, :] = subject_mask[clothes_top:target_height, :]
                    save_mask(arguments.output_dir / "target-clothes.png", clothes_mask)
                    body_skin_mask = np.zeros((target_height, target_width), dtype=bool)
                    body_skin_top = face_bounds_px["bottom"]
                    body_skin_bottom = min(target_height - 1, clothes_top)
                    if body_skin_bottom > body_skin_top:
                        body_skin_mask[body_skin_top:body_skin_bottom, :] = subject_mask[body_skin_top:body_skin_bottom, :]
                    save_mask(arguments.output_dir / "target-body_skin.png", body_skin_mask)
                    face_skin_mask = face_mask_from_bounds(
                        Bounds(
                            face_bounds_px["left"] + round(face_bounds_px["width"] * 0.12),
                            face_bounds_px["top"] + round(face_bounds_px["height"] * 0.18),
                            face_bounds_px["right"] - round(face_bounds_px["width"] * 0.12),
                            face_bounds_px["bottom"] - round(face_bounds_px["height"] * 0.08),
                        ),
                        target_width,
                        target_height,
                    )
                    save_mask(arguments.output_dir / "target-face_skin.png", face_skin_mask)
                    subject_bounds = {
                        "left": subject_left,
                        "top": subject_top,
                        "right": subject_right,
                        "bottom": target_height,
                        "width": subject_right - subject_left,
                        "height": target_height - subject_top,
                    }
                    target["bounds"]["subject"] = {
                        "pixels": subject_bounds,
                        "normalized": {
                            "left": round(subject_left / target_width, 6),
                            "top": round(subject_top / target_height, 6),
                            "right": round(subject_right / target_width, 6),
                            "bottom": 1.0,
                            "width": round((subject_right - subject_left) / target_width, 6),
                            "height": round((target_height - subject_top) / target_height, 6),
                        },
                    }
                    target["subjectCoverage"] = round(float(subject_mask.mean()), 6)
                    identity_bounds = face_bounds_px
                    target["bounds"]["identity"] = {
                        "pixels": identity_bounds,
                        "normalized": mapped_face,
                    }
                    target["bounds"]["hair"] = {
                        "pixels": {
                            "left": max(0, face_bounds_px["left"] - round(face_bounds_px["width"] * 0.5)),
                            "top": hair_top,
                            "right": min(target_width, face_bounds_px["right"] + round(face_bounds_px["width"] * 0.5)),
                            "bottom": face_bounds_px["bottom"],
                            "width": 0,
                            "height": 0,
                        },
                        "normalized": {
                            "left": round(max(0.0, mapped_face["left"] - mapped_face["width"] * 0.5), 6),
                            "top": round(max(0.0, mapped_face["top"] - mapped_face["height"] * 0.9), 6),
                            "right": round(min(1.0, mapped_face["right"] + mapped_face["width"] * 0.5), 6),
                            "bottom": mapped_face["bottom"],
                            "width": 0,
                            "height": 0,
                        },
                    }
                    target["faceLandmarks"] = generate_rule_based_landmarks(
                        face_bounds_px, target_width, target_height
                    )
                    target["faceCoverage"] = round(
                        float(face_hull_mask.mean()), 6
                    )
            # Anime faces have no MediaPipe landmarks; synthesize the 468-point
            # face mesh so the S refinement stages can emit facial-feature strokes.
            target_face_bounds = (target.get("bounds") or {}).get("face")
            if target_face_bounds and target_face_bounds.get("pixels") and not target.get("faceLandmarks"):
                target["faceLandmarks"] = generate_rule_based_landmarks(
                    target_face_bounds["pixels"],
                    target["width"],
                    target["height"],
                )
            target_issues = [
                f"TARGET_{code}" for code in target_issues if code != "FACE_TOO_SMALL"
            ]
            issue_codes.extend(target_issues)
            comparison, topology_issues = compare_target_topology(
                reference, reference_masks, target, target_masks
            )
            issue_codes.extend(topology_issues)
            if target_transform:
                # The target is a controlled reframe of the reference
                # (face geometry-mapped), so topology checks do not apply.
                issue_codes = [
                    code
                    for code in issue_codes
                    if code
                    not in (
                        "TARGET_NO_FACE",
                        "TARGET_FACE_DRIFT",
                        "TARGET_FACE_SCALE_DRIFT",
                        "TARGET_SUBJECT_DRIFT",
                        "TARGET_SUBJECT_LAYOUT_DRIFT",
                    )
                ]

    issue_codes = sorted(set(issue_codes))
    status, action = derive_status(issue_codes, arguments.attempt)
    report = {
        "schemaVersion": 1,
        "contract": CONTRACT,
        "status": status,
        "attempt": arguments.attempt,
        "action": action,
        "issueCodes": issue_codes,
        "models": validated_models,
        "reference": reference,
        "target": target,
        "targetComparison": comparison,
    }
    report_path = arguments.output_dir / arguments.report_name
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "report": str(report_path), "issueCodes": issue_codes}))
    return {"PASS": 0, "RETRY": 2, "UNSUPPORTED": 3}[status]


if __name__ == "__main__":
    raise SystemExit(main())
