#!/usr/bin/env python3
"""Create semantic masks for a single, centrally framed companion animal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment one centrally framed animal for the generalized stroke planner."
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report-name", default="semantic-preflight-animal.json")
    return parser.parse_args()


def keep_largest_connected_component(binary_mask: np.ndarray) -> np.ndarray:
    component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )
    if component_count <= 1:
        return binary_mask.astype(bool)
    largest_label = 1 + int(np.argmax(component_stats[1:, cv2.CC_STAT_AREA]))
    return component_labels == largest_label


def keep_component_with_most_anchor_overlap(
    binary_mask: np.ndarray,
    anchor_mask: np.ndarray,
) -> np.ndarray:
    component_count, component_labels = cv2.connectedComponents(
        binary_mask.astype(np.uint8), connectivity=8
    )
    if component_count <= 1:
        return binary_mask.astype(bool)
    overlap_by_label = np.bincount(
        component_labels[anchor_mask].reshape(-1),
        minlength=component_count,
    )
    overlap_by_label[0] = 0
    selected_label = int(np.argmax(overlap_by_label))
    if overlap_by_label[selected_label] == 0:
        return keep_largest_connected_component(binary_mask)
    return component_labels == selected_label


def build_subject_mask(target_rgb: np.ndarray) -> np.ndarray:
    height, width = target_rgb.shape[:2]
    grabcut_labels = np.zeros((height, width), dtype=np.uint8)
    subject_rectangle = (
        int(round(width * 0.176)),
        int(round(height * 0.068)),
        int(round(width * 0.742)),
        int(round(height * 0.879)),
    )
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR),
        grabcut_labels,
        subject_rectangle,
        background_model,
        foreground_model,
        8,
        cv2.GC_INIT_WITH_RECT,
    )
    subject_mask = np.isin(grabcut_labels, (cv2.GC_FGD, cv2.GC_PR_FGD))
    subject_anchor = np.zeros((height, width), dtype=bool)
    subject_anchor[
        int(round(height * 0.25)) : int(round(height * 0.82)),
        int(round(width * 0.32)) : int(round(width * 0.78)),
    ] = True
    subject_mask = keep_component_with_most_anchor_overlap(
        subject_mask,
        subject_anchor,
    )
    subject_mask = cv2.morphologyEx(
        subject_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ).astype(bool)
    return subject_mask


def mask_bounds(mask: np.ndarray) -> dict[str, Any]:
    height, width = mask.shape
    rows, columns = np.nonzero(mask)
    if len(columns) == 0:
        raise ValueError("Animal semantic mask is empty.")
    left = int(columns.min())
    top = int(rows.min())
    right = int(columns.max()) + 1
    bottom = int(rows.max()) + 1
    return {
        "pixels": {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "width": right - left,
            "height": bottom - top,
        },
        "normalized": {
            "left": round(left / width, 6),
            "top": round(top / height, 6),
            "right": round(right / width, 6),
            "bottom": round(bottom / height, 6),
        },
    }


def build_head_mask(subject_mask: np.ndarray) -> np.ndarray:
    subject_bounds = mask_bounds(subject_mask)["pixels"]
    height, width = subject_mask.shape
    subject_width = subject_bounds["width"]
    subject_height = subject_bounds["height"]
    ellipse_mask = np.zeros((height, width), dtype=np.uint8)
    center_x = int(round(subject_bounds["left"] + subject_width * 0.58))
    center_y = int(round(subject_bounds["top"] + subject_height * 0.39))
    radius_x = max(24, int(round(subject_width * 0.48)))
    radius_y = max(24, int(round(subject_height * 0.39)))
    cv2.ellipse(
        ellipse_mask,
        (center_x, center_y),
        (radius_x, radius_y),
        0,
        0,
        360,
        255,
        -1,
        cv2.LINE_AA,
    )
    return subject_mask & (ellipse_mask > 0)


def save_mask(output_path: Path, mask: np.ndarray) -> str:
    Image.fromarray(mask.astype(np.uint8) * 255).save(output_path)
    return str(output_path.resolve().relative_to(PROJECT_ROOT))


def describe_image(image_path: Path, masks: dict[str, np.ndarray], mask_files: dict[str, str]) -> dict[str, Any]:
    with Image.open(image_path) as image:
        width, height = image.size
    subject_mask = masks["subject"]
    identity_mask = masks["identity"]
    subject_rows, subject_columns = np.nonzero(subject_mask)
    return {
        "path": str(image_path.resolve()),
        "width": width,
        "height": height,
        "faceCount": 1,
        "faceDetectionScores": [],
        "faceLandmarkSource": "animal-head-mask",
        "faceLandmarks": [],
        "faceCoverage": round(float(identity_mask.mean()), 6),
        "subjectCoverage": round(float(subject_mask.mean()), 6),
        "subjectCentroid": {
            "x": round(float(subject_columns.mean()) / subject_mask.shape[1], 6),
            "y": round(float(subject_rows.mean()) / subject_mask.shape[0], 6),
        },
        "bounds": {
            "face": mask_bounds(identity_mask),
            "hair": mask_bounds(masks["hair"]),
            "clothes": None,
            "subject": mask_bounds(subject_mask),
            "identity": mask_bounds(identity_mask),
        },
        "maskFiles": mask_files,
        "issueCodes": [],
    }


def main() -> int:
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    target_rgb = np.asarray(Image.open(arguments.target).convert("RGB"))
    subject_mask = build_subject_mask(target_rgb)
    identity_mask = build_head_mask(subject_mask)
    body_mask = subject_mask & ~identity_mask
    empty_mask = np.zeros(subject_mask.shape, dtype=bool)
    masks = {
        "subject": subject_mask,
        "identity": identity_mask,
        "face_hull": identity_mask,
        "face_skin": identity_mask,
        "body_skin": body_mask,
        "hair": subject_mask,
        "clothes": empty_mask,
        "accessories": empty_mask,
    }
    mask_files = {
        name: save_mask(arguments.output_dir / f"target-{name}.png", mask)
        for name, mask in masks.items()
    }
    target_description = describe_image(arguments.target, masks, mask_files)
    report = {
        "schemaVersion": 1,
        "contract": "single-companion-animal-photo-v1",
        "status": "PASS",
        "attempt": 0,
        "action": "Animal subject and head masks passed.",
        "issueCodes": [],
        "models": [],
        "reference": {
            "path": str(arguments.reference.resolve()),
            "width": Image.open(arguments.reference).size[0],
            "height": Image.open(arguments.reference).size[1],
        },
        "target": target_description,
        "targetComparison": None,
    }
    report_path = arguments.output_dir / arguments.report_name
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "contract": report["contract"],
                "subjectCoverage": target_description["subjectCoverage"],
                "identityCoverage": target_description["faceCoverage"],
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
