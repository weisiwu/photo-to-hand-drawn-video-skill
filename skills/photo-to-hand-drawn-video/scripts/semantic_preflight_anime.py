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
    analyze_image,
    compare_target_topology,
    derive_status,
    validate_model_files,
)

CONTRACT = "single-person-anime-photo-v1"


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
    return parser.parse_args()


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
            target_issues = [
                f"TARGET_{code}" for code in target_issues if code != "FACE_TOO_SMALL"
            ]
            issue_codes.extend(target_issues)
            comparison, topology_issues = compare_target_topology(
                reference, reference_masks, target, target_masks
            )
            issue_codes.extend(topology_issues)

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
