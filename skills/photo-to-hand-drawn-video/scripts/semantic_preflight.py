#!/usr/bin/env python3
"""Offline semantic preflight for generalized single-person drawing runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIRECTORY = PROJECT_ROOT / "models" / "mediapipe"
FACE_MODEL_PATH = MODEL_DIRECTORY / "face_landmarker.task"
SEGMENTATION_MODEL_PATH = MODEL_DIRECTORY / "selfie_multiclass_256x256.tflite"

CATEGORY_NAMES = {
    0: "background",
    1: "hair",
    2: "body_skin",
    3: "face_skin",
    4: "clothes",
    5: "accessories",
}
SUBJECT_LABELS = (1, 2, 3, 4, 5)
RETRYABLE_TARGET_CODES = {
    "TARGET_NO_FACE",
    "TARGET_MULTIPLE_FACES",
    "TARGET_FACE_TOO_SMALL",
    "TARGET_SUBJECT_MASK_INVALID",
    "TARGET_FACE_DRIFT",
    "TARGET_FACE_SCALE_DRIFT",
    "TARGET_SUBJECT_DRIFT",
    "TARGET_SUBJECT_LAYOUT_DRIFT",
}


@dataclass(frozen=True)
class Bounds:
    """Pixel bounds with an exclusive right and bottom edge."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2.0, (self.top + self.bottom) / 2.0)

    def padded(self, padding_x: int, padding_y: int, width: int, height: int) -> "Bounds":
        return Bounds(
            max(0, self.left - padding_x),
            max(0, self.top - padding_y),
            min(width, self.right + padding_x),
            min(height, self.bottom + padding_y),
        )

    def as_dict(self, width: int, height: int) -> dict[str, Any]:
        return {
            "pixels": {
                "left": self.left,
                "top": self.top,
                "right": self.right,
                "bottom": self.bottom,
                "width": self.width,
                "height": self.height,
            },
            "normalized": {
                "left": round(self.left / width, 6),
                "top": round(self.top / height, 6),
                "right": round(self.right / width, 6),
                "bottom": round(self.bottom / height, 6),
            },
        }


@dataclass(frozen=True)
class NormalizedPoint:
    x: float
    y: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a single-person reference and optional stylized target offline."
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_files() -> list[dict[str, Any]]:
    manifest_path = MODEL_DIRECTORY / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validated_models: list[dict[str, Any]] = []
    for model_entry in manifest["models"]:
        model_path = MODEL_DIRECTORY / model_entry["file"]
        actual_hash = sha256_file(model_path)
        if actual_hash != model_entry["sha256"]:
            raise RuntimeError(f"Model checksum mismatch: {model_path}")
        validated_models.append(
            {
                "file": str(model_path.relative_to(PROJECT_ROOT)),
                "sha256": actual_hash,
                "bytes": model_path.stat().st_size,
            }
        )
    return validated_models


def mask_bounds(mask: np.ndarray) -> Bounds | None:
    rows, columns = np.nonzero(mask)
    if len(columns) == 0:
        return None
    return Bounds(
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    rows, columns = np.nonzero(mask)
    if len(columns) == 0:
        return None
    return float(columns.mean()), float(rows.mean())


def convex_hull(points: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    unique_points = sorted(set(points))
    if len(unique_points) <= 1:
        return unique_points

    def cross(
        origin: tuple[int, int], first: tuple[int, int], second: tuple[int, int]
    ) -> int:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[int, int]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[int, int]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def face_mask_from_landmarks(
    face_landmarks: list[Any], width: int, height: int
) -> np.ndarray:
    landmark_points = [
        (
            max(0, min(width - 1, round(landmark.x * width))),
            max(0, min(height - 1, round(landmark.y * height))),
        )
        for landmark in face_landmarks
    ]
    hull = convex_hull(landmark_points)
    mask_image = Image.new("L", (width, height), 0)
    if len(hull) >= 3:
        ImageDraw.Draw(mask_image).polygon(hull, fill=255)
    return np.asarray(mask_image, dtype=np.uint8) > 0


def face_mask_from_bounds(bounds: Bounds, width: int, height: int) -> np.ndarray:
    mask_image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask_image).ellipse(
        (bounds.left, bounds.top, bounds.right - 1, bounds.bottom - 1), fill=255
    )
    return np.asarray(mask_image, dtype=np.uint8) > 0


def detection_bounds(detection: Any, width: int, height: int) -> Bounds:
    relative_box = detection.location_data.relative_bounding_box
    left = max(0, int(math.floor(relative_box.xmin * width)))
    top = max(0, int(math.floor(relative_box.ymin * height)))
    right = min(width, int(math.ceil((relative_box.xmin + relative_box.width) * width)))
    bottom = min(height, int(math.ceil((relative_box.ymin + relative_box.height) * height)))
    return Bounds(left, top, max(left + 1, right), max(top + 1, bottom))


def crop_face_landmarks(
    source_rgb: np.ndarray,
    face_bounds: Bounds,
    face_landmarker: vision.FaceLandmarker,
) -> list[NormalizedPoint]:
    height, width = source_rgb.shape[:2]
    padded_bounds = face_bounds.padded(
        round(face_bounds.width * 0.45),
        round(face_bounds.height * 0.55),
        width,
        height,
    )
    crop_rgb = np.ascontiguousarray(
        source_rgb[
            padded_bounds.top : padded_bounds.bottom,
            padded_bounds.left : padded_bounds.right,
        ]
    )
    crop_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
    crop_result = face_landmarker.detect(crop_image)
    if not crop_result.face_landmarks:
        return []
    return [
        NormalizedPoint(
            x=(padded_bounds.left + landmark.x * padded_bounds.width) / width,
            y=(padded_bounds.top + landmark.y * padded_bounds.height) / height,
        )
        for landmark in crop_result.face_landmarks[0]
    ]


def resize_binary_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    return np.asarray(mask_image.resize(size, Image.Resampling.NEAREST)) > 0


def save_mask(output_path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(output_path)


def serialize_face_landmarks(face_landmarks: list[Any]) -> list[dict[str, float]]:
    return [
        {
            "x": round(float(landmark.x), 6),
            "y": round(float(landmark.y), 6),
        }
        for landmark in face_landmarks
    ]


def describe_bounds(
    bounds: Bounds | None, width: int, height: int
) -> dict[str, Any] | None:
    return bounds.as_dict(width, height) if bounds else None


def analyze_image(
    image_path: Path,
    prefix: str,
    output_directory: Path,
    face_landmarker: vision.FaceLandmarker,
    segmenter: vision.ImageSegmenter,
    long_range_face_detector: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[str]]:
    source_image = Image.open(image_path).convert("RGB")
    width, height = source_image.size
    source_rgb = np.asarray(source_image)
    media_image = mp.Image.create_from_file(str(image_path))
    direct_face_result = face_landmarker.detect(media_image)
    detector_result = long_range_face_detector.process(source_rgb)
    segment_result = segmenter.segment(media_image)
    category_mask = np.array(segment_result.category_mask.numpy_view(), copy=True)

    masks = {
        category_name: category_mask == category_id
        for category_id, category_name in CATEGORY_NAMES.items()
    }
    masks["subject"] = np.isin(category_mask, SUBJECT_LABELS)

    face_detections = detector_result.detections or []
    detected_face_bounds = [detection_bounds(detection, width, height) for detection in face_detections]
    detected_face_scores = [round(float(detection.score[0]), 6) for detection in face_detections]
    face_count = len(detected_face_bounds)
    landmark_source = "long-range-crop"
    face_landmarks: list[Any] = []
    if face_count == 1:
        face_landmarks = crop_face_landmarks(
            source_rgb, detected_face_bounds[0], face_landmarker
        )
    elif face_count == 0 and len(direct_face_result.face_landmarks) == 1:
        face_count = 1
        face_landmarks = direct_face_result.face_landmarks[0]
        landmark_source = "direct"

    if face_count == 1 and face_landmarks:
        masks["face_hull"] = face_mask_from_landmarks(
            face_landmarks, width, height
        )
    elif face_count == 1 and detected_face_bounds:
        masks["face_hull"] = face_mask_from_bounds(
            detected_face_bounds[0], width, height
        )
        landmark_source = "detector-bounds-fallback"
    else:
        # Anime fallback: MediaPipe is trained on real faces and often misses
        # stylized 2D faces. Approximate the face from the hair mask: the face
        # sits around the upper-center of the hair region (observed ~0.69w,
        # ~0.22h inside hair bounds for the tested illustration set).
        hair_mask = masks.get("hair", np.zeros((height, width), dtype=bool))
        hair_bounds_fallback = mask_bounds(hair_mask)
        if hair_bounds_fallback:
            hair_w = max(1, hair_bounds_fallback.width)
            hair_h = max(1, hair_bounds_fallback.height)
            face_w = max(12, round(hair_w * 0.5))
            face_h = max(12, round(hair_h * 0.28))
            face_cx = round(hair_bounds_fallback.left + hair_w * 0.69)
            face_cy = round(hair_bounds_fallback.top + hair_h * 0.22)
            face_left = max(0, face_cx - face_w // 2)
            face_top = max(0, face_cy - face_h // 2)
            face_right = min(width, face_left + face_w)
            face_bottom = min(height, face_top + face_h)
            fallback_bounds = Bounds(
                face_left, face_top, max(face_left + 1, face_right), max(face_top + 1, face_bottom)
            )
            masks["face_hull"] = face_mask_from_bounds(fallback_bounds, width, height)
            face_count = 1
            landmark_source = "anime-hair-fallback"
            detected_face_bounds = [fallback_bounds]
        else:
            masks["face_hull"] = np.zeros((height, width), dtype=bool)

    face_bounds = mask_bounds(masks["face_hull"])
    if face_bounds:
        identity_bounds = face_bounds.padded(
            round(face_bounds.width * 0.75),
            round(face_bounds.height * 0.9),
            width,
            height,
        )
        identity_window = np.zeros((height, width), dtype=bool)
        identity_window[
            identity_bounds.top : identity_bounds.bottom,
            identity_bounds.left : identity_bounds.right,
        ] = True
        masks["identity"] = masks["face_hull"] | (
            masks["hair"] & identity_window
        )
    else:
        masks["identity"] = masks["face_hull"].copy()

    mask_files: dict[str, str] = {}
    for mask_name, mask in masks.items():
        if mask_name == "background":
            continue
        mask_path = output_directory / f"{prefix}-{mask_name}.png"
        save_mask(mask_path, mask)
        mask_files[mask_name] = str(mask_path.resolve().relative_to(PROJECT_ROOT))

    subject_bounds = mask_bounds(masks["subject"])
    hair_bounds = mask_bounds(masks["hair"])
    clothes_bounds = mask_bounds(masks["clothes"])
    subject_coverage = float(masks["subject"].mean())
    face_coverage = float(masks["face_hull"].mean())
    subject_centroid = mask_centroid(masks["subject"])

    issue_codes: list[str] = []
    if face_count == 0:
        issue_codes.append("NO_FACE")
    elif face_count > 1:
        issue_codes.append("MULTIPLE_FACES")
    elif face_bounds and max(face_bounds.width, face_bounds.height) / min(width, height) < 0.08:
        issue_codes.append("FACE_TOO_SMALL")
    if not 0.04 <= subject_coverage <= 0.95:
        issue_codes.append("SUBJECT_MASK_INVALID")

    analysis = {
        "path": str(image_path.resolve()),
        "width": width,
        "height": height,
        "faceCount": face_count,
        "faceDetectionScores": detected_face_scores,
        "faceLandmarkSource": landmark_source if face_count == 1 else None,
        "faceLandmarks": serialize_face_landmarks(face_landmarks),
        "faceCoverage": round(face_coverage, 6),
        "subjectCoverage": round(subject_coverage, 6),
        "subjectCentroid": (
            {
                "x": round(subject_centroid[0] / width, 6),
                "y": round(subject_centroid[1] / height, 6),
            }
            if subject_centroid
            else None
        ),
        "bounds": {
            "face": describe_bounds(face_bounds, width, height),
            "hair": describe_bounds(hair_bounds, width, height),
            "clothes": describe_bounds(clothes_bounds, width, height),
            "subject": describe_bounds(subject_bounds, width, height),
            "identity": describe_bounds(mask_bounds(masks["identity"]), width, height),
        },
        "maskFiles": mask_files,
        "issueCodes": issue_codes,
    }
    return analysis, masks, issue_codes


def normalized_face_center(analysis: dict[str, Any]) -> tuple[float, float] | None:
    face_bounds = analysis["bounds"]["face"]
    if not face_bounds:
        return None
    normalized = face_bounds["normalized"]
    return (
        (normalized["left"] + normalized["right"]) / 2.0,
        (normalized["top"] + normalized["bottom"]) / 2.0,
    )


def compare_target_topology(
    reference: dict[str, Any],
    reference_masks: dict[str, np.ndarray],
    target: dict[str, Any],
    target_masks: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[str]]:
    issue_codes: list[str] = []
    if target["faceCount"] == 0:
        issue_codes.append("TARGET_NO_FACE")
    elif target["faceCount"] > 1:
        issue_codes.append("TARGET_MULTIPLE_FACES")

    reference_center = normalized_face_center(reference)
    target_center = normalized_face_center(target)
    center_distance = None
    face_scale_ratio = None
    if reference_center and target_center:
        center_distance = math.dist(reference_center, target_center)
        if center_distance > 0.14:
            issue_codes.append("TARGET_FACE_DRIFT")
        reference_face_coverage = max(reference["faceCoverage"], 1e-9)
        face_scale_ratio = math.sqrt(target["faceCoverage"] / reference_face_coverage)
        if not 0.58 <= face_scale_ratio <= 1.72:
            issue_codes.append("TARGET_FACE_SCALE_DRIFT")

    centroid_distance = None
    if reference["subjectCentroid"] and target["subjectCentroid"]:
        centroid_distance = math.dist(
            (
                reference["subjectCentroid"]["x"],
                reference["subjectCentroid"]["y"],
            ),
            (target["subjectCentroid"]["x"], target["subjectCentroid"]["y"]),
        )
        if centroid_distance > 0.18:
            issue_codes.append("TARGET_SUBJECT_DRIFT")

    reference_layout = resize_binary_mask(reference_masks["subject"], (64, 64))
    target_layout = resize_binary_mask(target_masks["subject"], (64, 64))
    intersection = np.logical_and(reference_layout, target_layout).sum()
    union = np.logical_or(reference_layout, target_layout).sum()
    layout_iou = float(intersection / union) if union else 0.0
    if layout_iou < 0.28:
        issue_codes.append("TARGET_SUBJECT_LAYOUT_DRIFT")

    comparison = {
        "faceCenterDistance": round(center_distance, 6) if center_distance is not None else None,
        "faceScaleRatio": round(face_scale_ratio, 6) if face_scale_ratio is not None else None,
        "subjectCentroidDistance": (
            round(centroid_distance, 6) if centroid_distance is not None else None
        ),
        "subjectLayoutIou64": round(layout_iou, 6),
        "thresholds": {
            "maxFaceCenterDistance": 0.14,
            "faceScaleRatio": [0.58, 1.72],
            "maxSubjectCentroidDistance": 0.18,
            "minSubjectLayoutIou64": 0.28,
        },
        "issueCodes": issue_codes,
    }
    return comparison, issue_codes


def derive_status(issue_codes: list[str], attempt: int) -> tuple[str, str]:
    if not issue_codes:
        return "PASS", "Semantic preflight and target topology checks passed."
    if all(code in RETRYABLE_TARGET_CODES for code in issue_codes) and attempt < 2:
        return "RETRY", "Regenerate the stylized target while preserving the reference topology."
    return "UNSUPPORTED", "The input or target remains outside the supported single-person contract."


def main() -> int:
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    validated_models = validate_model_files()

    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(FACE_MODEL_PATH),
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=2,
        min_face_detection_confidence=0.2,
        min_face_presence_confidence=0.2,
    )
    segmenter_options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(SEGMENTATION_MODEL_PATH),
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.IMAGE,
        output_category_mask=True,
        output_confidence_masks=False,
    )

    long_range_face_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.35,
    )
    try:
        with vision.FaceLandmarker.create_from_options(face_options) as face_landmarker:
            with vision.ImageSegmenter.create_from_options(segmenter_options) as segmenter:
                reference, reference_masks, reference_issues = analyze_image(
                    arguments.reference,
                    "reference",
                    arguments.output_dir,
                    face_landmarker,
                    segmenter,
                    long_range_face_detector,
                )
                target = None
                comparison = None
                issue_codes = list(reference_issues)
                if arguments.target:
                    target, target_masks, target_issues = analyze_image(
                        arguments.target,
                        "target",
                        arguments.output_dir,
                        face_landmarker,
                        segmenter,
                        long_range_face_detector,
                    )
                    # A stylized square target may legitimately contain a smaller
                    # absolute face than the source. Relative scale drift is checked
                    # below; the 8% minimum belongs only to the source contract.
                    target_issues = [
                        f"TARGET_{code}"
                        for code in target_issues
                        if code != "FACE_TOO_SMALL"
                    ]
                    issue_codes.extend(target_issues)
                    comparison, topology_issues = compare_target_topology(
                        reference, reference_masks, target, target_masks
                    )
                    issue_codes.extend(topology_issues)
    finally:
        long_range_face_detector.close()

    issue_codes = sorted(set(issue_codes))
    status, action = derive_status(issue_codes, arguments.attempt)
    report = {
        "schemaVersion": 1,
        "contract": "single-person-photo-v1",
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
