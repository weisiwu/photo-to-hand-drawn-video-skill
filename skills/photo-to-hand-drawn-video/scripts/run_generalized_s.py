#!/usr/bin/env python3
"""Run the generalized S planning and rendering pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
RENDERER_HTML = PROJECT_ROOT / "marker-brush-animation-s-upper-bound-7x.html"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight, semantically plan, optimize, and optionally render a generalized S run."
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--subject-type",
        choices=("auto", "human", "animal", "anime"),
        default="auto",
        help="Use MediaPipe for a person, the central companion-animal adapter, "
        "or the hair-mask anime adapter for stylized 2D illustrations.",
    )
    parser.add_argument("--attempt", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--budget-scale", type=float, default=2.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--duration", type=float, default=None,
                        help="Output duration in seconds; auto-computed from --playback-rate when omitted.")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--music", type=Path)
    parser.add_argument(
        "--playback-rate",
        type=float,
        default=3.0,
        help="Drawing pace: source seconds consumed per playback second. "
        "3.0 is the original fast pace; 1.5 draws at half speed (longer video).",
    )
    parser.add_argument(
        "--background-image",
        type=Path,
        help="Optional image drawn underneath the board (cover-fit, 85% opacity).",
    )
    parser.add_argument(
        "--dynamic-brush",
        type=int,
        choices=(0, 1),
        default=1,
        help="1 (default): per-stroke brush pool + width/alpha jitter for a "
        "livelier hand-drawn feel; 0: fixed phase brushes.",
    )
    parser.add_argument(
        "--stage-width",
        type=int,
        default=1080,
        help="Output video width in px (1080 for 3:4 portrait, 1080 for 9:16 Douyin).",
    )
    parser.add_argument(
        "--stage-height",
        type=int,
        default=1440,
        help="Output video height in px (1440 for 3:4 portrait, 1920 for 9:16 Douyin).",
    )
    parser.add_argument(
        "--artboard-size",
        type=int,
        default=3840,
        help="Internal supersampled render canvas edge (3840 default, 2880 legacy).",
    )
    parser.add_argument(
        "--view-size",
        type=int,
        default=960,
        help="On-screen artboard size in px (960 default; 1080 fills a 1080-wide Douyin frame).",
    )
    parser.add_argument(
        "--brush-style",
        choices=("marker", "pencil", "ink", "airbrush"),
        default="marker",
        help="Brush style: marker (default), pencil (graphite), ink (sumi wash), airbrush.",
    )
    parser.add_argument("--skip-video-validation", action="store_true")
    return parser.parse_args()


def run_command(
    command: list[str],
    accepted_exit_codes: set[int] | None = None,
    environment: dict[str, str] | None = None,
) -> int:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        env=environment,
    )
    accepted = accepted_exit_codes or {0}
    if completed.returncode not in accepted:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed.returncode


def stage_image(source_path: Path, destination_path: Path, format_name: str) -> None:
    image = Image.open(source_path).convert("RGB")
    save_options = {"quality": 95, "subsampling": 0} if format_name == "JPEG" else {}
    image.save(destination_path, format=format_name, **save_options)


def project_relative_url(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def compute_duration(playback_rate: float) -> float:
    """Mirror the renderer's VIDEO_DURATION_SECONDS for a given playback rate.

    Source timeline: 0-60s normal, 60-230s detail (7x), 230-334s S optimization
    (8x), 334-354s normal. All rates scale with the user playback rate.
    """
    rate_scale = playback_rate / 3.0
    normal = 3.0 * rate_scale
    detail = 7.0 * rate_scale
    s_optimization = 8.0 * rate_scale
    detail_start = 60.0 / normal
    detail_end = detail_start + (230.0 - 60.0) / detail
    s_optimization_end = detail_end + (334.0 - 230.0) / s_optimization
    return s_optimization_end + (354.0 - 334.0) / normal


def run_semantic_preflight(
    subject_type: str,
    reference_path: Path,
    target_path: Path,
    run_directory: Path,
    attempt: int,
) -> tuple[int, dict]:
    report_path = run_directory / "semantic-preflight.json"

    if subject_type == "anime":
        anime_command = [
            sys.executable,
            str(PROJECT_ROOT / "semantic_preflight_anime.py"),
            "--reference",
            str(reference_path),
            "--target",
            str(target_path),
            "--output-dir",
            str(run_directory),
            "--attempt",
            str(attempt),
        ]
        target_transform = target_path.with_name("target-transform.json")
        if target_transform.exists():
            anime_command.extend(["--target-transform", str(target_transform)])
        anime_exit_code = run_command(
            anime_command,
            accepted_exit_codes={0, 2, 3},
        )
        return anime_exit_code, json.loads(report_path.read_text(encoding="utf-8"))

    if subject_type in {"auto", "human"}:
        human_exit_code = run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "semantic_preflight.py"),
                "--reference",
                str(reference_path),
                "--target",
                str(target_path),
                "--output-dir",
                str(run_directory),
                "--attempt",
                str(attempt),
            ],
            accepted_exit_codes={0, 2, 3},
        )
        if human_exit_code == 0 or subject_type == "human":
            return human_exit_code, json.loads(report_path.read_text(encoding="utf-8"))

    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "semantic_preflight_animal.py"),
            "--reference",
            str(reference_path),
            "--target",
            str(target_path),
            "--output-dir",
            str(run_directory),
            "--report-name",
            report_path.name,
        ]
    )
    return 0, json.loads(report_path.read_text(encoding="utf-8"))


def main() -> int:
    arguments = parse_arguments()
    run_directory = arguments.run_dir.resolve()
    try:
        run_directory.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError("--run-dir must be inside the project directory") from error
    run_directory.mkdir(parents=True, exist_ok=True)

    staged_reference = run_directory / "reference.jpg"
    staged_target = run_directory / "target.png"
    stage_image(arguments.reference, staged_reference, "JPEG")
    stage_image(arguments.target, staged_target, "PNG")
    source_transform = arguments.target.with_name("target-transform.json")
    if source_transform.exists():
        (run_directory / "target-transform.json").write_bytes(source_transform.read_bytes())
    staged_background = None
    if arguments.background_image:
        staged_background = run_directory / "background.jpg"
        stage_image(arguments.background_image, staged_background, "JPEG")

    semantic_report = run_directory / "semantic-preflight.json"
    preflight_exit_code, preflight = run_semantic_preflight(
        arguments.subject_type,
        staged_reference,
        staged_target,
        run_directory,
        arguments.attempt,
    )
    if preflight_exit_code != 0:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return preflight_exit_code

    base_plan_json = run_directory / "generalized-base-plan.json"
    base_plan_javascript = run_directory / "generalized-base-plan.js"
    base_preview = run_directory / "generalized-base-preview.png"
    base_metrics = run_directory / "generalized-base-metrics.json"
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "build_generalized_s_base_plan.py"),
            "--target",
            str(staged_target),
            "--semantic-report",
            str(semantic_report),
            "--plan-json",
            str(base_plan_json),
            "--plan-js",
            str(base_plan_javascript),
            "--preview",
            str(base_preview),
            "--metrics",
            str(base_metrics),
            "--budget-scale",
            str(arguments.budget_scale),
        ]
    )

    final_plan_json = run_directory / "generalized-s-plan.json"
    final_plan_javascript = run_directory / "generalized-s-plan.js"
    final_preview = run_directory / "generalized-s-preview.png"
    final_metrics = run_directory / "generalized-s-metrics.json"
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "build_upper_limit_plan_s.py"),
            "--target",
            str(staged_target),
            "--base-plan",
            str(base_plan_json),
            "--base-preview",
            str(base_preview),
            "--semantic-report",
            str(semantic_report),
            "--plan-json",
            str(final_plan_json),
            "--plan-js",
            str(final_plan_javascript),
            "--preview",
            str(final_preview),
            "--metrics",
            str(final_metrics),
            "--budget-scale",
            str(arguments.budget_scale),
            "--javascript-global",
            "MARKER_PAINT_PLAN_GENERALIZED",
        ]
    )

    renderer_query = urlencode(
        {
            "reference": project_relative_url(staged_reference),
            "plan": project_relative_url(final_plan_javascript),
            "planGlobal": "MARKER_PAINT_PLAN_GENERALIZED",
            "stageW": arguments.stage_width,
            "stageH": arguments.stage_height,
            "artboard": arguments.artboard_size,
            "viewSize": arguments.view_size,
            "brushStyle": arguments.brush_style,
            "rate": arguments.playback_rate,
            "dynamic": arguments.dynamic_brush,
            **(
                {"bgImage": project_relative_url(staged_background)}
                if staged_background
                else {}
            ),
        }
    )
    raw_video = run_directory / "generalized-s-raw.mp4"
    final_video = run_directory / "generalized-s.mp4"
    if arguments.render:
        render_duration = arguments.duration or compute_duration(arguments.playback_rate)
        render_environment = os.environ.copy()
        bundled_node_modules = (
            Path.home()
            / ".cache"
            / "codex-runtimes"
            / "codex-primary-runtime"
            / "dependencies"
            / "node"
            / "node_modules"
        )
        if not render_environment.get("NODE_PATH") and bundled_node_modules.exists():
            render_environment["NODE_PATH"] = str(bundled_node_modules)
        run_command(
            [
                "node",
                str(PROJECT_ROOT / "render-video-chunked.cjs"),
                str(RENDERER_HTML),
                f"--duration={render_duration}",
                f"--fps={arguments.fps}",
                f"--width={arguments.stage_width}",
                f"--height={arguments.stage_height}",
                f"--query={renderer_query}",
                f"--output={raw_video}",
            ],
            environment=render_environment,
        )
        if arguments.music:
            fade_start = max(0.0, render_duration - 2.0)
            run_command(
                [
                    "ffmpeg",
                    "-y",
                    "-stream_loop",
                    "-1",
                    "-i",
                    str(arguments.music),
                    "-i",
                    str(raw_video),
                    "-map",
                    "1:v:0",
                    "-map",
                    "0:a:0",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-filter:a",
                    f"volume=0.22,afade=t=out:st={fade_start}:d=2",
                    "-shortest",
                    "-movflags",
                    "+faststart",
                    str(final_video),
                ]
            )
        else:
            final_video = raw_video

    validation_report = None
    if arguments.render and not arguments.skip_video_validation:
        validation_report = run_directory / "video-validation.json"
        view_left = max(24, (arguments.stage_width - arguments.view_size) // 2)
        view_top = max(24, round(arguments.stage_height * 0.27))
        validation_crop = f"{view_left},{view_top},{arguments.view_size},{arguments.view_size}"
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "validate_painting_video.py"),
                "--video",
                str(final_video),
                "--crop",
                validation_crop,
                "--target",
                str(staged_target),
                "--frame-step",
                "5",
                "--report",
                str(validation_report),
            ]
        )

    summary = {
        "schemaVersion": 1,
        "status": "PASS",
        "contract": preflight.get("contract"),
        "subjectType": "animal"
        if preflight.get("contract") == "single-companion-animal-photo-v1"
        else "anime"
        if preflight.get("contract") == "single-person-anime-photo-v1"
        else "human",
        "attempt": arguments.attempt,
        "budgetScale": arguments.budget_scale,
        "semanticReport": project_relative_url(semantic_report),
        "plan": project_relative_url(final_plan_json),
        "preview": project_relative_url(final_preview),
        "renderer": project_relative_url(RENDERER_HTML),
        "rendererQuery": renderer_query,
        "video": project_relative_url(final_video) if arguments.render else None,
        "videoValidation": project_relative_url(validation_report)
        if validation_report
        else None,
        "visualReviewRequired": True,
        "privacySanitizationRequired": True,
    }
    summary_path = run_directory / "run-summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
