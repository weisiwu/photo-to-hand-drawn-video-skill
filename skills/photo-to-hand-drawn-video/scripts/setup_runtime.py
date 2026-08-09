#!/usr/bin/env python3
"""Prepare external runtime files without redistributing them in this repository."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
MODEL_DIRECTORY = SCRIPT_DIRECTORY / "models" / "mediapipe"
BRUSHLIB_DIRECTORY = SCRIPT_DIRECTORY / "vendor" / "brushlib"
MODEL_MANIFEST = {
    "schemaVersion": 1,
    "models": [
        {
            "name": "MediaPipe Selfie Multiclass 256x256",
            "file": "selfie_multiclass_256x256.tflite",
            "source": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
            "sha256": "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0",
            "bytes": 16371837,
        },
        {
            "name": "MediaPipe Face Landmarker",
            "file": "face_landmarker.task",
            "source": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
            "sha256": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
            "bytes": 3758596,
        },
    ],
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--brushlib-dir",
        type=Path,
        help="Directory containing brushlib.js, brushes.js, and brushlib.wasm.",
    )
    parser.add_argument("--skip-models", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_models() -> None:
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for model in MODEL_MANIFEST["models"]:
        output_path = MODEL_DIRECTORY / model["file"]
        if output_path.exists() and sha256_file(output_path) == model["sha256"]:
            print(f"model ready: {output_path.name}")
            continue
        temporary_path = output_path.with_suffix(output_path.suffix + ".download")
        print(f"downloading: {model['name']}")
        urllib.request.urlretrieve(model["source"], temporary_path)
        actual_hash = sha256_file(temporary_path)
        if actual_hash != model["sha256"]:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError(f"checksum mismatch for {model['file']}")
        temporary_path.replace(output_path)
    (MODEL_DIRECTORY / "MANIFEST.json").write_text(
        json.dumps(MODEL_MANIFEST, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def install_brushlib(source_directory: Path) -> None:
    source_directory = source_directory.resolve()
    required_files = ["brushlib.js", "brushes.js", "brushlib.wasm"]
    missing_files = [name for name in required_files if not (source_directory / name).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"brushlib directory is missing: {', '.join(missing_files)}"
        )

    BRUSHLIB_DIRECTORY.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_directory / "brushlib.js", BRUSHLIB_DIRECTORY / "brushlib.js")
    shutil.copy2(source_directory / "brushes.js", BRUSHLIB_DIRECTORY / "brushes.js")
    wasm_base64 = base64.b64encode((source_directory / "brushlib.wasm").read_bytes()).decode("ascii")
    (BRUSHLIB_DIRECTORY / "brushlib-wasm-inline.js").write_text(
        f'window.BRUSHLIB_WASM_BASE64 = "{wasm_base64}";\n',
        encoding="utf-8",
    )
    print(f"brush runtime ready: {BRUSHLIB_DIRECTORY}")


def main() -> int:
    arguments = parse_arguments()
    if not arguments.skip_models:
        download_models()
    if arguments.brushlib_dir:
        install_brushlib(arguments.brushlib_dir)
    else:
        print("brush runtime not installed; pass --brushlib-dir before rendering")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
