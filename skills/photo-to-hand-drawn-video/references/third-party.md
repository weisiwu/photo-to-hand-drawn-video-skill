# Third-party runtime boundary

This repository contains the original S planning, orchestration, rendering integration, and validation code. It does not redistribute the following binary/runtime assets:

- [MediaPipe models](https://developers.google.com/mediapipe): downloaded from Google's official model storage by `setup_runtime.py` and verified with SHA-256.
- [brushlib-wasm](https://github.com/eliot-akira/brushlib-wasm): a MyPaint-compatible WebAssembly brush runtime used by the browser renderer.
- [MyPaint/libmypaint](https://github.com/mypaint/libmypaint): the underlying brush-engine family.

As of 2026-07-22, the `eliot-akira/brushlib-wasm` repository does not declare a license in its GitHub metadata and has no LICENSE file. For that reason this project does not bundle its JS/WASM output. Obtain and review a build yourself, then pass its directory to `setup_runtime.py`.

Music, user photos, generated target images, and production videos are not part of the open-source code license. Confirm your rights before publishing any generated result.
