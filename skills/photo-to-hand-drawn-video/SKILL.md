---
name: photo-to-hand-drawn-video
description: Turn a single-person or centrally framed single-pet photo plus a square stylized target into a deterministic, verifiable stroke-by-stroke hand-drawing video. Use when the user asks to 把照片变成逐笔手绘视频, 照片临摹动画, 真实画笔跟随, marker/dry-brush drawing process, or a no-reveal drawing video rendered with finite-width strokes rather than video generation or target-image compositing.
---

# Photo to hand-drawn video

Build the S release: semantic subject-first planning, finite-width residual refinement, MyPaint-compatible brush rendering, cursor tracking, pace compression, optional music, and anti-reveal validation.

## Read the applicable contract

- Read [references/pipeline.md](references/pipeline.md) before preparing a target or running the planner.
- Read [references/acceptance.md](references/acceptance.md) before accepting a preview or video.
- Read [references/third-party.md](references/third-party.md) before installing or redistributing runtime files.

## Respect the hard boundaries

- Never composite the target image into a runtime frame.
- Never use `drawImage`, clipping masks, `destination-in`, image patches, or whole-canvas reveal strokes to imitate painting.
- Never optimize only for SSIM. Keep process authenticity, identity similarity, edge quality, and anti-copy checks separate.
- Never claim arbitrary-image coverage. This release supports one person or one centrally framed companion animal.
- Keep the original photo, generated target, music, models, and production video out of Git unless the user owns the rights and explicitly asks to publish them.

## Prepare the target

1. Inspect the reference photo.
2. If the user did not supply a target, use an available image-generation tool to create one square stylized drawing.
3. Preserve identity, pose, crop, major clothing/fur markings, accessories, and primary scene objects.
4. Keep the target reproducible by marker, dry-brush, textured-ink, and liner strokes.
5. Sanitize image metadata with an available privacy skill before staging or publishing it.
6. Stop after three failed target attempts. Do not force manual coordinates.

## Prepare dependencies

Treat the directory containing this `SKILL.md` as `SKILL_DIR`.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r "$SKILL_DIR/scripts/requirements.txt"
npm install
npm run install:browser
python "$SKILL_DIR/scripts/setup_runtime.py" --brushlib-dir /path/to/brushlib-wasm-build
```

Require `ffmpeg` on `PATH`. The runtime setup downloads and checks MediaPipe models. It does not download brushlib-wasm automatically because the upstream repository has no declared license.

## Run a preview plan

Keep `--run-dir` inside `SKILL_DIR/scripts` so the file-based renderer can load its assets.

```bash
.venv/bin/python "$SKILL_DIR/scripts/run_generalized_s.py" \
  --reference /absolute/path/reference.jpg \
  --target /absolute/path/target.png \
  --subject-type human \
  --run-dir "$SKILL_DIR/scripts/runs/example" \
  --budget-scale 0.5
```

Use `--subject-type animal` for one centrally framed pet. Use `auto` only when MediaPipe models are installed.

Inspect `generalized-s-preview.png` and `semantic-preflight.json`. Reject the plan if the subject is not drawn before secondary background detail or if the identity region is malformed.

## Render the production video

Use `--budget-scale 2.0` for the retained S quality level. Add `--music` only when the user has publication rights.

```bash
.venv/bin/python "$SKILL_DIR/scripts/run_generalized_s.py" \
  --reference /absolute/path/reference.jpg \
  --target /absolute/path/target.png \
  --subject-type animal \
  --run-dir "$SKILL_DIR/scripts/runs/production" \
  --budget-scale 2.0 \
  --render \
  --duration 64.03 \
  --fps 25 \
  --music /absolute/path/music.mp3
```

Do not use `--skip-video-validation` for a final deliverable. Review `video-validation.json` and then perform the visual checks in [references/acceptance.md](references/acceptance.md).

## Report the result

Lead with the video and final preview paths. Then report:

- detected contract and subject type;
- stroke count and budget scale from metrics;
- video duration, resolution, and validation result;
- any target-generation retry;
- whether music and privacy sanitization were applied;
- the current composition and visual-quality limitations.

Call an automated pass an engineering pass, not proof that the artwork is visually good.
