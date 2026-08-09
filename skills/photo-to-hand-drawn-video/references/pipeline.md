# S pipeline

## Inputs

- `reference`: the original single-person or centrally framed single-pet photo.
- `target`: a square stylized drawing that preserves identity, pose, composition, and important objects.

The target is a planning reference. It is never composited into the animation at runtime.

## Stages

1. Run semantic preflight and create subject, identity, face/head, clothing, hair/fur, and background masks.
2. Build the base plan in subject-first order: subject structure, subject blocking, primary scene, background suggestion, then subject refinement.
3. Add continuous finite-width residual strokes only when they reduce local error.
4. Render every stroke with the MyPaint-compatible brush engine in a deterministic browser canvas.
5. Compress visually quiet detail phases to 7x or 8x while keeping early construction readable.
6. Optionally mix music with a short fade-out.
7. Validate the final video for reveal fingerprints, repainting behavior, and final-frame similarity.

## Target-image guidance

When a target is not supplied, use an image-generation tool to create a square drawing from the reference. Require:

- the same person or animal identity, pose, crop, and major accessories;
- large, readable structural shapes before fine texture;
- clean separation between subject and background;
- no text, watermark, extra limbs, or invented accessories;
- a finish that a real marker/dry-brush/liner workflow can reproduce.

Generate at most three target attempts. Stop if semantic preflight still fails; do not hand-write coordinates to force a pass.

## Supported compositions

- One visible person, including full-body or upper-body portraits.
- One centrally framed companion animal with a readable head and body.

Crowds, multiple faces, extreme occlusion, tiny subjects, and complex text/logos are unsupported by the current S release.
