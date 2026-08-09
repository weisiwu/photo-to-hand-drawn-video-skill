# Acceptance contract

## Engineering gates

- The semantic preflight status is `PASS`.
- No manual coordinate override is used.
- Runtime target compositing is disabled.
- No whole-canvas scan stroke is used.
- The video validator reports `passed: true`.
- The first visible color must not already equal the final image across most painted pixels.
- A meaningful fraction of painted pixels must be modified more than once.

## Visual review

Engineering gates do not prove that the drawing looks good. Inspect:

1. The subject becomes recognizable before the background is refined.
2. There is no broad horizontal reveal at the beginning.
3. The cursor follows a real active stroke and disappears when drawing stops.
4. The detail section does not look frozen or like empty drawing.
5. The final face/head, clothing/fur, hands/paws, and major accessories resemble the target.
6. The last camera movement shows the finished drawing without fake brush sweeps.

Reject the run if any visual item fails, even when all automated gates pass.

## Similarity boundary

Do not optimize only for full-image SSIM. A high score can be achieved by revealing or copying the target, which violates the purpose of the project. Judge together:

- process authenticity;
- subject and identity similarity;
- edge/structure quality;
- local-copy and reveal constraints.
