#!/usr/bin/env node
/**
 * Deterministic HTML-to-video renderer that restarts Chromium between chunks.
 * This bounds browser memory when a long animation uses a high-resolution
 * backing canvas while preserving exact frame timestamps.
 */

const { chromium } = require('playwright');
const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

function parseArguments() {
  const [, , htmlArgument, ...flags] = process.argv;
  if (!htmlArgument) {
    console.error('usage: node render-video-chunked.cjs <html-file> [--duration=81.67] [--fps=25] [--width=1080] [--height=1440] [--chunk-size=160] [--start-frame=0] [--frame-dir=/tmp/frames] [--query=reference=...] [--output=/path/video.mp4]');
    process.exit(1);
  }
  const options = {
    duration: 18.5,
    fps: 25,
    width: 1080,
    height: 1440,
    chunkSize: 160,
    startFrame: 0,
    frameDirectory: '',
    query: '',
    outputFile: '',
  };
  for (const flag of flags) {
    const numericMatch = flag.match(/^--(duration|fps|width|height|chunk-size|start-frame)=([\d.]+)$/);
    if (numericMatch) {
      const optionName = numericMatch[1].replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
      options[optionName] = Number(numericMatch[2]);
      continue;
    }
    const frameDirectoryMatch = flag.match(/^--frame-dir=(.+)$/);
    if (frameDirectoryMatch) options.frameDirectory = path.resolve(frameDirectoryMatch[1]);
    const queryMatch = flag.match(/^--query=(.+)$/);
    if (queryMatch) options.query = queryMatch[1];
    const outputMatch = flag.match(/^--output=(.+)$/);
    if (outputMatch) options.outputFile = path.resolve(outputMatch[1]);
  }
  return { htmlFile: htmlArgument.startsWith('http') ? htmlArgument : path.resolve(htmlArgument), options };
}

async function launchBrowser() {
  try {
    return await chromium.launch();
  } catch (launchError) {
    const cacheRoot = path.join(os.homedir(), 'Library/Caches/ms-playwright');
    const shellDirectory = fs.readdirSync(cacheRoot)
      .filter(name => name.startsWith('chromium_headless_shell-'))
      .sort()
      .pop();
    if (!shellDirectory) throw launchError;
    return chromium.launch({
      executablePath: path.join(cacheRoot, shellDirectory, 'chrome-headless-shell-mac-arm64/chrome-headless-shell'),
    });
  }
}

async function renderFrameChunk(htmlFile, options, frameDirectory, firstFrame, frameLimit) {
  const browser = await launchBrowser();
  try {
    const page = await browser.newPage({
      viewport: { width: options.width, height: options.height },
      deviceScaleFactor: 1,
    });
    const pageUrl = htmlFile.startsWith('http')
      ? `${htmlFile}${options.query ? `?${options.query}` : ''}`
      : `file://${htmlFile}${options.query ? `?${options.query}` : ''}`;
    await page.goto(pageUrl);
    await page.waitForFunction('window.__ready === true', null, { timeout: 20000 });
    await page.evaluate(() => window.__pause());
    for (let frameIndex = firstFrame; frameIndex < frameLimit; frameIndex += 1) {
      const timestamp = Math.min(frameIndex / options.fps, options.duration - 1e-4);
      await page.evaluate(seconds => window.__seek(seconds), timestamp);
      await page.screenshot({
        path: path.join(frameDirectory, `frame-${String(frameIndex).padStart(5, '0')}.png`),
        clip: { x: 0, y: 0, width: options.width, height: options.height },
      });
    }
  } finally {
    await browser.close();
  }
}

(async () => {
  const { htmlFile, options } = parseArguments();
  const outputFile = options.outputFile || htmlFile.replace(/\.html$/, '.mp4');
  const frameDirectory = options.frameDirectory
    || fs.mkdtempSync(path.join(os.tmpdir(), 'marker-chunked-frames-'));
  fs.mkdirSync(frameDirectory, { recursive: true });
  const frameCount = Math.round(options.duration * options.fps);

  console.log(`▸ Chunked render: ${path.basename(htmlFile)} → ${frameCount} frames @ ${options.fps}fps`);
  for (let firstFrame = options.startFrame; firstFrame < frameCount; firstFrame += options.chunkSize) {
    const frameLimit = Math.min(frameCount, firstFrame + options.chunkSize);
    await renderFrameChunk(htmlFile, options, frameDirectory, firstFrame, frameLimit);
    console.log(`  frames ${firstFrame}-${frameLimit - 1}/${frameCount - 1}`);
  }

  console.log('▸ ffmpeg encode…');
  execFileSync('ffmpeg', [
    '-y',
    '-framerate', String(options.fps),
    '-i', path.join(frameDirectory, 'frame-%05d.png'),
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-crf', '18',
    '-movflags', '+faststart',
    outputFile,
  ], { stdio: 'inherit' });
  fs.rmSync(frameDirectory, { recursive: true, force: true });
  console.log(`✓ Done: ${outputFile}`);
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
