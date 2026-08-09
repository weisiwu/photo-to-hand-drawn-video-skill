const { chromium } = require('playwright');
const fs = require('fs');

// Export the per-frame brush tip trajectory (video coordinates) from the
// renderer page without recording video: seek every frame, collect the trace.
// Usage: node export-trace.cjs "<full page url>" <duration> <fps> <out.json>
(async () => {
  const [, , url, durationArg, fpsArg, outPath] = process.argv;
  const duration = parseFloat(durationArg);
  const fps = parseFloat(fpsArg);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
  page.on('console', msg => {
    if (msg.type() === 'error' && !msg.text().includes('404')) console.log('[err]', msg.text().slice(0, 200));
  });
  await page.goto(url);
  await page.waitForFunction('window.__ready === true', null, { timeout: 60000 });
  await page.evaluate(() => window.__pause());
  const frames = Math.round(duration * fps);
  for (let i = 0; i < frames; i++) {
    await page.evaluate(seconds => window.__seek(seconds), i / fps);
    await page.waitForTimeout(20);
    if (i % 128 === 0) console.log(`seek ${i}/${frames}`);
  }
  const trace = await page.evaluate(() => window.__brushTrace);
  fs.writeFileSync(outPath, JSON.stringify(trace));
  console.log(`trace points: ${trace.length} -> ${outPath}`);
  await browser.close();
})();
