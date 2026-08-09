const { chromium } = require('playwright');

(async () => {
  const base = 'http://127.0.0.1:8765/marker-brush-animation-s-upper-bound-7x.html';
  const query = 'reference=runs/anime-v3/reference.jpg&plan=runs/anime-v3/generalized-s-plan.js&planGlobal=MARKER_PAINT_PLAN_GENERALIZED&stageW=1080&stageH=1920&artboard=3840&viewSize=1080&brushStyle=marker&rate=0.75&dynamic=1';
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
  page.on('console', msg => { if (msg.type() === 'error') console.log('[err]', msg.text().slice(0, 150)); });
  await page.goto(base + '?' + query);
  await page.waitForFunction('window.__ready === true', null, { timeout: 30000 });
  await page.evaluate(() => window.__pause());
  for (const t of [5, 15, 20, 30, 40, 50]) {
    await page.evaluate(seconds => window.__seek(seconds), t);
    await page.waitForTimeout(300);
    const state = await page.evaluate(() => ({
      t: window.__seek ? undefined : undefined,
      activeTip: window.__activeTip,
      cameraFollow: window.__cameraFollow,
      artTransform: document.getElementById('art').style.transform,
    }));
    console.log(`t=${t}s`, JSON.stringify(state));
  }
  await browser.close();
})();
