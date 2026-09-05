/* Optional real-browser QA; install Playwright and provide CHROME_PATH if needed. */
const {chromium} = require('playwright');
const path = require('node:path');
const fs = require('node:fs');
(async () => {
  const browser = await chromium.launch({headless:true, executablePath:process.env.CHROME_PATH || undefined});
  const page = await browser.newPage({viewport:{width:1024,height:900}});
  for (const name of ['rope','cache','batch']) {
    const errors=[]; page.on('pageerror', e => errors.push(e.message));
    await page.goto('file://'+path.resolve('figures/generated/'+name+'.html'));
    if (await page.locator('.frame:visible').count() !== 1) throw Error('frame visibility');
    await page.getByRole('button',{name:'Next',exact:true}).click();
    if (!(await page.locator('#state').textContent()).includes('2 of 4')) throw Error('next');
    await page.getByRole('button',{name:'Previous',exact:true}).click();
    await page.locator('#step').focus(); await page.keyboard.press('ArrowRight');
    if (!(await page.locator('#state').textContent()).includes('2 of 4')) throw Error('keyboard');
    await page.emulateMedia({reducedMotion:'reduce'});
    await page.getByRole('button',{name:'Play',exact:true}).click();
    if (!(await page.locator('#state').textContent()).includes('3 of 4')) throw Error('reduced motion');
    await page.emulateMedia({reducedMotion:'no-preference'});
    await page.getByRole('button',{name:'Play',exact:true}).click();
    await page.waitForFunction(() => document.getElementById('state').textContent === 'Frame 4 of 4');
    if (await page.getByRole('button',{name:'Play',exact:true}).count() !== 1) throw Error('play did not stop');
    for (const width of [1024,768,390]) {
      await page.setViewportSize({width,height:900});
      if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw Error('horizontal overflow '+name+' '+width);
    }
    await page.setViewportSize({width:1024,height:900});
    fs.mkdirSync('build/browser',{recursive:true});
    await page.screenshot({path:'build/browser/'+name+'.png',fullPage:true});
    if(errors.length) throw Error(errors.join('\n'));
  }
  await browser.close(); console.log('Browser QA passed: frames, keyboard, reduced motion, 1024/768/390px, no console errors');
})().catch(error => {console.error(error); process.exit(1);});
