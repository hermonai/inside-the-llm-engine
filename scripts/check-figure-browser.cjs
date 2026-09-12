/* Optional real-browser QA; install Playwright and provide CHROME_PATH if needed. */
const {chromium} = require('playwright');
const path = require('node:path');
const fs = require('node:fs');
(async () => {
  const browser = await chromium.launch({headless:true, executablePath:process.env.CHROME_PATH || undefined});
  const page = await browser.newPage({viewport:{width:1024,height:900}});
  for (const name of ['rope','cache','batch','ch06-row','ch06-loops','ch07-passes']) {
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
  // All canonical chapters must remain readable offline with embedded vectors.
  const failures=[];
  page.on('requestfailed', request => failures.push(request.url()));
  for (const [chapter,count] of [['05',8],['06',14],['07',10],['08',10],['09',10]]) {
  await page.goto('file://'+path.resolve('build/publication/chapter'+chapter+'.html'));
  if (await page.locator('img').count() !== count) throw Error('chapter figure count');
  if (await page.locator('math').count() === 0) throw Error('missing native math');
  if (!(await page.locator('img').evaluateAll(images => images.every(img => img.complete && img.naturalWidth > 0 && img.src.startsWith('data:image/svg+xml'))))) throw Error('chapter SVG not embedded/loaded');
  for (const width of [1024,768,390]) {
    await page.setViewportSize({width,height:900});
    if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw Error('chapter horizontal overflow '+width);
  }
  }
  if(failures.length) throw Error(failures.join('\n'));
  for (const entry of JSON.parse(fs.readFileSync('figures/manifest.json')).figures.filter(e=>e.id.startsWith('FIG-CH06-') || e.id.startsWith('FIG-CH07-') || e.id.startsWith('FIG-CH08-') || e.id.startsWith('FIG-CH09-') || e.id.startsWith('FIG-INDUSTRIAL-'))) {
    await page.goto('file://'+path.resolve(entry.generated[0]));
    const outside=await page.locator('text').evaluateAll(nodes=>nodes.filter(n=>{const r=n.getBBox();return r.x<0 || r.y<0 || r.x+r.width>1000 || r.y+r.height>720;}).map(n=>n.textContent));
    if(outside.length) throw Error(entry.id+' text outside canvas: '+outside.join('; '));
    if (entry.id.startsWith('FIG-CH08-') || entry.id.startsWith('FIG-CH09-')) {
      const crossings=await page.evaluate(() => {
        const rects=[...document.querySelectorAll('rect')].map(n=>n.getBBox()).filter(r=>r.width<1000);
        return [...document.querySelectorAll('text')].filter(n=>{
          const b=n.getBBox();
          return rects.some(r=>b.x<r.x+r.width && b.x+b.width>r.x && b.y<r.y+r.height && b.y+b.height>r.y &&
            !(b.x>=r.x && b.x+b.width<=r.x+r.width && b.y>=r.y && b.y+b.height<=r.y+r.height));
        }).map(n=>n.textContent);
      });
      if(crossings.length) throw Error(entry.id+' text crosses panel boundary: '+crossings.join('; '));
    }
  }
  await browser.close(); console.log('Browser QA passed: six animations, keyboard, reduced motion, offline Chapters 5/6/7/8/9 SVG/MathML, 1024/768/390px, Chapter 6/7/8/9 and industrial SVG bounds');
})().catch(error => {console.error(error); process.exit(1);});
