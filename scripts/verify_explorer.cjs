// Browser acceptance of the static shell and recorded-data player. No experiments run.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.CHEMWORLD_PREVIEW_URL||'http://127.0.0.1:8769/explore/index.html';
const out=process.env.CHEMWORLD_QA_OUTPUT||path.join(os.tmpdir(),'chemworld-skill-tree-qa');
const records=JSON.parse(fs.readFileSync(path.join(__dirname,'../site/explore/chemworld-site-records.json'),'utf8')).records;
(async()=>{
 fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_EXECUTABLE||undefined});
 const context=await browser.newContext({viewport:{width:1440,height:1000},permissions:['clipboard-read','clipboard-write']}),page=await context.newPage();
 const errors=[],requests=[],checks=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)errors.push(`${r.status()} ${r.url()}`);});
 page.on('request',r=>requests.push(r.url()));page.on('requestfailed',r=>errors.push(r.url()));
 const check=async(name,fn)=>{await fn();checks.push(name);console.log(`PASS ${checks.length}: ${name}`);};
 const ready=async route=>{await page.waitForFunction(()=>window.CHEMWORLD_CAPABILITIES&&document.querySelector('#site-page h1,#outline .tree-row,.site-error'));if(route?.startsWith('atlas'))await page.locator('#atlas-view:not([hidden]) [data-map]').first().waitFor();if(route?.startsWith('index'))await page.locator('#index-view:not([hidden]) h1').waitFor();if(route?.startsWith('guides'))await page.locator('#guide-reader h1').waitFor();};
 const go=async(route,lang='zh')=>{await page.goto(base+'?lang='+lang+'#'+route);await ready(route);};
 const hash=async route=>{await page.evaluate(r=>location.hash=r,route);await page.waitForURL('**#'+route);await ready(route);};
 const screenshot=name=>page.screenshot({path:path.join(out,name+'.png'),fullPage:true});
 try{
  await go('home');
  await check('Home exposes the real hierarchy and defers guide payloads',async()=>{
   assert.equal(await page.locator('.site-nav [data-nav]').count(),5);assert.equal(await page.locator('.site-explorer-toolbar').count(),0);
   assert.match(await page.locator('.home-scale').innerText(),/17/);assert.match(await page.locator('.home-scale').innerText(),/68/);
   assert.match(await page.locator('.architecture-captions').innerText(),/任务与评测/);
   assert.ok(!requests.some(u=>/chemworld-guide-(content|localized)\.js/.test(u)));await screenshot('site-home');
  });
  await check('Catalog filtering and player navigation work',async()=>{
   await page.locator('[data-nav="experiments"]').click();await page.locator('.case-card').first().waitFor();
   assert.equal(await page.locator('.case-card:visible').count(),3);await page.locator('[data-filter="separation"]').click();assert.equal(await page.locator('.case-card:visible').count(),2);
   await page.locator('[data-filter="reaction"]').click();await page.locator('.case-card:visible .site-link').click();
   await page.locator('#step-counter').waitFor();assert.equal(await page.locator('#step-counter').textContent(),'6 / 11');
   assert.equal(await page.locator('[data-nav="experiments"]').getAttribute('aria-current'),'page');
  });
  await check('Player honors the source visibility mask and never shows future chemical observations',async()=>{
   const record=records.find(r=>r.id==='reaction-to-assay'),fixture=JSON.parse(fs.readFileSync(path.join(__dirname,'../tests/fixtures/golden/core_scripted_trajectories.json'),'utf8')).summaries.find(r=>r.task_id===record.id);
   for(let i=0;i<record.steps.length;i++){
    await page.locator(`[data-step="${i}"]`).click();
    const expected=fixture.steps[i];for(const [key,value] of Object.entries(record.steps[i].observation)){assert.equal(expected.observed_mask[key],true);assert.equal(value,expected.observation[key]);}
    const points=await page.locator('[data-chart-step]').evaluateAll(es=>es.map(e=>Number(e.dataset.chartStep)));
    assert.deepEqual(points,record.steps.slice(0,i+1).flatMap((s,j)=>s.action.operation==='measure'&&s.status==='committed'&&Number.isFinite(s.observation.conversion)?[j]:[]));
   }
   await page.locator('[data-player-tab="data"]').click();assert.equal(await page.locator('.player-data tbody tr').count(),Object.keys(record.steps.at(-1).observation).length);
   const table=await page.locator('.player-data').innerText();for(const [key,value] of Object.entries(record.steps.at(-1).observation)){assert.ok(table.includes(key));assert.ok(table.includes(Number(value.toPrecision(5)).toLocaleString('zh-CN',{maximumFractionDigits:6})));}
  });
  await check('Transport, tabs, final step and speed controls work',async()=>{
   await page.locator('[data-player-tab="overview"]').click();await page.locator('[data-step="0"]').click();assert.equal(await page.locator('#previous-step').isDisabled(),true);
   await page.locator('#play-speed').selectOption('2');await page.locator('#play-record').click();await page.waitForFunction(()=>Number(document.querySelector('#record-progress').value)>=1);await page.locator('#play-record').click();
   const paused=await page.locator('#record-progress').inputValue();await page.waitForTimeout(750);assert.equal(await page.locator('#record-progress').inputValue(),paused);
   await page.locator('#record-progress').fill('9');await page.locator('#play-record').click();await page.waitForFunction(()=>document.querySelector('#step-counter').textContent==='11 / 11');
   assert.equal(await page.locator('#next-step').isDisabled(),true);assert.equal(await page.locator('#play-record').getAttribute('aria-label'),'播放记录');
   await page.locator('[data-player-tab="overview"]').focus();await page.keyboard.press('ArrowRight');assert.equal(await page.locator('[data-player-tab="signals"]').getAttribute('aria-selected'),'true');
   assert.match(await page.locator('#player-content').innerText(),/没有原始谱线/);await page.locator('#metric-select').selectOption('cost');assert.ok(await page.locator('[data-chart-step]').count()>1);
  });
  await check('Downloads retain public values and links retain the selected step across languages',async()=>{
   await page.locator('[data-step="5"]').click();await page.locator('#share-step').click();assert.equal(await page.evaluate(()=>navigator.clipboard.readText()),page.url());
   const pending=page.waitForEvent('download');await page.locator('#download-record').click();const download=await pending;const data=JSON.parse(fs.readFileSync(await download.path(),'utf8'));
   assert.deepEqual(data.steps,records.find(r=>r.id==='reaction-to-assay').steps);assert.ok(!JSON.stringify(data).includes('private_state_patch'));
   await page.locator('[data-site-lang="en"]').click();await page.locator('#step-counter').waitFor();assert.equal(await page.locator('html').getAttribute('lang'),'en');assert.equal(await page.locator('#step-counter').textContent(),'6 / 11');
   await screenshot('site-player-en');
  });
  await check('Leaving playback, including an invalid record, cancels the timer',async()=>{
   await page.locator('#play-record').click();await hash('experiment/unknown/1');await page.waitForTimeout(1400);assert.match(await page.locator('#site-page').innerText(),/Record not found/);assert.ok(page.url().endsWith('#experiment/unknown/1'));
   await hash('experiment/reaction-to-purification/6');await page.locator('#play-record').click();await page.locator('.site-brand').click();await page.waitForTimeout(1400);assert.ok(await page.locator('.home-hero').isVisible());assert.ok(page.url().endsWith('#home'));
  });
  await check('Global keyboard search reaches capabilities, guides and records',async()=>{
   await page.locator('.site-brand').focus();await page.keyboard.press('/');assert.equal(await page.locator('#site-search').evaluate(e=>e===document.activeElement),true);
   await page.locator('#site-search').fill('DFN');await page.locator('#site-search').press('ArrowDown');await page.keyboard.press('Enter');await page.locator('#document h1').waitFor();assert.match(page.url(),/#index\//);
   await page.locator('#site-search').fill('native_mechanisms');await page.locator('#site-search').press('Enter');await page.locator('#guide-reader h1').waitFor();assert.match(page.url(),/native_mechanisms/);
   await page.locator('#site-search').fill('reaction-to-assay');await page.locator('#site-search').press('Enter');await page.locator('#step-counter').waitFor();assert.equal(await page.locator('#step-counter').textContent(),'6 / 11');
  });
  await check('Guides include full prose, section navigation and source links',async()=>{
   await hash('guides/thermal-reaction/native_mechanisms');assert.ok((await page.locator('#guide-body').innerText()).length>3000);assert.ok(await page.locator('#guide-toc a').count()>=3);
   await page.locator('#guide-toc [data-guide-anchor="supported-finite-contract"]').click();assert.ok(page.url().endsWith('/supported-finite-contract'));await page.locator('[data-site-lang="zh"]').click();
   await page.locator('#supported-finite-contract').waitFor();assert.match(await page.locator('.guide-meta').innerText(),/中文预览译文/);assert.ok(page.url().endsWith('/supported-finite-contract'));await screenshot('site-guide');
  });
  await check('Atlas preserves selection, wheel zoom and clean dragging in the site shell',async()=>{
   await hash('atlas/materials');await page.locator('#focus-branch').click();await page.locator('#map-level').selectOption('5');await page.locator('[data-map="battery-dfn-model"]').waitFor();
   const viewport=await page.locator('#map-viewport').boundingBox(),before=await page.locator('#zoom-level').textContent();
   await page.mouse.move(viewport.x+80,viewport.y+100);await page.mouse.wheel(0,-120);await page.waitForFunction(old=>document.querySelector('#zoom-level').textContent!==old,before);
   const camera=()=>page.locator('#map-world').evaluate(e=>({left:e.style.left,top:e.style.top,zoom:e.style.zoom,transform:getComputedStyle(e).transform}));
   await page.mouse.move(viewport.x+80,viewport.y+100);await page.mouse.down();await page.mouse.move(viewport.x+150,viewport.y+140,{steps:8});await page.mouse.up();assert.equal(await page.evaluate(()=>getSelection().toString()),'');
   const state=await camera(),ids=await page.locator('[data-map]').evaluateAll(es=>es.map(e=>e.dataset.map));await page.locator('[data-site-lang="en"]').click();await page.locator('[data-map="battery-dfn-model"]').waitFor();
   assert.deepEqual(await page.locator('[data-map]').evaluateAll(es=>es.map(e=>e.dataset.map)),ids);assert.deepEqual(await camera(),state);
   for(let i=0;i<12;i++)await page.locator('#map-viewport').dispatchEvent('wheel',{deltaY:-240,clientX:200,clientY:300});
   assert.equal((await camera()).zoom,'2');assert.equal((await camera()).transform,'none');await page.locator('#fit-map').click();await screenshot('site-atlas');
  });
  await check('Browser history and return from another page keep the explorer usable',async()=>{
   await page.locator('[data-nav="research"]').click();await page.locator('.research-grid').waitFor();await page.goBack();await page.locator('#atlas-view:not([hidden])').waitFor();await page.locator('[data-view="index"]').click();assert.ok(await page.locator('#document h1').isVisible());
  });
  await check('Installation commands can be copied',async()=>{
   await hash('start');await page.locator('[data-copy="install"]').click();assert.match(await page.evaluate(()=>navigator.clipboard.readText()),/uv sync --locked --extra dev/);
  });
  for(const width of [1440,1024,736,390,320])await check(`Chinese and English routes fit at ${width}px`,async()=>{
   await page.setViewportSize({width,height:1000});
   for(const lang of ['zh','en'])for(const route of ['home','experiments','experiment/reaction-to-purification/6','atlas/materials','index/combustion','guides/thermal-reaction/native_mechanisms','research','start']){
    await go(route,lang);const overflow=await page.evaluate(()=>({page:document.documentElement.scrollWidth,viewport:innerWidth}));assert.ok(overflow.page<=overflow.viewport+1,`${width}/${lang}/${route}: ${JSON.stringify(overflow)}`);
    if(width===390&&lang==='en'&&['home','experiment/reaction-to-purification/6','guides/thermal-reaction/native_mechanisms'].includes(route))await screenshot('site-mobile-'+route.split('/')[0]);
    if(width===1440&&lang==='en'&&route==='home')await screenshot('site-home-en');
   }
  });
  await check('Mobile guide navigation, capability directory and player stay interactive',async()=>{
   await page.setViewportSize({width:390,height:844});await go('guides/thermal-reaction/native_mechanisms','en');await page.locator('#guide-mobile-nav').click();assert.equal(await page.locator('#guide-navigation').isVisible(),true);
   await page.locator('[data-view="index"]').click();await page.locator('#mobile-directory').click();assert.equal(await page.locator('#outline').isVisible(),true);
   await hash('experiment/reaction-to-purification/6');await page.locator('#next-step').click();assert.equal(await page.locator('#step-counter').textContent(),'7 / 21');
  });
  assert.deepEqual(errors,[]);assert.ok(!requests.some(u=>new URL(u).pathname.startsWith('/api/')));
  fs.writeFileSync(path.join(out,'site-verification.json'),JSON.stringify({checks:checks.length,passed:checks,errors},null,2));console.log(`Completed ${checks.length} checks; no browser errors, missing assets or backend calls.`);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
