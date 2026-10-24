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
 const screenshot=async name=>{await page.locator('img').evaluateAll(async images=>{for(const img of images)img.loading='eager';await Promise.all(images.map(img=>img.decode()));});await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));await page.screenshot({path:path.join(out,name+'.png'),fullPage:true});};
 try{
  await go('home');
  await check('Home exposes the real hierarchy and defers guide payloads',async()=>{
   assert.equal(await page.locator('.site-nav [data-nav]').count(),4);assert.equal(await page.locator('[data-nav="research"]').count(),0);assert.equal(await page.locator('.site-explorer-toolbar').count(),0);
   assert.match(await page.locator('.home-scale').innerText(),/17/);assert.match(await page.locator('.home-scale').innerText(),/72/);
   assert.equal(await page.locator('.architecture-visual').count(),0);await page.locator('.home-hero-art img').evaluate(img=>img.decode());
   assert.match(await page.locator('.home-hero-art img').getAttribute('src'),/assets\/home\/hero-/);assert.equal(await page.locator('.home-domain-card').count(),4);
   assert.ok(!requests.some(u=>/chemworld-guide-(content|localized)\.js|chemworld-api-reference\.js/.test(u)));await screenshot('site-home');
  });
  await check('Homepage domain links, featured record and getting-started entry work',async()=>{
   for(const id of ['reaction','materials','separation','bio']){await go('home');await page.locator(`.home-domain-card[href="#atlas/${id}"]`).click();await page.waitForURL('**#atlas/'+id);await page.locator('#atlas-view:not([hidden]) [data-map]').first().waitFor();}
   await go('home');await page.locator('.home-record-actions .primary').click();await page.locator('#step-counter').waitFor();assert.equal(await page.locator('#step-counter').textContent(),'6 / 21');
   await go('home');await page.locator('.home-next-actions .primary').click();await page.locator('.start-layout').waitFor();
  });
  await check('Milestones remain readable in both languages and link to current scope',async()=>{
   for(const lang of ['zh','en']){
    await go('home',lang);assert.equal(await page.locator('.home-milestone').count(),8);
    assert.deepEqual(await page.locator('.home-milestone-date time:first-child').evaluateAll(items=>items.map(item=>item.dateTime)),['2026-10-24','2026-10-23','2026-10-21','2026-10-20','2026-10-09','2026-09-16','2026-09-06','2026-08-13']);
    assert.match(await page.locator('.home-milestones').innerText(),/Thevenin/);
    assert.match(await page.locator('.home-milestones').innerText(),lang==='en'?/Milestones/:/迭代里程碑/);
    await page.locator('.home-milestones .site-link').click();await page.locator('#guide-reader h1').waitFor();
    assert.ok(page.url().endsWith('#guides/chemworld/current-foundation'));
    assert.match(await page.locator('#guide-body').innerText(),/a6d88a582/);
   }
   await go('home');
  });
  await check('Catalog filtering and player navigation work',async()=>{
   await page.locator('[data-nav="experiments"]').click();await page.locator('.case-card').first().waitFor();
   await page.locator('.case-card').last().scrollIntoViewIfNeeded();await page.waitForFunction(()=>[...document.querySelectorAll('.case-card-art img')].every(img=>img.complete&&img.naturalWidth>0));
   assert.equal(new Set(await page.locator('.case-card-art img').evaluateAll(images=>images.map(img=>img.currentSrc))).size,3);
   assert.equal(await page.locator('.case-card-art > svg').count(),0);await page.evaluate(()=>scrollTo(0,0));await screenshot('site-experiments-zh');
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
   await page.locator('.header-start').click();await page.locator('.start-layout').waitFor();await page.goBack();await page.locator('#atlas-view:not([hidden])').waitFor();await page.locator('[data-view="index"]').click();assert.ok(await page.locator('#document h1').isVisible());
  });
  await check('The retired research route resolves to the experiment collection',async()=>{
   await page.evaluate(()=>location.hash='research');await page.waitForURL('**#experiments');await page.locator('.experiment-collection').waitFor();assert.equal(await page.locator('.research-grid,[data-nav="research"]').count(),0);
  });
  await check('Installation commands can be copied',async()=>{
   await hash('start');await page.locator('[data-copy="install"]').click();assert.match(await page.evaluate(()=>navigator.clipboard.readText()),/uv sync --locked --extra dev/);
  });
  await check('All implemented capability and method pages have executable inline documentation',async()=>{
   await go('index/combustion');
   const coverage=await page.evaluate(()=>{
    const data=window.CHEMWORLD_API_REFERENCE, nodes=[];
    const walk=n=>{nodes.push(n);(n.children||[]).forEach(walk);};window.CHEMWORLD_CAPABILITIES.tree.forEach(walk);
    for(const lang of ['zh','en'])for(const node of nodes.filter(n=>['capability','method'].includes(n.kind))){
     const html=window.ChemWorldApiDocs.render(node,lang);
     if(node.state==='planned'){
      if(!html.includes('api-planned')||html.includes('data-api-block="python"'))throw Error('Fake planned API: '+node.id);
     }else if(!html.includes('data-api-block="python"')||!html.includes('data-api-block="input"')||!html.includes('data-api-block="output"'))throw Error('Missing inline docs: '+node.id);
    }
    return data.coverage;
   });
   assert.equal(coverage.implemented_capabilities,71);assert.equal(coverage.planned_capabilities,1);assert.equal(coverage.examples,81);
   const code=await page.locator('[data-api-block="python"] code').first().textContent();
   await page.locator('[data-api-block="python"] [data-api-copy]').first().click();assert.equal((await page.evaluate(()=>navigator.clipboard.readText())).replace(/\r\n/g,'\n'),code);
   const expected=await page.evaluate(()=>window.CHEMWORLD_API_REFERENCE.recipes.combustion.output);
   assert.deepEqual(JSON.parse(await page.locator('[data-api-block="output"] pre > code').first().textContent()),expected);
   assert.equal(await page.locator('#document a[href*="github.com"]').count(),0);
   await screenshot('site-api-zh');
  });
  await check('Parameter tables preserve every example value, precision and explicit quantity unit',async()=>{
   const checked=await page.evaluate(()=>{
    const data=window.CHEMWORLD_API_REFERENCE, host=document.createElement('div');let blocks=0,values=0,tiny=0,nulls=0,booleans=0;
    try {
     for(const [key,recipe] of Object.entries(data.recipes)){
      data.nodes.__table_qa={examples:[key]};host.innerHTML=window.ChemWorldApiDocs.render({id:'__table_qa',kind:'capability',state:'done'},'en','examples');
      for(const kind of ['input','output']){
       const source=kind==='input'?recipe.inputs:recipe.output, block=host.querySelector(`[data-api-block="${kind}"]`), found=new Set();
       if(JSON.stringify(JSON.parse(block.querySelector('pre > code').textContent))!==JSON.stringify(source))throw Error(`${key}/${kind}: JSON changed`);
       for(const row of block.querySelectorAll('[data-api-path]')){
        const path=JSON.parse(row.dataset.apiPath), value=path.reduce((item,part)=>item[part],source), cell=row.querySelector('.api-data-value');
        const expected=value===null?'null':typeof value==='object'?(Array.isArray(value)?'Empty array':'Empty object'):value===''?'""':String(value);
        const actual=typeof value==='number'?cell.textContent.replaceAll(',',''):cell.textContent;
        if(actual!==expected)throw Error(`${key}/${kind}/${path.join('.')}: ${actual} != ${expected}`);
        if(found.has(JSON.stringify(path)))throw Error('Duplicate field');found.add(JSON.stringify(path));values++;
        if(typeof value==='number'&&value!==0&&Math.abs(value)<1e-8)tiny++;
        if(value===null)nulls++;if(typeof value==='boolean')booleans++;
        const parent=path.slice(0,-1).reduce((item,part)=>item[part],source);
        if(path.at(-1)==='value'&&typeof parent?.unit==='string'&&row.querySelector('.api-data-unit').textContent!==(parent.unit||'—'))throw Error(`${key}/${kind}/${path.join('.')}: lost quantity unit ${parent.unit}`);
       }
       // Traverse the source independently; only a two-field quantity folds its unit into a column.
       const visit=(value,path=[])=>{
        if(value!==null&&typeof value==='object'&&Object.keys(value).length){
         const quantity=!Array.isArray(value)&&Object.keys(value).length===2&&Object.hasOwn(value,'value')&&typeof value.unit==='string'&&(value.value===null||typeof value.value!=='object');
         for(const [name,item] of Object.entries(value))if(!(quantity&&name==='unit'))visit(item,[...path,Array.isArray(value)?Number(name):name]);
        }else if(!found.delete(JSON.stringify(path)))throw Error(`${key}/${kind}: missing ${JSON.stringify(path)}`);
       };
       visit(source);if(found.size)throw Error('Unexpected fields');blocks++;
      }
     }
    } finally {delete data.nodes.__table_qa;}
    return {blocks,values,tiny,nulls,booleans};
   });
   assert.equal(checked.blocks,162);assert.ok(checked.values>1000);assert.ok(checked.tiny>0);assert.ok(checked.nulls>0);assert.ok(checked.booleans>0);
  });
  await check('Table/JSON switching, copy and nested groups work independently with the keyboard',async()=>{
   const input=page.locator('[data-api-block="input"]').first(),output=page.locator('[data-api-block="output"]').first();
   assert.equal(await input.locator('[data-api-panel="pretty"]').isVisible(),true);assert.equal(await output.locator('[data-api-panel="pretty"]').isVisible(),true);
   const expected=await page.evaluate(()=>window.CHEMWORLD_API_REFERENCE.recipes.combustion.output);
   await output.locator('[data-api-copy]').click();assert.deepEqual(JSON.parse(await page.evaluate(()=>navigator.clipboard.readText())),expected);
   await output.locator('[data-api-view="json"]').focus();await page.keyboard.press('Enter');assert.equal(await output.locator('[data-api-panel="json"]').isVisible(),true);
   assert.equal(await input.locator('[data-api-panel="pretty"]').isVisible(),true);
   await output.locator('[data-api-view="pretty"]').click();const group=output.locator('[data-api-group]').first();
   assert.equal(await output.locator('.api-data-row:visible').count(),6);await group.focus();await page.keyboard.press('Space');assert.equal(await output.locator('.api-data-row:visible').count(),3);
   await page.keyboard.press('Enter');assert.equal(await output.locator('.api-data-row:visible').count(),6);
   await hash('index/distillation');await page.locator('[data-tab="examples"]').click();
   const large=page.locator('[data-api-block="output"] [data-api-group]').filter({has:page.locator('.api-field-label',{hasText:/^timed_fractions$/})}).first();
   await large.waitFor();const before=await page.locator('[data-api-block="output"] .api-data-row:visible').count();await large.click();
   assert.ok(await page.locator('[data-api-block="output"] .api-data-row:visible').count()>before);
   await large.click();assert.equal(await page.locator('[data-api-block="output"] .api-data-row:visible').count(),before);
   await go('index/combustion');
  });
  await check('Documentation footer actions share font, line height and vertical alignment',async()=>{
   for(const lang of ['zh','en']){
    await go('index/combustion',lang);
    const styles=await page.locator('.doc-links > a,.doc-links > button').evaluateAll(items=>items.map(item=>{
     const css=getComputedStyle(item),box=item.getBoundingClientRect();return {font:css.fontFamily,size:css.fontSize,weight:css.fontWeight,line:css.lineHeight,color:css.color,top:box.top,height:box.height};
    }));
    assert.equal(styles.length,2);assert.deepEqual(styles[0],styles[1]);
    await page.locator('.doc-links').scrollIntoViewIfNeeded();await screenshot('site-api-footer-'+lang);
   }
   await go('index/combustion');
  });
  await check('API tabs expose parameter defaults, action units, additional examples and planned states',async()=>{
   await page.locator('[data-tab="io"]').click();assert.ok(await page.locator('.api-parameters tbody tr').count()>0);
   await page.locator('[data-tab="io"]').focus();await page.keyboard.press('ArrowRight');assert.equal(await page.locator('[data-tab="examples"]').getAttribute('aria-selected'),'true');
   await hash('index/surface');await page.locator('[data-tab="io"]').click();assert.ok(await page.locator('.api-action').count()>0);
   await page.locator('.api-action summary').first().click();assert.match(await page.locator('.api-action[open]').innerText(),/mol/);
   await hash('index/spme-dfn');await page.locator('[data-tab="examples"]').click();await page.locator('.api-example-more>summary').click();assert.equal(await page.locator('.api-example-more[open] [data-api-block="python"]').count(),1);
   await hash('index/bridge');assert.equal(await page.locator('.api-planned').count(),1);assert.equal(await page.locator('[data-api-copy]').count(),0);
   await go('index/combustion','en');assert.match(await page.locator('.api-reference').innerText(),/Complete Python example/);await screenshot('site-api-en');
  });
  await check('New foundation APIs render executed outputs in both languages',async()=>{
   for(const lang of ['zh','en'])for(const [id,key] of [['advanced-thermo','helmholtz-state'],['activity-thermo','activity-properties'],['fluid-engineering','flow-meter-inverse'],['working-fluids','water-if97'],['general-fluid-eos','molecular-pcsaft'],['descriptor-catalysis','surface-control'],['kinetic-expansion','mechanism-growth'],['flame-catalysis','gas-flow-column'],['flowsheet','process-path'],['phase-diagram','ternary-equilibrium'],['water-expansion','aqueous-filtration'],['bio-analysis','sbml-finite-time'],['battery-aging','battery-plating'],['impedance','msmr-impedance']]){
    await go('index/'+id,lang);
    assert.equal(await page.locator('.api-planned').count(),0);
    const output=page.locator('[data-api-block="output"]').first();
    await output.waitFor();assert.equal(await output.locator('[data-api-panel="pretty"]').isVisible(),true);
    const expected=await page.evaluate(k=>window.CHEMWORLD_API_REFERENCE.recipes[k].output,key);
    await output.locator('[data-api-view="json"]').click();
    // JSON.stringify serializes IEEE negative zero as 0; compare the JSON view
    // against that exact representation without rounding any nonzero value.
    assert.deepEqual(JSON.parse(await output.locator('pre > code').textContent()),JSON.parse(JSON.stringify(expected)));
    await output.locator('[data-api-view="pretty"]').click();
    const units=await output.locator('[data-api-path]').evaluateAll(rows=>Object.fromEntries(rows.map(row=>[JSON.parse(row.dataset.apiPath).at(-1),row.querySelector('.api-data-unit').textContent])));
    if(id==='advanced-thermo'){assert.equal(units.cp_J_mol_K,'J/(mol·K)');assert.equal(units.cv_J_mol_K,'J/(mol·K)');assert.equal(units.enthalpy_J_mol,'J/mol');}
    if(id==='fluid-engineering'){assert.equal(units.recovered_mass_flow_kg_s,'kg/s');assert.equal(units.differential_pressure_Pa,'Pa');}
    if(id==='working-fluids'){assert.equal(units.cp_J_kg_K,'J/(kg·K)');assert.equal(units.enthalpy_J_kg,'J/kg');await screenshot('site-water-'+lang);}
    if(id==='advanced-thermo')await screenshot('site-helmholtz-'+lang);
   }
  });
  for(const width of [1440,1024,736,390,320])await check(`Chinese and English routes fit at ${width}px`,async()=>{
   await page.setViewportSize({width,height:1000});
   for(const lang of ['zh','en'])for(const route of ['home','experiments','experiment/reaction-to-purification/6','atlas/materials','index/combustion','index/advanced-thermo','index/fluid-engineering','index/working-fluids','index/general-fluid-eos','guides/thermal-reaction/native_mechanisms','start']){
    await go(route,lang);const overflow=await page.evaluate(()=>({page:document.documentElement.scrollWidth,viewport:innerWidth}));assert.ok(overflow.page<=overflow.viewport+1,`${width}/${lang}/${route}: ${JSON.stringify(overflow)}`);
    const clippedNav=await page.locator('.site-nav a').evaluateAll(links=>links.filter(link=>{const box=link.getBoundingClientRect();return box.left<0||box.right>innerWidth;}).map(link=>link.textContent));assert.deepEqual(clippedNav,[],`${width}/${lang}: clipped navigation`);
    if(width===390&&lang==='en'&&['home','experiments','experiment/reaction-to-purification/6','guides/thermal-reaction/native_mechanisms','index/combustion'].includes(route))await screenshot('site-mobile-'+route.split('/')[0]);
    if(width===1440&&lang==='en'&&route==='home')await screenshot('site-home-en');
    if(width===1440&&lang==='en'&&route==='experiments')await screenshot('site-experiments-en');
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
