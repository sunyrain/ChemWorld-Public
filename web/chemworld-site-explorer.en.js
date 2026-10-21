// Generated English preview edition.
/* Local design preview. The atlas and index share one editorial capability snapshot. */
(() => {
  'use strict';
  const {tree, registry, snapshot, taxonomy} = window.CHEMWORLD_CAPABILITIES;
  const $ = (selector, scope = document) => scope.querySelector(selector);
  const $$ = (selector, scope = document) => [...scope.querySelectorAll(selector)];
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const states = {done:"Delivered", finite:"Finite model", partial:"In development", planned:"Planned"};
  const nodes = new Map();
  const root = {id:'chemworld',kind:'project',name:'ChemWorld',preview:"Recordable physical and chemical experimental environment",children:tree};
  const kindLabels={project:"Project",layer:"Architecture layer",domain:"Capability domain",family:"Function group",capability:"Capability",method:"Method / variant"};
  function register(node, parent = null, depth = 0) {
    node.parent = parent; node.depth = depth;
    nodes.set(node.id,node);
    (node.children || []).forEach(child => register(child,node,depth+1));
    node.leaves = node.kind==='capability' ? [node] : (node.children||[]).flatMap(child => child.leaves);
    node.methods = (node.kind==='method' ? [node] : (node.children||[]).flatMap(child=>child.methods));
  }
  register(root);
  const opened = new Set(['chemworld','physics','runtime','evaluation']);
  let selected = 'combustion', view = 'index', activeTab = 'overview';
  let guideReader, guideId=null, guideAnchor='', guideMatches=[];
  let mapSize = {width:1200,height:900}, positions = new Map();
  let camera = {x:24,y:50,k:1}, mapReady = false, searchMatches = [];
  let focusId='chemworld';
  const domainCount=[...nodes.values()].filter(n=>n.kind==='domain').length;
  const familyCount=[...nodes.values()].filter(n=>n.kind==='family').length;
  const nodeCountLabel=node=>node.kind==='capability'?`${node.methods.length} methods / variants`:`${node.leaves.length} capabilities`;
  const kindLabel=node=>`${kindLabels[node.kind]}${node.type?' · '+node.type:''}`;
  const special = {
    combustion: {
      intro:"Based on clearly defined feed and reaction specifications, we can determine the composition of the product and the corresponding heat output.",
      rows:[["Elemental quantification","The complete combustion products and the principle of conservation of atoms."],["Heat calculation","HHV / LHV, for reference against the specified standard."],["Flow specifications","Two-stream / Three-stream, wet base / dry base tail gas"],["Finite consumption","Feed rate → Product yield and heat"]],
      flow:["Feed and specifications","Measurement and thermal equilibrium","Products and Heat"],
      boundary:"Complete combustion and explicit reference to the reference state; the dynamics of the flame, product temperature, and general equilibrium are also extended.",
      io:[["Input","Composition of constituent elements, enthalpy or heat value, phase reference, limited feed rate; separate specifications for flow rate issues."],["Output","Measured product yield, oxygen consumption, HHV / LHV, heat corresponding to finite material quantities, and the selected flow rate specifications."],["Units & conventions","The Moore heat model uses J/mol for finite heat usage, and J for limited heat usage; heat release is defined using a negative enthalpy change convention."],["Failure conditions","Inconsistencies in enthalpy, reference state, or the absence of necessary product data will result in rejection."]],
      related:['assessment','conservation','caloric'],
      code:'from chemworld.foundation.units import Quantity\nfrom chemworld.physchem.combustion import combustion_data\n\nfuel = combustion_data("CH4", method="Stoichiometry", Hf=-74520)\nheat = fuel.heat(Quantity(2, "mol"), basis="HHV")\nprint(heat.value, heat.unit)',
      exampleNote:"Using a clearly defined example for the enthalpy of formation, calculate the complete combustion heat of 2 mol methane; without solving for the flame temperature.",
      validation:"This includes ensuring the conservation of elements and products, maintaining consistency with the HHV / LHV standard, limiting material heat, and adhering to specifications for two-stream and three-stream configurations, as well as rejecting inputs with conflicts or missing information.",
      source:'src/chemworld/physchem/combustion.py',test:'tests/test_combustion.py'
    },
    temperature: {
      intro:"Convert historical temperature readings to an explicitly chosen temperature scale for use in property calculations, inputs and observations.",
      rows:[["Supported temperature scales",'ITS-90 / ITS-68 / ITS-48 / ITS-27 / ITS-76'],["Forward & inverse conversion","Forward, inverse and inter-scale conversions within each scale's valid range"],["Native integration","Quantity, InputValue, and the observation channel"],["Source conventions","Literature conventions and differences between reference versions are explicitly distinguished"]],
      flow:["Historical readings and temperature scale","Bounded calibration & inverse conversion","Material property input or derived observation"],
      io:[["Input","Specify the temperature, original temperature scale, target temperature, and required source according to the agreed-upon units."],["Output","Temperature after calibration; the specific result can include the derivative of the same transformation."],["Boundary","Evaluate the true mapping endpoint; scalar observations retain the original values and sample times."]],
      related:['solubility','optics','conservation'],
      code:'from chemworld.foundation.units import Quantity\nfrom chemworld.physchem import temperature_quantity, water_refractive_index_iapws\n\ncalibrated = temperature_quantity(Quantity(26.85, "degC"), "IPTS-68", "ITS-90")\nindex = water_refractive_index_iapws(calibrated.value, 997)',
      exampleNote:"Calculate the refractive index of water after converting historical temperature data; the example is based on the current native field guidelines.",
      validation:"Verify nodes, intervals, forward and reverse directions, actual endpoints, and cross-functional consumption; retain the source value differences separately.",
      source:'docs/native_foundation.md',test:'tests/test_temperature_scales.py'
    }
  };
  const exampleFiles = {
    calibration:'native_calibration', 'thermal-reaction':'native_gas_mechanism', surface:'native_multisite_surface',
    flash:'native_flash_research', extraction:'native_liquid_research', distillation:'native_distillation_research',
    crystal:'native_crystal_research', adsorption:'native_sorbent_research', transport:'native_liquid_transport',
    electrode:'native_electrode_research', spm:'native_battery_cycles', 'spme-dfn':'battery_dfn',
    aqueous:'native_aqueous_research', mineral:'native_mineral_research', membrane:'native_membrane_fouling',
    handling:'native_surface_research', devices:'reuse_native_equipment', programmed:'programmed_chromatography',
    scheduler:'multi_device_lab', persistence:'shared_lifecycle_lab', continuation:'closed_shared_labs'
  };
  const sourceByBranch = {
    foundation:'docs/world-foundations.md', properties:'docs/native_foundation.md', reaction:'docs/native_mechanisms.md',
    separation:'docs/native_foundation.md', materials:'docs/native_domains.md', water:'docs/native_foundation.md',
    bio:'docs/native_biochemical.md', actions:'docs/operations.md', instruments:'docs/native_foundation.md',
    lifecycle:'docs/native_shared_lab.md', 'research-tasks':'docs/quickstart.md', protocol:'docs/agents.md',
    views:'docs/student-lab.md'
  };
  function lineage(node) { const out=[]; for(let n=node;n;n=n.parent)out.unshift(n); return out; }
  function source(node) { return special[node.id]?.source || lineage(node).reverse().find(n=>n.source)?.source || sourceByBranch[lineage(node).find(n=>n.kind==='domain')?.id] || 'docs/world-foundations.md'; }
  function status(node, badge=false) { return node.state ? `<span class="status ${node.state}${badge?' status-badge':''}"><i aria-hidden="true"></i>${states[node.state]}</span>` : ''; }
  function counts(node) { return Object.fromEntries(Object.keys(states).map(state=>[state,node.leaves.filter(n=>n.state===state).length])); }
  function announce(text) { $('#announcement').textContent=text; }
  function openAncestors(node) { lineage(node).slice(0,-1).forEach(n=>opened.add(n.id)); }
  function updateHash(mode='replace') {
    if(!mode)return;
    const next=view==='guides'?guideReader.route(guideId,guideAnchor,selected):`#${view}/${selected}`;
    if(location.hash!==next)history[mode==='push'?'pushState':'replaceState'](null,'',next);
    window.dispatchEvent(new Event('chemworld-route'));
  }
  function select(id, {reveal=true,resetTab=true,historyMode='replace'}={}) {
    if(!nodes.has(id))return;
    selected=id;
    if(!lineage(nodes.get(id)).some(n=>n.id===focusId))focusId='chemworld';
    if(resetTab)activeTab='overview';
    if(reveal)openAncestors(nodes.get(id));
    renderOutline(); renderDocument(); renderInspector();
    if(view==='atlas')renderMap(false);
    updateHash(historyMode); announce(`已选 ${nodes.get(id).name}`);
  }
  function openGuide(id,anchor='',nodeId=selected,{historyMode='push'}={}) {
    select(nodes.has(nodeId)?nodeId:selected,{historyMode:null});
    guideId=id;guideAnchor=anchor;
    switchView('guides',{historyMode,contextual:false});
    announce(`Currently reading ${guideReader.pages[id]?.title||"Domain guides"}`);
  }
  function switchView(next,{historyMode='push',contextual=true}={}) {
    view=['index','atlas','guides'].includes(next)?next:'index';
    if(view==='atlas'&&!mapReady){
      opened.clear();['chemworld','physics','runtime','evaluation'].forEach(id=>opened.add(id));
      renderOutline();
    }else if(view==='index'){
      openAncestors(nodes.get(selected));renderOutline();
    }
    if(view==='guides'&&(contextual||!guideId)){
      const guide=guideReader.resolve(nodes.get(selected));guideId=guide.id;guideAnchor=guide.anchor;
    }
    for(const name of ['index','atlas','guides'])$(`#${name}-view`).hidden=view!==name;
    $$('[data-view]').forEach(button=>button.tagName==='BUTTON'&&button.setAttribute('aria-pressed',String(button.dataset.view===view)));
    updateHash(historyMode);
    if(view==='guides')guideReader.render(guideId,guideAnchor,nodes.get(selected));
    if(view==='atlas')requestAnimationFrame(()=>{renderMap(!mapReady);mapReady=true;});
  }
  function renderOutline() {
    const focused=document.activeElement?.dataset.tree;
    const html=(node,depth)=>{
      const branch=!!node.children, isOpen=opened.has(node.id);
      return `<li><button type="button" class="tree-row ${branch?'branch':'leaf'} ${selected===node.id?'selected':''}" style="--depth:${depth}" data-depth="${depth}" data-kind="${node.kind}" data-tree="${node.id}" ${branch?`aria-expanded="${isOpen}"`:''} ${selected===node.id?'aria-current="true"':''} title="${esc(kindLabel(node))}">
        <span class="tree-toggle" aria-hidden="true">${branch?(isOpen?'▾':'▸'):' '}</span><span class="tree-name">${esc(node.name)}</span>${node.state?status(node):`<span class="tree-count" aria-label="${nodeCountLabel(node)}">${node.leaves.length}</span>`}</button>
        ${branch&&isOpen?`<ul>${node.children.map(child=>html(child,depth+1)).join('')}</ul>`:''}</li>`;
    };
    $('#outline').innerHTML=`<ul>${tree.map(node=>html(node,0)).join('')}</ul>`;
    if(focused)$(`[data-tree="${focused}"]`)?.focus({preventScroll:true});
  }
  $('#outline').addEventListener('click',event=>{
    const button=event.target.closest('[data-tree]');if(!button)return;
    const node=nodes.get(button.dataset.tree);
    if(node.children){opened.has(node.id)?opened.delete(node.id):opened.add(node.id);select(node.id,{reveal:false});}
    else{select(node.id);$('.sidebar').classList.remove('mobile-open');$('#mobile-directory').setAttribute('aria-expanded','false');}
  });
  function related(node) {
    return special[node.id]?.related || (node.parent?.children||[]).filter(n=>n.id!==node.id).slice(0,3).map(n=>n.id);
  }
  function documentationLinks(node) {
    const example=exampleFiles[node.id],guide=guideReader.resolve(node);
    return `<div class="doc-links"><a href="${guideReader.route(guide.id,guide.anchor,node.id)}" data-open-guide="${node.id}">Read the domain guide →</a>${special[node.id]?.source?.endsWith('.py')?`<a href="https://github.com/sunyrain/ChemWorld-Public/blob/main/${source(node)}" target="_blank" rel="noopener">Methods and interfaces ↗</a>`:''}${example?`<a href="https://github.com/sunyrain/ChemWorld-Public/blob/main/src/chemworld/examples/${example}.py" target="_blank" rel="noopener">Runnable example ↗</a>`:''}<button type="button" class="text-button" data-go-atlas="${node.id}">Locate in the atlas →</button></div>`;
  }
  function table(rows) { return `<table class="capability-table"><tbody>${rows.map(([key,value])=>`<tr><th scope="row">${esc(key)}</th><td>${esc(value)}</td></tr>`).join('')}</tbody></table>`; }
  function overviewContent(node) {
    if(!node.state)return `<h2>${kindLabels[node.children[0]?.kind]||"Subsequent content"}</h2>${childList(node)}<h2>Table of Contents (Hierarchical Structure)</h2><p>${taxonomy.levels.slice(1).join(' → ')}</p><p>Nodes at the same level use the same hierarchical classification.${taxonomy.counting}</p>`;
    const detail=special[node.id];
    return `<h2>Available capabilities</h2>${detail?.rows?table(detail.rows):`<p>${esc(node.ability)}</p>`}
      ${detail?.flow?`<h2>Example workflow</h2><div class="flow-path">${detail.flow.map(esc).join('<b aria-hidden="true">→</b>')}</div>`:node.example?`<h2>Representative experiment</h2><p>${esc(node.example)}</p>`:''}
      <h2>Scope & limits</h2><p>${esc(detail?.boundary||node.boundary)}</p>
      ${node.children?`<h2>Method / variant</h2>${childList(node)}`:''}
      <div class="related"><h2>Related capabilities</h2>${related(node).map(id=>`<button type="button" data-select="${id}">${esc(nodes.get(id).name)}</button>`).join('')}</div>`;
  }
  function childList(node){return `<div class="child-list">${(node.children||[]).map(child=>`<button type="button" data-select="${child.id}"><span>${esc(child.name)}</span>${child.state?status(child):`<small>${child.leaves.length} capabilities →</small>`}</button>`).join('')}</div>`;}
  function tabContent(node) {
    const detail=special[node.id];
    if(activeTab==='overview')return overviewContent(node);
    if(activeTab==='io')return detail?.io?`<h2>Inputs & outputs</h2>${table(detail.io)}`:`<h2>Interface scope</h2><p>${esc(node.ability||node.preview)}</p><h2>Conditions for use</h2><p>${esc(node.boundary||"Specific properties, states, and task permissions are defined separately for each capability.")}</p><p>For parameters, types, and return fields, please refer to the domain guide linked on this page.</p>`;
    if(activeTab==='examples') {
      const file=exampleFiles[node.id];
      return `<h2>${detail?.code?"Minimum example":file?"Runnable example":"Use cases"}</h2><p>${esc(detail?.exampleNote||node.example||(node.state==='planned'?"This refers to a planning capability, but there is currently no operational example that has been delivered.":"Specific usage details for this item can be found in the relevant field guide."))}</p>${detail?.code?`<pre><code>${esc(detail.code)}</code></pre>`:file?`<pre><code>uv run --no-sync python -m chemworld.examples.${file}</code></pre>`:''}${file?`<a href="https://github.com/sunyrain/ChemWorld-Public/blob/main/src/chemworld/examples/${file}.py" target="_blank" rel="noopener">View example source code ↗</a>`:''}`;
    }
    return `<h2>Current status</h2><p>${status(node,true)||"The branches include various delivery statuses."}</p><h2>Validation scope</h2><p>${esc(detail?.validation||node.boundary||"The scope of delivery is recorded separately for each specific capability, and it is not inferred that the entire domain is complete based on the classification branches.")}</p><p class="scope-statement">${node.state==='planned'?"The planned items represent potential directions for expansion, but they have not yet been implemented as delivered capabilities.":"The development acceptance within the specified scope has been completed and delivered. The limited model retains specific models, parameters, and the applicable physical domain."}</p>${detail?.test?`<a href="https://github.com/sunyrain/ChemWorld-Public/blob/main/${detail.test}" target="_blank" rel="noopener">View functionality and boundary checks ↗</a>`:''}<div class="doc-links"><a href="https://github.com/sunyrain/ChemWorld-Public/blob/main/docs/current-foundation.md" target="_blank" rel="noopener">View current delivery and acceptance records ↗</a></div>`;
  }
  function renderDocument() {
    const node=nodes.get(selected), detail=special[node.id];
    const tabs=[['overview',"Overview"],['io',"Inputs & outputs"],['examples',"Examples"],['validation',"Validation scope"]];
    $('#document').innerHTML=`<div class="doc-inner"><div class="breadcrumbs">${lineage(node).slice(1,-1).map(n=>`<button type="button" data-select="${n.id}">${esc(n.name)}</button>`).join('<span>/</span>')||"<span>Architecture overview</span>"}</div>
      <div class="node-kind">${esc(kindLabel(node))}</div><div class="title-row"><h1>${esc(node.name)}</h1>${status(node,true)}</div><p class="doc-intro">${esc(detail?.intro||node.preview||node.ability)}</p>
      <div class="doc-tabs" role="tablist" aria-label="Details of Abilities Section">${tabs.map(([id,label])=>`<button type="button" id="tab-${id}" role="tab" data-tab="${id}" aria-selected="${activeTab===id}" aria-controls="detail-panel" tabindex="${activeTab===id?0:-1}">${label}</button>`).join('')}</div>
      <div id="detail-panel" class="doc-content" role="tabpanel" aria-labelledby="tab-${activeTab}">${tabContent(node)}</div>
      ${documentationLinks(node)}<div class="source-note">Current development platform ${snapshot} Skill snapshot<br>Software declaration describing the number of registered users; the installation, task permissions, and current executable capabilities are evaluated separately for each specific world.</div></div>`;
  }
  $('#document').addEventListener('click',event=>{
    const button=event.target.closest('[data-tab]');if(!button)return;
    activeTab=button.dataset.tab;renderDocument();$(`[data-tab="${activeTab}"]`).focus({preventScroll:true});
  });
  $('#document').addEventListener('keydown',event=>{
    if(!event.target.matches('[data-tab]')||!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
    event.preventDefault();const ids=['overview','io','examples','validation'];let i=ids.indexOf(activeTab);
    i=event.key==='Home'?0:event.key==='End'?3:(i+(event.key==='ArrowRight'?1:3))%4;
    activeTab=ids[i];renderDocument();$(`[data-tab="${activeTab}"]`).focus({preventScroll:true});
  });
  function renderInspector() {
    const node=nodes.get(selected), detail=special[node.id];
    let content;
    if(!node.state){
      const c=counts(node);
      content=`<div class="inspector-section"><h3>${nodeCountLabel(node)}</h3><div class="distribution" aria-hidden="true">${Object.keys(states).map(s=>c[s]?`<span class="${s}" style="flex:${c[s]}"></span>`:'').join('')}</div><div class="distribution-labels">${Object.keys(states).map(s=>`<span class="status ${s}"><i></i>${states[s]} ${c[s]}</span>`).join('')}</div><p class="count-note">Method / Variant: Non-repeating count</p></div><div class="inspector-section"><h3>${kindLabels[node.children[0]?.kind]||"Branch"}</h3><div class="map-branch-list">${node.children.map(child=>`<button type="button" data-select="${child.id}">${esc(child.name)} ${!child.state?`· ${child.leaves.length}`:''} →</button>`).join('')}</div></div>`;
    }else content=`<div class="inspector-section"><h3>Ability</h3><p>${esc(node.ability)}</p></div>${node.example?`<div class="inspector-section"><h3>Representative experiment</h3><p>${esc(node.example)}</p></div>`:''}<div class="inspector-section"><h3>Scope & limits</h3><p>${esc(detail?.boundary||node.boundary)}</p></div>`;
    $('#inspector').innerHTML=`<div class="inspect-top"><div class="inspector-eyebrow">${esc(kindLabel(node))}</div><h2>${esc(node.name)}</h2>${status(node,true)}${node.preview?`<p class="group-preview">${esc(node.preview)}</p>`:''}</div>${content}${node.state&&node.children?`<div class="inspector-section"><h3>Method / variant</h3><div class="map-branch-list">${node.children.map(child=>`<button type="button" data-select="${child.id}">${esc(child.name)} →</button>`).join('')}</div></div>`:''}<button type="button" class="inspect-cta" data-open-index="${node.id}">Read the ability index →</button><button type="button" class="inspect-secondary" data-validate="${node.id}">Check the verification range</button>`;
  }
  function mapVisible() { const list=[]; function walk(n){list.push(n);if(opened.has(n.id))n.children?.forEach(walk);}walk(nodes.get(focusId));return list; }
  function renderMap(fit=false) {
    if($('#atlas-view').hidden)return;
    const previousFocus=document.activeElement?.dataset.map;
    const previousSelected=positions.get(selected);
    positions=new Map();let cursor=40;
    const compact=viewport.clientWidth<=600;
    const focused=nodes.get(focusId);
    $$('#map-level option').forEach(option=>{
      option.disabled=option.value==='custom'||Number(option.value)<=focused.depth;
    });
    const widthFor={project:190,layer:235,domain:315,family:310,capability:400,method:350};
    const maxDepth=Math.max(...mapVisible().map(n=>n.depth));
    const xAt={},widthAt={};let x=30;
    for(let depth=focused.depth;depth<=maxDepth;depth++){
      widthAt[depth]=Math.max(...mapVisible().filter(n=>n.depth===depth).map(n=>widthFor[n.kind]||350));
      xAt[depth]=x;x+=widthAt[depth]+55;
    }
    function layout(node) {
      let y;
      if(compact){
        y=cursor;cursor+=46;
        const indent=Math.min(node.depth-focused.depth,3)*16;
        positions.set(node.id,{x:18+indent,y,width:Math.max(150,viewport.clientWidth-40-indent)});
        if(node.children&&opened.has(node.id))node.children.forEach(layout);
        return y;
      }
      if(node.children&&opened.has(node.id)){
        const childPositions=node.children.map(layout);y=(childPositions[0]+childPositions.at(-1))/2;
        if(node.depth===1)cursor+=18;
      }else{y=cursor;cursor+=node.kind==='domain'?38:node.kind==='method'?42:46;}
      positions.set(node.id,{x:xAt[node.depth],y,width:widthAt[node.depth]});return y;
    }
    layout(focused);
    $('#map-scope').innerHTML=`<span>Scope</span>${lineage(focused).map(n=>`<button type="button" data-focus="${n.id}">${esc(n.name)}</button>`).join('<span>/</span>')}<small>${kindLabels[focused.kind]} · ${nodeCountLabel(focused)}</small>`;
    const visible=mapVisible();
    mapSize={width:Math.max(...visible.map(n=>positions.get(n.id).x+positions.get(n.id).width))+40,height:cursor+40};
    const world=$('#map-world');world.style.width=`${mapSize.width}px`;world.style.height=`${mapSize.height}px`;
    const svg=$('#map-lines');svg.setAttribute('width',mapSize.width);svg.setAttribute('height',mapSize.height);
    const pathNodes=new Set(lineage(nodes.get(selected)).map(n=>n.id));
    svg.innerHTML=visible.filter(n=>positions.has(n.parent?.id)).map(node=>{
      const p=positions.get(node.parent.id),c=positions.get(node.id), mid=c.x-24;
      const path=compact?`M${p.x+8},${p.y+17} V${c.y} H${c.x-3}`:`M${p.x+p.width+6},${p.y} H${mid} V${c.y} H${c.x-7}`;
      return `<path d="${path}" fill="none" stroke="${pathNodes.has(node.id)?'#86b3a5':'#c5cfd4'}" stroke-width="1"/>`;
    }).join('');
    let representative=nodes.get(selected);
    while(representative&&!positions.has(representative.id))representative=representative.parent;
    $('#map-nodes').innerHTML=visible.map(node=>{
      const p=positions.get(node.id),branch=!!node.children;
      return `<button type="button" class="map-node ${node.depth===0?'root':node.depth===1?'layer':''} ${representative?.id===node.id?'selected':''}" style="left:${p.x}px;top:${p.y-17}px;width:${p.width}px" data-map="${node.id}" data-kind="${node.kind}" ${branch?`aria-expanded="${opened.has(node.id)}"`:''} ${selected===node.id?'aria-current="true"':''} aria-label="${esc(node.name)}，${esc(kindLabel(node))}${branch?`，${nodeCountLabel(node)}，${opened.has(node.id)?"Fold away; put away":"Expand"}`:`，${states[node.state]}`}" title="${esc(kindLabel(node))}"><span aria-hidden="true" class="${branch?'expand-box':'leaf-mark'}">${branch?(opened.has(node.id)?'−':'+'):''}</span><span class="node-title">${esc(node.name)}</span>${node.state?status(node):`<span class="node-count">${node.leaves.length}</span>`}</button>`;
    }).join('');
    if(fit)fitMap();
    else{
      const newSelected=positions.get(selected);
      if(previousSelected&&newSelected)camera.y+=(previousSelected.y-newSelected.y)*camera.k;
      transformMap();
    }
    if(previousFocus)$(`[data-map="${previousFocus}"]`)?.focus({preventScroll:true});
  }
  function fitMap() {
    const box=$('#map-viewport').getBoundingClientRect();
    if(box.width===0)return;
    if(box.width<=600){camera={k:1,x:0,y:15};transformMap();return;}
    const k=Math.min(1.12,(box.width-40)/mapSize.width,(box.height-110)/mapSize.height);
    camera={k:Math.max(.12,k),x:(box.width-mapSize.width*k)/2,y:35+(box.height-110-mapSize.height*k)/2};
    transformMap();
  }
  function transformMap() {
    // Layout zoom repaints glyphs and SVG at the target resolution. Scaling a
    // permanently composited layer stretches its cached text raster at 200%.
    const world=$('#map-world');
    world.style.zoom=String(camera.k);
    world.style.left=`${camera.x/camera.k}px`;world.style.top=`${camera.y/camera.k}px`;
    $('#zoom-level').textContent=`${Math.round(camera.k*100)}%`;
    $('#zoom-out').disabled=camera.k<=.12;$('#zoom-in').disabled=camera.k>=2;
    const box=$('#map-viewport').getBoundingClientRect();
    const k=Math.min(164/mapSize.width,100/mapSize.height),ox=(180-mapSize.width*k)/2,oy=(116-mapSize.height*k)/2;
    const marks=[...positions.entries()].map(([id,p])=>`<rect x="${ox+p.x*k}" y="${oy+(p.y-7)*k}" width="${Math.max(2,p.width*k*.65)}" height="${Math.max(1.2,14*k)}" fill="${id===selected?'#176b5b':'#aebcc3'}"/>`).join('');
    $('#minimap').innerHTML=`<defs><clipPath id="mini-clip"><rect width="180" height="116"/></clipPath></defs><g clip-path="url(#mini-clip)">${marks}<rect x="${ox-camera.x/camera.k*k}" y="${oy-camera.y/camera.k*k}" width="${box.width/camera.k*k}" height="${box.height/camera.k*k}" fill="#176b5b08" stroke="#458b79" stroke-width="1"/></g>`;
  }
  function zoom(factor,x,y) {
    const box=$('#map-viewport').getBoundingClientRect();x??=box.width/2;y??=box.height/2;
    const next=Math.max(.12,Math.min(2,camera.k*factor)),ratio=next/camera.k;
    camera={x:x-(x-camera.x)*ratio,y:y-(y-camera.y)*ratio,k:next};transformMap();
  }
  function locateSelected() {
    let node=nodes.get(selected);while(node&&!positions.has(node.id))node=node.parent;
    const p=positions.get(node?.id);if(!p)return;
    const box=$('#map-viewport').getBoundingClientRect();camera.k=Math.min(1,(box.width-42)/p.width);
    camera.x=box.width/2-(p.x+p.width/2)*camera.k;camera.y=box.height/2-p.y*camera.k;transformMap();
  }
  let drag=null,suppressPointerClick=false;
  $('#map-nodes').addEventListener('click',event=>{
    const button=event.target.closest('[data-map]');if(!button)return;
    const node=nodes.get(button.dataset.map);
    if(node.children){opened.has(node.id)?opened.delete(node.id):opened.add(node.id);}
    $('#map-level').value='custom';
    select(node.id,{reveal:false});
  });
  const viewport=$('#map-viewport');
  // Suppress the click produced by the same pointer gesture, including on a node.
  // Keep keyboard activation (detail === 0) and the next genuine click available.
  viewport.addEventListener('click',event=>{
    if(suppressPointerClick&&event.detail!==0){
      suppressPointerClick=false;event.preventDefault();event.stopImmediatePropagation();
    }
  },true);
  viewport.addEventListener('dragstart',event=>event.preventDefault());
  viewport.addEventListener('pointerdown',event=>{
    if(event.button!==0||!event.isPrimary||event.target.closest('.map-controls'))return;
    suppressPointerClick=false;
    drag={id:event.pointerId,x:event.clientX,y:event.clientY,originX:camera.x,originY:camera.y,moved:false,threshold:event.pointerType==='touch'?9:6};
  });
  viewport.addEventListener('pointermove',event=>{
    if(!drag||event.pointerId!==drag.id)return;
    const dx=event.clientX-drag.x,dy=event.clientY-drag.y;
    if(!drag.moved){
      if(Math.hypot(dx,dy)<drag.threshold)return;
      drag.moved=true;viewport.setPointerCapture(event.pointerId);viewport.classList.add('dragging');
    }
    event.preventDefault();
    camera.x=drag.originX+dx;camera.y=drag.originY+dy;transformMap();
  });
  const endDrag=event=>{
    if(!drag||event.pointerId!==drag.id)return;
    suppressPointerClick=drag.moved;drag=null;viewport.classList.remove('dragging');
    if(viewport.hasPointerCapture(event.pointerId))viewport.releasePointerCapture(event.pointerId);
  };
  window.addEventListener('pointerup',endDrag);
  viewport.addEventListener('pointercancel',endDrag);viewport.addEventListener('lostpointercapture',endDrag);
  viewport.addEventListener('wheel',event=>{
    event.preventDefault();
    const box=viewport.getBoundingClientRect();
    const delta=event.deltaY*(event.deltaMode===1?16:event.deltaMode===2?box.height:1);
    zoom(Math.exp(-Math.max(-240,Math.min(240,delta))*.002),event.clientX-box.left,event.clientY-box.top);
  },{passive:false});
  $('#zoom-in').onclick=()=>zoom(1.25);$('#zoom-out').onclick=()=>zoom(.8);$('#fit-map').onclick=fitMap;$('#locate-node').onclick=locateSelected;
  function expandTo(depth){
    const focused=nodes.get(focusId);
    for(const node of nodes.values()){
      if(!lineage(node).includes(focused))continue;
      if(node.children&&node.depth<depth)opened.add(node.id);else opened.delete(node.id);
    }
    opened.add(focusId);$('#map-level').value=String(depth);renderOutline();renderMap(true);
    announce(`Has been expanded to${taxonomy.levels[depth]}，${nodeCountLabel(focused)}`);
  }
  function focusBranch(id){
    const node=nodes.get(id);focusId=node.children?node.id:node.parent.id;
    expandTo(Math.min(5,nodes.get(focusId).depth+2));renderInspector();
  }
  $('#focus-branch').onclick=()=>focusBranch(selected);
  $('#map-level').onchange=event=>expandTo(Number(event.target.value));
  $('#map-scope').onclick=event=>{const button=event.target.closest('[data-focus]');if(button)focusBranch(button.dataset.focus);};
  $('#show-overview').onclick=()=>{focusId='chemworld';selected='chemworld';expandTo(2);renderInspector();renderDocument();updateHash();announce(`Display the three-layer architecture and all its components. ${domainCount} A specific area of ability`);};
  $('#collapse-tree').onclick=()=>{opened.clear();opened.add('chemworld');renderOutline();};
  $('#mobile-directory').onclick=()=>{const open=$('.sidebar').classList.toggle('mobile-open');$('#mobile-directory').setAttribute('aria-expanded',String(open));};
  $$('[data-view]').forEach(button=>button.onclick=event=>{event.preventDefault();switchView(button.dataset.view);});
  document.addEventListener('click',event=>{
    const guide=event.target.closest('[data-open-guide]');
    if(guide&&!event.ctrlKey&&!event.metaKey&&!event.shiftKey){event.preventDefault();const node=nodes.get(guide.dataset.openGuide),entry=guideReader.resolve(node);openGuide(entry.id,entry.anchor,node.id);}
    const selectedButton=event.target.closest('[data-select]');
    if(selectedButton){select(selectedButton.dataset.select);if(view==='atlas')locateSelected();else $('#document').scrollTop=0;}
    const atlas=event.target.closest('[data-go-atlas]');if(atlas){select(atlas.dataset.goAtlas);switchView('atlas');requestAnimationFrame(locateSelected);}
    const index=event.target.closest('[data-open-index]');if(index){select(index.dataset.openIndex);switchView('index');$('#document').scrollTop=0;}
    const validate=event.target.closest('[data-validate]');if(validate){select(validate.dataset.validate);activeTab='validation';renderDocument();switchView('index');}
    if(!event.target.closest('.search-wrap'))$('#search-results').hidden=true;
  });
  function search() {
    const query=$('#search').value.trim().toLocaleLowerCase();
    const container=$('#search-results');
    if(!query){container.hidden=true;searchMatches=[];guideMatches=[];return;}
    const words=query.split(/\s+/);
    searchMatches=[...nodes.values()].filter(node=>node.id!=='chemworld').filter(node=>words.every(word=>`${node.name} ${node.preview||''} ${node.ability||''} ${node.id} ${lineage(node).map(n=>n.name).join(' ')}`.toLocaleLowerCase().includes(word))).sort((a,b)=>Number(b.name.toLocaleLowerCase().includes(query))-Number(a.name.toLocaleLowerCase().includes(query)));
    guideMatches=guideReader.search(query);
    const capabilities=searchMatches.slice(0,25).map(node=>`<button type="button" class="search-result" data-search="${node.id}"><strong>${esc(node.name)} ${node.state?`· ${states[node.state]}`:''}</strong><small>${esc(kindLabel(node))} · ${esc(lineage(node).slice(1,-1).map(n=>n.name).join(' / '))}</small></button>`).join('');
    const guides=guideMatches.map(page=>`<button type="button" class="search-result" data-search-guide="${esc(page.id)}"><strong>${esc(page.title)}</strong><small>Domain Guide · Methods, Scope, and Examples</small></button>`).join('');
    container.innerHTML=capabilities||guides?`${capabilities?"<div class=\"search-group-label\">Abilities and methods</div>"+capabilities:''}${guides?"<div class=\"search-group-label\">Domain guides</div>"+guides:''}`:`<div class="search-empty">"Not found"${esc($('#search').value)}”<br>Try: battery, Henry, chromatography, thermometer</div>`;
    container.hidden=false;announce(`Found ${searchMatches.length} These abilities are related to ${guideMatches.length} This guide`);
  }
  function chooseSearch(id) {const fromGuide=view==='guides';select(id,{historyMode:fromGuide?null:'replace'});if(fromGuide)switchView('index');$('#search-results').hidden=true;$('#search').value='';if(view==='atlas')locateSelected();else{requestAnimationFrame(()=>$(`[data-tree="${id}"]`)?.scrollIntoView({block:'nearest'}));}}
  function chooseGuide(id){openGuide(id);$('#search-results').hidden=true;$('#search').value='';}
  $('#search').addEventListener('input',search);$('#search').addEventListener('focus',()=>{if($('#search').value)search();});
  $('#search-results').onclick=event=>{const button=event.target.closest('[data-search]');if(button)chooseSearch(button.dataset.search);const guide=event.target.closest('[data-search-guide]');if(guide)chooseGuide(guide.dataset.searchGuide);};
  $('#search').addEventListener('keydown',event=>{if(event.key==='Enter'){if(searchMatches.length){event.preventDefault();chooseSearch(searchMatches[0].id);}else if(guideMatches.length){event.preventDefault();chooseGuide(guideMatches[0].id);}}if(event.key==='ArrowDown'&&!$('#search-results').hidden){event.preventDefault();$('.search-result')?.focus();}});
  $('#search-results').addEventListener('keydown',event=>{const results=$$('.search-result');const i=results.indexOf(document.activeElement);if(['ArrowDown','ArrowUp'].includes(event.key)){event.preventDefault();const next=i+(event.key==='ArrowDown'?1:-1);if(next<0)$('#search').focus();else results[Math.min(next,results.length-1)]?.focus();}});
  document.addEventListener('keydown',event=>{if(window.CHEMWORLD_SITE)return;if(event.key==='/'&&!['INPUT','TEXTAREA'].includes(document.activeElement.tagName)){event.preventDefault();$('#search').focus();}if(event.key==='Escape'){$('#search-results').hidden=true;if(document.activeElement.closest('.search-results'))$('#search').focus();}});
  function readHash() {
    const parts=location.hash.slice(1).split('/').map(part=>{try{return decodeURIComponent(part);}catch{return '';}});
    if(!['index','atlas','guides'].includes(parts[0]))return;
    if(nodes.has(parts[1])){selected=parts[1];openAncestors(nodes.get(selected));}
    if(['index','atlas','guides'].includes(parts[0]))view=parts[0];
    if(view==='guides'){const fallback=guideReader.resolve(nodes.get(selected));guideId=parts[2]||fallback.id;guideAnchor=parts[3]||'';}
    renderOutline();renderDocument();renderInspector();switchView(view,{historyMode:'replace',contextual:false});
  }
  window.addEventListener('hashchange',readHash);
  $('.registry').innerHTML=[['modules',"modules"],['components',"component types"],['operations',"operations"],['measurements',"measurements"]].map(([key,label])=>`<span><strong>${registry[key]}</strong>${label}</span>`).join('');
  $('#directory-count').textContent=`${root.leaves.length} capabilities`;
  $('.nav-count').textContent=root.leaves.length;
  $('#atlas-summary').textContent=`3 architecture layers · ${domainCount} capability domains · ${familyCount} function groups · ${root.leaves.length} Capability`;
  let lastCompact=null;
  new ResizeObserver(()=>{
    if(view!=='atlas'||!mapReady||viewport.clientWidth===0)return;
    const compact=viewport.clientWidth<=600;
    if(lastCompact===null){lastCompact=compact;transformMap();}else if(compact!==lastCompact){lastCompact=compact;renderMap(true);}else transformMap();
    $('#fit-map').textContent=compact?"Fit width":"Fit canvas";
  }).observe(viewport);
  guideReader=window.createChemWorldGuideReader({nodes,lineage,onNavigate:openGuide,
    onCapability(id){select(id,{historyMode:null});switchView('index');$('#document').scrollTop=0;},
    onAtlas(id){select(id,{historyMode:null});switchView('atlas');requestAnimationFrame(locateSelected);},announce});
  window.addEventListener('chemworld-language-save',event=>{event.detail.atlas={opened:[...opened],camera,focusId,mapReady,activeTab};});
  const resume=window.CHEMWORLD_LANGUAGE_RESUME;
  if(resume?.hash===location.hash&&resume.atlas){const state=resume.atlas;opened.clear();state.opened.forEach(id=>opened.add(id));camera=state.camera;focusId=nodes.has(state.focusId)?state.focusId:'chemworld';mapReady=state.mapReady;activeTab=state.activeTab;}
  readHash();
})();
