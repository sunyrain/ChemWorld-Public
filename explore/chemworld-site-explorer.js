/* Local design preview. The atlas and index share one editorial capability snapshot. */
(() => {
  'use strict';
  const {tree, registry, snapshot, taxonomy} = window.CHEMWORLD_CAPABILITIES;
  const $ = (selector, scope = document) => scope.querySelector(selector);
  const $$ = (selector, scope = document) => [...scope.querySelectorAll(selector)];
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const states = {done:'已交付', finite:'有限模型', partial:'开发中', planned:'规划'};
  const nodes = new Map();
  const root = {id:'chemworld',kind:'project',name:'ChemWorld',preview:'可回放的因果物理化学实验环境',children:tree};
  const kindLabels={project:'项目',layer:'架构层',domain:'能力领域',family:'功能组',capability:'具体能力',method:'方法 / 变体'};
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
  const nodeCountLabel=node=>node.kind==='capability'?`${node.methods.length} 个方法 / 变体`:`${node.leaves.length} 项具体能力`;
  const kindLabel=node=>`${kindLabels[node.kind]}${node.type?' · '+node.type:''}`;
  const special = {
    combustion: {
      intro:'从明确进料与反应规格，得到产物组成和有限热量。',
      rows:[['元素计量','完全燃烧产物与原子守恒'],['热量计算','HHV / LHV 与明确相参考'],['流量规格','两流 / 三流、湿基 / 干基尾气'],['有限消耗','进料量 → 产物量与热量']],
      flow:['进料与规格','计量与热量平衡','产物与热量'],
      boundary:'完全燃烧与显式参考态；火焰动力学、产物温度和通用平衡另行扩展。',
      related:['assessment','conservation','caloric'],
      validation:'覆盖元素与产物守恒、相参考、HHV / LHV 一致性、有限物料热量、两流与三流规格，以及冲突和缺失输入的拒绝。',
      test:'tests/test_combustion.py'
    },
    temperature: {
      intro:'将历史温度读数校准到明确温标，继续供物性、输入和观测使用。',
      rows:[['温标覆盖','ITS-90 / ITS-68 / ITS-48 / ITS-27 / ITS-76'],['变换与反求','各自有效范围内的正向、逆向与温标间转换'],['原生消费','Quantity、InputValue 与观测通道'],['来源约定','文献约定和参考版本差值行为显式区分']],
      flow:['历史读数与温标','有界校准与反求','物性输入或派生观测'],
      related:['solubility','optics','conservation'],
      validation:'检查节点、区间、正反方向、真实端点和跨功能消费；来源差值约定分别保留。',
      test:'tests/test_temperature_scales.py'
    }
  };
  function lineage(node) { const out=[]; for(let n=node;n;n=n.parent)out.unshift(n); return out; }
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
    updateHash(historyMode); announce(`已选择 ${nodes.get(id).name}`);
  }
  function openGuide(id,anchor='',nodeId=selected,{historyMode='push'}={}) {
    select(nodes.has(nodeId)?nodeId:selected,{historyMode:null});
    guideId=id;guideAnchor=anchor;
    switchView('guides',{historyMode,contextual:false});
    announce(`正在阅读 ${guideReader.pages[id]?.title||'领域指南'}`);
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
    const guide=guideReader.resolve(node);
    return `<div class="doc-links"><a href="${guideReader.route(guide.id,guide.anchor,node.id)}" data-open-guide="${node.id}">阅读领域指南 →</a><button type="button" class="text-button" data-go-atlas="${node.id}">在总览图谱中定位 →</button></div>`;
  }
  function table(rows) { return `<table class="capability-table"><tbody>${rows.map(([key,value])=>`<tr><th scope="row">${esc(key)}</th><td>${esc(value)}</td></tr>`).join('')}</tbody></table>`; }
  function overviewContent(node) {
    if(!node.state)return `<h2>${kindLabels[node.children[0]?.kind]||'下级内容'}</h2>${childList(node)}<h2>目录层级</h2><p>${taxonomy.levels.slice(1).join(' → ')}</p><p>同级节点使用相同分类层次；${taxonomy.counting}</p>`;
    const detail=special[node.id];
    return `<h2>可用能力</h2>${detail?.rows?table(detail.rows):`<p>${esc(node.ability)}</p>`}
      ${detail?.flow?`<h2>代表路径</h2><div class="flow-path">${detail.flow.map(esc).join('<b aria-hidden="true">→</b>')}</div>`:node.example?`<h2>代表实验</h2><p>${esc(node.example)}</p>`:''}
      <h2>适用边界</h2><p>${esc(detail?.boundary||node.boundary)}</p>
      ${node.children?`<h2>方法 / 变体</h2>${childList(node)}`:''}
      <div class="related"><h2>关联能力</h2>${related(node).map(id=>`<button type="button" data-select="${id}">${esc(nodes.get(id).name)}</button>`).join('')}</div>`;
  }
  function childList(node){return `<div class="child-list">${(node.children||[]).map(child=>`<button type="button" data-select="${child.id}"><span>${esc(child.name)}</span>${child.state?status(child):`<small>${child.leaves.length} 项能力 →</small>`}</button>`).join('')}</div>`;}
  function tabContent(node) {
    const detail=special[node.id];
    const api=window.ChemWorldApiDocs.render(node,'zh',activeTab);
    if(activeTab==='overview')return api===null?overviewContent(node):`<p class="scope-statement">${esc(detail?.boundary||node.boundary)}</p>${api}${node.children?childList(node):''}`;
    if(activeTab==='io'||activeTab==='examples')return api===null?overviewContent(node):api;
    return `${api||''}<h2>当前状态</h2><p>${status(node,true)||'分支包含多种交付状态。'}</p><h2>验证范围</h2><p>${esc(detail?.validation||node.boundary||'交付范围按具体能力分别记录，不从分类分支推断整域已完成。')}</p><p class="scope-statement">${node.state==='planned'?'规划项代表扩展方向，尚未作为已交付能力。':'已交付指明示范围内的开发验收；有限模型保留具体模型、参数与物理适用域。'}</p>`;
  }
  function renderDocument() {
    const node=nodes.get(selected), detail=special[node.id];
    const tabs=[['overview','使用文档'],['io','接口与参数'],['examples','示例与输出'],['validation','验证范围']];
    $('#document').innerHTML=`<div class="doc-inner"><div class="breadcrumbs">${lineage(node).slice(1,-1).map(n=>`<button type="button" data-select="${n.id}">${esc(n.name)}</button>`).join('<span>/</span>')||'<span>架构总览</span>'}</div>
      <div class="node-kind">${esc(kindLabel(node))}</div><div class="title-row"><h1>${esc(node.name)}</h1>${status(node,true)}</div><p class="doc-intro">${esc(detail?.intro||node.preview||node.ability)}</p>
      <div class="doc-tabs" role="tablist" aria-label="能力详情章节">${tabs.map(([id,label])=>`<button type="button" id="tab-${id}" role="tab" data-tab="${id}" aria-selected="${activeTab===id}" aria-controls="detail-panel" tabindex="${activeTab===id?0:-1}">${label}</button>`).join('')}</div>
      <div id="detail-panel" class="doc-content" role="tabpanel" aria-labelledby="tab-${activeTab}">${tabContent(node)}</div>
      ${documentationLinks(node)}<div class="source-note">当前开发基座 · ${snapshot} 能力快照<br>注册数量描述软件声明；具体世界的安装、任务许可与当前可执行能力分别判断。</div></div>`;
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
      content=`<div class="inspector-section"><h3>${nodeCountLabel(node)}</h3><div class="distribution" aria-hidden="true">${Object.keys(states).map(s=>c[s]?`<span class="${s}" style="flex:${c[s]}"></span>`:'').join('')}</div><div class="distribution-labels">${Object.keys(states).map(s=>`<span class="status ${s}"><i></i>${states[s]} ${c[s]}</span>`).join('')}</div><p class="count-note">方法 / 变体不重复计数</p></div><div class="inspector-section"><h3>${kindLabels[node.children[0]?.kind]||'分支'}</h3><div class="map-branch-list">${node.children.map(child=>`<button type="button" data-select="${child.id}">${esc(child.name)} ${!child.state?`· ${child.leaves.length}`:''} →</button>`).join('')}</div></div>`;
    }else content=`<div class="inspector-section"><h3>能力</h3><p>${esc(node.ability)}</p></div>${node.example?`<div class="inspector-section"><h3>代表实验</h3><p>${esc(node.example)}</p></div>`:''}<div class="inspector-section"><h3>适用边界</h3><p>${esc(detail?.boundary||node.boundary)}</p></div>`;
    $('#inspector').innerHTML=`<div class="inspect-top"><div class="inspector-eyebrow">${esc(kindLabel(node))}</div><h2>${esc(node.name)}</h2>${status(node,true)}${node.preview?`<p class="group-preview">${esc(node.preview)}</p>`:''}</div>${content}${node.state&&node.children?`<div class="inspector-section"><h3>方法 / 变体</h3><div class="map-branch-list">${node.children.map(child=>`<button type="button" data-select="${child.id}">${esc(child.name)} →</button>`).join('')}</div></div>`:''}<button type="button" class="inspect-cta" data-open-index="${node.id}">在能力索引中阅读 →</button><button type="button" class="inspect-secondary" data-validate="${node.id}">查看验证范围</button>`;
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
    $('#map-scope').innerHTML=`<span>范围</span>${lineage(focused).map(n=>`<button type="button" data-focus="${n.id}">${esc(n.name)}</button>`).join('<span>/</span>')}<small>${kindLabels[focused.kind]}起 · ${nodeCountLabel(focused)}</small>`;
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
      return `<button type="button" class="map-node ${node.depth===0?'root':node.depth===1?'layer':''} ${representative?.id===node.id?'selected':''}" style="left:${p.x}px;top:${p.y-17}px;width:${p.width}px" data-map="${node.id}" data-kind="${node.kind}" ${branch?`aria-expanded="${opened.has(node.id)}"`:''} ${selected===node.id?'aria-current="true"':''} aria-label="${esc(node.name)}，${esc(kindLabel(node))}${branch?`，${nodeCountLabel(node)}，${opened.has(node.id)?'收起':'展开'}`:`，${states[node.state]}`}" title="${esc(kindLabel(node))}"><span aria-hidden="true" class="${branch?'expand-box':'leaf-mark'}">${branch?(opened.has(node.id)?'−':'+'):''}</span><span class="node-title">${esc(node.name)}</span>${node.state?status(node):`<span class="node-count">${node.leaves.length}</span>`}</button>`;
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
    announce(`已展开到${taxonomy.levels[depth]}，${nodeCountLabel(focused)}`);
  }
  function focusBranch(id){
    const node=nodes.get(id);focusId=node.children?node.id:node.parent.id;
    expandTo(Math.min(5,nodes.get(focusId).depth+2));renderInspector();
  }
  $('#focus-branch').onclick=()=>focusBranch(selected);
  $('#map-level').onchange=event=>expandTo(Number(event.target.value));
  $('#map-scope').onclick=event=>{const button=event.target.closest('[data-focus]');if(button)focusBranch(button.dataset.focus);};
  $('#show-overview').onclick=()=>{focusId='chemworld';selected='chemworld';expandTo(2);renderInspector();renderDocument();updateHash();announce(`显示三层架构及全部 ${domainCount} 个能力领域`);};
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
    const guides=guideMatches.map(page=>`<button type="button" class="search-result" data-search-guide="${esc(page.id)}"><strong>${esc(page.title)}</strong><small>领域指南 · 方法、范围与示例</small></button>`).join('');
    container.innerHTML=capabilities||guides?`${capabilities?'<div class="search-group-label">能力与方法</div>'+capabilities:''}${guides?'<div class="search-group-label">领域指南</div>'+guides:''}`:`<div class="search-empty">未找到“${esc($('#search').value)}”<br>试试：电池、Henry、色谱、温标</div>`;
    container.hidden=false;announce(`找到 ${searchMatches.length} 项能力与 ${guideMatches.length} 篇指南`);
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
  $('.registry').innerHTML=[['modules','模块'],['components','类部件'],['operations','操作'],['measurements','测量接口']].map(([key,label])=>`<span><strong>${registry[key]}</strong>${label}</span>`).join('');
  $('#directory-count').textContent=`${root.leaves.length} 项具体能力`;
  $('.nav-count').textContent=root.leaves.length;
  $('#atlas-summary').textContent=`3 架构层 · ${domainCount} 能力领域 · ${familyCount} 功能组 · ${root.leaves.length} 具体能力`;
  let lastCompact=null;
  new ResizeObserver(()=>{
    if(view!=='atlas'||!mapReady||viewport.clientWidth===0)return;
    const compact=viewport.clientWidth<=600;
    if(lastCompact===null){lastCompact=compact;transformMap();}else if(compact!==lastCompact){lastCompact=compact;renderMap(true);}else transformMap();
    $('#fit-map').textContent=compact?'适应宽度':'适应画布';
  }).observe(viewport);
  guideReader=window.createChemWorldGuideReader({nodes,lineage,onNavigate:openGuide,
    onCapability(id){select(id,{historyMode:null});switchView('index');$('#document').scrollTop=0;},
    onAtlas(id){select(id,{historyMode:null});switchView('atlas');requestAnimationFrame(locateSelected);},announce});
  window.addEventListener('chemworld-language-save',event=>{event.detail.atlas={opened:[...opened],camera,focusId,mapReady,activeTab};});
  const resume=window.CHEMWORLD_LANGUAGE_RESUME;
  if(resume?.hash===location.hash&&resume.atlas){const state=resume.atlas;opened.clear();state.opened.forEach(id=>opened.add(id));camera=state.camera;focusId=nodes.has(state.focusId)?state.focusId:'chemworld';mapReady=state.mapReady;activeTab=state.activeTab;}
  readHash();
})();
