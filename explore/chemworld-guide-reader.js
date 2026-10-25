/* Reader chrome over the repository-rendered Markdown; no duplicate guide prose. */
'use strict';
window.createChemWorldGuideReader = function({nodes,lineage,onNavigate,onCapability,onAtlas,announce}) {
  const sourcePages=window.CHEMWORLD_GUIDES.pages,language=window.CHEMWORLD_LANGUAGE||'zh';
  const pages=Object.fromEntries(Object.entries(sourcePages).map(([id,page])=>{
    const canonical=id.replace(/\.en$/,''),localized=window.CHEMWORLD_GUIDE_LOCALES?.[canonical]?.[language];
    return [id,{...(localized||page),id,alias:id!==canonical}];
  }));
  const $=(s,r=document)=>r.querySelector(s);
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const titles={native_mechanisms:'原生气相反应机制',native_foundation:'原生实验与领域模型',native_domains:'跨领域原生实验',native_capabilities:'已安装能力与声明',native_shared_lab:'共享实验室与设备调度',native_stochastic:'整数随机反应网络',native_biochemical:'生化网络与 SBML','world-foundations':'共同物理化学基础',operations:'实验操作与控制','native_foundation':'仪器与测量合同','agents':'Agent 与程序接口','student-lab':'交互实验与观察工作台','quickstart':'研究任务目录','agents':'离线研究流程','limitations':'真实世界接入与迁移','world-foundations':'ChemWorld 系统架构'};
  const entry=(id,anchor='',label='')=>({id,anchor,label:label||titles[id]||pages[id]?.title||id});
  const domains={
    foundation:[entry('world-foundations'),entry('native_foundation','parameter-and-measurement-inputs','科学输入与有限标定')],
    properties:[entry('native_foundation','properties-at-the-current-state','当前状态下的物性'),entry('native_capabilities')],
    'fuel-assessment':[entry('native_foundation','properties-at-the-current-state','物性与热量接口（共同指南）')],
    reaction:[entry('native_mechanisms'),entry('native_foundation','two-site-surface-interactions','多位点表面催化')],
    separation:[entry('native_foundation','tpph-flash-and-finite-phase-inventories','相平衡与分离单元'),entry('native_domains')],
    'process-systems':[entry('native_foundation','finite-pump-pipe-and-heat-exchanger-transport','泵、管与换热'),entry('native_shared_lab')],
    materials:[entry('native_foundation','native-single-particle-lithium-battery','电池与电化学模型'),entry('native_foundation','selected-native-tdbcef-material','固相材料与 CALPHAD'),entry('native_domains')],
    water:[entry('native_foundation','finite-carbonate-water-chemistry','水相与矿物'),entry('native_foundation','finite-dilute-reverse-osmosis','膜分离与污染')],
    bio:[entry('native_biochemical'),entry('native_stochastic')],
    actions:[entry('operations'),entry('native_foundation')],
    instruments:[entry('native_foundation'),entry('native_foundation','programmed-chromatography-and-persistent-instrument-state','程序色谱与持久状态')],
    lifecycle:[entry('native_shared_lab'),entry('native_foundation','readings-time-and-reuse','时间与设备复用')],
    records:[entry('native_foundation'),entry('native_shared_lab')],
    access:[entry('environment'),entry('environment-authoring'),entry('agents'),entry('legacy'),entry('native_cpu')],
    'research-tasks':[entry('quickstart'),entry('native_domains')],
    protocol:[entry('agents'),entry('agents')],
    'learning-resources':[entry('quickstart'),entry('environment-authoring')]
  };
  const specific={
    'thermal-reaction':entry('native_mechanisms'),surface:entry('native_foundation','two-site-surface-interactions'),
    temperature:entry('native_foundation','historical-temperature-calibration'),calibration:entry('native_foundation','finite-native-calibration'),
    flash:entry('native_foundation','tpph-flash-and-finite-phase-inventories'),extraction:entry('native_foundation','strict-liquid-stability-and-finite-washing'),
    distillation:entry('native_foundation','finite-dynamic-distillation'),crystal:entry('native_foundation','finite-crystal-growth-and-dissolution'),
    adsorption:entry('native_foundation','persistent-finite-sorbent-contact'),transport:entry('native_foundation','finite-pump-pipe-and-heat-exchanger-transport'),
    electrode:entry('native_foundation','finite-transient-working-electrode'),spm:entry('native_foundation','native-single-particle-lithium-battery'),
    calphad:entry('native_foundation','selected-native-tdbcef-material'),aqueous:entry('native_foundation','finite-carbonate-water-chemistry'),
    mineral:entry('native_foundation','finite-calcite-kinetics-and-material-surface-ownership'),membrane:entry('native_foundation','finite-dilute-reverse-osmosis'),
    ssa:entry('native_stochastic'),programmed:entry('native_foundation','programmed-chromatography-and-persistent-instrument-state'),
    bridge:entry('limitations'),observatory:entry('environment-authoring')
  };
  const sourceToId=new Map(Object.values(sourcePages).map(p=>[p.source,p.id.replace(/\.en$/,'')]));
  const flatten=tokens=>tokens.flatMap(t=>[t,...flatten(t.children||[])]);
  const route=(id,anchor,nodeId)=>`#guides/${encodeURIComponent(nodeId)}/${encodeURIComponent(id)}${anchor?'/'+encodeURIComponent(anchor):''}`;
  let current=null,observer=null;
  function resolve(node){
    const ancestors=lineage(node);
    const override=[...ancestors].reverse().find(n=>specific[n.id]);if(override)return specific[override.id];
    const domain=ancestors.find(n=>n.kind==='domain');if(domain&&domains[domain.id])return domains[domain.id][0];
    const source=[...ancestors].reverse().find(n=>sourceToId.has(n.source));
    return source?entry(sourceToId.get(source.source)):entry('world-foundations');
  }
  function href(item,nodeId){return route(item.id,item.anchor,nodeId);}
  function renderNav(node,id,anchor){
    const activeDomain=lineage(node).find(n=>n.kind==='domain');
    $('#guide-navigation').innerHTML=[...nodes.values()].filter(n=>n.kind==='layer').map(layer=>`<section class="guide-nav-group"><h2>${esc(layer.name)}</h2>${layer.children.map(domain=>{
      const items=domains[domain.id]||[resolve(domain)],active=activeDomain?.id===domain.id;
      return `<div class="guide-domain ${active?'active':''}"><a class="guide-domain-link" href="${href(items[0],domain.id)}" data-guide-id="${items[0].id}" data-guide-anchor="${items[0].anchor}" data-guide-node="${domain.id}" ${active?'aria-current="true"':''}>${esc(domain.name)}</a>${active?`<div class="guide-subnav">${items.map(item=>`<a href="${href(item,node.id)}" data-guide-id="${item.id}" data-guide-anchor="${item.anchor}" data-guide-node="${node.id}" class="${id===item.id&&(!anchor||anchor===item.anchor)?'current':''}">${esc(item.label)}</a>`).join('')}</div>`:''}</div>`;
    }).join('')}</section>`).join('');
  }
  function installLinks(page,node){
    const base=new URL('https://github.com/sunyrain/ChemWorld-Public/blob/main/'+page.source);
    for(const link of $('#guide-body').querySelectorAll('a[href]')){
      const raw=link.getAttribute('href');
      if(!raw||/^javascript:/i.test(raw)){link.removeAttribute('href');continue;}
      const target=new URL(raw,base);
      if(raw.startsWith('#')){
        const anchor=decodeURIComponent(raw.slice(1));link.href=route(page.id,anchor,node.id);link.dataset.guideId=page.id;link.dataset.guideAnchor=anchor;link.dataset.guideNode=node.id;
      }else if(target.origin===base.origin&&target.pathname.endsWith('.md')){
        const repoRoot=new URL('https://github.com/sunyrain/ChemWorld-Public/blob/main/').pathname;
        const source=decodeURIComponent(target.pathname.slice(repoRoot.length));const id=sourceToId.get(source);
        if(id){link.href=route(id,decodeURIComponent(target.hash.slice(1)),node.id);link.dataset.guideId=id;link.dataset.guideAnchor=decodeURIComponent(target.hash.slice(1));link.dataset.guideNode=node.id;}
        else{link.href=target.href;link.setAttribute('download',source.split('/').at(-1));link.title='下载参考文档';}
      }else{link.href=target.href;link.target='_blank';link.rel='noopener';}
    }
    for(const img of $('#guide-body').querySelectorAll('img[src]'))img.src=new URL(img.getAttribute('src'),base).href;
  }
  function decorateCode(){
    for(const pre of $('#guide-body').querySelectorAll('pre')){
      const container=pre.closest('.codehilite')||pre;
      if(container.parentElement.classList.contains('guide-code-block'))continue;
      const wrap=document.createElement('div');wrap.className='guide-code-block';container.before(wrap);
      const bar=document.createElement('div');bar.className='guide-code-toolbar';
      const label=document.createElement('span');label.textContent=container.dataset.lang||'代码示例';
      const button=document.createElement('button');button.type='button';button.className='guide-copy';button.textContent='复制';button.setAttribute('aria-label','复制代码');
      button.addEventListener('click',async()=>{
        try{await navigator.clipboard.writeText(pre.textContent);button.textContent='已复制';announce('代码已复制');}
        catch{const selection=getSelection(),range=document.createRange();range.selectNodeContents(pre);selection.removeAllRanges();selection.addRange(range);button.textContent='已选中，请复制';announce('代码已选中，可使用系统复制快捷键');}
        setTimeout(()=>{if(button.isConnected)button.textContent='复制';},1800);
      });
      bar.append(label,button);wrap.append(bar,container);
    }
    for(const table of $('#guide-body').querySelectorAll('table')){const wrap=document.createElement('div');wrap.className='guide-table-wrap';table.before(wrap);wrap.append(table);}
    for(const set of $('#guide-body').querySelectorAll('.tabbed-set')){
      const labels=set.querySelector(':scope > .tabbed-labels'),panels=[...set.querySelectorAll(':scope > .tabbed-content > .tabbed-block')];
      if(!labels||!panels.length)continue;
      labels.setAttribute('role','tablist');labels.setAttribute('aria-label','示例视图');
      const tabs=[...labels.children].map((label,i)=>{
        const button=document.createElement('button');button.type='button';button.textContent=label.textContent;button.id=label.htmlFor+'-tab';button.setAttribute('role','tab');
        panels[i].id=label.htmlFor+'-panel';panels[i].setAttribute('role','tabpanel');panels[i].setAttribute('aria-labelledby',button.id);button.setAttribute('aria-controls',panels[i].id);label.replaceWith(button);return button;
      });
      set.querySelectorAll(':scope > input').forEach(input=>input.remove());
      const activate=i=>tabs.forEach((tab,j)=>{tab.setAttribute('aria-selected',String(i===j));tab.tabIndex=i===j?0:-1;panels[j].hidden=i!==j;});
      tabs.forEach((button,i)=>{button.onclick=()=>activate(i);button.onkeydown=event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;event.preventDefault();const next=event.key==='Home'?0:event.key==='End'?tabs.length-1:(i+(event.key==='ArrowRight'?1:tabs.length-1))%tabs.length;activate(next);tabs[next].focus();};});activate(0);
    }
  }
  function scrollToAnchor(anchor){
    const reader=$('#guide-reader'),target=anchor?reader.querySelector(`#${CSS.escape(anchor)}`):null;
    if(target&&reader.contains(target)){
      if(getComputedStyle(reader).overflowY==='auto')reader.scrollTop+=target.getBoundingClientRect().top-reader.getBoundingClientRect().top-24;
      else target.scrollIntoView({block:'start'});
    }else if(!anchor){reader.scrollTop=0;if(getComputedStyle(reader).overflowY!=='auto')reader.scrollIntoView({block:'start'});}
    for(const a of $('#guide-toc').querySelectorAll('a'))a.classList.toggle('active',a.dataset.guideAnchor===anchor);
  }
  function watchSections(){
    observer?.disconnect();const reader=$('#guide-reader');
    observer=new IntersectionObserver(entries=>{
      const first=entries.filter(e=>e.isIntersecting).sort((a,b)=>a.boundingClientRect.top-b.boundingClientRect.top)[0];if(!first)return;
      for(const a of $('#guide-toc').querySelectorAll('a'))a.classList.toggle('active',a.dataset.guideAnchor===first.target.id);
    },{root:getComputedStyle(reader).overflowY==='auto'?reader:null,rootMargin:'-10px 0px -65% 0px'});
    reader.querySelectorAll('#guide-body h2,#guide-body h3').forEach(h=>observer.observe(h));
  }
  function render(id,anchor,node){
    const page=pages[id];if(!page){current=null;observer?.disconnect();$('#guide-reader').innerHTML='<div class="guide-empty" role="alert">未找到这篇指南。请从左侧选择领域。</div>';$('#guide-toc').innerHTML='';$('#guide-related').innerHTML='';renderNav(node,id,anchor);return;}
    renderNav(node,id,anchor);
    if(current?.id===id&&current.node===node.id){requestAnimationFrame(()=>{scrollToAnchor(anchor);watchSections();});return;}
    current={id,node:node.id};
    const chain=lineage(node).filter(n=>['layer','domain'].includes(n.kind));
    const title=titles[id.replace(/\.en$/,'')]||page.title;const language=page.translation==='local-preview'?(page.language==='en'?'英文预览译文':'中文预览译文'):(page.language==='en'?'英文原文':'中文原文');
    $('#guide-reader').innerHTML=`<div class="guide-reading-width"><div class="guide-breadcrumbs">${chain.map(n=>esc(n.name)).join('<span>/</span>')}<span>/</span>领域指南</div><div class="guide-meta">${page.reference?'参考与验收文档':'领域指南'}<span>·</span>${language}</div><h1>${esc(title)}</h1>${title!==page.title&&page.language==='en'?`<p class="guide-original-title" lang="en">${esc(page.title)}</p>`:''}<div class="guide-reading-actions"><button type="button" data-guide-back="${node.id}">← 返回${node.kind==='capability'||node.kind==='method'?'当前能力':'能力索引'}</button><button type="button" data-guide-atlas="${node.id}">在图谱中定位</button><button type="button" id="guide-source-toggle" aria-expanded="false" aria-controls="guide-source">查看原文</button></div><div id="guide-source" hidden><p>文档来源：<code>${esc(page.source)}</code></p><pre>${esc(page.markdown)}</pre></div><div id="guide-body" class="guide-prose" lang="${page.language}">${page.html}</div><div class="guide-end"><span>已读至本篇末尾</span><button type="button" data-guide-back="${node.id}">返回能力索引 →</button></div></div>`;
    const first=$('#guide-body h1');if(first){$('#guide-reader h1').id=first.id;first.remove();}
    if(page.translation==='local-preview'){const note=document.createElement('p');note.className='guide-translation-note';note.textContent='预览译文由本地模型生成；代码、公式与接口名保留原文。';$('.guide-reading-actions').before(note);}
    const headings=flatten(page.toc).filter(t=>t.level>=2&&t.level<=3);
    $('#guide-toc').innerHTML=headings.length?headings.map(h=>`<a href="${route(id,h.id,node.id)}" data-guide-id="${id}" data-guide-anchor="${esc(h.id)}" data-guide-node="${node.id}" class="toc-depth-${h.level}">${esc(h.name)}</a>`).join(''):'<span class="guide-no-sections">本篇为单页说明</span>';
    if(headings.length){const toc=document.createElement('details');toc.className='guide-inline-toc';toc.innerHTML=`<summary>本页目录 · ${headings.filter(h=>h.level===2).length} 个章节</summary><nav aria-label="本页章节">${$('#guide-toc').innerHTML}</nav>`;$('#guide-body').before(toc);}
    const group=node.kind==='capability'||node.kind==='method'?node.parent:node;
    let related=(group.children||[]).filter(n=>n.id!==node.id).slice(0,5);
    if(!related.length)related=(lineage(node).find(n=>n.kind==='domain')?.leaves||[]).filter(n=>n.id!==node.id).slice(0,4);
    $('#guide-related').innerHTML=`<h2>关联能力</h2>${related.map(n=>`<button type="button" data-guide-back="${n.id}">${esc(n.name)} →</button>`).join('')}<p>指南解释方法与范围；具体安装和执行条件见能力索引。</p>`;
    installLinks(page,node);decorateCode();
    requestAnimationFrame(()=>{scrollToAnchor(anchor);watchSections();});
  }
  $('#guides-view').addEventListener('click',event=>{
    const link=event.target.closest('[data-guide-id]');
    if(link&&!event.ctrlKey&&!event.metaKey&&!event.shiftKey){event.preventDefault();onNavigate(link.dataset.guideId,link.dataset.guideAnchor||'',link.dataset.guideNode);$('.guide-sidebar').classList.remove('mobile-open');$('#guide-mobile-nav').setAttribute('aria-expanded','false');}
    const back=event.target.closest('[data-guide-back]');if(back)onCapability(back.dataset.guideBack);
    const atlas=event.target.closest('[data-guide-atlas]');if(atlas)onAtlas(atlas.dataset.guideAtlas);
    if(event.target.closest('#guide-source-toggle')){const source=$('#guide-source');source.hidden=!source.hidden;$('#guide-source-toggle').setAttribute('aria-expanded',String(!source.hidden));}
  });
  $('#guide-mobile-nav').onclick=()=>{const open=$('.guide-sidebar').classList.toggle('mobile-open');$('#guide-mobile-nav').setAttribute('aria-expanded',String(open));};
  return {resolve,render,route,pages,sourceToId,
    search(query){return Object.values(pages).filter(p=>!p.reference&&!p.alias&&`${titles[p.id]||''} ${p.title} ${flatten(p.toc).map(t=>t.name).join(' ')}`.toLowerCase().includes(query)).slice(0,8).map(p=>({id:p.id,title:titles[p.id]||p.title}));},
    references:[...Object.values(domains).flat(),...Object.values(specific)]
  };
};
