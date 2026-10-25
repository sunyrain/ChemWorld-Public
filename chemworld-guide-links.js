/* Capability links resolve to the single documentation site. */
'use strict';
window.createChemWorldGuideLinks = function({lineage}) {
  const language=window.CHEMWORLD_LANGUAGE||'zh';
  const pages=Object.fromEntries(Object.entries(window.CHEMWORLD_DOC_INDEX).map(([id,editions])=>[id,editions[language]]));
  const sourcePages=Object.fromEntries(Object.entries(window.CHEMWORLD_DOC_INDEX).flatMap(([id,editions])=>Object.entries(editions).map(([lang,p])=>[id+lang,p])));
  const entry=(id,anchor='',label='')=>({id,anchor,label:label||pages[id]?.title||id});
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
  const sourceToId=new Map(Object.values(sourcePages).map(p=>[p.source,p.id]));
  function resolve(node){
    const ancestors=lineage(node);
    const override=[...ancestors].reverse().find(n=>specific[n.id]);if(override)return specific[override.id];
    const domain=ancestors.find(n=>n.kind==='domain');if(domain&&domains[domain.id])return domains[domain.id][0];
    const source=[...ancestors].reverse().find(n=>sourceToId.has(n.source));
    return source?entry(sourceToId.get(source.source)):entry('world-foundations');
  }

  const route=(id,anchor='')=>window.chemworldDocsUrl(id,anchor);
  return {resolve,route,pages,
    search(query){return Object.values(pages).filter(p=>`${p.title} ${p.id}`.toLowerCase().includes(query)).slice(0,8).map(p=>({id:p.id,title:p.title}));},
    references:[...Object.values(domains).flat(),...Object.values(specific)]
  };
};
