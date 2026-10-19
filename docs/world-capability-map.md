# ChemWorld capability map

This is the frozen release map. See [current foundation](current-foundation.md)
for the October update, including vessels, control, streams and configurable spectra.

This table is the reader-facing summary of the published surface. It describes what the
world substrate and instrument contract expose; it does not claim that every row has been
qualified in every possible combination.

## 1. Reusable components and interfaces

| Component | Public role | Main operations | Interfaces carried into composition |
| --- | --- | --- | --- |
| Reaction | Material transformation, selectivity and degradation | add reagent, add solvent, add catalyst, heat, wait, sample, quench | material, temperature, time, reaction event |
| Thermal | Heating, cooling, residence time and energy bookkeeping | heat, wait, cool crystallize, evaporate, distill, run flow | temperature, time, energy, safety |
| Phase | Phase creation, settling and phase identity | add phase, mix, settle, separate phase | phase, volume, composition, state identity |
| Separation | Extraction, washing, drying, concentration and transfer | add extractant, wash, dry, concentrate, transfer | material, phase, volume, sample identity |
| Crystallization | Seeding, cooling, crystal growth and filtration | seed crystals, cool crystallize, filter crystals | temperature, phase, crystal state, sample |
| Distillation | Evaporation, fractionation and fraction collection | evaporate, distill, collect fraction | temperature, phase, volatility, fraction |
| Continuous flow | Flow configuration and residence-time execution | set flow rate, run flow | flow rate, residence time, temperature, material |
| Electrochemistry | Potential/current control and electrochemical conversion | set potential, electrolyze | potential, current, charge, energy, material |
| Observation | Public measurements, masks, noise and terminal assay | measure | observable values, missingness, cost, latency, sample |

The common composition paths currently exposed by the vocabulary are:

| Composition pattern | Required interface chain | Typical use |
| --- | --- | --- |
| Reaction + thermal + observation | material → temperature/time → measurement | batch reaction and characterization |
| Reaction + phase + separation + observation | material → phase/volume → separation → measurement | reaction followed by extraction or purification |
| Reaction + thermal + crystallization + observation | material → temperature/time → crystal state → measurement | reaction to crystallization |
| Reaction + thermal + distillation + observation | material → temperature/phase → fraction → measurement | reaction to distillation |
| Reaction + thermal + continuous flow + observation | material → flow/residence time → temperature → measurement | continuous-flow reaction |
| Reaction + electrochemistry + observation | material → potential/current/charge → measurement | electrochemical conversion |
| Phase + observation | phase/composition → measurement | equilibrium or partition characterization |

These are composition patterns, not a closed list of tasks. A new pattern is legal only when
its component dependencies, units, state ownership, resource debits and lifecycle path are
declared and accepted by the compatibility checker.

## 2. Public operation surface

The current public operation language contains 28 typed operation kinds, grouped as follows.

| Group | Operations |
| --- | --- |
| Material and reaction setup | add reagent, add solvent, add catalyst |
| Batch control | heat, wait, sample, quench |
| Phase and separation | add phase, add extractant, mix, settle, separate phase, wash, dry, concentrate, transfer |
| Crystallization | seed crystals, cool crystallize, filter crystals |
| Distillation | evaporate, distill, collect fraction |
| Continuous flow | set flow rate, run flow |
| Electrochemistry | set potential, electrolyze |
| Lifecycle and observation | terminate, measure |

Every operation declares typed fields and preconditions. The public runtime rejects invalid
fields, missing prerequisites, out-of-range values, incompatible units, exhausted resources
and post-termination actions before they can mutate the committed physical state.

## 3. Continuous and categorical parameter axes

The following axes are available to authoring and task contracts. Bounds may be narrowed by a
particular world, task, vessel or resource card; values are never silently remapped.

| Axis | Unit or type | Current public domain |
| --- | --- | --- |
| Reagent amount | mol | non-negative and limited by the active safety/volume envelope |
| Catalyst amount | mol | 0–0.005 mol per operation contract |
| Added, phase, extractant, wash and sample volume | L | positive values bounded by operation, vessel and remaining sample volume |
| Target temperature | K | operation-specific; current public families span approximately 250–430 K |
| Duration and residence time | s | duration operations 1–14,400 s; flow residence time 1–7,200 s |
| Stirring speed | rpm | 100–1,200 rpm where the operation exposes stirring |
| Transfer fraction | fraction | 0.0001–1.0 |
| Seed mass | g | 0.000001–0.050 g, with a cumulative cap |
| Reflux ratio | ratio | 0–10 |
| Flow rate | mL min⁻¹ | 0.01–20.0 |
| Potential | V | −3.0–3.0 V, with electrochemical coupling checks |
| Current | mA | 0.001–500 mA |
| Solvent, phase, extractant, instrument and electrolyte profile | categorical | finite declared choices; unknown choices fail closed |
| Operation, instrument and termination selection | categorical | finite contract-declared choices; no implicit fallback |

Continuous axes are suitable for space-filling coverage. Categorical axes are suitable for
covering-array coverage. The combination generator must report the selected coverage target
and generated count rather than implying enumeration.

## 4. Instrument surface

All instruments are bounded virtual measurement contracts. Their cost, sample consumption,
latency, observable channels, noise/missingness rules and termination requirements are part of
the task contract.

| Instrument | Main observable families | Cost | Sample consumption | Lifecycle rule |
| --- | --- | ---: | ---: | --- |
| HPLC | conversion, yield, selectivity, purity, recovery, phase and product distribution | 0.08 | 0.00020 L | non-final measurement |
| GC | by-product, degradation and distillate-purity signals | 0.06 | 0.00015 L | non-final measurement |
| UV–vis | yield, conversion, selectivity, flow, electrochemical and transport signals | 0.025 | 0.00005 L | non-final measurement |
| pH meter | pH, dissociation, precipitation and equilibrium signals | 0.018 | 0.00003 L | non-final measurement |
| Final assay | endpoint channels across reaction, separation, crystallization, distillation, flow, electrochemistry and equilibrium | 0.16 | 0.00030 L | requires prior termination |

An instrument exposes only the channels declared by the task. Private world laws, hidden
mechanism parameters and world/fork lineage are not part of the agent-visible observation.

## 5. Resources, termination and evaluation

The public surface also includes the non-physical interfaces that make a composition an
instrument rather than a collection of equations.

| Surface | What is explicit |
| --- | --- |
| Resources | operation attempts, vessel starts, stock, time, sample, instrument uses and terminal-assay budget |
| Transactions | precondition checks, atomic commit, rollback on failure, and no ghost state |
| Termination | explicit termination, final-assay precondition, discard/failure handling and no post-terminal actions |
| Observations | public fields, masks, noise, missingness and measurement cost/latency |
| Evaluation | endpoint definition, success metrics, safety limits, threshold semantics and declared task objective |
| Replay | committed actions, keyed observations, state transitions and resource consequences reconstructed from the same contract and seed |

## 6. Reader boundary

This map is a capability summary, not a benchmark scoreboard. It does not assert that all
listed axes have been jointly qualified, that the world space is finite at the task level,
or that the virtual modules predict arbitrary laboratory chemistry. Qualification results
must state their tested compositions, denominators and failures separately from this map.
