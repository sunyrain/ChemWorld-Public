# ChemWorld world-composition contract

This page describes the frozen public release. Current main supports component
networks; see [current foundation](current-foundation.md) for the updated runtime
and compatibility-report fields. The original qualification remains historical.

This document defines the reader-facing authoring boundary for the published release. It is
a bounded composition contract, not a claim that every possible chemical workflow or every
combination of modules has been implemented or qualified.

The external claim is deliberately bounded:

> ChemWorld supports open construction within the declared component vocabulary and
> compatibility domain. Qualification uses coverage-guided compositions and frozen unseen
> compositions; it is not an exhaustive proof over all possible tasks.

## 1. Object hierarchy

The authoring language keeps six objects separate.

| Object | Meaning |
| --- | --- |
| Component | A reusable physical, instrument, transactional, resource or observation module. |
| World | A compatible set of components, parameters and private laws that share one runtime state. |
| Task contract | The public goal and operating surface attached to a world. |
| Scenario | One task contract with concrete initial state, parameters, seed and intervention choice. |
| Trajectory | The committed operation--observation sequence produced by an agent in one scenario. |
| World fork | A controlled parent--child intervention that changes one declared private component while holding the public contract fixed. |

The task contract is written as

\[
T = (W, S_0, A, I, O, R, \tau, E),
\]

where `W` is the world, `S0` the initial state, `A` the allowed operations, `I` the
instruments, `O` the public observations, `R` the resource ledger, `τ` the termination
rule and `E` the evaluation surface.

## 2. Component vocabulary

The published vocabulary is finite and named at the level a reader can understand. A concrete
world may use a subset, provided all dependencies and interfaces are satisfied.

| Component kind | Main responsibility | Typical public interfaces |
| --- | --- | --- |
| `reaction` | Material transformation and reaction events | material, temperature, time, state transition |
| `thermal` | Heating, cooling, residence time and energy accounting | temperature, heat, time, energy |
| `phase` | Phase identity, equilibrium or phase-state transitions | phase, volume, composition, state identity |
| `separation` | Mixing, settling, extraction, washing, transfer and drying | material, phase, volume, sample identity |
| `crystallization` | Seeding, cooling crystallization and filtration | temperature, phase, crystal state, sample |
| `distillation` | Evaporation, fractionation and collection | temperature, phase, volatility, fraction |
| `continuous_flow` | Flow configuration and residence-time execution | flow rate, residence time, material, energy |
| `electrochemistry` | Potential/current control and electrochemical conversion | charge, potential, current, energy, material |
| `observation` | Public measurements, masks, noise and assay outputs | observable values, masks, cost, latency |

The vocabulary is extensible by versioning the contract. Adding a new component kind or
interface is a new contract version, not an undocumented field in an existing authoring
file.

## 3. Declarative authoring form

The following is the public shape of a composition request. It is intentionally readable;
implementation-specific identities, hashes and private-law payloads are not part of the
agent-facing contract.

```yaml
schema_version: chemworld-world-composition-0.1
composition_id: authored-reaction-purification
world_split: public-test
components:
  - kind: reaction
    role: transformation
    parameters:
      family: declared-reaction-family
      controls: [reagent, catalyst, solvent, temperature, time]
  - kind: thermal
    role: temperature-and-energy
    parameters:
      temperature_range_K: [280.0, 420.0]
  - kind: phase
    role: phase-state
    parameters:
      phases: [aqueous, organic]
  - kind: separation
    role: downstream-processing
    parameters:
      operations: [add_phase, mix, settle, separate_phase, wash, dry]
  - kind: observation
    role: public-measurement
    parameters:
      instruments: [hplc, final_assay]

task:
  objective: balanced
  budget: 10
  operations: [add_solvent, add_reagent, heat, add_phase, mix, settle,
               separate_phase, terminate, measure]
  instruments: [hplc, final_assay]
  observations: partial-instrument-observation
  resources:
    operation_budget: 10
    sample_volume_L: 0.001
    time_s: 3600
    instrument_uses: 2
    final_assays: 1
    # Optional for generated/open-agent tasks:
    # process_time_policy: required workflow envelope plus explicit repeat/implicit-time allowances
  termination: final-assay-or-budget
  evaluation:
    metrics: [score, purity, recovery, process_mass_balance_error]
    threshold: 0.55
```

The request is a construction description, not a trajectory. A scenario supplies concrete
parameter values, initial-state values and a seed after the world and task contract have
been compiled.

For coverage-generated tasks, `resources.process_time_policy` is the executable explanation of
`resources.time_s`. It decomposes the envelope into `timed_stage_max_s`, an explicit
`implicit_stage_reserve_s` for quench/transfer/filter/separation operations, and
`repeat_allowance_s`; it also records required and additional repeat counts for every operation.
The runtime checks this policy before committing an action and exposes `used_s`, `limit_s`, and
`remaining_s` on the public campaign state. A hand-set single cap without this derivation is not
sufficient for an open agent action space.

### Runtime entry point

The same request is accepted by the public compiler and environment constructor:

```python
import gymnasium as gym
import chemworld

request = {
    "schema_version": "chemworld-world-composition-0.1",
    "composition_id": "composed-reaction-assay-demo",
    "world_split": "public-dev",
    "components": [
        {"kind": "reaction", "role": "transformation", "parameters": {}},
        {"kind": "thermal", "role": "temperature-and-energy", "parameters": {}},
        {"kind": "observation", "role": "public-measurement", "parameters": {}},
    ],
    "task": {"budget": 8, "resources": {"operation_budget": 8}},
}

compiled = chemworld.compile_world_composition(request)
env = gym.make("ChemWorld", composition=compiled, seed=0)
observation, info = env.reset(seed=0)
```

`compiled.to_public_dict()` and `info["composition"]` expose the same component,
interface, operation, instrument, resource, termination, evaluation and accepted
compatibility surface. `chemworld.check_world_composition_compatibility(request)` returns the
same pre-execution decision without constructing an environment. Rejected compilation raises
`WorldCompositionError`; its diagnostics identify a stable rejection class and request path.

Complete public requests for a single process module, a cross-module world and a multi-stage
world are collected in [Public world-authoring examples](world-composition-examples.md). The same
page maps all 15 registered reference tasks onto the component and task-contract hierarchy and
keeps controlled world forks separate from general composition.

## 4. Interface and parameter rules

Every component declares the inputs it consumes, the outputs it produces, the units of
numeric fields, and the state/event identities it may update. Interfaces are checked before
execution and remain visible at the public-contract level only where the task permits them.

Parameters fall into four classes:

1. **Categorical** choices, such as a solvent family, phase role or instrument kind;
2. **Continuous** values with declared units and closed bounds, such as temperature, time,
   volume, current or flow rate;
3. **Discrete** values, such as stage counts, fraction identifiers or operation limits; and
4. **Seeded** values controlling initial state and declared randomness, which are replay keys
   rather than public hidden-law fields.

The compiler must reject a request before execution when a parameter is missing, outside its
declared domain, expressed in an incompatible unit, or attached to a component that does not
own that field. A valid composition must expose a complete public surface for operations,
instruments, observations, resources, termination and evaluation.

The published authoring parameters are deliberately finite:

| Component | Accepted parameter fields |
| --- | --- |
| Reaction | `family`, `controls` |
| Thermal | `temperature_range_K`, `duration_range_s` |
| Phase | `phases` |
| Separation | `operations` |
| Crystallization | `temperature_range_K`, `seed_mass_range_g` |
| Distillation | `temperature_range_K`, `reflux_ratio_range`, `fraction_count` |
| Continuous flow | `flow_rate_range_mL_min`, `residence_time_range_s`, `temperature_range_K` |
| Electrochemistry | `potential_range_V`, `current_range_mA`, `duration_range_s` |
| Observation | `instruments` |

Plain numeric values use the unit named by the field. A unit-bearing value uses
`{value: [...], unit: degC}` or the corresponding scalar form; compatible units are converted
before bounds are checked. Accepted authored ranges narrow the runtime operation validator,
so a value outside the declared range is not merely documented—it is rejected before state
mutation.

## 5. Compatibility and exclusion rules

Compatibility is defined by interfaces and state ownership, not by a task name.

- A state-changing component must have a typed material or process-state input and a declared
  output event.
- A downstream component may consume an upstream output only when material identity, quantity,
  units and phase/state meaning are compatible.
- Components that require a phase, thermal, electrical or flow interface cannot be used without
  the corresponding provider or an explicit adapter in the contract.
- Every committed transition must remain inside the shared constitution: non-negative amounts,
  applicable mass/charge/energy checks, safety limits, resource debits and lifecycle rules.
- Public observations may expose only fields declared by the task contract. Private world laws,
  component identities and lineage are evaluator-visible but not agent-visible.
- A final evaluation must have a reachable termination path and a valid terminal observation;
  a composition that can never close its lifecycle is rejected before qualification.

The following are representative rejection classes, not a promise that the list is exhaustive:

| Rejection | Example |
| --- | --- |
| Missing dependency | crystallization requested without a phase or thermal path |
| Interface mismatch | a volume-valued output wired to a temperature input |
| Conflicting ownership | two modules both claim authority over the same phase transition |
| Invalid parameter | negative volume, out-of-range potential or unsupported instrument |
| Lifecycle hole | no reachable termination or final-assay path |
| Resource impossibility | required operation path exceeds the declared sample, time or attempt budget |

## 6. Composition versus world forks

General composition selects multiple compatible components. A world fork is narrower: it is
an attribution experiment derived from a valid parent world and changes exactly one declared
private component while preserving the public action, observation, instrument, resource,
failure, scoring and task contract. Fork syntax must not be used to imply arbitrary
multi-component authoring.

## 7. Coverage boundary

The composition space is open within the declared vocabulary, but qualification is finite and
coverage-guided. Discrete axes are sampled with covering-array logic; continuous axes use
space-filling samples; ordered operation paths cover declared interactions such as
reaction-to-separation or reaction-to-crystallization. The sample count is determined by the
coverage target and declared depth, not by a claim of enumeration.

The 15 registered tasks are reference points in this space. A frozen unseen composition must
not occur in that reference set, must be generated after the constructor and compatibility
rules are frozen, and must run through the same construction, execution, termination and
replay path without a core-runtime patch.

## 8. Claim boundary

This contract supports claims about:

- declared components and their public interfaces;
- fail-closed construction and compatibility checking;
- coverage-guided composition and finite unseen-composition qualification;
- transactional, resource, observation-boundary and exact-replay semantics; and
- agent use of a newly generated world as an instrument demonstration.

It does not support claims about exhaustive task coverage, arbitrary third-party worlds,
universal agent intelligence, causal model rankings, physical-laboratory transfer or the
validity of every possible chemical law outside the declared model-card domains.
