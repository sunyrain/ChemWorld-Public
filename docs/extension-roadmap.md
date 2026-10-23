# Native extensions and model ranges

The current Python backend includes the models below. Scientific kernels expose
explicit parameters and application ranges; connected worlds use ChemWorld's native
state, operation, observation and transaction contracts. Optional upstream packages
serve as numerical references, not production solver dependencies.

| Area | Implemented scope | Guide |
| --- | --- | --- |
| Shared foundation | Declared modules, owned state, common residual/DAE infrastructure, native units, finite calibration, saved continuation and replay across Python/Gym/CLI/Agent | [Native sessions](native_foundation.md) |
| Thermodynamics and properties | Activity/excess properties, six UNIFAC variants, cubic alpha/mixing rules, pure-fluid Helmholtz properties/roots/saturation/flash, property derivatives and fitting | [Thermodynamic API examples](native_foundation.md#thermodynamic-and-engineering-apis) |
| Engineering fluids | Friction, fittings, meters, valves, pumps, packed beds/towers, gas and two-phase flow, open channels, particle distributions, equipment geometry and atmospheric/solar calculations | [Engineering API](native_foundation.md#engineering-flow-meter) |
| Surface and bulk reactions | Finite gas/bulk/surface reactions, multisite adsorption and oxidation, coverage interactions and common thermal dynamics | [Mechanisms](native_mechanisms.md) |
| Separation and transport | Finite TP/PH flash, liquid stability, dynamic columns, adsorption/regeneration, crystallization, membrane flow/fouling/rinsing and conservative diffusion | [Continuous workflows](native_domains.md) |
| Batteries and electrodes | SPM, bounded SPMe/DFN, particle distributions, finite-solvent SEI, selected impedance/SOH/OCV diagnostics, and persistent Thevenin RC/two-node thermal models | [Electrochemistry](native_foundation.md) |
| Aqueous/mineral chemistry | Dilute carbonate equilibrium, finite calcite dissolution/precipitation and phase-owned surface geometry | [Continuous workflows](native_domains.md) |
| Solid phases | Selected CALPHAD/ChemSage models and source inputs, finite equilibrium and thermal consumers | [Solid-phase models](native_foundation.md) |
| Biochemistry | Deterministic and stochastic networks; supported SBML rules/events and SED-ML/COMBINE contracts | [Biochemical worlds](native_biochemical.md) |
| Shared laboratory time | Finite heterogeneous vessel groups, declared open-flow systems, multiple persistent devices, FIFO measurement queues and delayed readings | [Shared clock](native_shared_lab.md) |

## Current boundaries

Sources, units, phase assumptions and validity ranges are part of each model's input.
An available equation does not imply a validated parameter set for every material.
Source-table identity review, broader battery families and experiment interoperability
remain active development. The implemented models do not replace every feature of
CoolProp, thermo, chemicals, fluids, PyBaMM, pycalphad or COPASI.

General associating/multifluid EOS, unrestricted phase envelopes and general flowsheet
solution remain future work. The unified CPU kernel redesign is a plan: compiled
execution, SIMD and a new default backend are not included in this snapshot.
Local vessel time remains the default; shared time is enabled explicitly and only
advances the model combinations supported by its guide.

## Reference and reuse practice

Models are checked against analytical limits, conservation and finite reference cases.
Independent implementation and numerical agreement do not remove attribution or
license obligations on reused source material or data. Engineering reference methods
include source-derived tables and formulas with provenance; the public project does
not claim that every method was developed without source access. Caller-supplied
fluid/data files retain their own licenses and are not implicitly bundled here.

See [current foundation](current-foundation.md) for synchronization and validation
details, and [installation](getting_started.md) for the supported environment.
