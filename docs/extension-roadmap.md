# Native extensions and model ranges

The public Python backend now includes the following bounded model families.
The [current foundation](current-foundation.md) records source and qualification status;
the [API guide](native_foundation.md) gives executable input/output examples.

| Area | Implemented scope |
| --- | --- |
| Shared foundation | Owned state, residual/DAE infrastructure, local/shared clocks, finite resources, queued measurements, exact continuation and replay |
| Thermodynamics | Activity/UNIFAC/cubic models, Helmholtz, IF97/ice/humid air/incompressible fluids, selected molecular/association/multifluid EOS, local stability and flash |
| Engineering processes | Fluid correlations, serial molecular units, single recycle, finite receivers and bounded design |
| Reactions and transport | Imported gas/surface chemistry, finite graph/rule growth, energy grains, reactor networks, wall/piston/heat coupling and bounded 1D transport |
| Aqueous and separation | Sourced aqueous/mineral chemistry, reactive transport, headspace, RO/NF/MD/ED, adsorption, resin/GAC, precipitation, clarification, filtration and washing |
| Solid phases | Selected CALPHAD/ChemSage models, multiphase CEF TP/PH, finite binary/ternary diagram searches |
| Batteries | SPM/SPMe/DFN, particle distributions, composite/MSMR/half-cell and other finite families, selected controls/thermal models, SEI/plating, impedance and SOH |
| Stochastic and biochemical analysis | Direct/next-reaction, explicit-partition hybrid dynamics and bounded tau-leap; SBML/SED-ML subsets, steady/control analysis, fitting, profiles, flux modes and finite-time growth |

## Current boundaries

Each model declares its sources, units, phases and validity ranges. Some are synthetic
demonstrations; others carry sourced parameters. A delivered equation does not validate
every material. Battery and molecular parent programmes remain active even where
individual submodels have passed their fixed comparisons.

The public package retains bounded serial and single-recycle process consumers.
General finite process graphs and their inventory/control integrations have advanced
in main development; see [latest progress](current-foundation.md#development-progress).
General freely propagating flames, unrestricted global phase envelopes, complete upstream
databases and universal long-term aging remain outside the delivered scope. The optional
C++ build skeleton is available; scientific kernels and SIMD migration remain in progress. See [shared time](native_shared_lab.md) for supported couplings.

## Reference and reuse practice

Production models run through the native Python backend. Optional upstream solvers serve
as independent references. Analytical limits, conservation and fixed comparison cases
check the implemented slices; numerical agreement does not remove source/data attribution
or license obligations. Caller-supplied databases retain their own licenses.
