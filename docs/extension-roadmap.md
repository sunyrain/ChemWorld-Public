# Extension to-do list

These are proposed extensions, not implemented capabilities. References were
reviewed on 19 October 2026. Estimates cover a bounded independent model,
ChemWorld operations/observations, examples and reference comparisons by one
developer familiar with the project. They exclude real-laboratory validation,
large agent campaigns and wholesale reimplementation of the upstream packages.

| Priority | Work item and reference | First deliverable | Estimate |
| --- | --- | --- | --- |
| P0 | Surface microkinetics: [Cantera](https://github.com/Cantera/cantera) | One site-balanced adsorption/desorption/reaction network with dynamic coverages and coverage-dependent rates; compare temperature/feed perturbations with Cantera. | 8–12 person-days |
| P1 | Battery cycling and ageing: [PyBaMM](https://github.com/pybamm-team/PyBaMM) | A bounded single-particle model with lumped thermal dynamics and one SEI-growth mechanism; charge, discharge, rest and diagnostic pulses share persistent cell state. | 15–25 person-days |
| P1 | Aqueous/mineral chemistry: [Reaktoro](https://github.com/reaktoro/reaktoro) | One carbonate–water–gas system with activities, phase appearance and kinetic mineral dissolution/precipitation. | 10–15 person-days |
| P1 | Membrane separation: [WaterTAP](https://github.com/watertap-org/watertap), [IDAES](https://github.com/IDAES/idaes-pse) | One reverse-osmosis unit with finite feed, pressure/flux/salt rejection and concentration polarization; explicit material and energy accounting. | 8–12 person-days |
| P2 | Biochemical stochastic dynamics: [COPASI](https://github.com/copasi/COPASI) | One enzyme network with inhibition and feedback, deterministic and Gillespie reference paths, and intrinsic noise separated from observation noise. | 8–12 person-days |
| P2 | Richer reaction networks: [RMG-Py](https://github.com/ReactionMechanismGenerator/RMG-Py) | Import a small, curated generated mechanism into a supported reaction-network subset, check species/units and compare trajectories; record omitted rate laws explicitly. | 6–10 person-days |
| P2 | Coupled battery degradation: PyBaMM | Add lithium plating and active-material loss after the first battery slice is verified; compare diagnostic identifiability under matched capacity loss. | 10–15 additional person-days |
| P2 | More faithful catalysis: Cantera; [CatMAP](https://github.com/SUNCAT-Center/catmap) as a scientific reference | Multiple site types, coverage interactions and mechanism competition after the first surface network. | 8–12 additional person-days |

## Reference details

- Cantera documents [surface reactions and coverage dependencies](https://www.cantera.org/stable/yaml/reactions.html).
  ChemWorld already has Arrhenius-family and lumped catalytic laws; explicit
  surface-site state and coverage-dependent kinetics are the intended increment.
- PyBaMM demonstrates [coupled degradation](https://docs.pybamm.org/en/v25.6.0/source/examples/notebooks/models/coupled-degradation.html),
  including SEI growth, lithium plating, cracking and active-material loss. Only
  the bounded first slice is proposed initially; a full DFN multiphysics model
  is outside that estimate.
- Reaktoro documents [mineral kinetics](https://reaktoro.org/tutorials/kinetics/kinetics-minerals-using-palandri-kharaka-model.html)
  and coupled equilibrium/kinetics. Parameter and thermodynamic database rights
  must be recorded separately from solver code.
- WaterTAP's [unit catalogue](https://watertap.readthedocs.io/en/stable/technical_reference/unit_models/index.html)
  includes reverse osmosis, ion exchange and electrodialysis. The first task is
  one membrane unit; fouling and ageing require additional specified models.
- COPASI supports [deterministic, stochastic and hybrid time courses](https://copasi.org/Support/User_Manual/Tasks/Time_Course_Simulation/).
  Existing Michaelis–Menten kinetics alone do not provide stochastic biochemical
  networks or a distinction between process and detector noise.
- RMG generates reaction mechanisms and thermochemical/kinetic inputs. It is a
  reference/source of bounded mechanism assets, not a proposed replacement for
  ChemWorld's interaction runtime or a general reaction-prediction claim.

## Independent implementation workflow

- [ ] Select one finite phenomenon, operating range and observable task.
- [ ] Pin the reference release, equations, public interface, parameter sources
      and numerical tolerances before implementation.
- [ ] Write an equation/interface specification and implement the selected slice
      independently in ChemWorld's state, operation and ledger contracts.
- [ ] Run analytical limits, conservation and numerical-convergence checks, then
      compare with the pinned reference across declared conditions, including failures.
- [ ] Add measurement/resource choices and a provider-free research example.
- [ ] Publish the supported range, reference results and unresolved differences.

No upstream solver code has been copied as part of this synchronization. PyBaMM
uses BSD-3-Clause; Cantera has a three-clause BSD-style license; IDAES and WaterTAP
have BSD-style terms with additional enhancement provisions; RMG-Py uses MIT;
Reaktoro uses LGPL-2.1, COPASI Artistic-2.0 and CatMAP GPL-3.0. Consult each pinned
license and data terms for any actual reuse. Independent implementation is not
a claim of legal clean-room certification or a way to remove obligations on
copied code/data.

Suggested order: surface microkinetics for the next enhancement to existing
chemistry; battery cycling for the first distinct new simulation domain.
