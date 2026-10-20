# Native extensions and model ranges

Current development source includes the bounded models below. Each uses ChemWorld's
native state, operation, observation and transaction contracts. Upstream packages
provide equation references or optional numerical comparisons; they are not required
to execute these models. These synthetic examples do not replace the full upstream
packages or establish fidelity to a particular laboratory material.

| Area | Implemented native slice | Example or guide |
| --- | --- | --- |
| Shared foundation | One session behind Gym, CLI and Agent; declared modules, owned material/equipment state, pre-commit balances, native units and serialized continuation | [Native sessions](native_foundation.md) |
| Surface chemistry | Finite CO/O2 adsorption and oxidation, optional bulk reaction, shared thermal dynamics; reversible two-site model with coverage interactions and site exchange | `python -m chemworld.examples.native_multisite_surface` |
| Thermal and phase processes | State-dependent properties, TP/PH flash, binary/ternary two-liquid stability, finite dynamic columns, concentration, adsorption/regeneration and crystal growth/redissolution | `python -m chemworld.examples.native_distillation_research`; `python -m chemworld.examples.native_crystal_research` |
| Transport and electrodes | Finite tanks, pressure/power-limited pumping and heat exchange; conservative 1D planar/spherical diffusion and a finite diffusion/BV electrode | `python -m chemworld.examples.native_liquid_transport`; `python -m chemworld.examples.native_electrode_research` |
| Batteries | Spherical single-particle diffusion, CC/CV/rest control, lumped synthetic thermodynamics and finite-solvent SEI/lithium loss | `python -m chemworld.examples.native_battery_degradation` |
| Aqueous and mineral chemistry | Dilute carbonate/water equilibrium and finite calcite, with seeded dissolution/precipitation and phase-owned surface geometry at 298.15 K | `python -m chemworld.examples.native_mineral_research` |
| Membranes | Finite feed/collection tanks, isothermal dilute-NaCl reverse osmosis, concentration polarization, persistent synthetic fouling and finite-water rinsing | `python -m chemworld.examples.native_membrane_fouling` |
| Stochastic biochemistry | Constant-volume/temperature Gillespie events, reversible and explicit enzyme networks; integer additions, molecular capture and separate process/detector randomness | [Biochemical worlds](native_biochemical.md), [SSA](native_stochastic.md) |
| Mechanisms and instruments | Bounded YAML/JSON import with P-log/Chebyshev and NASA7 gas thermodynamics; programmed chromatography with retained instrument state | [Mechanisms](native_mechanisms.md); `python -m chemworld.examples.programmed_chromatography --output chromatography-run` |
| Binary solids | One/two substitutional phases, ideal mixing and L0/L1 Redlich–Kister terms; global supporting-line checks and thermal local-equilibrium paths | `python -m chemworld.examples.native_solid_solution` |
| Shared laboratory time | One/two independent thermal or SSA vessels, one switched heater and one offline analyzer; FIFO requests, atomic time slices and delayed readings | [Shared laboratory clock](native_shared_lab.md) |

In a checkout, prefix Python commands with `uv run --no-sync`. The package examples
run without model credentials. The biochemical checkout example is
`uv run --no-sync python examples/native_biochemical.py`.

## Remaining integration and scientific extensions

Shared time is an explicit session configuration. All enabled vessels then age together,
and snapshots retain queued jobs, pending SSA events and unpublished readings. Local
time remains the default. Shared mode currently excludes open flow, alternate phase
strategies and keyed detector noise; see the guide for its full scope. The final
cross-domain combination and installation acceptance is in progress.

Further scientific extensions need their own finite equations, parameters and reference
cases: electrolyte-resolved battery models, plating/cracking, nonideal concentrated
aqueous chemistry, membrane backflow and chemical cleaning, spatial surface kinetics,
crystal breakage/agglomeration, non-isothermal hybrid SSA/ODE, and multi-sublattice solids.
The implemented SPM, membrane and binary-solid slices are not complete
PyBaMM, WaterTAP or CALPHAD equivalents.

## Reference and reuse practice

Relevant scientific references include [Cantera](https://github.com/Cantera/cantera),
[PyBaMM](https://github.com/pybamm-team/PyBaMM),
[Reaktoro](https://github.com/reaktoro/reaktoro),
[WaterTAP](https://github.com/watertap-org/watertap),
[IDAES](https://github.com/IDAES/idaes-pse),
[COPASI](https://github.com/copasi/COPASI),
[RMG-Py](https://github.com/ReactionMechanismGenerator/RMG-Py),
[CatMAP](https://github.com/SUNCAT-Center/catmap) and
[pycalphad](https://github.com/pycalphad/pycalphad).

New models start with explicit equations, operating ranges, ownership and exchange
rules. Native implementations share the existing solver/session and are checked against
analytical limits, conservation and finite reference cases. A copied data fixture keeps
its source and license, such as the Cantera 3.1.0 mechanism under
`tests/fixtures/mechanisms/`. Upstream solver dependencies remain optional.
Independent implementation does not remove license obligations on reused code or data.
