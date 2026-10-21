# Installed native capability declarations

Generated from `DomainModuleRegistry`; not a second module registry. Operation availability and numeric domains are further narrowed by the actual composition, physical specifications, device and task. A module declaration does not guarantee arbitrary cross-domain coupling.

| Module | Components / strategies | Scope |
| --- | --- | --- |
| `aqueous_mineral` | aqueous_mineral, finite-carbonate-calcite-equilibrium | Authored finite aqueous acid/base/complexation and mineral equilibrium; explicit thermodynamic data and ionic-strength domain, not a geochemical database. |
| `battery_spm` | battery_spm, single-particle-battery | Synthetic SPM with calorics/SEI, or explicit isothermal SPMe-v1 with conservative electrolyte transport; shared native cycling. Explicit isothermal DFN has coupled spatial reactions and a fixed synthetic qualification at 24 through-plane cells per region / 64 radial cells. Coarser meshes and other parameters need convergence checks; not wet-lab validation. No PyBaMM runtime backend. |
| `continuous_flow` | continuous_flow | See the authored physical spec and runtime contract; no universal validity claim. |
| `core` | material_input, sample_storage, reaction | See the authored physical spec and runtime contract; no universal validity claim. |
| `crystallization` | crystal_thermal, crystallization, finite-crystal-population-temperature | See the authored physical spec and runtime contract; no universal validity claim. |
| `distillation` | distillation | See the authored physical spec and runtime contract; no universal validity claim. |
| `dynamic_distillation` | dynamic_distillation, finite-equilibrium-column | Finite tray/receiver inventories, heat and timed cuts; authored small dynamic column, not arbitrary plant design. |
| `electrical_control` | transient_electrode, finite-transient-electrode | See the authored physical spec and runtime contract; no universal validity claim. |
| `electrochemistry` | electrochemistry | See the authored physical spec and runtime contract; no universal validity claim. |
| `finite_transport` | finite_transport, finite-pump-transport | Finite pump, pipe, tanks and authored heat exchange; bounded working point; not arbitrary multiphase process-network simulation. |
| `lab_schedule` | lab_schedule | Single process; finite independent closed systems with explicitly declared shared-clock strategies and atomic finite pump/membrane paths; explicit finite heater/analyzer identities with FIFO occupancy; no same-vessel hybrid or general loops. |
| `liquid_equilibrium` | liquid_equilibrium, isothermal-liquid-equilibrium | Strict finite liquid phase equilibrium using declared component properties and supported composition domain; no silent extrapolation. |
| `mineral_kinetics` | mineral_kinetics, finite-calcite-kinetics | Finite calcite dissolution/precipitation with fast aqueous equilibrium; fixed 298.15 K, explicit surface geometry; no mineral nucleation. |
| `observation` | observation | See the authored physical spec and runtime contract; no universal validity claim. |
| `phase` | phase | See the authored physical spec and runtime contract; no universal validity claim. |
| `programmed_chromatography` | programmed_chromatography | Authored peak/retention response, finite sampling and persistent carryover; not a fitted universal chromatographic transport model. |
| `ro_membrane` | ro_membrane, finite-dilute-ro | Finite ideal dilute NaCl RO at 298.15 K; synthetic deposited foulant and finite rinse; no reverse drive, membrane holdup or full WaterTAP flowsheet. |
| `separation` | separation | See the authored physical spec and runtime contract; no universal validity claim. |
| `solid_solution` | solid_solution, calphad_phase, binary-solid-equilibrium | Finite binary equilibrium or explicitly selected native TDB/CEF phase; local constitution and thermal equilibrium; no global phase selection. |
| `sorbent_contact` | sorbent_contact | Competitive finite sorbent kinetics, owned carrier and finite wash; explicit condensed calorics and temperature domain. |
| `stochastic` | stochastic, gillespie-direct | Integer Gillespie direct; constant volume/temperature mass action; persistent event and RNG; not spatial kMC or automatic ODE/SSA switching. |
| `surface_kinetics` | surface_reaction | Finite owned sites; declared single/multisite mean-field mechanisms and symmetric interactions; not spatial kMC or a real catalyst fit. |
| `thermal` | thermal, heater_connection, owned-inventory-temperature, static-finite-carrier-stock, legacy-liquid-enthalpy | Finite-power or explicitly isothermal device; shared energy accounting; manual reconnection requires one disabled heater in local-clock mode. |
| `vacuum_concentration` | vacuum_concentration, finite-vacuum-concentration-enthalpy | Finite variable-composition concentration with authored VLE/calorics; unsupported kinetic coupling is rejected. |
| `vapor_liquid_equilibrium` | vapor_liquid_equilibrium, constant-pressure-flash-enthalpy | Finite TP/PH closure with explicit phase/caloric models; no simultaneous undeclared kinetic process. |

For declared action units, parameter rules, state ownership and dependencies:
`uv run --no-sync python -m chemworld.examples.native_capabilities --format json`
