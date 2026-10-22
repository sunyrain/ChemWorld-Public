# Native physical sessions

## Pure virial coefficients

`virial_correlation` declares seven second-virial methods (`Pitzer_Curl`,
`Abbott`, `Tsonopoulos`, `Tsonopoulos_extended`, `Oconnell_Prausnitz`, `Xiang`,
`Meng`) and two third-virial methods (`Orbey_Vera`, `Liu_Xiang`) for the existing
property package and input card. For example:

```python
from chemworld.physchem import virial_correlation, virial_temperature_report

curve = virial_correlation(
    "Tsonopoulos",
    parameters={"Tc": 425.2, "Pc": 3.8e6, "omega": 0.193},
    temperature_range_K=(300, 900),
    source_note="Explicit example critical parameters; application range chosen by caller",
)
value, first, second, third = virial_temperature_report(curve, temperature_K=510)
```

Inputs use K, Pa and m³/mol; B is m³/mol and C is m⁶/mol². Signed coefficients
are valid. `virial_derivatives` and `virial_fast` return the value through the
third temperature derivative; `virial_vector` and `virial_matrix` apply the same
kernel to explicit parameter sets. They do not generate cross parameters or
perform mixture rules. All parameters remain fixed during differentiation.

`virial_value(..., order=-1)` and `order=-2` provide the documented indefinite
integrals for Pitzer–Curl, Abbott and the two Tsonopoulos variants, with zero
constants in the dimensional-temperature basis (`log(T / 1 K)`). Other methods
do not declare these indefinite integrals. Normal `integrate_correlation`
remains available for definite integrals within the declared application range.

Extended Tsonopoulos requires explicit `a` and `b`, or a `polar` selection with
`species_type` and the dipole in debye where required. The default `documented`
polar convention includes methanol and the zero-dipole alkanol limit; explicitly
select `convention="chemicals_1_5_2"` to reproduce that API's different behavior.
Meng requires explicit `a`, or `polar={"dipole": ..., "haloalkane": ...}`.
Choices and source notes survive JSON. Unknown classes and conflicting choices
are rejected. Xiang/Liu constants were recovered from public calls in their
documented mathematical basis and checked on separate inputs; their original
paper tables have not been directly verified. Development mathematical agreement
does not establish material accuracy or the validity of a truncated gas EOS.

## External density and volume sources

`chemworld.data.volume.VolumeDatabase(data_root)` reads explicitly supplied
Density tables and the existing ChemSep/VDI sources. It uses the same lazy
identity and source database as other property inputs. Call
`db.volumes("64-17-5", method="PERRY_VOLUME")` to inspect candidates, then
`db.property_input("64-17-5", component, method="PERRY_VOLUME")` to obtain a
normal `PropertyInputCard`; its `evaluate_state` consumes liquid `density`
and `molar_volume` with the component's molecular weight.

Select `curve_id` if a source has multiple candidates. Missing critical inputs
must be supplied in `parameters`; unknown temperature bounds require
`temperature_bounds_K`. Source coefficients cannot be overwritten. CRC constant
observations require `application_temperature_K` and `application_note` and
remain single points. VDI tables require `interpolation="linear"`; missing
intervals and extrapolation are rejected. These temperature paths do not imply
an independent pressure equation or caloric data. Source identity and molecular
weight must agree with the component. External scientific datasets are not
bundled or silently discovered through an upstream library.

## Virial mixtures and gas consumers

The public virial API also provides cross-parameter constructors, B/C mixture
rules and composition derivatives. `virial_gas` exposes density- and
pressure-form forward/inverse calculations. These mathematical kernels do not
by themselves establish a mixture's material accuracy or phase stability.

`chemworld.data.virial_inputs.sourced_virial_input` consumes an explicitly
selected `ConstantDatabase` record (including HEOS constants), preserving source
identity, units and missing fields. The caller declares the application range;
critical constants alone do not supply a measured validity interval. The volume
database also exposes CRC virial polynomials with the same explicit range rule.

For a pure single-phase gas, `PropertyInputCard.evaluate_state(gas_model="virial", ...)`
uses the B/C equations for pressure, density and compressibility. With explicit
ideal caloric data it also evaluates consistent internal energy, enthalpy,
entropy and heat capacities. `VirialGasThermodynamics` and
`virial_gas_batch_system` connect this state to the common `CoupledSystem` for
closed, fixed-volume heating, energy accounting and continuation. General
nonideal mixture calorics, open-flow processes and phase equilibrium are outside
this consumer contract. See `tests/test_virial_consumers.py` for runnable inputs.

## Run native sessions

These development examples use the installed current runtime, without a model
API key or upstream solver. Start with the finite synthetic CO oxidation model
below, then use the domain-specific examples in the later sections. Python 3.11
or 3.12 is the validation target.

Public functional checks live under `tests/`; bounded optional reference runners
live under `scripts/`. The `workstreams/world_foundation/` reports cited in this
guide identify development records in the authoring repository. Those run outputs
are not part of this public source synchronization; the included commands create
new comparison results at the output path you choose.

For the current ten cross-domain workflows, shared-clock boundaries and the
separate source/install denominators, start with [native domains](native_domains.md).
Both sequential and keyed detection are supported on the qualified shared-clock
histories. Explicitly declared multi-device queues and idle-system material
replacement use the same runtime, transaction and continuation machinery.
The sections below also describe later scientific extensions; their individual
acceptance scope does not follow automatically from stage-A completion.

After installing the wheel or sdist in an isolated environment:

```sh
python -m chemworld.examples.native_surface_research --bulk --mobile --terminal-assay
python -m chemworld.examples.reuse_native_equipment
```

In a development checkout, prefix Python commands with `uv run --no-sync`.
The first example feeds finite stocks, heats, measures, routes material and ends
the task. `--no-surface` runs the same apparatus as an inert or bulk-only system;
`--bulk` adds the declared bulk reaction. `--mobile` attaches a material-borne
surface. Without it, the stationary surface remains on its vessel when gas moves.

The second example uses the same heater model in inert, bulk, surface and combined
systems. It initializes fresh material in the reactor between two batches and
checks that the retained heater keeps its identity and calibration. It also binds
one compiled physical world to two tasks: direct termination and termination
followed by a required final assay. Their physical response before ending agrees.

## Use the same session through other entry points

```python
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.surface_world import surface_world_spec

spec = surface_world_spec(terminal_assay=True)
session = ChemWorldRuntime(composition=spec, observation_noise_mode="keyed")
session.reset(seed=12)
result = session.execute({"operation": "add_component", "component": "CO-feed",
                          "amount_mol": 2e-5})
reading = session.execute({"operation": "measure", "instrument": "gas_pressure"})
for channel in reading.observation.channels:
    print(channel.channel_id, channel.value, channel.unit, channel.quality)
session.close()
```

`ChemWorldEnv(composition=spec)` provides the Gym `reset`/`step` interface over
the same runtime. `chemworld.eval.runner.run_agent` accepts `composition=spec.to_dict()`
for an Agent. Both entry points retain transaction outcomes, physical duration,
resource accounting and the native observation in `info`.

To execute JSON actions from the CLI, save `spec.to_dict()` as `world.json` and
each action as a separate JSON file, then run:

```sh
chemworld render --composition world.json --seed 12 --noise-mode keyed \
  --actions feed.json pressure.json --format json --trajectory run.jsonl \
  --snapshot continuation.json
chemworld verify --submission run.jsonl
```

The CLI refuses to overwrite an existing trajectory or snapshot. Snapshots and
trajectory provenance contain private simulator state/configuration for evaluator
continuation and replay; they are not Agent observations. JSON can reorder map
keys, so physical replay uses a numerical tolerance (the CLI default is `1e-5`),
while categorical outcomes must agree. Replay requires the original runtime
semantics; current main does not substitute for frozen paper physics.

## Readings, time and reuse

Native channels own physical values and units, uncertainty, quality, axes and
acquisition times. Invalid or unobserved channels are explicit; a valid physical
reading need not fit in `[0, 1]`. The fixed Gym scalar view is a derived
compatibility projection. Use `info["native_observation"]` or session results for
the complete channel record. Calling `observe()` returns cached readings and
does not acquire another sample.

In-situ acquisition advances the world through the same physical integrator.
Offline analysis keeps its acquisition snapshot and accounts for analysis time
separately. A task requiring a final assay first freezes production, then ends
after that assay; it does not advance the frozen production vessel again.

`session.snapshot().save(path)` and
`ChemWorldRuntime.from_snapshot(SessionSnapshot.load(path))` support
private continuation with compatible bindings. `session.reset()` creates a fresh
world. Evaluator-only `reset(options={"scope": "material", "vessel_ids": ["reactor"]})`
replaces that material while retaining attached stationary equipment and its
calibration. Partial initialization retains clocks, budgets and RNG state; it is
not a free Agent operation.
`SessionSnapshot` is also exported from `chemworld.runtime`.

Current private sessions use `chemworld-session-v2`. Their phase identities and
material coordinate order survive sorted JSON, including retained samples;
pending events, apparatus queues, RNG and resource events remain part of the
same continuation. Configuration comparison uses the installed model codecs.
Supply the same `domain_modules` registry when restoring a local extension;
callable names in a saved binding describe installed code and are never imports.

Older v1 sessions require an explicit `SessionSnapshot.migrate_v1(payload)` call.
This converts the encoding only when runtime semantics match and omitted
assembly order is unambiguous. Multiple records without saved order require
the original ordering information. Restoration still checks the complete
execution binding. Older physical-world inputs retain their supported boundary
conversions; this does not authorize reinterpretation of frozen session evidence.

## Model and acceptance boundaries

The gas reaction, five-step surface rates, finite material transfers, shared
internal-energy/heat-capacity balance and heater model are executable physics.
The tests' injected numerical failure, process failure and invalid readout are
interface fixtures, not implemented failure mechanisms. Thermal stopping uses a
real event. Property tables and rate parameters are finite synthetic contracts.
Surface flow, arbitrary thermodynamics and new electrode kinetics remain outside
this foundation slice.

The native acceptance covers fixed native/Gym/CLI/Agent histories, both observation
noise modes, direct/assay endings, replay and retained equipment. Installation
checks build both archives and run the installed modules outside the checkout,
with Cantera absent and model keys removed from the child environment. Reported
platform results apply only to actual runs; this is development acceptance, not a
formal release or a claim of general chemical validity. See
[contribution guide](https://github.com/sunyrain/ChemWorld-Public/blob/main/CONTRIBUTING.md) for packaging and frozen releases.

## Properties at the current state

`ComponentPropertyPackage.evaluate_state(...)` evaluates a pure state;
`evaluate_property_state(packages, amounts_mol, ...)` evaluates a composition.
Both use the existing component IDs, declared aliases and CAS identifiers.
Supply `temperature_K`, `pressure_Pa`, `phase` and the requested `quantities`.
The report retains units and validity warnings for each property. For example:

```python
from chemworld.physchem.curated_properties import curated_property_package

report = curated_property_package("7732-18-5").evaluate_state(
    temperature_K=400, pressure_Pa=1000, phase="gas",
    quantities=("cp", "enthalpy", "internal_energy", "density"),
)
cp = report.properties["cp"].quantity().to("J/(mol*K)")
```

Energy values are molar J/mol, heat capacities J/(mol K), volume m³/mol and
mass density kg/m³. The default caloric datum is explicitly sensible H=0 at
298.15 K; it supplies no formation enthalpy or absolute entropy. Supply a
`CaloricReference` when another datum is needed. Entropy and chemical potential
require an entropy datum. The six curated packages currently supply gas Cp and
vapor pressure; their default material phase does not imply liquid Cp or
transport data. Missing quantities raise an error.

Gas states default to the ideal-gas closure. An explicit existing `CubicEOSSpec`
selects PR/SRK volume, fugacity, departures and derivatives on the requested
gas/liquid root. This is a single-phase evaluation; phase equilibrium belongs
to the flash model. Its Cp derivative follows the selected outer root and does
not support root-topology changes. `validity_policy="warn"` or `"ignore"` can
evaluate beyond a correlation range, but this API retains the invalid-domain
record under either policy.

Native reactor and equipment consumers can share
`PropertyPackageThermodynamics` with `SpeciesCaloricModel` data. They obtain
U(n,T), Cv and composition derivatives from the same evaluations at the current
candidate state. The common integrator owns time and heat accounting; material
mixing solves the same U relation for temperature. Accepted states and authored
operating intervals must remain inside the declared caloric domain. This
additive caloric runtime does not claim a nonideal closed-volume EOS balance.
Existing constant-Cv declarations delegate to this evaluation path; the older
liquid volumetric-Cp model and fixed-capacity phase plateaus remain explicitly
selected models. Variable-caloric phase change requires a flash/latent closure.

Run `python -m chemworld.examples.state_property_research` in an installed
environment for a property query and an actual variable-Cv reactor history.
Its temperature dependence is synthetic and retains the original native U
datum; it is an interface example rather than new material data. Production
evaluation imports no reference backend. The optional `physchem-ref` tools are
used only for the separate same-equation and model-bias checks documented in
`workstreams/world_foundation/NATIVE_PROPERTY_ACCEPTANCE.md`.

## TP/PH flash and finite phase inventories

`VLEPropertyPackage` combines existing property packages, explicit gas enthalpy
references and a declared liquid activity model. `tp_flash(...)` updates K from
the current liquid and vapor compositions; an optional existing gas EOS updates
vapor fugacity and departure enthalpy. Liquid enthalpy is the temperature
derivative of the same Psat/activity free energy, sharing the gas datum.
`ph_flash(..., pressure_Pa=..., enthalpy_J=...)` holds total H fixed and solves T.
Pure saturation uses an enthalpy lever rule. Missing/vanished phases carry zero
inventory. Unreachable H and nonconvergence raise explicitly.

TP returns a state query when no inlet H is supplied; its heat duty is then
unavailable. With explicit `feed_enthalpy_J`, TP reports the heat required to
maintain the specified T. PH reports zero added heat and its actual enthalpy
residual. It does not close the balance by assigning the residual to a heater.

This first slice supports one liquid plus one gas, two/three components and a
convex ideal liquid or certified convex binary Margules liquid. Other activity
models and uncertified/nonconvex liquids require the separate LLE/stability
solver. Critical transitions and three-phase states are outside this model.
The Raoult liquid thermodynamic model neglects liquid volume/Poynting work;
its nominal liquid holdup is a separate geometric filling approximation.
No exact liquid internal-energy/PV-work balance is claimed.

`FlashSystemSpec` declares a finite constant-pressure thermal system. A native
`vapor_liquid_equilibrium` component selects its enthalpy strategy through the
existing physical advance coordinator. Candidate H is integrated once, PH closes
each candidate, and accepted liquid/gas quantities occupy the normal phase
ledger. The normal device, feed, material routing, measurement, transaction,
resource and snapshot services consume that same inventory. Registered liquid
species explicitly declare their carrier representation, preventing a second
legacy solvent-volume inventory from being inferred from geometric volume.
Element and carrier balance checks remain active.

Run `python -m chemworld.examples.native_flash_research` for a PH query followed
by actual finite feeding, cross-phase heating, a timed phase probe, transfer and
termination. `phase_state` is an explicitly noiseless synthetic sensor, not a
claim of experimental phase identification. See
`workstreams/world_foundation/NATIVE_FLASH_ACCEPTANCE.md` for the fixed 27-state
thermo comparison and development boundaries.

## Strict liquid stability and finite washing

`liquid_tpd_minimum(activity_model, composition, temperature_K=...)` minimizes
the actual tangent-plane distance on a deterministic set of simplex starts.
`equilibrate_liquids(activity_model, amounts_mol, temperature_K=...)` minimizes
constrained mixing Gibbs energy, refines equality of phase chemical potentials,
checks component balances and repeats TPD searches on both resulting phases.
A stable feed returns one liquid. A remaining negative post-split TPD rejects
an unsupported additional phase. Binary/ternary models support at most two
liquids; the finite numerical search is not a proof of exhaustive global
optimization for arbitrary activity functions.

`mixing_chemical_potentials_RT` reports `ln(x_i gamma_i)`, relative to the
common pure-liquid standard and divided by RT. Common pure reference terms
cancel across equal-T liquid phases; these are not fabricated absolute chemical
potentials. Activity coefficients use the existing parameter model, shared with
the property and contact calculations.

`StrictLiquidSystemSpec` declares an isothermal material law, component molar
holdups and finite wash policy. A `liquid_equilibrium` component and matching
`CompositionPhysics.liquid_models` entry enable the normal mix/wash, feed,
clock, measurement, material routing and snapshot services. Actual wash liquid
comes from a finite named supply; discarded material is an explicit molecular
outlet. A miscible wash retains one mixed liquid. Droplet entrainment uses the
same inventory transfer function as the empirical contact model. The phase
probe reports liquid count and second-liquid fraction as synthetic native
channels. Mixing heat and PV work remain explicitly unmodeled in this
isothermal slice; no heat-control operation is implied.

The older distribution-coefficient contact model remains a separate model.
Its `minimum_tpd_like` includes an empirical partition drive and is explicitly
marked `thermodynamic_stability_certified=False`. Strict liquid equilibrium
does not use that drive or fall back to it after a solver failure.

Run `python -m chemworld.examples.native_liquid_research` for finite mixing,
washing, a timed observation, transfer and termination. The example's Margules
parameters and component identities are synthetic. See
`workstreams/world_foundation/NATIVE_LLE_ACCEPTANCE.md` for common-tangent,
ternary and actual PhasePy validation.

## Conservative diffusion primitive

`chemworld.physchem.finite_volume` provides planar and spherical one-dimensional
Fick operators in SI units. `DiffusionMesh` defines physical volumes and face
areas; `ConservativeDiffusion.evaluate(amounts_mol)` returns molar derivatives
and signed face rates without advancing time. Positive prescribed boundary flux
means outward flux in mol/(m² s). A spherical center requires symmetry.

`operator.couple(existing_inventory_keys, process_id=..., owner_id=...,
boundary_enthalpies_J_mol=...)` supplies a process and material ports to an
existing `CoupledSystem`. Explicit boundary enthalpies are required for open
boundaries. BDF/Radau use the assembled sparse stencil; outward fixed-flux
depletion stops at its actual event. The caller retains inventory ownership.
This is the shared spatial primitive for subsequent electrode/particle models;
these mathematical consumers do not implement a battery or a new PDE runtime.

## Finite dynamic distillation

`DynamicColumnSpec` reuses `ShortcutColumnSpec` for the hardware envelope and
`VLEPropertyPackage` for the thermodynamic law. It has 2–8 equilibrium trays,
a finite reboiler, condenser and product collector. Each compartment advances
actual component amounts and H; task-18 PH closes its phase allocation at each
candidate. Vapor withdrawal and liquid overflow use declared finite holdup laws.
Heat and material streams enter one shared BDF advance. This model excludes
full tray hydraulics, pressure-drop networks and simultaneous reactions.

A `dynamic_distillation` component and matching `CompositionPhysics.column_models`
entry select this strategy. `dynamic_column_world_spec(spec, initial, ...)`
constructs a native apparatus from explicitly supplied properties and initial
inventories. Real `distill`, heat/wait and timed observations use the common
coordinator; dry, flood, temperature, collection and native vessel-volume
boundaries stop at their actual roots. Product history records actual collector
increments over time. `collect_fraction` moves actual liquid n and H into its
separate retained phase; it never resets the column or manufactures recovery.

Distill controls are target temperature, duration and finite `reflux_ratio`.
Native event JSON additionally accepts numeric `full_reflux: 1` (with the usual
required reflux field present); `full_reflux: 0` resumes finite collection.
The unchanged fixed Gym numeric carrier explicitly rejects this additional
field and directs callers to event JSON. The old `distillation` component
remains a labelled `shortcut_estimate`, without stage histories or a fallback
from failed dynamic integration.

PhaseLedger alone owns bulk n. Material and stationary support H occupy separate
typed slots. Replacing material retains apparatus heat and device calibration,
then equilibrates the new fluid plus retained body through PH. Batch reset uses
the declared initial thermal boundary while retaining hardware settings.
Snapshots retain all compartments and controls. The synthetic phase probe
observes the reboiler compartment. Liquid volume remains the geometric holdup
approximation of task 18, with no exact liquid U/PV-work claim.

Run `python -m chemworld.examples.native_distillation_research` for five actual
actions: distill, timed measurement, finite collection, reflux change and stop.
The example needs no optional reference packages. Its Cp, latent heat and vapor
pressure data are explicitly synthetic. The fixed dynamic grid, IDAES MESH
steady comparisons, Fenske limit and time refinement are documented in
`workstreams/world_foundation/NATIVE_DYNAMIC_DISTILLATION_ACCEPTANCE.md`.

## Finite vacuum concentration

`CoupledConcentrationSpec` uses the same state-dependent VLE properties and PH
closure as flash and distillation. A finite pot and receiving bottle each own
component inventory and H. Actual retained pot vapor supplies the withdrawal
port. The existing equipment capture fraction routes vapor to the bottle;
uncaptured vapor crosses an explicit component/H boundary. Captured vapor
remains bottle inventory until PH and the finite cooling port condense it.
There is no separate instantaneous condensate calculation or latent-heat debit.

A `vacuum_concentration` component and matching
`CompositionPhysics.concentration_models` entry select the strategy.
`concentration_world_spec(spec, initial, ...)` constructs a native apparatus.
`concentrate` enables heating and vapor transport. Heat/wait and timed in-situ
measurements continue the same physical advance. Heater pause leaves the
transport setting unchanged. An operating stop closes transport and heat at its
actual root; subsequent passive measurement can cool the bottle. Duration
exhaustion with an unmet solvent endpoint is a valid result.

Pot, bottle, condenser gas saturation, residual liquid, target recovery,
temperature and vessel volume are finite boundaries. Accepted time, heater
delivery, cooling and outlet integrals settle once. Native material receipts
authenticate the capture split, component withdrawal and transported enthalpy;
reusing an action name does not authorize a new outlet. Numerical failures roll
back current material, H and time. PhaseLedger owns all current bulk material;
typed material and vessel slots distinguish fluid H from stationary heat
capacity. Material replacement retains body heat and calibration, and PH
reconciles the fresh charge without advancing time. Snapshots retain actual
inventories, controls and cumulative boundary receipts.

An explicitly nonvolatile component uses a liquid Cp/H reference at the model
pressure and exactly zero gas allocation. Missing gas properties remain
unavailable. The shared RR report retains a trace liquid fraction independently
when its vapor fraction rounds to one; the undefined beta=1 objective is null
for positive nonvolatile material. This slice supports ideal mixtures with one
or two volatile species and an optional nonvolatile target, at fixed pressure,
without simultaneous reactions. Liquid volume remains a declared geometric
holdup approximation; liquid PV/Poynting work is neglected.

`simulate_vacuum_concentration(CoupledConcentrationRequest(...))` and the native
operation use the same contribution and common compartment solver. Existing
`VacuumConcentrationRequest` inputs explicitly prescribe a fixed evaluated
operating profile. Their differential approximation remains labelled
`declared_fixed_operating_profile`, shares phase-advection and solver utilities,
and is never a fallback for a failed native solve. Its legacy service now charges
actual elapsed time and delivered heat, including early stops.

Run `python -m chemworld.examples.native_concentration_research` for concentrate,
timed measurement, heater pause, wait and termination. The source/installable
example uses synthetic properties and needs no optional reference package. The
fixed nine-control grid, two material fixtures, Rayleigh holdup limit and native
snapshot continuation are recorded in
`workstreams/world_foundation/NATIVE_CONCENTRATION_ACCEPTANCE.md`.

## Persistent finite sorbent contact

`SorbentKineticsSpec` supplies competitive Langmuir affinities, local LDF rates
and adsorption energies. Fluid concentration and temperature come from the
same candidate used by `CoupledSystem`; bound molecules enter its inventory
once. Temperature-dependent affinity and bulk/bound U offsets must use the
same adsorption energy. Carrier heat capacity joins the common temperature
balance; no separate adsorption heat is charged. Unequal component mobilities
use the documented finite-vacancy uptake constraint to preserve shared sites.
This is a zero-dimensional, single-site-class liquid contact with two competing
components, without axial bed, pressure-drop or intraparticle resolution.

`sorbent_contact` components select `CompositionPhysics.sorbent_models` and
the existing coupled thermal strategy. A fixed carrier owns its loading in an
equipment slot; a mobile carrier moves actual mass, sites, heat capacity and
bound molecules with its material portion. Material replacement retains fixed
loading and apparatus heat, mixing fresh material through the common U model.
Actual dry/heat/wait advances share one clock and solver. Solver failure rolls
back fluid, loading, temperature and physical time; established failed-action
cost/risk penalties still apply.

`CondensedBackgroundSpec` explicitly defines volumetric Cv and U for an L-only
carrier, with condensed PV/Poynting work neglected. It does not invent molecular
carrier or gas data. A native sorbent fixture must declare a finite carrier
source vessel. Existing add_solvent debits that actual stock and transports its
energy; routing carries solute U, carrier U and mobile bound material. The static
wash source has no active thermal process. Source exhaustion rejects the input,
and normal material/batch initialization preserves finite supplies. Explicit
entity replacement and new-world reset remain lifecycle boundaries.

The legacy standalone isothermal drying request now delegates to the same
instantaneous finite-fluid LDF contribution. It retains an explicit finite fresh
bed request and the final spent outlet, without certifying unprovided caloric
data. The legacy laboratory runtime owns a 20 g unused-media kit per vessel;
each actual dry packet debits it and leaves through an authenticated kg entity
receipt. Repeated dry or material reset cannot manufacture fresh media. No
second current spent-bed stock is maintained.

Run `uv run --no-sync python -m chemworld.examples.native_sorbent_research`
for eight existing operations through adsorption, finite wash, heating, timed
measurement and stop. All constitutive data are explicitly synthetic. The fixed
12-condition contact grid, independent equilibrium references and three native
cycles are recorded in
`workstreams/world_foundation/NATIVE_SORBENT_REGENERATION_ACCEPTANCE.md`.

## Finite transient working electrode

`electrode_world_spec()` creates a native finite planar O + e ↔ R system.
The shared conservative diffusion operator and `CoupledSystem` advance both
species profiles, Butler–Volmer exchange and double-layer charge together.
The Robin boundary includes the first half-cell diffusion resistance; the
outer boundary is closed. Positive current is anodic. Material and signed
charge relations are checked before a candidate is returned for transaction
validation. Electrical work is recorded, without claiming complete cell
energy closure or counterelectrode chemistry.

```python
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.electrode_world import electrode_world_spec

session = ChemWorldRuntime(composition=electrode_world_spec())
session.reset(seed=7)
session.execute({"operation": "configure_electrical_control", "potential_V": 0.1})
session.execute({"operation": "wait", "duration_s": 5})
reading = session.execute({"operation": "measure", "instrument": "electrical_meter"})
print(reading.observation.native_record())
```

Set `ramp_duration_s` on the control action for a linear potential ramp from
the current setpoint. Configuration does not advance time; `wait` and the
meter's one-second in-situ acquisition use the same physical coordinator.
Measurements expose terminal V and total A, without internal concentrations
or interfacial potential. This synthetic meter is noiseless and nonconsuming.

Profiles live once in material phase compartments. Charge and cumulative
electrical work belong to the electrode equipment; material reset preserves
them, while explicit equipment replacement leaves the actual electrolyte
profile intact. Snapshots preserve the control programme, inventories and
charge. Static equipment supplies remain separate from moving liquid; dynamic
or unsupported material state still requires a compatible advance strategy.

The first model uses constant diffusivities, sufficient supporting electrolyte,
isothermal operation and finite series resistance. It excludes migration,
convection, homogeneous reactions, current control and full-cell thermal
physics. Existing fixed diffusion-layer and RC models remain explicit simpler
alternatives sharing the same electrochemical constants and BV factors.

Run `uv run --no-sync python -m chemworld.examples.native_electrode_research`
for three controls, waits and native measurements followed by termination.
Fixed numerical checks and development failures are recorded in
`workstreams/world_foundation/NATIVE_ELECTRODE_ACCEPTANCE.md`.

## Programmed chromatography and persistent instrument state

`chemworld.world.chromatography_world.chromatography_world_spec("hplc")`
(or `"gc"`) creates a finite synthetic three-analyte world with source bottles,
a receiver, material routing, waste and one persistent chromatograph. It uses the
same runtime, resource ledger, measurement records, snapshots and replay as other
native worlds; no model provider or API key is required.

```python
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.chromatography_world import chromatography_world_spec

session = ChemWorldRuntime(composition=chromatography_world_spec("hplc"))
session.reset(seed=12)
session.execute({"operation": "configure_chromatography",
                 "instrument": "programmed_hplc", "program_index": 1})
# Initially the receiver contains blank solvent. Route a finite source aliquot
# into it to measure a standard or sample (see the complete example below).
result = session.execute({"operation": "measure", "instrument": "programmed_hplc"})
print(result.observation.channel("peak_1_area"))
```

Program 0 is constant (298.15 K, HPLC mobile-phase fraction 0.2); program 1
ramps over 30 s to fraction 0.8 for HPLC or 338.15 K for GC. Program 2 uses
knots at 0, 15 and 30 s, reaching 20% of that change at 15 s. These programmes
reuse the existing retention equations, with explicit programme delay and a
time-integrated elution criterion; the constant case regresses to the existing
isocratic/isothermal method. Curves reuse the Gaussian peak basis.

Each acquisition consumes 0.0002 L, costs 0.05 currency and adds 60 s of offline
analysis time without evolving the host. Actual analyte residue belongs to an
equipment-owned state slot. A blank washes this inventory into the recorded
external waste outlet; it is not a reset. Use count, gain/baseline drift,
programme and calibration persist through material replacement and snapshot
restore. Full reset restores the declared initial instrument.

The public record contains a signal curve, fitted peak areas and concentration
estimates with units and quality, plus the declared peak templates and response
calibration needed to reproduce the inverse fit without simulator state. This
finite instrument assumes known peak identities/shapes; it does not identify an
unknown retention mechanism. `calibrate_chromatography` records the last
acquired public peak areas with declared `standard_1_M`, `standard_2_M` and
`standard_3_M`; at least three standard acquisitions fit the existing response
calibration. It does not inspect hidden concentrations, detector gain or baseline.
Saturation, overlap and incomplete elution invalidate estimates, not completed
sampling. Re-reading cached observations consumes no sample, time or random draw.

Run the complete standard–sample–blank sequence twice for both instruments and
all three programmes, including JSONL replay and independent residue/waste checks:

```sh
uv run --no-sync python -m chemworld.examples.programmed_chromatography --output /tmp/chromatography-example
```

Choose a new output directory; existing results are never overwritten. The same
module runs from an installed wheel. This is a finite development example, not
a fitted real column: three synthetic analytes, symmetric peaks, lumped residue,
fixed offline duration and a non-evolving host. Continuous mobile-phase usage,
column spatial transport and instrument thermal utilities are not modelled, so
no energy-closure or general gradient-column claim is made. See
authoring development record `NATIVE_CHROMATOGRAPHY_ACCEPTANCE.md`.

## Native single-particle lithium battery

`battery_world_spec()` installs a finite battery with one spherical particle
population per electrode. Each population uses the same conservative diffusion
operator as the planar electrode. An internal lithium transfer couples the two
surfaces to the applied current; voltage follows the synthetic OCV curves and
symmetric Butler–Volmer overpotentials. Current, lithium and delivered electrical
work advance in the common solver, with actual surface and voltage cutoff events.

```python
from chemworld.physchem.battery_models import BatterySPMSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.battery_world import battery_world_spec

spec = BatterySPMSpec()
session = ChemWorldRuntime(composition=battery_world_spec(spec))
session.reset(seed=4)
session.execute({"operation": "configure_electrical_control",
                 "current_A": spec.current_at_c_rate(0.5)})
session.execute({"operation": "wait", "duration_s": 600})
reading = session.execute({"operation": "measure", "instrument": "battery_meter"})
```

The same electrical-control action accepts exactly one supported quantity:
`current_A` or `potential_V` for this battery, `potential_V` for the finite working electrode.
Both share scalar ramp interpolation and the native meter contract; a source
rejects modes and limits it does not support. Positive battery current means
discharge. C-rate is defined from the initial usable lithium capacity, not
recomputed from subsequent measured capacity. The native meter exposes only
terminal V and A. Its one-second acquisition advances the same battery state.

Lithium profiles, signed charge and work belong to the battery equipment and
are saved only once. Material reset retains them; replacing the actual battery
requires explicit new device state. A surface cutoff preserves the real elapsed
time and inventory. Reconfiguring to zero current allows subsequent relaxation.
The shared ledger uses positive electrical work into the system, so delivered
battery work is recorded with the opposite sign.

The parameters are synthetic and use uniform unresolved electrolyte. The default
isothermal option remains available; electrolyte transport remains unresolved,
and the optional finite SEI extension is described below. Run
`uv run --no-sync python -m chemworld.examples.native_battery_research` for a
discharge/rest/recharge history. PyBaMM is used only by the optional reference
block; the ordinary installation runs this example without it. Numerical scope,
all development failures and fixed comparisons are recorded in
`workstreams/world_foundation/NATIVE_BATTERY_SPM_ACCEPTANCE.md`.

### Cycles and a consistent synthetic thermal model

Pass `BatterySPMSpec(thermal=BatteryThermalSpec(...))` to evolve temperature.
The cell's chemical energy is integrated from the same linear OCV functions,
and the common solver balances terminal electrical work, chemical storage,
heater input and ambient loss. Kinetic, ohmic and diffusion dissipation are
independently derived and checked; they are not added again as reaction heat.
The OCV curves have no explicit temperature dependence, so reversible entropy
heat is zero. Positive OCV slopes are excluded from this convex thermal model.
`series_resistance_ohm` optionally adds actual voltage drop and Joule heating.

`battery_world_spec(spec, thermal_device=ThermalDeviceSpec(...))` attaches the
existing heater or isothermal bath. `heat`, `wait` and thermal pause/resume
use that same device; `stirring_speed_rpm` is optional on `heat`. A finite-power
heater changes actual temperature gradually. An ideal bath must start at its
setpoint and records its actual compensation heat.

Use `queue_electrical_stage` with exactly one of `current_A`/`potential_V`, a
finite `stage_duration_s`, and optionally one native terminal condition:
`stop_voltage_below_V`, `stop_voltage_above_V`, or `stop_abs_current_A`.
Zero-current stages are rests. Stages append to the source-owned queue;
`wait` advances the common controller, whose interval loop also serves the
existing thermal/gas programmes. No battery-specific scheduler is installed.
Completed queues switch to zero current. Explicit `configure_electrical_control`
clears the queue and installs a manual step/ramp. Voltage control solves for
the current from the actual profile; an unreachable voltage remains at the
source current limit, with the actual voltage and limit status recorded.

Each stage transition records the terminal reading and actual elapsed time.
Physical cutoff events retain the reached state; numerical failures roll back
the complete candidate action, including preceding candidate stages, while
retaining diagnostics. Snapshots preserve the queue, cursor, temperature and
inventories exactly. Material initialization is an evaluator boundary: it
retains battery SOC/profiles and uses the declared initial temperature, recorded
in its lifecycle receipt. Continuous cycling uses no such initialization.

Run `uv run --no-sync python -m chemworld.examples.native_battery_cycles` for two
CC/CV/rest/charge cycles. The nine thermal comparisons against optional PyBaMM
and two native cycles are documented in
`workstreams/world_foundation/NATIVE_BATTERY_THERMAL_ACCEPTANCE.md`.

### Finite SEI growth and retained degradation

Add `sei=SEISpec()` to `BatterySPMSpec` to enable solvent-diffusion-limited SEI.
The explicit total reaction is `2 Li + EC -> Li2CO3 + C2H4`, corresponding to
two lithium ions and two electrons at the negative interface. EC, carbonate,
retained ethylene and both active-lithium profiles live once in the cell's
equipment slot. Film thickness and resistance are derived from carbonate stock;
there is no independent thickness inventory or per-cycle capacity decrement.

The side current is `-2 F r`. Negative intercalation carries `I + 2 F r`, while
the positive electrode carries I. Both the particle boundary and negative BV
relation use that main current. Film resistance changes measured terminal
voltage. Its dissipation and reaction heat use the same chemical state function
as the thermal model. The irreversible side process stops at its electrochemical
driving-force boundary; high-rate or unfavorable conditions are not silently
extrapolated into an endergonic growth regime.

The initial film is explicitly present in the initial elemental inventory. A
closed cell consumes its actual finite EC; no hidden solvent reservoir is used.
The analytic maintained-solvent check adds a declared external feed instead.
Species formulas make Li/C/H/O checks meaningful at the runtime transaction.
Material initialization preserves the whole cell's chemical history; replacing
the cell uses a new explicit equipment inventory. Snapshot and Agent replay
retain film, solvent, gas, active lithium, electrical controls and temperature.

The source queue bounds pending work at 32 stages. Completed stages remain in
operation history and are removed from the pending list when new work is added;
`stage_index` remains absolute and `stage_offset` locates the retained list.
Thus repeated finite programmes do not exhaust a lifetime queue limit.

Run `uv run --no-sync python -m chemworld.examples.native_battery_degradation`
for ten continuous native cycles. The synthetic model includes finite solvent,
film resistance and the two-electron accounting; it excludes plating, cracking,
electrolyte transport and gas-pressure mechanics. Fixed analytic, nine-condition
PyBaMM and ten-cycle checks are in
`workstreams/world_foundation/NATIVE_BATTERY_SEI_ACCEPTANCE.md`. Windowed discharge
charge and lost-lithium equivalent charge are reported separately: neither is
used as a direct substitute for the other.

### Finite pump, pipe and heat-exchanger transport

`transport_world_spec()` installs a real finite feed and receiver connected by
a bounded pump, pipe and optional Ergun bed resistance. `set_flow_rate` sets a
request in mL/min without advancing time; `run_flow` or `wait` advances both
inventories together. Zero request stops pumping. `run_flow` may set the utility
temperature through `target_temperature_K`; otherwise the installed value is used.

The pump solves the greatest flow permitted by the requested rate, pressure
head and shaft-power limits. `PumpSpec` and `pump_operating_point` accept the
connected device's monotone pressure-load law for reuse by a membrane. The
example reaches about 12.21 mL/min for a 20 mL/min request and stops at the finite
pickup heel without charging the unexecuted time tail.

The common finite-inventory solver advances component amounts and enthalpy.
Liquid properties use native packages at the candidate T/x; the exchanger uses
the actual mass rate after pump heating. Hydraulic work heats the fluid, shaft
inefficiency is explicit external loss, and utility heat uses the existing
Nu/U/NTU functions. Source and receiver share pressure and elevation; the
pipe/exchanger has negligible holdup and the utility is a declared constant
temperature boundary, not an untracked coolant inventory.

Material slots own fluid enthalpy; vessel slots own container heat. The pump
owns its specification and control. Default batch initialization preserves the
finite feed and its local age. Explicit receiver material replacement retains
the actual container heat and recloses temperature without advancing time.
Source exhaustion, full receiver and outlet property limits stop the process;
numerical failure rolls back the candidate with a diagnostic.

`flow_meter` reports mL/min, Pa, W, K and L through native channels. Native, Gym,
CLI and Agent interfaces share execution, accounting and exact replay. Run
`uv run --no-sync python -m chemworld.examples.native_liquid_transport`.
The bounded synthetic model excludes reactions, recirculation, cavitation,
two-phase flow and an empirical industrial pump curve. Existing spatial PFR
experiments retain their declared geometry model. Validation and optional
fluids/ht comparisons are in
`workstreams/world_foundation/NATIVE_TRANSPORT_ACCEPTANCE.md`.

## Finite crystal growth and dissolution

`crystal_thermal` components install `CompositionPhysics.crystal_models` beside
an explicit thermal model. Existing seed, cool_crystallize, heat, wait, filter
and resuspend operations advance the same finite slurry at its actual
temperature. A requested setpoint supplies bounded heating or cooling; it does
not allocate dissolved molecules in advance. PhaseLedger owns current bulk
molecules. Cohort constituent rows decompose the solid into particle counts,
diameters, seed provenance and occluded impurity; they are not additional stock.

The declared van't Hoff curve and finite radial growth/recession law share
liquid/solid caloric references. One rate/energy assembler and ODE backend
advance material, temperature and utility integrals. Geometry coordinates have
units m and mol/mol, reconstructing actual molecular stocks for each candidate.
Zero-diameter roots remove hosts and release their actual occluded material.
Finite impurity capture tapers only below the serialized inventory resolution
(default 1e-8 of total impurity); actual remaining stock and numerical
corrections are retained and checked against unchanged independent balances.

Nucleation uses explicit temporal birth quadrature, normally 30 s: actual
molecules enter nucleus-size reservoirs and join growing cohorts at interval
ends. This finite-resolution approximation has a separate interval-refinement
check. There is no aggregation, breakage, polymorph or axial population model.
Authored condensed Cv and U require negligible condensed PV work; missing
phase calorics remain unavailable. Simultaneous bulk reaction and crystal
thermal processes are outside this slice.

The native ideal filter retains all particles, solid target, seed and occluded
impurity, and 8% wet mother liquor. Its 480 s advances the same active thermal
control. Resuspension stores historical filtrate at its own temperature and
mixes declared incoming carrier through U, without instant dissolution. Existing
resource cards limit fresh carrier and seed. Material replacement retains
apparatus heat; snapshots retain cohorts, controls and resource continuation.
Current container transfers require equal temperatures when mixing occupied
native crystal compartments. Numerical failure rolls back physical state and
time, retaining the established failed-action penalties.

The legacy laboratory slurry also uses the finite radial contribution, with its
explicit condensed enthalpy boundary. Its carrier Cp is not relabelled Cv. The
legacy 96%/92% mechanical retention approximation remains separate and does not
certify a complete native U balance. Quench legacy bulk reactions before slurry
thermal holds. The standalone prescribed-temperature crystallization provider
remains an explicit reference approximation, not a failed-native-solver fallback.

Run `uv run --no-sync python -m chemworld.examples.native_crystal_research`
for eight operations and 901 s of actual time using synthetic properties and
no optional reference library. Fixed contact, birth-refinement and two-cycle
coverage and all attempted outcomes are recorded in
`workstreams/world_foundation/NATIVE_CRYSTAL_THERMAL_ACCEPTANCE.md` and its JSON.

## Finite carbonate water chemistry

`aqueous_world_spec` installs a declared `aqueous_mineral` component with actual
water, H+/OH-, CO2*, bicarbonate/carbonate, Ca, Na, Cl and finite pure calcite.
Molality uses actual remaining water kg; geometric liquid volume is a separate
declared holdup. Element, carbon, charge, water dissociation and carbonate
mass action close together. Finite calcite may appear or deplete; no balancing
ion or mineral reservoir is supplied after the solve. Derived speciation is
checked against PhaseLedger before a read and is not another material stock.

The same closure serves `add_component` from finite neutral solution stocks,
`wait`, aqueous-only `sample` and the non-destructive `aqueous_ph` electrode.
Sampling retains the separate mineral phase. The existing pH meter model uses
hydrogen activity and the session RNG; the older simplified acid/base model
remains explicit for historical tasks. Both observation noise schedules support
exact snapshots and resource continuation. Invalid inputs and numerical failure
preserve physical inventories and time under the existing transaction rules.

This synthetic system is restricted to 298.15 K, low pressure and ionic strength
<=.1 mol/kg. Constants are pKw=14, pKa1=6.35, pKa2=10.33, log10 Ksp=-8.48,
Davies A=.509/b=.3, neutral gamma=1 and water activity=1. Actual water still
participates in the reaction element balance. These equilibrium constants do
not provide caloric data, temperature dependence or mineral kinetics. Mineral
advection and thermal mixing are outside this slice.

Run `uv run --no-sync python -m chemworld.examples.native_aqueous_research`
without optional libraries. `scripts/check_native_aqueous.py --reaktoro-prefix
<compatible-prefix>` runs the fixed 33-condition block, including 32 comparisons
against Reaktoro 2.13.0 with the same custom standard states and activities,
one domain rejection and four analytical limits. The reference lives outside
the default runtime dependency set. Its CLI flushes results before bypassing
an observed Windows callback-finalization failure; calculation failures retain
nonzero exit status. Coverage and all outcomes are in
`workstreams/world_foundation/NATIVE_AQUEOUS_MINERAL_ACCEPTANCE.md` and its JSON.

## Finite calcite kinetics and material surface ownership

`mineral_world_spec` installs a `mineral_kinetics` owner beside the same bounded
carbonate chemistry. A single shared BDF solve advances finite calcite while
every candidate closes fast aqueous reactions at the current solid amount.
The shrinking-area law is dn/dt=-A k(1-Omega), A=A_ref(n/n_ref)^(2/3).
PhaseLedger owns molecules; a phase-bound material slot owns reference area
and reference mol, without registering that reference as another atom stock.

The native aqueous and calcite ports move their own phases. A solid aliquot
carries proportional reference geometry; aqueous transfer/sampling retains
the solid and its surface. Equipment stays in place. The first model supports
one evolving mineral family per compartment. Different origins or different
size histories require separate compartments. Same-progress portions of one
lot can split and recombine without changing their reference area or stock.

At exhaustion, actual dissolved atoms remain and the exhausted surface is
removed. Seedless supersaturation has zero area and does not nucleate. A
declared zero rate/surface retains its stock. Chemical exhaustion or a numerical
saturation tolerance stops reaction while the supported isothermal hold fills
the remaining physical wait; these times are recorded separately. Finite acid
input changes the same water state before subsequent kinetic time. The pH
electrode uses the committed speciation after its actual acquisition interval.

The temperature, activity and caloric limits of task 32 still apply. This is
neither a reactive-transport database nor a distribution of independently
evolving crystal sizes. Run `uv run --no-sync python -m
chemworld.examples.native_mineral_research` for ten native actions, phase
transfer and a 1402 s shared-clock history without an upstream dependency.

The optional `scripts/check_native_minerals.py` compares twelve fixed
trajectories/120 states and nine early-time limits. Its independent reference
uses Reaktoro 2.13.0 constrained Gibbs chemistry and DOP853 integration; it does
not call the native algebraic solver/ODE backend or treat an upstream implicit
finite time step as an exact continuous trajectory. Coverage, all failures and
results are in `workstreams/world_foundation/NATIVE_MINERAL_KINETICS_ACCEPTANCE.md`
and its JSON summary.

## Finite dilute reverse osmosis

`membrane_world_spec()` installs a finite, well-mixed feed chamber, a persistent
membrane/hydraulic drive, and a finite permeate collector. The initial example
contains 1 L of water plus dilute NaCl. Native physical-world capacities come
from the declared vessel hardware; the legacy template size range is unchanged.

`set_flow_rate` requests a swept water flow in mL/min. `run_flow` and `wait`
advance the same common finite-inventory solver. The task-24 pump function
limits actual flow by required membrane pressure, pipe resistance, maximum
head and shaft power. Pressure is derived from the coupled solution-diffusion
and concentration-polarization equations, not from requested rejection. The
remaining feed is the concentrate: there is no duplicate concentrate stock.

The native `membrane_meter` exposes water flux in m/s, membrane pressure in Pa,
shaft power in W, feed/outlet/collected salinity in mol/m^3 and collected L.
Instantaneous permeate salinity differs from the collector's accumulated
composition. `collected_fraction` is collected water divided by all currently
retained water; after explicit material replacement it is not cumulative
recovery against all historical feed. Task-specific recovery uses the actual
input/output history. Zero-flow outlet salinity reports the model's right limit.

With finite salt permeability, zero-drive permeate concentration tends to the
bulk concentration. A perfectly rejecting membrane instead has a finite
osmotic threshold. These are distinct limits. Reverse drive is unsupported;
finite pickup, collector capacity, no forward pump drive and concentration
domain limits retain the reached state/time. Numerical failure rolls back the
candidate. Batch initialization preserves both actual liquid inventories and
their local ages. Explicit material replacement records its exchange while
preserving membrane identity and the other vessel.

This first model is isothermal at 298.15 K with ideal dilute osmotic pressure,
water density 1000 kg/m^3 and negligible salt partial volume. Bulk/interface
concentrations stay within 0–100 mol/m^3. Hydraulic displacement pressurizes a
dead-end chamber with negligible compression storage and membrane/pipe holdup.
No recirculation, compressible transient or collector backflow is implied.
Fouling is an explicit option described below. Ideal mixing enthalpy is zero; the boundary removes actual pump work
as heat. This scope does not predict solution heats or high-pressure calorics.

Run `uv run --no-sync python -m chemworld.examples.native_membrane` for two
successive half-hour intervals and native readings. Independent WaterTAP local
SD/CP equation comparisons use the same imposed ideal boundary properties;
they do not claim a full WaterTAP flowsheet or its default property package.
Coverage and all attempts are in
`workstreams/world_foundation/NATIVE_MEMBRANE_ACCEPTANCE.md`.

## Persistent membrane fouling and finite rinsing

Pass `MembraneSpec(fouling=MembraneFoulingSpec())` to `membrane_world_spec` to
include an inert SiO2 pollutant, a finite pure-water wash tank and a finite
waste tank. Feed pollutant moves into an equipment-owned film; the same film
inventory sets resistance in the same RO flux/pump solve. Desorption returns
material to feed. No second membrane solver, clock or mass copy is introduced.

`set_flow_rate` accepts optional `flow_path="process"` or `"rinse"`. Omitting
the path retains its current selection. The clean-only membrane, finite
transport path and old PFR reject a rinse path they do not implement. Switching
the ideal valve consumes no time or liquid; `wait` or `run_flow` performs the
actual operation. The rinse bypass isolates feed, carries wash over the
membrane surface and sends all detached pollutant into waste. It has zero
holdup and no membrane permeation. Water use and pump work follow actual flow;
zero flow does not clean. Empty wash or full waste stops at the reached time.

Batch boundaries and replacing feed preserve the membrane and bound film.
Replacing a membrane uses the common equipment lifecycle and records the
removed bound inventory. This evaluator initialization is not free agent
cleaning. The public meter adds wash/waste volumes, rinse flow and waste
pollutant mass; it does not expose hidden fouling parameters or bound inventory.

This is a synthetic dilute suspension model: SiO2 has zero partial volume and
adds only hydraulic resistance, with no osmotic, reaction or caloric term.
The 298.15 K boundary removes shaft work. The rinse route is not a model of
chemical cleaning, backwash or pump/valve transients. Clean limits retain the
task-34 WaterTAP comparison; fouling uses independent mass balances, analytic
desorption and time-step refinement.

Run `uv run --no-sync python -m chemworld.examples.native_membrane_fouling` for
three process/rinse cycles using the same finite tanks. The nine fixed
development cycle units and all attempted outcomes are in
`workstreams/world_foundation/NATIVE_MEMBRANE_FOULING_ACCEPTANCE.md` and its JSON.

## Finite binary solid equilibrium

`solid_world_spec()` installs a finite A/B specimen, the existing thermal
device and a synthetic phase meter. `SolidSolutionSpec` declares one or two
substitutional phases with ideal mixing and L0/L1 Redlich–Kister terms. The
solver checks common tangents against every stationary point of every declared
phase. A nominal phase can split into two composition domains; a grid only
initializes the tangent solve. The existing phase ledger stores each domain's
actual A/B inventory, with no separate mass copy in equipment settings.

`heat` and `wait` use the common finite-compartment enthalpy integrator. At each
candidate enthalpy, stable phase equilibrium determines temperature and phase
composition. The existing finite-power heater/cooler supplies the actual heat;
time and utility use are settled once. Material and batch replacement retain
the specimen holder's sensible heat and equilibrate it with the fresh material;
vessel replacement initializes the holder too. These are recorded evaluator
initializations, not free experimental actions.

The meter reads committed phase fractions/compositions and temperature in K.
Native/Gym/CLI/Agent sessions and independent-process JSON continuation share
the same implementation. The module declares its own 290–850 K validity range;
the historical liquid action range does not silently cap a solid at 430 K.
Authored limits and device/vessel limits still narrow the permitted request.

Run `uv run --no-sync python -m chemworld.examples.native_solid_solution` to
heat a demixed solid and then cool it back into two composition domains.
This is a synthetic binary, constant-pressure, instantaneous local-equilibrium
model with common constant Cp. Al/Cu formulas provide elemental counters only;
it is not a fitted Al–Cu alloy, a nucleation/growth model, a magnetic or
multisublattice description, or a runtime TDB interpreter. The optional
pycalphad reference uses the same synthetic energies and explicitly aligns its
rounded gas constant with the native constant. No upstream solver is required
by the runtime. The fixed 33-unit development block and all attempts are in
`workstreams/world_foundation/NATIVE_SOLID_SOLUTION_ACCEPTANCE.md` and its JSON.

## Selected native TDB/CEF material

`TdbDatabase` accepts declarative TDB text with an explicit `InputSource`.
`CefPhaseModel` evaluates the selected phase's sublattices, vacancies, charged
constituents, ordering, magnetic and caloric contributions without an upstream
runtime dependency. `CefMaterialSpec` adds an authored temperature interval,
fixed pressure, holder heat capacity and mechanical atom-molar volume.
`calphad_world_spec(model, element_amounts_mol=..., temperature_K=...)` installs
that material on the existing solid/thermal strategy and finite-power device.

Inventory keys are real TDB elements and amounts are mol of nonvacant atoms.
Charged constituents and vacancies belong to the committed phase's site
fractions; they do not create another material inventory. The model is stored
once in the versioned `calphad_material` physical declaration. Its equipment
record only references that binding. `calphad_phase_meter` reads committed
temperature, element amounts and site fractions. It does not synthesize a
diffraction spectrum or pretend that real elements are the old A/B counters.

`heat`/`wait` integrate the same finite-inventory total enthalpy. Internal
constitution and relaxed heat capacity come from the common constrained root;
temperature comes from the common caloric root. Constant element amounts are
eliminated from numerical unknowns in this heat-only process, and every
contribution must have zero matter rate. Holder heat is retained on material
replacement; snapshot continuation, history replay, failure settlement and
thermal controls retain their existing contracts.

This consumer holds one explicitly selected solid or liquid phase at a local
internal minimum. It does not select a globally stable multiphase assemblage.
Line compounds may have valid thermal response but underdetermined individual
element potentials; no arbitrary multiplier is reported as a determined
chemical potential. Nonpositive effective heat capacity, failed constraints
and unsupported derivative boundaries reject the closure. Gas EOS and general
phase selection require their separately declared models. The mechanical
volume is authored geometry, not a pressure derivative inferred from TDB.

`ChemSageDatabase.from_file(path, source=...)` accepts original declarative
ChemSage DAT input on that same Gibbs/material interface. The immutable input
retains source provenance, phase occurrences, suppression flags, original
stoichiometry, all intervals and model metadata. Repeated labels use
`NAME#OCCURRENCE1`, etc.; `phase_records` exposes the original labels and records.
G and Cp descriptions compile to the common caloric evaluator, including
reference H/S, latent heat, extended powers and magnetic terms. QKTO uses the
declared chemical groups and directional Kohler/Toop exponents. Auxiliary
electron balances constrain CEF internal constitution without adding atoms.

Source suppression requires explicit `activate_suppressed=True`. Nonzero
excess pressure coefficients require an explicit `excess_pressure_unit="Pa"`
or `"bar"`. Select `MQMQAPhaseModel` for SUBG/SUBQ quadruplet phases. It consumes
source coordination numbers and their mixed/reciprocal recurrences, reference
pair energies, both entropy models, and directional G/Q/R interactions with
chemical groups and ternary terms. Quadruplet fractions are internal
constitution; actual atom amounts remain in the existing material inventory.
It shares Gibbs responses, constrained constitution and `CefMaterialSpec` with
CEF. Original pair atoms and quadruplet charge must be consistent; the model
rejects contradictory input and preserves its original declarations.
Legacy annotations with no defined energy effect remain visible in diagnostics.
Nonzero MQMQA source auxiliary balances need a separately declared consumer.
The CEF consumer rejects MQMQA input. Unsupported real-gas, aqueous, volume and group
override physics are also rejected explicitly. JSON stores original input
and declared options, then recompiles through known native codecs.

The complete seven-file ChemSage qualification covers 114 original phase
declarations: 111 valid phases pass 666 response points; three inconsistent
source phases are explicitly rejected, leaving 18 original numerical points
unexecuted. A separate 96-input, 576-point synthetic block covers additional-ion
chemical groups on both ionic lattices. These are software development
qualifications, not a claim of measured accuracy for every imported material.
The fixed coverage, independent reference and preserved failures are documented
in `workstreams/world_foundation/NATIVE_CONV41D3_EXPERIMENT.md`.

## Two-site surface interactions

Pass `MultisiteSurfaceSpec` to the existing `surface_world_spec(surface_spec=...)`.
Both site classes belong to the same carrier, with separate finite capacities,
CO/O inventories and CO exchange between classes. The existing surface process
factory, thermal coordinator, gas instruments and material-routing operations
perform the experiment. There is no second catalyst service or time loop.

The new mechanism uses reversible adsorption/oxidation and a common transition
state for each forward/reverse pair. Standard Gibbs potentials and caloric
energy share the declared U/Cv table and reference entropy. A symmetric
`interaction_matrix_J_mol` contributes G_ex=U_ex=1/2 n.epsilon.n/N_ref.
The same energy and its gradient enter the heat balance; it is not an extra
external reaction-heat source. State-slot declarations reference existing
adsorbate/site amounts, so carrier splitting, merging and material transfer
retain extensive energy. A conflicting authored caloric table is rejected.

The historical five-step mechanism remains the default and retains its exact
rates. It contains irreversible oxidation. Selecting that legacy specification
restores the original model; removing B from an explicitly reversible model
does not silently turn its A reactions into the old irreversible kinetics.
The optional old bulk reaction can still contribute through the common solver;
the reversible-pair validation concerns the declared new surface channels.

Run `uv run --no-sync python -m chemworld.examples.native_multisite_surface`.
The example uses finite gas sources and the existing in-situ detector latency.
It has 120 s of requested heating/waiting and 10 s of acquisition, counted once.
The fixed 57-unit development comparison uses CatMAP's actual local interaction
and rate-constant kernels with independently solved steady equations; it does
not exercise descriptor fitting or CatMAP's mapper. All attempts remain in
`workstreams/world_foundation/NATIVE_MULTISITE_RESULTS.json`. This is a finite
synthetic mean-field model, with no spatial kMC or real-catalyst prediction.

## Cross-domain examples

The installed [native domain workflows](native_domains.md) exercise shared
equipment, finite process inventories, delayed stochastic readouts, and replay.
The [capability table](native_capabilities.md) is generated from the installed
module declarations; supported composition domains remain explicitly bounded.

## Parameter and measurement inputs

Run `uv run --no-sync python -m chemworld.examples.scientific_inputs` for a complete
import example: an existing water property package, six synthetic cooling-curve
records in two experiment groups, and two heat-loss parameters applied through
the native model declaration. The example emits JSON with sources, uncertainties,
train/validation counts and public pressure readings. It performs no fitting.

`chemworld.data.scientific_inputs.ScientificTable.from_json(text)` reads either
parameter or measurement tables. Columns declare a quantity, unit, role, finite
valid range and reference state. Each cell contains `value`, `unit`, standard
`uncertainty`, `uncertainty_note` and `quality`. A source has an explicit citation,
version, licence declaration and `synthetic`, `measured` or `literature` label.
Unknown uncertainty stays `None`; it must have an explanatory note. Celsius
values receive the temperature offset, while uncertainty uses the scale only.

Each row supplies `row_id`, `source_id`, `group_id`, `sample_id`, `split` and all
declared cells. `sample_id` identifies one acquisition, including all its channels;
repeat acquisitions need separate IDs. Measurement splits are `train`,
`validation`, `test` or `unused`. All rows in one experiment group must share a
split. Parameter rows use `parameters`. Missing or invalid observations remain
in the table with their quality flags; conditions and parameters must be valid.
The importer rejects duplicate identities, unknown columns, incompatible units
and valid values outside their declared range.

`table.to_csv()` writes a wide CSV with the row identifiers and five cell fields
per quantity, for example `pressure.value` and `pressure.uncertainty`.
`ScientificTable.from_csv(text, metadata)` reads it using the same contract;
`metadata` is `table.to_dict()` with the `rows` entry removed. No split is inferred
or randomized. `table.summary()` reports complete counts, and
`table.rows_for("validation")` selects the declared holdout.

`chemworld.physchem.property_inputs.PropertyInputCard` imports the existing
component/correlation format with per-correlation source and uncertainty records.
Its `evaluate_state` method uses the existing property evaluator. Energy and
entropy requests require an explicit imported caloric reference. The curated
example preserves unknown upstream table redistribution terms and unquantified
uncertainty; successful parsing does not establish a licence or calibration.

`chemworld.world.parameter_inputs.bind_parameter_row(physics, table, row_id, targets)`
applies one parameter row to new model configuration. The application supplies
`ModelParameterTarget` objects independently of the input, defining the installed
model/version, binding, numeric field path and quantity contract. The function
returns new `physics` and an import `receipt`; pass that configuration through
the usual composition/session constructor. Model codecs validate coupled changes
together. Measurement tables cannot be bound as parameters, and imports never
write observations or inventory into a running world.

### Finite native calibration

`chemworld.data.calibration.NativeCalibration` fits a finite parameter whitelist
by executing fixed histories in fresh native sessions. Supply a `ScientificTable`,
`CalibrationParameter` declarations and one `NativeCalibrationExperiment` per
data group. Each `CalibrationReadout` binds an observation column to an actual
measurement step and public channel, with an optional sampling-time column.
Parameter paths may address existing continuous model configuration fields;
network structure, task targets and live inventory cannot be fitted.

Only training rows enter uncertainty-weighted, bounded least squares. Successful
fits execute the validation groups separately and report errors in original
units, weighted errors, active bounds and local Jacobian rank. Non-valid input
observations remain in the exclusion counts; valid observations require positive
declared standard uncertainties. Failed forward histories retain actual readings,
operation outcomes, time, cost and sampling. They stop the fit instead of becoming
artificial residual penalties. The forward limit counts actual group executions,
including finite-difference evaluations.

Run the fixed synthetic thermal, SPM and confounded thermal examples with:

```powershell
uv run --no-sync python -m chemworld.examples.native_calibration --output $env:TEMP/chemworld-calibration.json
```

The thermal example uses an explicitly ideal synthetic pressure gauge and fixed
data perturbations. The SPM example fits one diffusion coefficient from public
terminal voltage. Joint heat-loss/heat-capacity fitting demonstrates that a good
curve fit can still leave parameters unidentifiable. These examples establish a
calibration workflow; empirical material validation requires separately sourced
data and a fixed validation design.

### DIPPR correlations and caloric requests

`PropertyCorrelation` accepts `dippr100`, `101`, `102`, `104`, `105`, `106`,
`107`, `114`, `115`, `116`, `127` (each with the `dippr` prefix), plus
`dippr100_reciprocal`, `dippr105_reciprocal`, and `dippr106_reciprocal`.
Declare a supported `property_id`, output unit, absolute-K input, coefficients
and a material validity interval. `property_equation_contracts()` exposes the
required/optional parameters, property dimensions, equation source and domains.
Material parameters still need their own source and uncertainty metadata.

The existing `evaluate_correlation` and property package use these declarations.
`correlation_temperature_derivative` returns the analytic first derivative;
`integrate_correlation` returns a definite integral or, with
`over_temperature=True`, its integral divided by T. Its numerical error estimate
and quadrature evaluation count describe the adaptive integration only, excluding
domain checks. Failed quadrature and intervals crossing poles are rejected.

`caloric_correlation_change`, `evaluate_caloric_correlation` and the existing
phase-path enthalpy reports consume these same equations. Output-unit conversion
scales the result while retaining characteristic temperatures and exponents.
Gas Cv-to-entropy conversion uses the declared ideal-gas reference. Other phases
still need an explicit closure before H and U can be interchanged.

The fixed development block compares 142 cases with chemicals 1.5.2; production
evaluation does not import chemicals. This covers formula behavior, not a new
material database. Polynomial inventory admission stays unchanged pending the
later process-integration task; the full chemicals expansion remains in progress.

### Read-only battery impedance (development)

The existing isothermal SPM, SPMe and DFN expose local rest-point analysis through
`chemworld.physchem.battery_impedance.battery_impedance`. It consumes their actual
inventory rates and electrical constraints, not an independently fitted circuit:

```python
from chemworld.physchem.battery_models import BatterySPMSpec
from chemworld.physchem.battery_impedance import battery_impedance

cell = BatterySPMSpec()
state = cell.initial_state()
spectrum = battery_impedance(cell, [1e-4, 1e-3, 1e-2], state=state)
report = spectrum.to_dict()  # per-frequency real/imaginary ohm and failures
```

This does not advance time, spend experimental resources, expose a new agent
instrument, or modify a snapshot. Positive current means discharge; passive
impedance is `-delta_V/delta_I`, with `exp(+i*2*pi*f*t)` phasors. Frequency must be
strictly positive. A nonstationary concentration profile or a stopping boundary
is rejected rather than silently relaxed. Thermal and side-reaction combinations
are not yet supported. The present quasi-static BV models have no double-layer
capacitance, so their spectra must not be interpreted as including its RC arc.

The shared `residual_frequency_response` in `physchem.residual_frequency` also
supports general declared residuals, non-diagonal mass matrices and algebraic
feedthrough. It uses the existing derivative modes and sparse linear solver,
retaining singular-frequency failures rather than substituting a least-squares
answer. Physical input/output units and derivative provenance remain in the
report. This is local response, not a nonlinear stability certificate.

The first independent development block compares six SPM cases against continuous
spherical diffusion, including four radial resolutions and a flat-OCV limit.
The subsequent SPMe/DFN block checks terminal impedance and the complex mean salt
concentration response in each of the three electrolyte regions against released
PyBaMM 25.4.2 equations, independently differentiated with CasADi and solved with
SciPy. Ten cases, each at 13 frequencies, pass the fixed comparison and native /
reference refinement bounds. The original uniform-radial reference was insufficiently
resolved for five SPMe cases (5/10 overall); that result is retained separately.
Changing only the reference radial mesh to boundary refinement gives 10/10 without
changing native physics or error bounds. This is a finite, synthetic, constant-property
domain, not validation for arbitrary materials, nonstationary points or wet experiments.
Other model families in NATIVE-CONV-45 remain outstanding. The fixed inputs, limits and
all attempts are linked from `NATIVE_CONV45_IMPEDANCE_EXPERIMENT.md` in the workstream.

### Sourced nonlinear electrode OCV

`physchem.battery_materials.ElectrodeOCV` represents a bounded polynomial plus
exponential and tanh terms. It stores the material, source, model compatibility and
stoichiometry domain. Use `from_parameter_row` to bind explicitly named columns from
the existing `ScientificTable`; amplitudes use V and scales/shifts use `dimensionless`.
The original parameter values, units and uncertainty notes remain in its JSON.

```python
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import BatterySPMSpec

# table is a validated ScientificTable of kind "parameters".
ocv = ElectrodeOCV.from_parameter_row(
    table, "electrode-row", material_id="my-electrode",
    polynomial=("constant_V", "linear_V"),
    tanh_terms=(("transition_V", "transition_scale", "transition_center"),),
)
spec = BatterySPMSpec(negative_ocv=ocv)
```

An explicit curve replaces that electrode's linear OCV fields. The actual SPM,
SPMe and DFN voltage laws consume it, and DFN uses its analytic slope in the coupled
Jacobian. Chemical potential and integrated electrode energy derive from the same
expression. Public curve evaluation rejects a point outside its declared domain;
solver trials may explicitly evaluate its analytic continuation while the battery's
physical stopping surfaces still govern accepted states.

The fixed eight-unit development block compares two Chen2020 OCV relationships,
three continuous discharge/rest/charge trajectories and three local spectra against
released PyBaMM 25.4.2 functions and models: 8/8 passed. Geometry and transport remain
synthetic in that block. It is not a complete LGM50 parameterization. Nonlinear OCV
with thermal or side-reaction models is currently rejected; the full material,
transport, thermal and other model-family scope of CONV-45 remains open. Inputs and
all outcomes are in `workstreams/world_foundation/NATIVE_CONV45_OCV_EXPERIMENT.md` and
`NATIVE_CONV45_OCV_RESULTS.json`.

### Equilibrium electrode SOH (development)

`physchem.battery_soh` solves the voltage-window geometry using those same OCV
functions. `ElectrodeSOHSpec` accepts known electrode capacities and either
`known="cyclable_lithium"` (active electrode Li charge) or `known="cell_capacity"`.
`HalfCellSOHSpec` is explicitly a positive working electrode versus Li/Li+ at 0 V;
this is an equilibrium analysis, not yet a native half-cell dynamical process.
Capacities are C; `ScientificTable` parameter bindings can explicitly convert Ah
or mAh while retaining original units, uncertainty and source. No estimator is
added: native public-observation calibration remains the existing data calibration
workflow, and local implicit sensitivities do not prove parameter identifiability.

```python
from chemworld.physchem import soh_from_battery, battery_spec_at_soc
from chemworld.physchem.battery_models import BatterySPMSpec

cell = BatterySPMSpec(minimum_voltage_V=3.38, maximum_voltage_V=3.92)
state = cell.initial_state()
soh = soh_from_battery(cell, state)  # active Li only, not electrolyte salt or locked SEI
if soh.status == "unique":
    solution = soh.branch()
    energy_J = soh.theoretical_energy_j()  # reversible OCV integral, not loaded work
    fresh_cell = battery_spec_at_soc(cell, soh, .5)  # a NEW initial configuration
```

This does not advance time, spend resources, relax a live concentration profile,
or reset an existing device. SPM, SPMe and DFN read their actual electrode bins.
`at_soc()` and `soc_at_voltage()` use the selected equilibrium branch. Complete
multiple results require an explicit branch index; incomplete isolation retains
unresolved boxes and cannot be treated as unique. Known capacity at a declared
electrode face is handled without widening the domain. Impossible inventory,
no voltage-window solution, and numerical/rank uncertainty have distinct statuses.
The shared bounded interval/Krawczyk search uses the existing point-root solver;
outward floating-point enclosures are not a formal exact-real proof. Cutoff roots
must delimit a complete physical path, not merely solve two endpoint equations.

Offline example: `uv run --no-sync python examples/native_battery_soh.py`.
The fixed 12-unit development block compares released PyBaMM 25.4.2 SOH equations,
hand algebra and independent polynomial roots, including multiple/infeasible/
nonidentifiable cases. Inputs and all attempts are in
`workstreams/world_foundation/NATIVE_CONV45_SOH_EXPERIMENT.md` and the matching
`NATIVE_CONV45_SOH_RESULTS*.json`. MSMR-specific SOH, ageing inference, wet-cell
accuracy and the remaining task-45 model families are not covered by this block.

### Many-particle electrodes (development; scientific qualification pending)

`BatteryMPMSpec` is an explicit ideal-uniform-electrolyte MPM. Each electrode
retains its total active volume and material coefficients in `ParticleSpec`.
Optional `ParticleSizeDistribution` declarations allocate that volume across
positive radii. Every size has its own actual radial inventory, diffusion and
interfacial current; currents are solved together at the shared electrode
potentials, never assigned in advance from the volume fractions. With no size
distribution, the declared particle radius gives the monodisperse SPM limit.

```python
from chemworld.physchem import BatteryMPMSpec, ParticleSizeDistribution, mpm_fields
from chemworld.world.battery_world import battery_world_spec

# source is the common InputSource for this distribution.
sizes = ParticleSizeDistribution((5e-6, 15e-6), (.5, .5), source)
cell = BatteryMPMSpec(negative_distribution=sizes, positive_distribution=sizes)
fields = mpm_fields(cell, cell.initial_state(), current_A=.2)
world = battery_world_spec(cell)
```

Direct weights are normalized volume fractions. `from_weights(..., basis=...)`
explicitly converts and normalizes positive relative number, area or volume
weights. `from_parameter_table` retains original common source rows, units and
uncertainty; `um` is an explicit length conversion, not an inferred unit.
`lognormal_size_distribution` discretizes a stated truncated density in the
specified input basis, preserving median, log-standard-deviation, bounds and
quadrature count. It does not recover unobserved distribution tails.

All sizes use the existing battery equipment slot, source controls, shared
clock, native measurements, snapshot and replay. Surface readouts are area
averages; bulk stoichiometry is volume weighted. Stops check every size, not
only an average. Rest can redistribute Li between sizes at zero terminal
current. Electrode chemical energy, dissipation, SOH totals and read-only local
impedance consume the same bins. A sourced nonlinear OCV must explicitly declare
`MPM` compatibility; stored older material compatibility lists are not widened.

Offline workflow: `uv run --no-sync python -m chemworld.examples.battery_mpm --output /tmp/mpm.jsonl`.
Functional checks include true CC/CV/source limits,
analytic derivatives, conservation, energy balance, both measurement noise modes
and cross-process continuation. The fixed nine-unit released-model comparison is
prepared in `workstreams/world_foundation/NATIVE_CONV45_MPM_EXPERIMENT.md` but has
**not yet run**. Do not infer qualified mesh resolution, measured-cell accuracy,
composite electrodes or thermal/ageing capability from these functional checks.

### Historical temperature calibration

`temperature_convert(value_K, current, desired)` supports ITS-90 and the complete
tabulated ITS-68, ITS-48, ITS-27 and ITS-76 ranges. Historical IPTS/EPT aliases are
accepted explicitly. Each historical scale uses one cubic calibration and the
shared bounded root solver for its inverse; domains use actual mapped endpoints.
`temperature_scale_conversion` also returns the derivative of that same mapping.
The Kelvin/Celsius unit system and world temperature semantics remain unchanged.

```python
from chemworld.foundation.units import Quantity
from chemworld.physchem import temperature_quantity, water_refractive_index_iapws

calibrated = temperature_quantity(Quantity(26.85, "degC"), "IPTS-68", "ITS-90")
index = water_refractive_index_iapws(calibrated.value, 997)
```

`temperature_input` preserves original input and scale provenance in the existing
InputValue note. `temperature_observation` retains the raw scalar channel and adds
a derived channel in the original Observation, with sample/timestamps and native
JSON roundtrip. Measurement uncertainty uses first-order derivative propagation;
unknown calibration uncertainty stays unknown. Invalid numeric readings and curves
requiring per-point uncertainty are explicitly rejected by this scalar adapter.

`its90_68_difference` is a separate approximate polynomial method. Its default
`published_overlap` fills the old documentation's low-temperature gap with the
published middle polynomial. Explicit `convention="chemicals_1_5_2"` reproduces
the older API's zero correction in that gap and its junction choices. Such zeros
are compatibility behavior, not evidence that the scales coincide. Neither method
asserts thermodynamic-temperature accuracy or silently modifies caloric references.
Full node/interval, cross-function and source-inventory development evidence is
recorded in the foundation acceptance document; earlier source failures and the
285 unresolved primary-CAS component mappings remain visible.

### Formal SBML models on the native world

Given a provenance-bearing `SBMLModel`, declare its numerical profile and the
quantities that an instrument may observe:

```python
from chemworld.physchem.sbml_system import SBMLObservable
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.sbml_world import sbml_world

world = sbml_world(model, maximum_step_s=0.02,
                   observables=(SBMLObservable("amount", "A"),))
runtime = ChemWorldRuntime(composition=world, observation_noise_mode="keyed")
runtime.reset(seed=4)
runtime.execute({"operation": "wait", "duration_s": 0.25})
runtime.execute({"operation": "measure", "instrument": "formal_quantity_meter"})
reading = runtime.observe()  # declared mol channel with detector uncertainty
saved = runtime.snapshot().to_dict()  # private continuation; keep outside agent payloads
restored = ChemWorldRuntime.from_snapshot(saved)
```

The bounded Level 3 Version 2 profile supports explicit 3D compartments, source
units, kinetic laws, initial/assignment/rate rules and transverse inequality
events with delays, cancellation, captured assignments and execution priorities.
Unsupported semantics are rejected. Events require an authored finite maximum
step; arbitrarily fast crossings and tangencies have no general guarantee.

One existing equipment entity owns all live amounts, compartment volumes,
parameters and pending events. The existing shared laboratory clock owns time.
`numerical_form="residual"` selects the common residual solver. Source quantities
become mol, L and seconds at explicit boundaries; model formulas retain their
source basis. A concentration rate rule includes the volume chain term, and an
event concentration assignment uses the volume immediately before execution.

The workspace has no physical chemical stock or liquid heat capacity. Formal
coordinates do not establish elemental, charge, energy or carrier-mass closure,
and heat/material sampling operations are unavailable. The gauge is instantaneous
and consumes no sample; its allowed amount/concentration/volume channels carry
mol, mol/L or L with declared Gaussian uncertainty. Public state omits evolving
coordinates and queues. Material preparation preserves the formal entity; a full
experiment reset initializes it again. A failed event cascade rolls back the
complete advance. Full private snapshots preserve pending events and detector
continuation across processes.

The fixed native-world block passed all 36 runtime contracts and 770 independent
closed-form comparisons. Full SDK parity remains explicitly false: the 18-model
COPASI block has three retained differences, and RoadRunner has one source-model
initialization difference. Official SBML volume/event cases and all raw failed
attempts are retained in the foundation evidence records. SED-ML/COMBINE protocols
and complete entry-point qualification remain separate tasks.
