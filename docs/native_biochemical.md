# Finite low-copy worlds

Original SBML inputs can use the same integer kernels through `sbml_world`:

```python
from chemworld.world.sbml_world import sbml_world

# model is an SBMLModel with original XML, provenance and explicit source units.
world = sbml_world(model, numerical_form="stochastic",
                   solver_method="gibson-bruck-next-reaction-v1", seed=17,
                   maximum_events=100000, controlled_parameters=("k",))
```

The other choices are `gillespie-direct-v1` and
`adaptive-explicit-tau-leap-v1`; tau requires an explicit `TauLeapPolicy` through
`tau_policy`. The builder seed selects the process stream. Session reset seeds
select observation noise independently. The existing `wait`,
`set_model_parameters` and `formal_quantity_meter` operations use the original
formal equipment coordinates and shared laboratory clock. Pending reaction
clocks or tau packets survive full JSON and new-process continuation.

This mode admits explicitly compiled nonnegative elementary source monomials
of total molecular order at most three, including full regenerated reactants,
declared modifiers and held boundary pools. Original substance, extent, volume,
time, local-parameter and conversion-factor semantics determine each molecular
quantum. Amounts must resolve to integer molecules; absent chemical formulas,
temperature dependence and reaction energies are not invented. The
`reaction_rate` readout retains the source deterministic law in mol/s; it is
not a discrete propensity in events/s.

Signed net laws, saturation or fractional species powers, reversible single
channels and continuously changing geometry remain outside this integer adapter.
Existing deterministic SBML modes retain their source event/rule support and old
v1/v2 JSON. Integer declarations use v3.
Independent distribution, source-SDK and installed-package qualification of
this adapter are recorded separately from its functional checks and from the
authored chemical-network results below.

The existing `SEDMLProtocol` and `SEDMLExperimentSession` also accept these
integer processes in finite SED-ML experiments and COMBINE archives. Declare
KiSAO `0000029` for direct, `0000027` for Gibson–Bruck, or `0000084` for the
implemented nonnegative Poisson tau strategy. Basic Poisson `0000040` is a
different variant and is rejected. Algorithm parameter `0000488` supplies the
process seed (default 0); `0000415` supplies the internal-step budget. Tau also
accepts `0000467` (maximum step in source time units), `0000228` (epsilon),
`0000230` (SSA fallback threshold) and `0000249` (critical threshold).

These declarations use the existing task graph, controls, measurements, reports
and full experiment JSON cursor. `resetModel="false"` continues the same process;
`resetModel="true"` restarts its declared process seed and initial inputs. Equal
fixed seeds across resets do not create independent replicates. The experiment
caller's seed controls the separate detector stream. Item-based archive models
still need an explicit `items_per_mole` policy when using `from_archive`.

The developing integer source-event profile also consumes piecewise constant
global parameter assignment rules, state-only constraints and the existing SBML
event queue. State guards are checked after each actual reaction or tau batch;
affine source-time guards and delayed queue deadlines split the advance. Event
capture, cancellation, priorities, cascades and source concentration/volume
assignments use the existing formal coordinates and unit conversion. The new
queue continuation survives complete runtime and experiment JSON. Tau observes
whole batches; this does not locate source events inside an approximate leap.

Continuous rate rules, non-affine time guards, time equality pulses, assignment
rules for species/geometry and time-dependent constraints are rejected explicitly.
The event profile is in development: functional and compatibility checks are
available, while independent scientific/source and installed-package acceptance
remain open. See the [51K evidence and scope](https://github.com/sunyrain/ChemWorld/blob/823b6b01f/workstreams/world_foundation/reports/convergence/51K/README.md).

Two synthetic worlds run on the native session and task-36 direct SSA strategy:
`biochemical_world("reversible")` (A↔B) and `biochemical_world("enzyme")`
(E+S→ES, ES→E+S, ES→E+P). These are numerical examples, not fitted biological
systems. Synthetic E is N2 and ES is CH2N2O so binding preserves atoms.
There is no Michaelis–Menten reduction and no implicit ODE/SSA switch.

Authored integer networks can explicitly supply full `forward_orders` for
elementary autocatalysis (`A+B → 2A`), partial regeneration (`2A → A+B`) and
catalysts with zero net change. Direct, next-reaction and tau-leap share these
integer reactants, with total molecularity at most three and unchanged
falling-factorial rates. Supply the full orders explicitly; an existing net-only
declaration keeps its original meaning. In tau-leap, newly admitted channels also
use a conservative initial-stock firing bound without crediting other channels'
same-leap production. Invalid proposals are rejected, never clipped; this remains
a biased Poisson approximation. See the
[implementation and qualification boundaries](https://github.com/sunyrain/ChemWorld/blob/823b6b01f/workstreams/world_foundation/reports/convergence/51H/README.md).
The fixed fourteen-model development block completed 98,312 trajectories across
the exact and two approximate strategies against independent finite-CME laws.
The tau acceptance tests include declared approximation allowances; they do not
establish exact-law equivalence. Kernel timing in this block found both tau
policies slower than direct SSA. Actual source-SDK qualification is separate.

Run without a model key or COPASI:

```sh
uv run --no-sync python examples/native_biochemical.py
uv run --no-sync pytest tests/test_biochemical_worlds.py tests/test_stochastic_kinetics.py -q
```

The ordinary `add_component` action accepts mol units, but every dose must
represent an exact integer number of molecules (within floating-point conversion
roundoff). Feed index 0/1 is A for reversible, E/S for enzyme. No fractional
rounding, carrier addition, thermal control or flow is supported in this slice.
Elemental transitions are checked in exact integer atoms, not a macroscopic mol
tolerance. The mol ledger remains the sole inventory; counts are a derived view.

`molecule_counter` reports independent Gaussian-noisy, signed calibrated counts
(standard deviation 1 count, unit response per molecule), never event truth.
`molecule_capture` independently selects each solute molecule with probability
0.1, exports selected molecules to instrument waste, and measures them with the
same detector noise. It is an **idealized molecular capture**, not a withdrawn
liquid aliquot: carrier volume stays constant. Analysis effort is 1 s on the
offline instrument clock, with no reactor evolution. Both reads cost 0.01.
Captured truth and sampling RNG remain private. Empty/noisy channels are not
silently replaced by latent values. No saturation or unknown-species identification
is claimed.

Process seed is declared in the world, detector seed at session reset, and capture
uses a separate persisted stream derived from the world seed. Reads do not change
the next reaction or capture; actual dosing/capture invalidates a pending event.
Snapshots retain all continuations, and material/batch/vessel initialization
restores the declared discrete state (including its initial process/capture seeds).
Thus repeated resets are deterministic reinitializations, not independent
replicates: author a world with a new seed for an independent process replicate.
Stationary instrument usage survives material resets. Snapshots contain evaluator
truth and are not agent observations.

Optional external-reference block (no default dependency, choose a fresh output):

```sh
uv run --no-sync --with python-copasi==4.48.309 python scripts/validate_native_biochemical.py --output /tmp/biochemical-attempt.json
```

The fixed development design and complete 8,288 endpoints are in
`workstreams/world_foundation/archive/20261020-native-extension/evidence/NATIVE_BIOCHEMICAL_ACCEPTANCE.md` and
`NATIVE_BIOCHEMICAL_RESULTS.json`. It compares transient/effectively equilibrated
binomial moments, finite COPASI endpoint distributions and volume/count-scaled
ODE convergence. Finite distribution agreement is not proof of all kinetic
regimes or wet-lab predictive validity. A generic molecule counter and synthetic
capture are explicit simplifications, not models of a named physical assay.

## Controlled fixed-partition ODE / stochastic chemistry

`StochasticHybridSpec` and `CrossPartitionHybridSpec` accept `ode_method="RK45"`
when authoring a world with `stochastic_hybrid_world`. This selects the shared
solver backend's RK45 policy for the continuous state and integrated Poisson
hazards. The historical default is DOP853; default JSON declarations keep their
original field set and execution signature. The selected method is bound to the
numerical continuation and retained across full JSON restoration.

Both methods use the original network, phase inventory, transaction manager and
shared clock. The fixed partition keeps its integer and continuous closed stocks;
the cross-partition variant additionally records real stoichiometric molecular
transfers and enforces whole-jump availability. Existing isothermal, fixed-volume
and unsupported-operation boundaries still apply. See the
[cross-partition example](https://github.com/sunyrain/ChemWorld-Public/blob/main/examples/native_cross_partition.py); pass
`ode_method="RK45"` when constructing its physical specification.

Full checkpoints with the same future actions reproduce the complete state and
diagnostics exactly. Arbitrary wait splits retain numerical agreement rather
than a bitwise trajectory guarantee. A fixed ten-model RK45 development block
completed 12,289 native and 81,921 independent analytic trajectories with its
physical, marginal/joint distribution and moment checks passing. Separate paired
kernel measurements against DOP853 found RK45 slower on most tested models;
explicit selection does not imply acceleration. Actual COPASI method-31 output
has retained negative-stock and last-bit repeat discrepancies, so full source
qualification remains open in task 51G. See the
[complete results and limits](https://github.com/sunyrain/ChemWorld/blob/823b6b01f/workstreams/world_foundation/reports/convergence/51G/README.md).

## Continuous reaction SDE (development)

`stochastic_sde_world` installs continuous Itô reaction noise on the same native
session and mol inventory. This is a continuous population approximation. It
does not quantize populations or doses into integer molecules.

```python
from dataclasses import replace
from chemworld.foundation.discrete_inventory import AVOGADRO
from chemworld.physchem import StochasticSDESpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.biochemical_worlds import biochemical_network
from chemworld.world.stochastic_sde_world import stochastic_sde_world

network = replace(biochemical_network("reversible"), stochastic=None)
spec = StochasticSDESpec(volume_L=1e-15, step_s=0.125)
world = stochastic_sde_world(
    network, spec, {"A": 100.5 / AVOGADRO, "B": 20.25 / AVOGADRO}, seed=123,
)
session = ChemWorldRuntime(composition=world)
try:
    session.reset(seed=7)
    session.execute({"operation": "wait", "duration_s": 0.0625})
    result = session.execute({"operation": "measure", "instrument": "sde_amount_meter"})
    print(result.observation.to_dict())
finally:
    session.close()
```

The default drift uses the original elementary mass-action concentration law;
each reaction gets independent noise with amplitude `sqrt(abs(particle_flux))`.
`ReactionFlux` and `ReactionNoise` can bind safe MathML with typed `SDEParameter`
values in coherent SI. Flux outputs require particle/s and noise outputs require
particle/sqrt(s). Populations `n0...` are dimensionless, amounts `a0...` are mol,
concentrations `c0...` are mol/m³, and `volume_m3` is m³. Default native kinetic
constants retain their original mol/L concentration basis. Unknown symbols,
incompatible branch units and cyclic flux dependencies are rejected.

RI5 defaults to fixed mesh endpoints and linear output within each accepted interval.
Snapshots retain the already drawn weak variables, numerical endpoints and mesh
anchor. Splitting waits retains the same numerical trajectory. The weak-order
claim concerns smooth interior mesh endpoints; the linear output is not a
Brownian bridge. `boundary_policy="physical"` uses bounded feasible-side step
reduction and redraws; `"reject"` fails on a negative candidate. Neither policy
clips negative stock. Work limits fail the original transaction without consuming
its process stream. Boundary reduction/redrawing can bias distributions and has
no boundary weak-order or exact chemical-master-equation guarantee.

Set `output_policy="ri5_fractional"` explicitly to evaluate the RI5 map at the
requested elapsed time from the saved left endpoint and weak variables. This
avoids the variance loss caused by linear interpolation between mesh endpoints.
It consumes no additional process draw. Actual callbacks, including saved-output
and endpoint verification, count toward the evaluation budget. This remains a
weak numerical output, with no strong-path or Brownian-bridge claim. A negative
fractional output fails the whole transaction; a feasible full endpoint does not
certify every intermediate time. The solver does not clip that output or replace
it with linear interpolation. Historical declarations without an output policy
retain their original linear codec and saved numerical path.

Set `output_policy="ri5_truncated"` to make each requested output time an
accepted RI5 endpoint. Before drawing a numerical candidate, the step is capped
at the requested time; physical acceptance therefore checks the endpoint that
will actually be committed. This policy carries no pending dense output. A
final interval shorter than `min_step_s` is attempted as a whole; if it is
infeasible, the bounded redraw rule applies without further sub-minimum
backtracking. Stock is never clipped or replaced with another output strategy.

With this policy, splitting a wait can change its numerical grid and weak
trajectory. The same complete checkpoint and same subsequent actions still
reproduce the full state exactly. Instantaneous nonconsuming measurement does
not split time or consume the process stream. The linear and fractional options
keep their existing fixed-mesh rules and historical snapshot bindings.

Continuous solute dosing uses `add_component` and the existing feed index (0 is
A in the example). It discards numerical lookahead and retains the consumed RNG.
The fixed-temperature declaration excludes thermal evolution and carrier addition.
Each numerical interval has fixed volume. `sde_amount_meter` has independent signed Gaussian detector
noise with SD `1/AVOGADRO` mol and no sampling or reactor time. Integer molecular
instruments and unbound consuming samplers are rejected.

For bounded chemical inputs, declare `SDEControl` entries in `spec.controls`.
Targets are `rate_multiplier` (dimensionless), `flux_parameter`, or
`noise_parameter` (the named parameter's coherent SI units). For example:

```python
from chemworld.physchem import SDEControl

spec = StochasticSDESpec(
    volume_L=1e-15, step_s=0.125, minimum_volume_L=0.75e-15,
    controls=(SDEControl("forward_scale", "forward", "rate_multiplier", 0, 4),),
)
```

Build the world with this spec, then call
`session.execute({"operation": "set_sde_parameters", "values": {"forward_scale": 0.5}})`.
The action schema exposes each declared input's bounds, dimensions and current
value. A multi-input update is atomic. Unknown names, invalid dimensions,
out-of-range values or a currently undefined flux/noise expression are rejected.
Changes discard numerical lookahead while retaining the consumed process stream.

An explicit `minimum_volume_L` enables `sde_aliquot_meter` when its fixed
`1e-16 L` withdrawal fits. It uses the original representative liquid sampling
and material receipts: actual mol and volume leave the reactor, with no rounding
to molecular counts. It measures `sample_amount_*` in mol, with independent
Gaussian SD `1/AVOGADRO` mol, and charges 1 s of offline instrument time and 0.01.
The reactor clock does not advance during this offline analysis. The sampler
has deterministic volume; capture and pipetting noise are not modelled. The
volume floor and numerical representability are enforced. Subsequent chemistry
uses the remaining real volume and a freshly bound numerical interval.

Material/vessel initialization replaces the declared stocks and volume, retains
the world clock, current inputs and process RNG, and invalidates lookahead.
Batch initialization reinstalls the declared initial process seed and inputs,
so it is deterministic reinitialization rather than an independent replicate.
Instrument replacement retains reaction state and pending work. Snapshots and
ordinary replay preserve these boundaries. Advance work reports count actual
drift/diffusion callbacks, including numerical verification of a saved packet;
snapshot restoration retains full numerical endpoint checks.

Native wait/feed/read, shared clocks, rollback, JSON and new-process continuation
have functional checks. A wheel rebuilt from the sdist and genuinely installed
outside the checkout passes all four native/Gym and detector-noise protocol
combinations, exact replay and fresh-process continuation without COPASI. Its
dependencies come from the locked caller environment. Independent SDE
distributions, moment and step convergence, and boundary statistics remain open
in task 51E. Separate sdist-built and genuinely installed wheels also pass all
four protocols for each explicit fractional and truncated output policy,
including exact replay, material initialization and fresh-process continuation.
These results do not close parent task 51.

## Adaptive stochastic / deterministic partition (development)

`stochastic_adaptive_world` uses the existing chemical world, phase inventory,
transaction manager and shared clock. `AdaptiveHybridSpec` selects lower and
upper particle thresholds, the classification interval, and RK4 or LSODA for
the deterministic subnet. Classification retains its previous value inside
the hysteresis band. A reaction belongs to the stochastic subnet if any of its
participants is low, including products and catalysts. Its persistent
next-reaction target survives deactivation and reactivation.

Full integer `forward_orders` may include regenerated reactants. For
`A + B => 2 A`, declare `forward_orders={"A": 1, "B": 1}`: rates use both
reactants and material changes use the net stoichiometry once. The adaptive
variant admits elementary molecularity up to three, including partial
regeneration, and rejects missing consumed reactants or fractional orders.

```python
from dataclasses import replace

from chemworld.physchem import AdaptiveHybridSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.biochemical_worlds import biochemical_network
from chemworld.world.stochastic_adaptive_world import stochastic_adaptive_world

network = replace(biochemical_network("reversible"), stochastic=None)
quantum = 1 / 6.02214076e23
spec = AdaptiveHybridSpec(
    volume_L=1e-15, lower_limit=45, upper_limit=55, step_s=0.125, method="LSODA"
)
world = stochastic_adaptive_world(
    network, spec, {"A": 50.25 * quantum, "B": 50.75 * quantum}, seed=17
)
session = ChemWorldRuntime(composition=world)
try:
    session.reset(seed=7)
    session.execute({"operation": "wait", "duration_s": 0.375})
    session.execute({"operation": "add_component", "component": 0, "amount_mol": 0.5 * quantum})
    result = session.execute({"operation": "measure", "instrument": "sde_amount_meter"})
    print(result.observation.to_dict())
finally:
    session.close()
```

All material remains real mol in the original phase ledger; classification
never rounds stocks to molecules. The stochastic rates use generalized
falling factorials and require an available whole reaction jump. Fractional
low populations therefore retain an explicit hybrid approximation. Rates are
frozen within numerical slabs and refreshed at events, grid boundaries and
material interventions. This is distinct from continuously integrated hazards.

This world currently fixes temperature and volume, supports instantaneous
solute additions and nonconsuming mol measurements, and rejects consuming
meters or other unbound couplings. Material and vessel initialization preserve
the process stream and world time while rebinding channel targets; batch
initialization reinstates the declared initial state. A numerical failure rolls
back physical changes and keeps the existing failed-attempt accounting.

Complete JSON restoration and the same future actions reproduce the full
state exactly, including in a new process. Arbitrary observation-time splits
can change the deterministic numerical integration. An sdist-built wheel,
genuinely installed outside the checkout, passes eight protocols across both
ODE methods, native/Gym entry points and detector noise modes: 80 committed
steps, exact replay and eight fresh-process continuations. Dependencies come
from the locked caller environment. Independent finite-grid qualification covers
11 families across both methods and two step sizes: 44 units, 40,964 native and
163,844 reference trajectories. These tests cover linear ODE subnets and elementary
SSA, including fractional stocks, switching and channel reactivation.

Measured resource comparisons show gains and regressions: RK4 is faster on the
tested mixed networks, while low-copy and small ODE cases are slower. Replacing
high-copy stochastic channels with ODE preserves their tested mean and removes
their stochastic variance. Actual COPASI methods 29/30 discard material in the
tested switching cases; native stocks remain unrounded.

Nonlinear fixed-grid qualification adds dimerization, trimerization,
autocatalytic growth, feedback catalysis and mild/stiff enzyme reactions. All
36 native units pass against independent closed flows: 24,588 native and
98,316 reference trajectories, including two-step-size checks for pure ODEs.
The 20,488 previously successful recorded endpoints, event counts, partitions,
work and RNG states remain exact after expanding reactant admission. Both the
original and autocatalytic installed profiles pass eight protocols each,
including real stochastic events, ODE work, replay and fresh-process restoration.

The actual source comparison remains incomplete: 19/36 units pass; 270 SDK
calls stop before their requested endpoint and other cases lose material.
All failures and partial outputs are retained. Continuous-hazard convergence
and full source equivalence remain unproven; this adaptive extension does not
expand the separate integer SSA or fixed-partition model admission rules.
See the [complete qualification and limitations](https://github.com/sunyrain/ChemWorld/blob/823b6b01f/workstreams/world_foundation/reports/convergence/51F/README.md).
