# Run the integrated native examples

These are fixed synthetic development workflows, not autonomous model campaigns
or fits to real laboratory data. They use the same `ChemWorldRuntime`, Gym, CLI,
transaction system and replay as the individual native domains. No model key or
optional chemistry backend is required. Installed entry points ship in the wheel
and sdist; no test helpers or repository-only configuration are imported.

From the locked repository environment:

```bash
uv run --no-sync python -m chemworld.examples.native_domains --output /tmp/native-domains
uv run --no-sync chemworld verify --submission /tmp/native-domains/heater.jsonl
uv run --no-sync python -m chemworld.examples.native_capabilities --format json
```

After installing the built wheel in a virtual environment, the equivalent entry
is `python -m chemworld.examples.native_domains --output /tmp/native-domains`.
Use `--workflow NAME` to select one or repeat it to select several. Outputs are
actual action/readout trajectories and terminal snapshots, not invented scores.

| Workflow | What it exercises |
| --- | --- |
| `heater` | One calibrated finite heater: gas surface reaction → liquid adsorption → thermal SPM battery. The same device moves, not copies. Each vessel retains its own material and elapsed local time. |
| `membrane` | Three process/rinse cycles on finite feed/pump/RO/permeate and wash/waste inventories; fouling persists, with no refill/reset. |
| `mineral` | Carbonate/calcite kinetics, finite acid input, phase-owned transfer, sampling and pH. |
| `shared-clock` | Independent low-copy A/B SSA and deterministic A/B vessel, two competing requests to one offline molecular counter, delayed results and persistent RNG. |
| `distillation` | Finite tray holdup, retained receiver, timed fraction collection, full reflux and reflux change. |
| `crystal` | Seeding, finite cooling/growth, filtration, resuspension, warm redissolution and a second cooling interval. |
| `solid` | Binary demixing → heating → cooling with local equilibrium and native phase readout. |
| `heterogeneous` | Four closed systems (reaction, SPM battery, adsorption and SSA) advance on one clock with a finite heater. |
| `open-closed` | A finite transport apparatus and a closed thermal bottle share event endpoints, including source exhaustion. |
| `device-lifecycle` | Multiple heaters/analyzers, FIFO contention and local material replacement while an unrelated analysis remains active. |

## Device and time boundaries

`reconnect_heater` is a zero-physical-time connection action in an explicitly
authored **local-clock** composition. Pause control first; there must be exactly
one thermal device in the world and a destination with a compatible native
temperature domain. Its identity, calibration and settings are retained; material
does not move. Use `select_vessel` then `heat` to operate at its new destination.
It does not implement robot motion, spatial collision avoidance or general
instrument hot plugging. A rejected reconnection follows normal procedure-penalty
semantics but does not change material, equipment or physical time.

The `shared-clock` workflow is different: all enabled vessels age together under
the common finite scheduler. It does **not** couple SSA and ODE inside one vessel,
and molecular capture is zero-carrier-volume Bernoulli sampling, not a liquid
aliquot. Do not interpret repeated resets with the same seeds as independent runs.

Use `--noise-mode keyed` to select the existing keyed detector primitive. Examples
use world/process seed 7 and independent observation seed 12. Each output includes
the parsed composition and literal action sequence in a `*-recipe.json` file.
The shared-clock example explicitly declares subsecond waits so a snapshot can
retain a busy one-second analyzer and an unexecuted SSA event; it does not shorten
the analyzer's physical analysis time.

An authored shared clock uses `LabClockSpec(vessel_capacity=4, ...)` to allow
four independent closed thermal/SSA vessels. The default is two; the explicit
integer capacity is 1–64, separate from the finite job queue `capacity`.
The 1-, 2- and 4-vessel paths have direct functional coverage. All declared
vessels evolve to the same event endpoint, including unselected vessels.
Changing declaration order or selecting another vessel does not choose the
numerical assembly order. One physical heater and one offline analyzer serve
persistent FIFO requests. A failed slice discards sampling and RNG changes;
the existing rejected-action cost/risk penalty remains in history. Open
connections beyond declared finite apparatus and same-vessel ODE/SSA mixing
remain unsupported. Multiple devices use explicit identities and per-device
FIFO queues; choose a device when the request would otherwise be ambiguous.
Local material replacement is evaluator-only and requires the affected system
to be idle. It retains the shared clock, resource history, calibration and
unrelated queued analyses; it is not an Agent reset action.

`chemworld.examples.closed_shared_labs.reactive_closed_lab()` authors four
independent synthetic systems: a thermal A/B reaction, thermal SPM battery,
finite adsorption contact and low-copy SSA. `phase_closed_lab()` combines
finite crystal growth with a separate Al/Cu solid specimen. Construct a
`ChemWorldRuntime(composition=...)` with either recipe and use `select_vessel`,
`queue_lab_heat`, optional `queue_lab_measure` and `wait`. Electrical drive is
an explicitly authored initial program; these recipes do not add a controller.
The one finite heater moves between systems, and each model keeps its own
temperature domain. An isothermal battery advances but cannot accept heating.
All strategies must explicitly declare `shared_clock=True`; installing a module
alone does not grant this capability. A physical cutoff aligns every system to
that actual time and stops the laboratory. These combinations use the existing
native equations and synthetic parameters; they do not establish material fits.

`chemworld.examples.open_shared_labs.open_shared_lab("transport")` combines
the finite source–pump/pipe/heat-exchanger–receiver apparatus with a closed
thermal bottle. Choose `"membrane"` for finite RO, retained fouling, wash/waste
and a separate bottle. Select the apparatus's `reactor`, call `set_flow_rate`,
then use `wait` to advance the complete laboratory. Select `bottle` and
`queue_lab_measure` to occupy its one offline chromatograph while flow continues.
The existing local `run_flow` action cannot introduce a second clock here.

Each finite apparatus advances once, including its source and collection
containers. The configured vessel capacity counts physical containers: three
for the transport example and five for the fouling membrane example. A finite
source, collection limit or wash exhaustion aligns all systems to the actual
stop. Global material, pump work and heat remain in their original ledgers;
connected-system references appear in the outcome diagnostics. Extra paths,
shared reservoirs, unmodelled loops and overlapping strategy ownership are
rejected during compilation. These recipes reuse the original bounded flow,
dilute/isothermal membrane and thermal equations.

## Residual and caloric solver choices

`BatterySPMSpec` and `TransientElectrodeSpec` default to
`numerical_form="residual"`. Set `numerical_form="ode"` to select their original
ODE path explicitly. The selection is saved with the physical model and survives
snapshot restoration. A residual solve failure remains a numerical failure;
it does not select another algorithm.

Both paths use the same finite inventories, transport and reaction equations,
energy conventions, device bounds and runtime transactions. For SPM, the residual
path solves current and voltage constraints together, including constant-current,
constant-voltage and finite-source current limits. Nonterminal events locate
the limit transitions. Algebraic values appear in solver diagnostics and are
recomputed from physical state; they are not another persistent inventory.
The planar electrode retains its existing redox, diffusion and double-layer
charge model and its isothermal energy scope.

The common backend reduces regular block index-1 residual systems to BDF or
Radau. It is not an arbitrary-index DAE solver. Diagnostics identify analytic
and finite-difference derivatives separately; the common physical assembly uses
an analytic rate matrix and explicit numerical state/time derivatives.

`FiniteTransportSpec.caloric_algorithm` defaults to `"residual"`, consuming the
same liquid enthalpy, analytic heat capacity and apparatus heat capacity. The
explicit `"brentq"` choice remains available. `caloric_residual_system` also
provides native temperature responses to target energy and declared physical
parameters. A missing analytic parameter derivative is reported as unavailable.
None of these numerical choices expands the underlying synthetic material scope.

## Capability and validation scope

See [the generated module table](native_capabilities.md). Its source is the
installed declarations and their scope notes, not a duplicate capability list.
Actual numeric domains come from authored physics and device specifications and
are narrowed by task permissions. Presence of two modules is not evidence that
every coupling between them is supported. Small SPM, dilute 0D RO and finite
binary equilibrium are not full PyBaMM, WaterTAP or CALPHAD replacements.

Export the actual bound public action ranges, material connections and installed
instrument declarations for a recipe (or a raw composition file):

```bash
uv run --no-sync python -m chemworld.examples.native_capabilities --composition /tmp/native-domains/heater-recipe.json
```

This reports the current task and active vessel at reset; future vessel choices
or physical state changes may change availability. It reuses the runtime's public
contracts, not a separate hand-maintained range table.

Validation commands:

```bash
uv run --no-sync pytest tests/test_native_domains_end_to_end.py -v
uv run --no-sync python scripts/smoke_test_wheel.py --native-foundation-only --output /tmp/native-install.json
```

The latter builds both distributions and installs each outside the repository
using locked default dependencies. It checks absence of optional upstream
libraries, runs the ten examples, and verifies each trajectory in another
process. This is a local installation test, **not** a PyPI upload or formal release.
No claim is made about platforms not actually tested in the acceptance record.

The stage-A source qualification passed 10 histories x 2 noise modes x 4 actual
entry points: 80 trajectories, 80 independent JSONL replays and 20 fresh-process
continuations. These are separate from the Windows installation qualification:
Python 3.11/3.12 x wheel/sdist, four passed units with 40 histories and 40
fresh-process replays using default sequential noise. Linux/macOS installation
is deferred; its eight planned units remain unexecuted. This does not qualify
every later scientific extension or establish real-material accuracy.
