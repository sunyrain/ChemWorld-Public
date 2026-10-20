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
item 39's finite scheduler. It does **not** couple SSA and ODE inside one vessel,
and molecular capture is zero-carrier-volume Bernoulli sampling, not a liquid
aliquot. Do not interpret repeated resets with the same seeds as independent runs.

Use `--noise-mode keyed` to select the existing keyed detector primitive. Examples
use world/process seed 7 and independent observation seed 12. Each output includes
the parsed composition and literal action sequence in a `*-recipe.json` file.
The shared-clock example explicitly declares subsecond waits so a snapshot can
retain a busy one-second analyzer and an unexecuted SSA event; it does not shorten
the analyzer's physical analysis time.

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
libraries, runs the seven examples, and verifies each trajectory in another
process. This is a local installation test, **not** a PyPI upload or formal release.
No claim is made about platforms not actually tested in the acceptance record.
