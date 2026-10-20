# ChemWorld

**Programmable chemical worlds for autonomous experimentation.**

[Quick start](#quick-start) · [Native API](docs/native_foundation.md) ·
[Model ranges](docs/extension-roadmap.md) · [Connect an agent](docs/agents.md) ·
[中文基座说明](docs/extension-roadmap.zh.md)

ChemWorld is a simulation foundation for agents that choose and carry out experiments.
Researchers define the physical world, available equipment and task; agents add material,
heat, separate, measure and decide what to do next. Each operation changes persistent
state, consumes resources and produces a record that can be inspected and replayed.

The current development branch (`0.5.0.dev0`) brings chemical reactions, phase processes,
electrochemistry, membranes and stochastic biochemistry into one experimental runtime.
Python, Gym, CLI and Agent entry points share the same execution and accounting.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
Python 3.11 and 3.12 are supported; these examples run without model credentials or
upstream chemistry solvers.

```bash
git clone https://github.com/sunyrain/ChemWorld-Public.git
cd ChemWorld-Public
uv sync --locked --extra dev
uv run --no-sync python -m chemworld.examples.native_surface_research --bulk --mobile --terminal-assay
```

This complete experiment feeds finite stocks, couples bulk and surface reactions to a
heater, measures gas composition, transfers material and finishes with an assay.
Its output includes each operation's outcome, physical duration and native readings.

For a browser workbench, run `uv run --no-sync chemworld lab`.
[Student Lab](#student-lab-no-model-required) provides a local interface to the existing
tutorial tasks.

## One foundation across experimental domains

- **Compose worlds and tasks separately.** Declare materials, vessels, devices and
  physical models, then bind goals, allowed operations, budgets and ending conditions.
- **Reuse equipment and state.** A heater acts through a thermal connection; material
  transfers follow declared phase and ownership rules. Calibration, catalyst sites,
  membrane fouling and battery degradation persist with their owning entities.
- **Advance coupled physics together.** Compatible processes share a solver and energy
  balance. An optional [shared laboratory clock](docs/native_shared_lab.md) adds equipment
  queues, simultaneous aging of independent vessels and delayed analyzer results.
- **Measure physical quantities.** Native observations retain units, quality and sampling
  times. Gym's fixed scalar view is a compatibility projection of those observations.
- **Keep execution accountable.** Transactions check declared exchanges and conservation
  before commit. Outcomes distinguish invalid requests, numerical failure and physical
  stops; snapshots and trajectories support continuation and replay.

| Domain | Available native models | Start here |
| --- | --- | --- |
| Reactions and catalysis | Bulk kinetics, gas mechanisms, finite surface sites, reversible two-site interactions and coupled heat | [Surface example](src/chemworld/examples/native_surface_research.py), [mechanism import](docs/native_mechanisms.md) |
| Thermodynamics and phase processes | State-dependent properties, TP/PH flash, liquid–liquid equilibrium, finite distillation, concentration, adsorption and crystallization | [Physical models guide](docs/native_foundation.md#properties-at-the-current-state) |
| Transport and separation | Finite tanks, pressure/power-limited pumps, heat exchange, reverse osmosis, persistent fouling and rinsing | [Transport example](src/chemworld/examples/native_liquid_transport.py), [membrane example](src/chemworld/examples/native_membrane_fouling.py) |
| Electrochemistry and batteries | Conservative diffusion, finite electrodes, single-particle batteries, CC/CV/rest cycling, thermal response and SEI | [Battery example](src/chemworld/examples/native_battery_degradation.py) |
| Water and minerals | Dilute carbonate equilibrium, finite calcite dissolution/precipitation and phase-owned surface geometry | [Mineral example](src/chemworld/examples/native_mineral_research.py) |
| Stochastic biochemistry | Gillespie events, reversible/enzyme networks, integer inventories and separate process/measurement randomness | [Biochemical worlds](docs/native_biochemical.md) |
| Solid phases | Finite binary substitutional equilibrium and heating/cooling paths | [Solid-solution example](src/chemworld/examples/native_solid_solution.py) |
| Instruments and scheduling | Programmed chromatography, persistent instrument history, shared heater/analyzer queues and timed readouts | [Shared laboratory example](examples/native_shared_lab.py) |

Each model has explicit equations and operating limits; combinations must satisfy the
declared interfaces. See [model ranges and extension directions](docs/extension-roadmap.md)
for the implemented scope and optional upstream references.

## Use the native session

```python
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.surface_world import surface_world_spec

session = ChemWorldRuntime(composition=surface_world_spec())
session.reset(seed=12)
session.execute({
    "operation": "add_component", "component": "CO-feed", "amount_mol": 2e-5,
})
reading = session.execute({"operation": "measure", "instrument": "gas_pressure"})
for channel in reading.observation.channels:
    print(channel.channel_id, channel.value, channel.unit, channel.quality)
session.close()
```

Use `ChemWorldEnv(composition=spec)` for Gym, JSON actions through `chemworld render`
for the CLI, or `run_agent(..., composition=spec.to_dict())` for a custom policy.
The [native session guide](docs/native_foundation.md#use-the-same-session-through-other-entry-points)
covers these paths, physical observations, snapshot continuation and trajectory replay.
The [agent guide](docs/agents.md) covers offline policies and optional live-model adapters.

## Student Lab (no model required)

Launch the animated browser workbench after installing the package:

```bash
uv run --no-sync chemworld lab
```

The animated workbench runs on `127.0.0.1` and uses the same session and action schemas.
Its tutorial catalog exposes public action effects and observations without a model
account or API key. The domain examples above provide the broader native API coverage.

## Develop and extend

New capabilities enter through ChemWorld's module declarations, state ownership,
physical processes and instrument contracts. Reference tools inform equations and
bounded comparisons; production simulation uses the native implementation.
See the [extension guide](docs/extension-roadmap.md) and
[contribution guide](CONTRIBUTING.md).

Run the focused foundation checks:

```bash
uv run --no-sync pytest tests/test_domain_modules.py tests/test_native_foundation_end_to_end.py tests/test_lab_clock.py -q
```

Full cross-domain combination and installation acceptance is still in progress.
Current development source and the fixed historical runtime below have separate
validation records.

<details>
<summary>Earlier tutorials and frozen evidence</summary>

The tutorials require no provider key and remain pinned to `v0.3.0`:

| Notebook | What you do | Run |
| --- | --- | --- |
| `01_first_experiment` | Inspect a Reaction-to-Assay contract, validate actions, use HPLC and complete a final assay | [Open in Colab](https://colab.research.google.com/github/sunyrain/ChemWorld-Public/blob/v0.3.0/notebooks/01_first_experiment.ipynb) |
| `02_reaction_to_purification` | Continue through extraction, wash, drying and concentration | [Open in Colab](https://colab.research.google.com/github/sunyrain/ChemWorld-Public/blob/v0.3.0/notebooks/02_reaction_to_purification.ipynb) |
| `03_controlled_world_change` | Hold the public experiment fixed while one registered world component changes | [Open in Colab](https://colab.research.google.com/github/sunyrain/ChemWorld-Public/blob/v0.3.0/notebooks/03_controlled_world_change.ipynb) |

Checked-in notebook outputs are deterministic public demonstrations, not benchmark results or optimized laboratory procedures.

### Frozen evidence

The following results and illustration belong to the frozen August runtime. They remain
unchanged historical evidence, not a qualification of current main. Reproduce them with the
[pinned snapshot](https://github.com/sunyrain/ChemWorld-Public/tree/db98e8002fac0512ed9084a3c8c27985cb9fc405);
the notebook links above also stay pinned:

![Evidence-derived agent lifecycle, rollback recovery and controlled world changes](docs/assets/representative-agent-and-world-change.svg)

- 64/64 registered task–world units and 1,786/1,786 reference recipes completed.
- 52/52 coverage-generated compositions completed, including 8/8 non-reference reaction–distillation worlds.
- Eight deterministic lifecycles checked all 89 submitted actions: 88 committed and one planned rollback.
- Six controlled parent–child world pairs produced 24 provider-free traces.
- One independent agent completed a 15-action public-instrument lifecycle.

These are finite software-model qualification results. They do not establish universal chemical fidelity, general agent intelligence or transfer to a physical laboratory.

The original whole-tree release verifier belongs to the pinned August checkout;
see [evidence documentation](docs/evidence.md). Frozen illustration consistency
can be checked with `uv run --no-sync python scripts/build_readme_visuals.py --check`.

</details>

## Release boundary

This repository contains the current development runtime, public protocols, historical sanitized evidence, pinned tutorial outputs, documentation, examples and focused tests. It excludes manuscript files and assets, development matrices, draft/interim/pilot results, raw provider responses, provider session identifiers, private evaluator configuration, credentials and study-specific development artifacts.

See [CITATION.cff](CITATION.cff) for citation metadata. The software is available under the [MIT License](LICENSE).
