# Current foundation

## Source synchronization {#source-synchronization}

The **23 October 2026** update advances the public foundation from `8a17e23b`
to committed source `963d23f1`. The Python backend now includes:

- **Thermodynamic models:** activity-derived excess and partial-molar properties,
  finite calibration, six UNIFAC variants, Twu/PPR78 and GE cubic mixing rules,
  and a common homogeneous-phase property interface.
- **Pure-fluid Helmholtz:** native derivatives, thermodynamic properties, density
  roots, saturation, flash and bounded thermal/process consumers. Fluid coefficients
  and their provenance are supplied explicitly; no upstream solver is required.
- **Property methods:** temperature and parameter derivatives, correlation fitting,
  scientific conversions and explicit property-selection policies. Data identity
  review remains in progress; this is not complete coverage of every source table.
- **Engineering calculations:** friction and fittings, meters, valves, pumps,
  packed beds/towers, two-phase and open-channel flow, particle distributions,
  equipment/tank geometry and atmosphere/solar methods. Correlations have individual
  application ranges; they do not collectively constitute a general flowsheet solver.
- **Maintenance:** shared composition/action helpers replace repeated code. Golden
  regression expectations have been attributed to their runtime versions; frozen
  historical reports and their original results remain unchanged.

The [native API guide](native_foundation.md) includes runnable inputs and outputs.
The capability website adds examples for Helmholtz, activity thermodynamics,
property derivatives and flow-meter inversion, retaining JSON and readable tables.
The CPU kernel redesign remains a **plan**; the shipped implementation uses the
native Python backend. This synchronization is a development source update,
not a PyPI release, new scientific qualification or performance record.

Windows / Python 3.11 focused integration initially passed **769/771** checks.
The two failures were an optional reference-data setup in the install test and a
SciPy-version-dependent golden value. After separating optional empirical inputs
and retaining version-specific expectations at the original tolerance, the affected
selection passed **17 checks**, with **1 optional reference check skipped**. That
reference install check separately passed on Python 3.12 with reference packages.
These overlapping runs are not added together. The actual sdist-to-wheel install,
all **70 executable documentation examples**, three public-boundary checks, Ruff,
the strict bilingual build and **26 browser checks** pass.

Repository-wide typing remains unfinished: mypy reports 3,378 errors versus 3,255
on the preceding public snapshot in the same environment. This update does not
claim a green full quality pipeline or new cross-platform scientific qualification.

### Previous public synchronization: 22 October

The **22 October 2026** update imports committed foundation work through source
revision `8a17e23b`, advancing the previous `8dc31912` snapshot. It includes:

- Explicit external density/volume sources, including CRC virial data, and
  source-preserving critical-constant inputs.
- Seven second- and two third-virial methods, cross parameters, mixture rules
  and derivatives, plus density/pressure forward and inverse calculations.
- Pure single-phase gas PVT and caloric consumers and closed, fixed-volume
  heating through the common coupled solver. General nonideal mixture calorics
  and open-flow processes remain outside this contract.
- Thevenin RC and two-node thermal models connected to persistent worlds,
  shared-clock acquisition, instrument queues, continuation and replay.

See the [native API](native_foundation.md) and [workflow examples](native_domains.md).
Sources and application ranges must be explicit. Thevenin parameters are empirical;
the model retains separate charge and heat balances rather than inferring chemical species.
This synchronization distributes implementation, tests and comparison runners;
it does not republish raw research runs or claim new material validation.

Windows/Python 3.11 integration selected 217 tests. Initially 215 passed, including
the real sdist-to-wheel installation; two old instrument tests failed identically
on the preceding public source. Their test double and readout-failure expectations
were updated to the existing transaction contract: physical sampling stays paid,
while an invalid readout creates no valid signal or reward. All 12 tests in that
affected file then passed. These overlapping results are reported separately.

Changed Python files pass Ruff. The strict bilingual documentation build and
20 browser checks pass, including the milestone links, both languages and
viewport widths from 320 to 1440 pixels. The website and both READMEs present
six grouped milestones drawn from the main ChemWorld development history.

### Milestone dates and distribution scope

The homepage follows development delivery dates in the main ChemWorld repository,
independently of the public synchronization dates recorded here. It groups August
campaign/resource work, September visualization and phase-resolved workflows,
and October laboratory composition, research records and native scientific models.
The optional Blender laboratory and local research-workbench application are
main-repository prototypes; they are not included in this public distribution.
Public runtime commands and supported models remain described in the guides above.

### Previous public synchronization: 21 October

The 21 October update imports committed foundation work through source revision
`8dc31912`. Uncommitted development changes are outside this snapshot. This is a
development source update, not a new PyPI release or a rerun of frozen scientific evidence.

- Module declarations now drive model codecs, advancement, state ownership and snapshot continuation.
- Shared time supports finite heterogeneous vessel groups, declared open flow systems,
  multiple devices, FIFO measurement queues, maintenance and retained equipment state.
- Native parameter/measurement inputs and finite calibration use public observations.
- Scientific additions include residual/DAE infrastructure, bounded SPMe/DFN and particle
  distributions, expanded property/data methods, selected CALPHAD/ChemSage models,
  and supported SBML/SED-ML experiment contracts. Each retains its documented limits.
- The [new website](https://sunyrain.github.io/ChemWorld-Public/explore/?lang=en)
  adds a capability atlas, searchable index, bilingual domain reader and saved-record player.

Repository tests and runnable examples are distributed with their implementations.
References to development qualification notes describe their original source context;
those research-workspace records are not a new public evidence release. Historical
public evidence below remains tied to its original runtime.

Local synchronization checks on Windows/Python 3.11: test collection found 3,468
cases. The focused 710-case run initially passed 708 and failed two archive-import
checks before newly synchronized modules were added to Git's packaging selection.
After inclusion, both affected checks and the atomic-data archive check passed
(3/3); the packaging progress/error-reporting checks passed separately (6/6).
The strict bilingual documentation build and all 17 browser checks passed, including
five viewport widths, language preservation, observed-only plotting and downloads.
These focused checks do not claim a new full-suite or cross-platform qualification.

The October 2026 development snapshot brings the public runtime up to the current
ChemWorld foundation. Package metadata is `0.5.0.dev0`; this is a source update,
not a new frozen benchmark or a PyPI release.

## What changed

The native session and domain extensions are now available on top of these
existing capabilities. See the [native guide](native_foundation.md) for execution,
state ownership and model-specific examples, and the [extension scope](extension-roadmap.md)
for the delivered models and shared-clock scope.

| Capability | Current public behavior |
| --- | --- |
| Materials and vessels | Create containers, transfer identified samples, select vessels and route material through declared component ports. |
| Component composition | Compile compatible capability networks after dependency, state-owner and resource checks; composition reports use `component-network` and vessel-qualified state owners rather than the old fixed pattern names. |
| Full-process crystallization | Retain phase inventories, crystal populations and seed provenance through sampling, filtration, resuspension and repeated crystallization. Recovery excludes charged seed and uses tracked initial reagent charge. |
| Process control | Configure finite ramp/hold programs, queue stages, set feedback and pause/resume advancement. Gas pressure and purge have finite mass and energy accounting. |
| Continuous streams | Advance finite feed reservoirs, a mixed tank and collected outlets with coupled reaction and thermal dynamics, plus a bounded electrochemical channel. |
| Measurement design | Configure NMR, IR and MS scan count, resolution and dilution, then acquire a paid, sample-consuming synthetic spectrum. Particle-size measurements are available in compatible crystallization tasks. |
| Controlled chemistry | Use coupled weak-acid/precipitation worlds with or without an aqueous ion-pair intermediate, and registered crystallization population/occlusion and purification interventions. |
| Resources and replay | Retain attempted versus committed actions, physical consumption, rejected requests and exactly replayable current-runtime trajectories. |
| Scientific records | Reusable document storage, versioned notebooks and public-history access are included; study-specific campaign runners are not distributed. |

Operation, component and instrument declarations come from the native module
registry; each compiled world and task exposes its compatible subset. Native
physical channels retain units while the historical Gym vector is a compatibility
projection. Existing classic optimization and optional reinforcement-learning
interfaces remain available.

## Try the current runtime

```bash
uv sync --locked --extra dev
uv run --no-sync chemworld tasks list
uv run --no-sync python examples/demo_configurable_spectra.py
uv run --no-sync python examples/demo_minimal_agent.py --output runs/minimal-agent.jsonl
uv run --no-sync chemworld verify --submission runs/minimal-agent.jsonl --tolerance 0
uv run --no-sync chemworld lab --no-browser
```

Current source checks use `uv run --no-sync pytest -q`. The targeted suite covers
operation failures, resource accounting, exact replay, multi-vessel routing,
control, streams, spectra, crystallization and retained historical evidence.
Optional external reference checks are reported as skipped when their reference
installation is absent. Install/build checks are described in
[contributing](https://github.com/sunyrain/ChemWorld-Public/blob/main/CONTRIBUTING.md).

The preceding foundation synchronization on Windows AMD64 completed with 466 tests passed and
1 skipped (the optional external IDAES checkout was unavailable), plus Ruff,
mypy and the strict bilingual documentation build. Wheel and source-distribution
installs were checked independently on Python 3.11.15 and 3.12.10, including all
15 task initializations, packaged resources, Student Lab and exact replay of the
offline examples. These are software integration checks, not a new scientific
qualification of the historical reports.

The native extension synchronization adds 21 further installed examples, covering
667 committed operations per archive. Wheel and sdist both passed on Windows
Python 3.12.10 with ten upstream solver/reference packages absent, alongside the
eight surface variants, four equipment-reuse variants, task binding, saved
continuation and JSONL replay. The Python 3.11.15 integration first found two
failures among 721 cases: an inconsistent temperature fixture and single-phase
inventory roundoff. Both were corrected without relaxing assertions; all 147
affected flash, column, concentration, measurement and public-entry checks then
passed. The unchanged checks and the follow-up overlap and are not added into a
new denominator.

The latest integrated examples exercise seven continuous workflows through
Python, Gym, CLI and Agent interfaces, with sequential and keyed detector noise
and fresh-process snapshot continuation. Their scope includes a shared heater,
three membrane process/rinse cycles, mineral chemistry, competing analyzer jobs,
full-reflux distillation, repeated crystallization and binary solid phases.
See the [workflow guide](native_domains.md) for commands and model limits.

On Windows Python 3.11.15, the final public selection passed 81/81 checks, including
56 cross-interface histories and replays and 14 fresh-process continuations.
Shared-material regressions passed 71/71. This integration found and corrected
order-sensitive finite-feed summation during JSON replay; the complete affected
selection then ran again from its first case. Independent wheel and sdist checks
both passed on Windows Python 3.12.10, each running all seven default-noise workflows
and their new-process replays. These overlapping checks are reported separately.

## Model scope

Spectra are calibrated synthetic reporter signals, not predictions of arbitrary
molecular structures. Local time remains the default: operating a selected vessel
does not advance other vessels. An explicitly configured shared laboratory clock
advances all enabled vessels and schedules equipment queues and delayed readings;
its supported combinations are listed in the [shared laboratory guide](native_shared_lab.md).
The gas plenum does not dissolve or react with the liquid. The mixed-tank electrical
channel is an idealized two-electron model; the separate native battery domain uses
bounded SPM, SPMe/DFN and equivalent-circuit models with explicit parameter sets. World-noise shift modes can include a simultaneous
physical-parameter change; they should not be treated as pure noise interventions.

## Historical reproduction

The checked-in evidence reports, original manifest and tutorial outputs remain
unchanged. They describe their original runtime. The home-page animation also
shows a historical trace. Use the pinned notebook links or check out
[`db98e800`](https://github.com/sunyrain/ChemWorld-Public/tree/db98e8002fac0512ed9084a3c8c27985cb9fc405)
before running `scripts/verify_release.py`. That script verifies the frozen whole
tree; it is not a development preflight for current main.

Current chemistry does not reproduce every old numerical outcome. In particular,
the original midpoint partition-world comparison still executes and replays,
but no longer meets its old divergence threshold. Current tests retain that
negative outcome instead of relabelling the historical qualification as current.

## Synchronization provenance

Runtime source was selected from ChemWorld commit
`5156c5c137c29c945759f24f13595eef7ef3cda0` (19 October 2026). Public-specific CLI,
provider adapters, documentation and frozen-evidence consumers are retained;
manuscripts, private evaluation inputs, study runners and provider payloads are
not copied. Version metadata, type annotations and public CLI error handling are integrated here.
The later native foundation updates include the domain implementations and
cross-domain integration delivered on 20 October 2026. Finite feed normalization
and accounting use stable summation so JSON key ordering does not change the
material packet or its subsequent replay.
The [extension roadmap](extension-roadmap.md) distinguishes delivered capabilities
from further scientific extensions.
