# Current foundation

## 25 October 2026 update

The public Python foundation now follows committed development source
[`823b6b01f`](https://github.com/sunyrain/ChemWorld/commit/823b6b01f).
Start with [your own environment](environment-authoring.md): choose apparatus,
declare action and resource limits, connect an agent, and define a reward only if needed.
The three-action [user reward example](environment.md) ships with the package.

| Area | Current addition |
| --- | --- |
| Thermodynamics and interfaces | Further molecular/association/multifluid models, multiphase caloric flash, finite-process consumers and planar interface calculations |
| Reactions and biochemical systems | Reactor networks, coupled energy transport, mechanism constraints, SBML/COPASI interoperability, controlled hybrid dynamics and event continuation |
| Water and electrochemistry | Finite membrane/treatment flows, activated sludge, battery parameter and hysteresis families, additional half-cell and thermal workflows |
| Environment use | User-defined objectives and rewards, environment authoring, explicit legacy application entry points, persistent state and replay |
| Optional CPU development | Semantic declarations, resource/history infrastructure and a C++20 ABI/binding skeleton; full numerical model execution remains on Python |

The installed registry declares **31 modules, 38 component types, 52 operations
and 30 instruments**. These are interface declarations, not independent domains
or capabilities available in every world. Each model and composition has its own limits.
The development board records **39 of 56 parent tasks complete**, with 4 claimed
and 13 pending; the separate CPU programme has 4 of 31 tasks complete.
Task completion is not an independent material-validation count.

The old Student Lab command is retired. Research Workbench and Blender Lab remain
internal prototypes; the public distribution provides Python, CLI and agent interfaces.
Frozen notebooks, traces and their evidence retain their original runtime versions.
This update does not release a new PyPI version or reproduce the paper experiments.

### Integration checks for this update

The current public source runs all **82 executable API examples**. Focused environment,
runtime, observation, snapshot and website checks pass; one stale README-layout
assertion was updated and its eight-test documentation group passes. **10,592 tests
collect without errors**; this is not a full-suite pass. Optional independent-reference
tests require their separately installed upstream libraries.

The strict bilingual documentation build passes. All **73 HTML pages and 2,013 local
links** were checked, with no missing targets. Browser connections were unavailable
for this update, so no new visual/browser-interaction pass is claimed.

Both wheel and source-distribution installations pass independently on Windows
Python 3.11.15. Each covers fifteen task initializations, ten native workflows, six
documentation commands, user-defined rewards, retained equipment and fresh-process
replay, with optional upstream solvers absent. This does not establish new Linux or
macOS installation coverage.

## Historical evidence

Earlier installation and qualification results retain their original execution
versions. See the [24 October public record](https://github.com/sunyrain/ChemWorld-Public/blob/096de6de32be5b016ca481dcd07a91ca0796f9c8/docs/current-foundation.md)
for previous source snapshots and their exact scopes. Frozen notebook and trace
demonstrations remain available through [Evidence](evidence.md) and [Notebooks](notebooks.md).
Current physics is not a promise of numerical compatibility with those historical records.

Repository-wide typing, full upstream-library qualification, broader platform
installation coverage and most CPU execution tasks remain separate ongoing work.
