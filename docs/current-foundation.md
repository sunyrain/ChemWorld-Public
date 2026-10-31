# Current foundation

## Current maintenance — 31 October 2026 {#development-progress}

Current work consolidates the reusable product, its documentation and installation
paths. Retired study runners and their dedicated evaluations are excluded from the
product. The current paper's notebook organizes its study through the installed
public API; historical results retain their original execution versions and data.

New foundation expansion tasks are paused. Delivered models remain available within
their documented scope. The supported full scientific-model execution path remains
Python; optional CPU work does not establish complete numerical-model coverage.
Software-reference comparisons do not establish universal material accuracy.

## Available in the public source {#source-synchronization}

The [public runtime source](https://github.com/sunyrain/ChemWorld-Public/tree/main/src/chemworld)
is synchronized from the maintained product. Shared technical guides follow the
same source; the website retains its own navigation and presentation.
Choose apparatus and action/resource limits with [environment authoring](environment-authoring.md),
then [connect an agent and define an optional reward](environment.md).
The three-action user reward example is included.

| Area | Publicly available scope |
| --- | --- |
| Thermodynamics and interfaces | Molecular/association/multifluid models, multiphase caloric flash, finite-process consumers and planar interface calculations |
| Reactions and biochemical systems | Reactor networks, energy transport, imported mechanisms, SBML/COPASI interoperability and bounded hybrid dynamics |
| Water and electrochemistry | Finite membrane/treatment flows, activated sludge, battery parameter and hysteresis families, selected half-cell and thermal workflows |
| Environment use | User-defined objectives and rewards, persistent state, resource limits, observations and replay |
| Optional CPU development | Semantic declarations, resource/history infrastructure and a C++20 ABI/binding skeleton; full numerical-model coverage is not established |

The website reads module, component, operation and instrument counts from the public
registry when it builds. These are software declarations, not independent scientific
domains or capabilities available in every world. The capability map is a reading guide.

Use the Python, CLI or Agent interfaces. The old Student Lab command is retired;
Research Workbench and Blender Lab remain internal prototypes.

## Verification and historical evidence

At the 25 October public runtime synchronization, **82 executable API examples passed**, and
wheel and source-distribution installations passed independently on Windows Python
3.11.15, including native workflows and fresh-process replay. These results belong
to that snapshot; they are not a complete validation of every subsequent product change.
The [record of the public synchronization](https://github.com/sunyrain/ChemWorld-Public/blob/c21fe86cb739b9bed021069ba98921b868576df4/docs/current-foundation.md)
retains the exact test scopes and denominators.

The homepage, capability atlas and domain guides share the original website interface
at `/`. Documentation opens in the integrated reader with domain navigation, section
links and copyable code. Old `/explore/` and guide addresses still work.
This page update changes documentation, not the runtime, PyPI release or paper results.

Frozen notebooks and trace demonstrations retain their original runtime versions;
see [Evidence](evidence.md) and [Notebooks](notebooks.md). Current physics is not a
promise of numerical compatibility with those historical records. Complete upstream
qualification, wider platform coverage and full CPU execution remain unestablished;
the paused expansion programme is not an active delivery queue.
