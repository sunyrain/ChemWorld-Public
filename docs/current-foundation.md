# Current foundation

## Latest development — 25 October 2026 {#development-progress}

The main development repository has advanced beyond the current public package.
The additions below are committed development work; they are not yet included in
the public installation described in the next section.

| Area | Latest development | Scope |
| --- | --- | --- |
| Process networks | Finite graphs connect explicit feeds, splitters, mixers, heat exchange, separation and multiple recycle connections. Inventory and trajectory-control consumers use the same world state. | Boundary and initialization fixes are being checked; the complete fixed-scope qualification remains in progress. |
| Battery observations | Half-cell DFN readouts can be serialized and restored with explicit model and field checks. | Measurement records remain derived observations; this does not establish completion of the whole half-cell programme. |
| Mechanism interoperability | Supported imported Cantera gas networks can be exported with their original species, reaction and unit declarations. | Complete source declarations are required; unsupported or incomplete networks are rejected. |
| Property fitting | Physical sensitivity diagnostics now require successful finite-difference evaluations of the original model. | An optimizer penalty alone cannot establish parameter identifiability; failed trials remain visible. |

The development board currently records **39 of 56 parent tasks complete**, five
active and twelve pending. The separate CPU programme has **4 of 31 tasks complete**.
Task completion and software-reference comparisons do not establish universal
material accuracy. Full scientific-model execution remains on Python.

## Available in the public source {#source-synchronization}

The public Python runtime still follows the 25 October source snapshot
[`823b6b01f`](https://github.com/sunyrain/ChemWorld/commit/823b6b01f).
Choose apparatus and action/resource limits with [environment authoring](environment-authoring.md),
then [connect an agent and define an optional reward](environment.md).
The three-action user reward example is included.

| Area | Publicly available scope |
| --- | --- |
| Thermodynamics and interfaces | Molecular/association/multifluid models, multiphase caloric flash, finite-process consumers and planar interface calculations |
| Reactions and biochemical systems | Reactor networks, energy transport, imported mechanisms, SBML/COPASI interoperability and bounded hybrid dynamics |
| Water and electrochemistry | Finite membrane/treatment flows, activated sludge, battery parameter and hysteresis families, selected half-cell and thermal workflows |
| Environment use | User-defined objectives and rewards, persistent state, resource limits, observations and replay |
| Optional CPU development | Semantic declarations, resource/history infrastructure and a C++20 ABI/binding skeleton; full numerical-model migration remains in progress |

The installed registry declares **31 modules, 38 component types, 52 operations and
30 instruments**. These are software declarations, not independent scientific domains
or capabilities available in every world. The website reads these counts from the
public registry when it builds; the capability map remains an editorial guide.

Use the Python, CLI or Agent interfaces. The old Student Lab command is retired;
Research Workbench and Blender Lab remain internal prototypes.

## Verification and historical evidence

At the public runtime synchronization, **82 executable API examples passed**, and
wheel and source-distribution installations passed independently on Windows Python
3.11.15, including native workflows and fresh-process replay. These results belong
to that public runtime; they do not qualify the newer development additions above.
The [record of the public synchronization](https://github.com/sunyrain/ChemWorld-Public/blob/c21fe86cb739b9bed021069ba98921b868576df4/docs/current-foundation.md)
retains the exact test scopes and denominators.

The homepage, capability atlas and domain guides share the original website interface
at `/`. Documentation opens in the integrated reader with domain navigation, section
links and copyable code. Old `/explore/` and guide addresses still work.
This update changes website content, not the runtime, PyPI release or paper results.

Frozen notebooks and trace demonstrations retain their original runtime versions;
see [Evidence](evidence.md) and [Notebooks](notebooks.md). Current physics is not a
promise of numerical compatibility with those historical records. Complete upstream
qualification, wider platform coverage and most CPU execution tasks remain ongoing.
