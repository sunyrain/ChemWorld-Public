# Evidence

> Current main: [October foundation update](current-foundation.md). Existing figures and numerical evidence below describe the frozen August runtime.

ChemWorld v0.3.0 retains the four frozen, sanitized evidence groups from v0.2.0. The presentation layer adds no new benchmark experiment.

| Evidence group | Frozen result | What it supports |
| --- | ---: | --- |
| Composition qualification | 64/64 units; 1,786/1,786 recipes | Registered world composition and reference execution |
| Deterministic use cases | 8/8 cases; 89 submitted actions | Lifecycle, failure, resource and exact-replay semantics |
| Controlled world forks | 6 pairs; 24 traces | Single-component world changes under fixed public experiments |
| Agent instrument use | 1 lifecycle; 15 committed actions | Persistent use of the public action and instrument interface |

## Why the visual is auditable

`scripts/build_readme_visuals.py` reads only the sanitized compressed reports. It regenerates the README SVG, the interactive site payload and the detailed static tables. `--check` fails when any checked-in output is stale.

## Provenance and sanitization

Every compressed report binds its original SHA-256, sanitizer version, public protocol and a digest of removed metadata. Canonical JSON and gzip `mtime=0` make repeated sanitization reproducible.

See the [complete report map](https://github.com/sunyrain/ChemWorld-Public/blob/main/evidence/README.md), [representative behavior tables](representative-behavior.md) and [release manifest](https://github.com/sunyrain/ChemWorld-Public/blob/main/release/manifest.json).

!!! warning "Finite claim boundary"
    Passing these checks does not establish universal chemical fidelity, general agent intelligence or safe transfer to real laboratory practice.
