# Contributing

The maintained product is developed in
[ChemWorld](https://github.com/sunyrain/ChemWorld) and exported into this repository.
Changes to the website, documentation and frozen public evidence are kept separate
from product exports. Propose product changes upstream so that the two copies do
not evolve independently. The development version is defined once in
`src/chemworld/__init__.py`; Hatch reads it when building package metadata.

Retired study runners and their dedicated tests no longer ship with the product.
The notebook tutorials pinned to v0.3.0 and the frozen evidence remain historical
examples; use their declared release for historical reproduction.

Develop against current main with the locked environment:

```bash
uv sync --locked --extra dev --extra docs
uv run --no-sync ruff check src tests scripts examples
uv run --no-sync mypy src/chemworld
uv run --no-sync pytest -q
uv run --no-sync mkdocs build --strict
uv run --no-sync python scripts/build_explorer_site.py --output site/explore
```

Use focused tests while implementing. For packaging changes, run
`uv run --no-sync python scripts/smoke_test_wheel.py --python 3.12` to build both
wheel and sdist and install them outside the checkout with locked dependencies.
This checks the installed CLI, resources, Student Lab and replay examples.
Run the corresponding check on other platforms before claiming their validation.

Preserve material/energy accounting, rejected requests and exact replay. A model
change may invalidate historical numerical evidence; do not overwrite that
evidence or regenerate its manifest as a development preflight. A source sync is
not a new qualification campaign. See [current foundation](docs/current-foundation.md).

The original `scripts/verify_release.py` and `scripts/build_release_manifest.py`
belong to the frozen v0.4.0 checkout, linked in that guide. Current tests separately
verify that retained evidence bytes and denominators are unchanged.

Keep credentials, provider payloads, personal sessions, private evaluation inputs
and generated run directories out of Git. New scientific experiment blocks need
a concise question, coverage, measurements and fixed acceptance rules before execution.
Proposed external-model work starts with documented equations and interface
behavior, then bounded independent implementations checked against reference
calculations. Preserve licensing and provenance for any reused code or data.

The static explorer source lives in `web/`; its guide prose lives in `docs/`.
Chinese native guides marked as preview translations should be checked against
the English source when editing scientific terms. Build outputs stay in ignored
`site/` and are published only on the separate `gh-pages` hosting branch.
The player exports observed public fields from the committed historical test
fixture. It does not execute experiments or provide exact-replay submissions.

The capability index embeds API signatures and executed examples from
`scripts/capability_reference_recipes.py`. When changing these examples or their
public interfaces, regenerate the committed `web/chemworld-api-reference.js`:

```bash
uv run --no-sync python scripts/build_capability_reference.py --report /path/outside/repo/api-docs.json
```

The command checks every documented recipe, records all failures in the requested
report, and publishes the asset only when execution and capability coverage pass.
Use `--only <recipe>` to check one example without changing the asset. These are
synthetic software demonstrations, not scientific qualification runs. Ordinary
site builds copy the committed asset and do not rerun experiments. The shared
bilingual presentation lives in `web/chemworld-api-docs.js` and its matching CSS.
