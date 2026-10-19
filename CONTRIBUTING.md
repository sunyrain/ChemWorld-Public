# Contributing

Develop against current main with the locked environment:

```bash
uv sync --locked --extra dev --extra docs
uv run --no-sync ruff check src tests scripts examples
uv run --no-sync mypy src/chemworld
uv run --no-sync pytest -q
uv run --no-sync mkdocs build --strict
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
