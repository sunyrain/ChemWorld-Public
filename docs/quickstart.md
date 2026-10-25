# 5-minute quickstart

Install the environment, run three operations and receive an instrument reading.
These examples use synthetic models and need no model key or optional upstream solver.

```bash
git clone https://github.com/sunyrain/ChemWorld-Public.git
cd ChemWorld-Public
uv sync --locked --extra dev
uv run --no-sync chemworld capabilities --format json
uv run --no-sync python -m chemworld.examples.user_reward
```

The last command adds two feeds and measures pressure. Its optional reward
is an ordinary user function; changing it does not change the acquired observation.

For a longer experiment with retained material, transfer and a terminal assay:

```bash
uv run --no-sync python -m chemworld.examples.native_surface_research --bulk --mobile --terminal-assay
```

Next, [design an environment](environment-authoring.md), [connect your own agent](environment.md),
or browse the [native workflows](native_domains.md). Python 3.11 and 3.12 are the
documented installation targets. The package is a development source distribution.

Historical tutorial notebooks retain their pinned runtime and recorded outputs;
use the [notebook guide](notebooks.md) for those examples.
