# Historical applications

ChemWorld is a programmable physical environment. Rankings, preset policy factories
and operation-by-operation LLM adapters no longer define the default entry point.
Start with the [programmable environment](environment.md).

For new environments, use the [current authoring guide](environment-authoring.md).
The retained [Work I world-authoring contract](https://github.com/sunyrain/ChemWorld/blob/main/docs/world-authoring-contract.md)
is a frozen controlled-fork protocol. Its single-variable interventions, certificates and
freeze procedures do not govern general environment development. Its original path and
contents remain unchanged for bound historical evidence; the retained v1 composition
contract likewise describes its first-paper research scope.

| Historical feature | Explicit archive entry |
| --- | --- |
| run, evaluate, leaderboard, suite, baselines, submission and old release commands | `python -m chemworld.legacy.cli --help` |
| Ranking aggregation | `chemworld.legacy.leaderboard` |
| Preset policy factory | `chemworld.legacy.agent_registry.make_agent` |
| Eleven baseline/adapter modules | `chemworld.legacy.agents`: bo, greedy, lhs, random, event, rl, llm, live_llm, diagnostic_live_llm, documented_codex_g2, structured_g2 |

These archived applications remain distributed for explicit research consumers.
They are not recommended environment APIs or default scoring methods. Old imports
are not silently redirected. The shared runner, free-research MCP tools, notebooks
and used research implementations remain available. Runtime/trajectory fields with
existing consumers retain their semantics in this cleanup.

Previous documentation and source are in Git commit
`3e5fe32598d96c4ab2b83fcfb26b48939c26c967`. Frozen experiments must still use their
original bound version and data: these archive entry points do not recreate old
physics. No historical experiment was rerun or recertified by this cleanup.
