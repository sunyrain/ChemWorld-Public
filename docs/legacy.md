<!-- Generated from https://github.com/sunyrain/ChemWorld/blob/main/docs/legacy.en.md; edit the upstream guide. -->

# Historical applications

ChemWorld is a programmable physical environment. Rankings, preset policy factories
and operation-by-operation LLM adapters no longer define the default entry point.
Start with the [programmable environment](environment.md).

For new environments, use the [current authoring guide](environment-authoring.md).
The old Work I controlled-fork protocol describes frozen research. Its single-variable
interventions, certificates and freeze procedures do not govern general environment
development. The retained [v1 composition contract](world-composition-contract.md)
likewise describes its first-paper research scope.

The old Work I, Work II, S0/G2, mechanism-adaptation and Experiment 1 applications,
including their study runners and evidence pipeline, have been removed from the
current product. Current NCS organization and analyses live beside its notebook.
Their bound guides, configurations and original implementations remain historical
evidence, not maintained entry points.

| Historical feature | Explicit archive entry |
| --- | --- |
| run, evaluate, leaderboard, suite, baselines and submission | `python -m chemworld.legacy.cli --help` |
| Ranking aggregation | `chemworld.legacy.leaderboard` |
| Preset policy factory | `chemworld.legacy.agent_registry.make_agent` |
| Eleven baseline/adapter modules | `chemworld.legacy.agents`: bo, greedy, lhs, random, event, rl, llm, live_llm, diagnostic_live_llm, documented_codex_g2, structured_g2 |

The compatibility interfaces listed above remain distributed for existing consumers;
retired study runners do not. These interfaces are not recommended environment APIs
or default scoring methods. Old imports
are not silently redirected. The generic agent runner, document workspace, public
history and versioned notebooks remain available. The study-specific MCP adapter
and its typed-belief protocol are archived with their research consumers. Runtime/trajectory fields with
existing consumers retain their semantics in this cleanup.

Frozen experiments must still use their
original bound version and data: these archive entry points do not recreate old
physics. No historical experiment was rerun or recertified by this cleanup.
