# Native integer stochastic chemistry

Declare `ReactionNetworkSpec(..., stochastic={"volume_L": 0.01,
"temperature_K": 298.15})` and pass integer counts to
`chemworld.world.stochastic_world.stochastic_world(network, counts, seed=7)`.
Use the returned composition with `ChemWorldRuntime`, `reset()`, then
`execute({"operation": "wait", "duration_s": 10})`. This is the same runtime,
transaction and accounting path as the deterministic worlds.

The default algorithm is `gillespie-direct-v1`. Select
`"algorithm": "gibson-bruck-next-reaction-v1"` in the same stochastic declaration
to use exact dependency-updated reaction clocks. Both methods use the same
`stochastic_world`, actions and `advance_stochastic` API. Pure callers also pass
the chosen algorithm to `initial_continuation(seed, algorithm=...)`.

Each elementary direction is a separate irreversible `mass_action` reaction
with `parameters={"k": ...}`. A↔B is two channels, not a potentially negative
net propensity. The rate constants retain the network's mol/L and seconds
convention. For molecularity m, the stochastic constant multiplying combinations
is `k*(N_A*V)**(1-m)*prod(factorial(r_i))`. Non-integer stoichiometry, custom
orders, activities, continuous controls, implicit ODE/SSA mixtures and molecularity
above three are outside this finite implementation.

The phase ledger owns all material. Its mol values represent exact integer
molecule counts divided by the exact Avogadro constant; conversion allows only
four floating-point ULPs, not fractional dosing. Counts are bounded by 2**40
to keep integer resolution unambiguous. Process continuation belongs to the
vessel's ProcessLedger and contains no second inventory: PCG64 state, process
time, event count and a rate/inventory signature. Direct stores one absolute
pending event; next-reaction stores each channel's absolute event time, cached
propensity and any dormant channel's remaining internal Poisson hazard. Public
state projections omit it. A real instantaneous intervention invalidates its
pending event while keeping the process RNG stream; a read does neither.

`snapshot().to_dict()` / `ChemWorldRuntime.from_snapshot(payload)` preserves the
pending clocks and exact continuation, including in another process.
The pure `advance_stochastic` API also returns individual event times/identities
for numerical verification. A numerical event budget rolls the entire candidate
advance back, rather than reporting chemical failure or charging physical time.
The common runtime still applies its declared request/measurement resource rules.
`StochasticAdvance.work` and runtime `stochastic_work` expose propensity
evaluations and random draws. Sparse dependencies can reduce work per event;
this is not a universal wall-time speedup or an approximate time-leaping method.

Only constant volume and temperature are supported. There is no claim of a
caloric/energy model or fitted biochemical accuracy. Fractional destructive
instruments are rejected at compilation; dedicated integer sampling and the
two presets are available through [biochemical worlds](native_biochemical.md). No COPASI or other
upstream simulator is loaded by the runtime.

Development qualification: `uv run --no-sync pytest -q tests/test_stochastic_kinetics.py`;
`uv run --no-sync python scripts/validate_native_ssa.py --output /tmp/ssa-new-attempt.json`.
The latter refuses to overwrite a previous attempt. The fixed 4000-run design,
all outcomes and statistical criteria are in the world_foundation workstream.

## Additional stochastic algorithms

The snapshot also includes explicit-partition hybrid ODE/jump dynamics and a bounded
adaptive tau-leap implementation. Each has its own continuation and validity contract;
the direct/next-reaction description above does not make tau-leaping exact.
Use `chemworld.examples.stochastic_tau_leap.tau_world(seed=51, copies=10000)`
with `ChemWorldRuntime` for the packaged approximate model. No automatic arbitrary
partitioning or spatial kMC is claimed.
