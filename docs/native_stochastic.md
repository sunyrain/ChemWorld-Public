# Native integer stochastic chemistry

Declare `ReactionNetworkSpec(..., stochastic={"volume_L": 0.01,
"temperature_K": 298.15})` and pass integer counts to
`chemworld.world.stochastic_world.stochastic_world(network, counts, seed=7)`.
Use the returned composition with `ChemWorldRuntime`, `reset()`, then
`execute({"operation": "wait", "duration_s": 10})`. This is the same runtime,
transaction and accounting path as the deterministic worlds.

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
vessel's ProcessLedger and contains no second inventory: PCG64 state, absolute
pending event, process time, event count and a rate/inventory signature. Public
state projections omit it. A real instantaneous intervention invalidates its
pending event while keeping the process RNG stream; a read does neither.

`snapshot().to_dict()` / `ChemWorldRuntime.from_snapshot(payload)` preserves the
pending event and exact continuation, including in another installed process.
The pure `advance_stochastic` API also returns individual event times/identities
for numerical verification. A numerical event budget rolls the entire candidate
advance back, rather than reporting chemical failure or charging physical time.
The common runtime still applies its declared request/measurement resource rules.

Only constant volume and temperature are supported. There is no claim of a
caloric/energy model or fitted biochemical accuracy. Fractional destructive
instruments are rejected at compilation; dedicated integer sampling and the
two presets are available through [biochemical worlds](native_biochemical.md). No COPASI or other
upstream simulator is loaded by the runtime.

Development qualification: `uv run --no-sync pytest -q tests/test_stochastic_kinetics.py`;
`uv run --no-sync python scripts/validate_native_ssa.py --output /tmp/ssa-new-attempt.json`.
The latter refuses to overwrite a previous attempt. The fixed 4000-run design,
all outcomes and statistical criteria are in the world_foundation workstream.
