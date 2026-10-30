<!-- Generated from https://github.com/sunyrain/ChemWorld/blob/main/docs/native_stochastic.md; edit the upstream guide. -->

# Native stochastic and mixed chemistry

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
the direct/next-reaction cost comparison does not establish a universal wall-time speedup.

Only constant volume and temperature are supported. There is no claim of a
caloric/energy model or fitted biochemical accuracy. Fractional destructive
instruments are rejected at compilation; dedicated integer sampling and the
two biochemical presets are the separate NATIVE-EXT-37 task. No COPASI or other
upstream simulator is loaded by the runtime.

Development qualification: `uv run --no-sync pytest -q tests/test_stochastic_kinetics.py`;
`uv run --no-sync python scripts/validate_native_ssa.py --output /tmp/ssa-new-attempt.json`.
The latter refuses to overwrite a previous attempt. The fixed 4000-run design,
all outcomes and statistical criteria are in the world_foundation workstream.

The next-reaction development block completed 49,152 trajectories across both
methods; all 29 units and 353 fixed distribution, continuation, world and work
comparisons passed. Its fixed design and statistical resolution are recorded in the
[51A delivery](https://github.com/sunyrain/ChemWorld/blob/main/workstreams/world_foundation/reports/convergence/51A/README.md).

## Approximate adaptive tau-leaping

The same `StochasticSpec` also accepts
`algorithm="adaptive-explicit-tau-leap-v1"` with an explicit `TauLeapPolicy`.
It samples integer Poisson firing totals, protects critical channels and falls
back to direct SSA when leaping is unsuitable. Negative candidates are rejected
and retried with a smaller step; accepted counts are never clipped. Ordinary
rejection is not claimed to give rejection-unbiased paths, and epsilon is a
local step-selection control rather than a global error bound.

Pending batches retain their absolute interval, firing counts and process RNG
in the original continuation. Observations are step-held until a batch endpoint;
`output_hold_s` exposes that lag. Snapshot restoration preserves this approximate
numerical path exactly. A batch records aggregate firings, without inventing
the individual event times of an SSA trajectory.

`chemworld.examples.stochastic_tau_leap.run_example()` exercises original world
waits and restored continuation. The [51C delivery](https://github.com/sunyrain/ChemWorld/blob/main/workstreams/world_foundation/reports/convergence/51C/README.md)
records 30 units, 192 comparisons and 49,152 distribution trajectories in each
of two retained attempts, including non-grid observation error and fallback
costs. These results do not qualify implicit tau methods, SDE or automatic
partition selection.

## Fixed-partition continuous and discrete chemistry

`StochasticHybridSpec` declares the integer species and stochastic reaction
channels of one reaction network. The remaining material uses continuous mol
coordinates in the same phase ledger. Integrated Poisson hazards respond to the
continuous trajectory; pending hazards and RNG survive ordinary snapshots.
The original profile allows rate feedback between the partitions while each
partition's net elemental stock remains closed. Its actual world and shared-clock
qualification is recorded in the [51B delivery](https://github.com/sunyrain/ChemWorld/blob/main/workstreams/world_foundation/reports/convergence/51B/README.md).

`CrossPartitionHybridSpec` explicitly selects real material transfer: one jump
can change integer molecules and continuous mol by the corresponding molecular
quantum. The original `stochastic_hybrid_world` and native wait entry remain the
consumer. The original process continuation records cumulative cross-partition
firings; the material ledger remains the only inventory. A continuous reactant
must contain enough material for the whole jump, and the numerical domain must
resolve a single molecule. Transfer receipts and both partitions' atomic changes
are checked together before committing a time slice.

This profile retains fixed volume, temperature and partition assignments.
It does not supply a heat model, consuming hybrid sampling or automatic
repartitioning. Same-prefix snapshot continuation and numerical agreement under
different adaptive integration splits are distinct guarantees. The
[51D delivery](https://github.com/sunyrain/ChemWorld/blob/main/workstreams/world_foundation/reports/convergence/51D/REPORT.md)
records 36/36 development units and 338/338 checks, with a separate complete
shared-world/runtime snapshot consumer and all retained failures.
