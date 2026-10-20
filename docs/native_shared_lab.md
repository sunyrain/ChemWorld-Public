# A finite shared laboratory clock

Run the self-contained example from the locked repository environment:

```sh
uv run --no-sync python examples/native_shared_lab.py
```

`shared_lab_world()` authors two distinct liquid bottles, one finite-power heater
and one programmed HPLC. It uses the existing session, native local solvers,
sampling law, apparatus memory and transaction. No external chemistry backend,
model key or distributed scheduler is involved.

```python
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.lab_world import shared_lab_world

lab = ChemWorldRuntime(composition=shared_lab_world())
lab.reset(seed=17)
lab.execute({"operation": "queue_lab_heat", "duration_s": 30,
             "target_temperature_K": 350})
lab.execute({"operation": "queue_lab_measure"})
lab.execute({"operation": "select_vessel", "vessel": "sample"})
lab.execute({"operation": "queue_lab_measure"})
result = lab.execute({"operation": "wait", "duration_s": 120})
print(result.info["actual_duration_s"])  # 30: next heater completion
print(lab.lab_status())                  # analyzer result is not yet visible
```

The default `CompositionPhysics.lab_clock=None` is the original **local** clock.
An explicit `LabClockSpec` selects shared mode for the whole session. All enabled
vessels advance together, whether selected or not. A requested wait is an upper
bound: it returns at the next completion event. Another wait starts the next FIFO
job at that same physical time. Zero-time requests between waits can be queued,
but never retroactively change a specimen already taken. At equal timestamps,
persistent sequence numbers determine completion order.

Analysis samples at **service start**, not at queue submission or result release.
The existing sampler handles true withdrawals and persistent chromatograph
carryover. Results carry `sampled_at` and `available_at` on the `lab` clock; only
completed packets appear in `lab_status()` or `info["lab_clock"]`. Polling consumes
no resource or randomness. Detector failures produce an invalid completed job,
not a restored sample. Process, sampling and detector RNG states are independent.

Physical seconds are charged once across vessels. Analysis effort counts actual
busy seconds separately, and sampling cost is charged once at service start.
Queue/control/wait calls consume the ordinary action budget; job records expose
the separate acquisition history. A numerical failure discards the entire
uncommitted slice. A physical stop first realigns all systems to the earliest
endpoint, commits that slice, then requires a full lab reset before further
advancement. Existing committed events remain in the record.

Private `snapshot().to_dict()` and `ChemWorldRuntime.from_snapshot(...)` restore
device ownership, queue order, pending SSA events and unpublished packets exactly.
Never provide a private continuation snapshot to an agent as observation evidence.

## Explicit limits

One or two independent closed native thermal/SSA vessels; at most one heater and
one nonterminal native offline analyzer. The default finite session admits at most
64 jobs (configurable to 256), preserving all completion records. Thermal and SSA
may be separate vessels, never an implicit hybrid solver in one vessel. Only
queue, select, wait and terminate actions are available in this initial shared
mode. Partial entity reset, terminal/in-situ instruments,
inter-vessel/open flow and alternate phase strategies are rejected rather than
silently mixing clocks. Chromatograph method configuration is authored before
the shared session. These are synthetic development examples, not fitted
laboratory chemistry or a claim of robot orchestration.

Queued detector readouts support sequential and keyed noise. Keyed mode uses the
existing observation-coordinate primitive with a laboratory namespace and the
persistent queue sequence; polling does not change the key, physical sampling or
SSA stream. The detector seed and coordinate inputs persist in private snapshots.
The optional `lab_schedule.duration_range_s` declares the public wait range down
to 1 microsecond; without it the ordinary one-second lower bound is unchanged.
This allows stopping inside a short offline acquisition without altering the
instrument's declared analysis time or the underlying physical rates.
