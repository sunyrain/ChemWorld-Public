# Finite low-copy worlds

Two synthetic worlds run on the native session and task-36 direct SSA strategy:
`biochemical_world("reversible")` (A↔B) and `biochemical_world("enzyme")`
(E+S→ES, ES→E+S, ES→E+P). These are numerical examples, not fitted biological
systems. Synthetic E is N2 and ES is CH2N2O so binding preserves atoms.
There is no Michaelis–Menten reduction and no implicit ODE/SSA switch.

Run without a model key or COPASI:

```sh
uv run --no-sync python examples/native_biochemical.py
uv run --no-sync pytest tests/test_biochemical_worlds.py tests/test_stochastic_kinetics.py -q
```

The ordinary `add_component` action accepts mol units, but every dose must
represent an exact integer number of molecules (within floating-point conversion
roundoff). Feed index 0/1 is A for reversible, E/S for enzyme. No fractional
rounding, carrier addition, thermal control or flow is supported in this slice.
Elemental transitions are checked in exact integer atoms, not a macroscopic mol
tolerance. The mol ledger remains the sole inventory; counts are a derived view.

`molecule_counter` reports independent Gaussian-noisy, signed calibrated counts
(standard deviation 1 count, unit response per molecule), never event truth.
`molecule_capture` independently selects each solute molecule with probability
0.1, exports selected molecules to instrument waste, and measures them with the
same detector noise. It is an **idealized molecular capture**, not a withdrawn
liquid aliquot: carrier volume stays constant. Analysis effort is 1 s on the
offline instrument clock, with no reactor evolution. Both reads cost 0.01.
Captured truth and sampling RNG remain private. Empty/noisy channels are not
silently replaced by latent values. No saturation or unknown-species identification
is claimed.

Process seed is declared in the world, detector seed at session reset, and capture
uses a separate persisted stream derived from the world seed. Reads do not change
the next reaction or capture; actual dosing/capture invalidates a pending event.
Snapshots retain all continuations, and material/batch/vessel initialization
restores the declared discrete state (including its initial process/capture seeds).
Thus repeated resets are deterministic reinitializations, not independent
replicates: author a world with a new seed for an independent process replicate.
Stationary instrument usage survives material resets. Snapshots contain evaluator
truth and are not agent observations.

Optional external-reference block (no default dependency, choose a fresh output):

```sh
uv run --no-sync --with python-copasi==4.48.309 python scripts/validate_native_biochemical.py --output /tmp/biochemical-attempt.json
```

The authoring repository retains the fixed design and 8,288 development endpoints in
`workstreams/world_foundation/NATIVE_BIOCHEMICAL_ACCEPTANCE.md` and
`NATIVE_BIOCHEMICAL_RESULTS.json`. These run outputs are not distributed here;
the optional command generates a fresh complete summary. It compares transient/effectively equilibrated
binomial moments, finite COPASI endpoint distributions and volume/count-scaled
ODE convergence. Finite distribution agreement is not proof of all kinetic
regimes or wet-lab predictive validity. A generic molecule counter and synthetic
capture are explicit simplifications, not models of a named physical assay.
