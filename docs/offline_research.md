<!-- Generated from https://github.com/sunyrain/ChemWorld/blob/main/docs/offline_research.en.md; edit the upstream guide. -->

# Complete offline research example

Run from the [locked development environment](getting_started.md):

```bash
uv run --no-sync python examples/demo_offline_research.py --output runs/offline-example
```

This fixed development demonstration asks whether local temperature-yield
interpolation transfers to unseen jacket setpoints. It is not a paper rerun,
provider experiment, mechanism-identification benchmark or real-chemistry test.

The design is written before execution. Three training setpoints (330, 350,
370 K) determine a piecewise-linear rule, clipped to [0,1], and an observed-mean
baseline. Both sets of predictions are sealed before testing 360 and 400 K.
All other actions are held fixed. Each episode resets a fresh vessel with the
same world seed 0; these are not independent worlds or random replicates.
The jacket setpoint is not the instantaneous sample temperature.

An empty-vessel HPLC refusal is intentionally logged. Two additional two-action
episodes demonstrate an open record and an operation-budget truncation. Fixed
totals are seven recorded batches, 45 attempted actions, five final assays, one
refusal, one open batch and one truncated batch. Missing assay values stay null.

Acceptance requires complete records and exact replay, **not favorable errors**.
Unexpected failures are retained and cause a nonzero exit. Existing output
directories are not overwritten. Outputs include the design, notebook,
append-only host ledger, seven native trajectories, sealed predictions, actual
absolute errors, dataset JSONL/card and seven independent-process zero-tolerance
replay results. Replay each source trajectory separately, not the concatenated
multi-episode dataset as a single vessel.

The host retains replay provenance separately from model-facing observations.
Notebook access is an API ownership boundary, not an OS sandbox. The final note
records limitations and competing explanations: thermal lag, kinetics and
measurement depletion. Two errors cannot calibrate a prediction interval or
establish a general law. An unexecuted follow-up is labelled unexecuted.

See the [source](https://github.com/sunyrain/ChemWorld-Public/blob/main/examples/demo_offline_research.py)
and the [bilingual detailed design](offline_research.md).
