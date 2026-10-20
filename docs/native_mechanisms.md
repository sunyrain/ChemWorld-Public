# Finite native gas mechanisms

This is a development feature, not a general Cantera/RMG interpreter or a claim
of experimentally validated chemistry. It uses ChemWorld's existing reaction
network, common inventory/energy integrator, thermal device, transactions and
replay. Native execution does not import Cantera.

## Import and run

From the repository's locked environment:

```sh
uv run --no-sync python -m chemworld.physchem.mechanism_import \
  tests/fixtures/mechanisms/cantera-3.1.0/h2o2.yaml \
  --phase ohmech --output /tmp/native-h2o2.json
uv run --no-sync python -m chemworld.examples.native_gas_mechanism \
  tests/fixtures/mechanisms/cantera-3.1.0/h2o2.yaml \
  --phase ohmech --amounts '{"H2":0.00002,"O2":0.00001,"AR":0.00007}'
```

The converter refuses to overwrite an existing output. `load_mechanism` reads the
exported native JSON. Source thermo, units, original reaction declarations and
source provenance survive the round trip. When installed outside the checkout,
the same module commands accept an explicit local source path; test fixtures are
not bundled as hidden runtime data.

The example authors a 10-mL vessel at 700 K, with finite-power heating to 750 K,
`heat → pressure measurement → wait → pressure measurement → terminate`.
The pressure instrument uses the existing physical acquisition duration, so the
total physical time includes measurements. Only the generic pressure instrument
is exposed: the existing CO/O2/CO2 assay is **not** relabelled as a generic H2/O2
or arbitrary-species analyzer. Inventory is private state, not a fabricated assay.

## Supported finite contract

- One explicitly selected ideal-gas phase, 1–64 local species and 1–256 reactions.
  Every selected species/reaction is retained, or the entire import fails.
- NASA7 standard-state thermo, elementary Arrhenius, third-body, Lindemann/Troe,
  P-log and Chebyshev. Explicit collision partners on both equation sides retain
  their kinetic orders, even when they cancel from net stoichiometry. Duplicate
  channels remain separate reactions.
- P-log sums all terms at the same pressure **before** log interpolation.
  Chebyshev uses inverse-temperature and log-pressure scaled coordinates.
  Both reject T/P outside the declared domain. At most eight machine ULPs at an
  endpoint are normalized for inventory-to-pressure floating-point roundoff;
  no physical extrapolation or nearest-pressure clamping is provided.
- The native concentration/rate basis is mol/L and mol/(L·s). Input length
  m/cm/mm, amount mol/kmol, time s/min and supported activation-energy units are
  converted once. A finite extension `units: {volume: L, quantity: mol}` can be
  used **instead of** a length declaration; L is never treated as a length.
  P-log requires an explicit `temperature-range` (an additional finite-domain
  contract not required by Cantera's unrestricted P-log format).
- Pressure is computed from the current ideal-gas inventory and temperature.
  Reversible gas rates use NASA7 and `C° = P°/(1000 RT)` in mol/L, rather than
  silently reusing the historical liquid standard concentration. Internal
  energy is `Σ n_i (h_i(T)−RT)` and heat capacity its derivative; reaction heat
  is counted once by the shared energy balance.
- Native thermal component limits derive from the vessel's explicitly declared
  thermal model. Historical liquid defaults and the public minimum 1-second
  heat/wait duration are unchanged. The mathematical batch integrator also
  supports subsecond stiff trajectories.

Executable Python, external species includes, custom reaction orders,
negative-A fits, unsupported thermo/kinetics and unbalanced mechanisms fail
closed. Transport and non-selected phases remain labelled source metadata, not
implemented models. Source `h2o2.yaml` and its BSD license are preserved under
`tests/fixtures/mechanisms/cantera-3.1.0/`.

## Validation

```sh
uv run --no-sync pytest tests/test_pressure_dependent_mechanisms.py -q
uv run --no-sync --with 'cantera==3.1.0' python scripts/validate_native_mechanisms.py \
  --output /tmp/mechanism-comparison.json
```

The optional reference command runs 45 rate-grid cases, nine heated trajectories
and three equilibrium limits. The development note fixes denominators and
tolerances before execution. Final evidence and the first failed endpoint run
are retained beside `NATIVE_MECHANISM_ACCEPTANCE.md` in the authoring repository.
The optional command generates a fresh comparison without those archived outputs.
This does not certify other
mechanisms, platforms, wet-lab outcomes or a release build.
