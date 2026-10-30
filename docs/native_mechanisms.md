<!-- Generated from https://github.com/sunyrain/ChemWorld/blob/main/docs/native_mechanisms.md; edit the upstream guide. -->

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

- One explicitly selected ideal-gas phase; no artificial species/reaction count
  ceiling. Every selected species/reaction is retained, or the import fails.
  An explicitly nonreacting gas (`reactions: none`) retains its thermochemistry.
- NASA7, NASA9, Shomate, constant-Cp and piecewise-Gibbs standard-state thermo;
  Arrhenius, third-body, Lindemann/Troe/SRI/Tsang, chemically activated falloff,
  P-log, Chebyshev and the declared Blowers–Masel branch. Collision partners retain
  their kinetic orders, even when they cancel from net stoichiometry. Duplicate
  channels remain separate reactions.
- P-log sums all terms at the same pressure **before** log interpolation.
  Chebyshev uses inverse-temperature and log-pressure scaled coordinates.
  The imported source semantics clamp P-log at pressure endpoints and evaluate
  Chebyshev's polynomial outside its fitted pressure interval. These source
  rules do not extend species thermochemistry beyond its declared temperature
  interval or establish physical accuracy outside the fitted rate domain.
- The native concentration/rate basis is mol/L and mol/(L·s). Input length
  m/cm/mm, amount mol/kmol, time s/min and supported activation-energy units are
  converted once. A finite extension `units: {volume: L, quantity: mol}` can be
  used **instead of** a length declaration; L is never treated as a length.
  Source file, list and entry unit scopes are retained during section resolution.
- Pressure is computed from the current ideal-gas inventory and temperature.
  Reversible gas rates use the selected thermo and `C° = P°/(1000 RT)` in mol/L, rather than
  silently reusing the historical liquid standard concentration. Internal
  energy is `Σ n_i (h_i(T)−RT)` and heat capacity its derivative; reaction heat
  is counted once by the shared energy balance.
- Native thermal component limits derive from the vessel's explicitly declared
  thermal model. Historical liquid defaults and the public minimum 1-second
  heat/wait duration are unchanged. The mathematical batch integrator also
  supports subsecond stiff trajectories.

Custom/negative/nonreactant orders and signed pre-exponential coefficients require
their explicit source permissions. Executable rate code, unsupported
thermo/kinetics and unbalanced mechanisms reject. Transport and non-selected
phases remain source metadata. Source `h2o2.yaml` and its BSD license remain under
`tests/fixtures/mechanisms/cantera-3.1.0/`.

## Named sections and local source files

The Python entry accepts explicit library directories. The current file's
directory has priority; no Cantera installation path or network download is
implicitly consulted:

```python
from chemworld.physchem import import_mechanism

network = import_mechanism(
    "my-mechanism/main.yaml", phase_name="gas", search_paths=["my-library"]
)
```

The YAML/JSON CLI accepts repeated `--search-path` arguments in lookup order,
after the input file's own directory:

```sh
uv run --no-sync python -m chemworld.physchem.mechanism_import \
  my-mechanism/main.yaml --phase gas \
  --search-path my-library --search-path shared-library \
  --output native-gas.json
```

A phase can select species and reactions from local named sections or files:

```yaml
phases:
- name: gas
  thermo: ideal-gas
  kinetics: gas
  species:
  - thermo.yaml/species: [H2, H, O, O2, OH, H2O, HO2, H2O2, AR, N2]
  reactions:
  - reactions.yaml/hydrogen: all
```

For an in-memory document, `convert_mechanism(..., source_documents={...})`
accepts an explicit mapping from referenced filenames to document dictionaries.
Named sections preserve source order and each source's unit system. Duplicate
species selections, ambiguous file keys, missing files or sections, and
unsupported selected reactions produce errors.

`declared-species`, `skip-undeclared-elements` and
`skip-undeclared-third-bodies` apply only when explicitly requested by the source.
The returned network's `metadata["source_resolution"]` records selected entries,
omissions, source phase selection and document digests. It survives JSON replay;
the resulting network uses the original inventory, heat and runtime consumers.
External element/adjacent-phase sections and other phase families remain outside
this gas importer.

A separate `import_chemkin(mechanism_path, thermo_path)` entry implements a
bounded NASA7 gas subset for elementary, third-body, Lindemann, Troe, PLOG,
CHEB, REV, FORD, and RORD declarations. It expands recursive local INCLUDE
files, accepts both `ELEM` and `ELEMENTS`, and preserves explicit source units
and reaction-order metadata. Unsupported directives and phase families still
reject the complete import rather than being silently dropped; current limits
include named reaction sections, surface/plasma/ionic phases, and nonideal
models. MOLECULES concentration headers are converted to native mol/L kinetics
using the exact Avogadro constant and recorded source metadata. Cantera 3.2
`ck2yaml` currently rejects RORD continuations,
so that upstream round-trip limitation remains recorded in the CHEMKIN report.
This entry is not selected by the YAML/JSON CLI, and its bounded source contract
does not qualify the broader comparison below.

## Validation

```sh
uv run --no-sync pytest tests/test_pressure_dependent_mechanisms.py tests/test_reaction_rate_contracts.py -q
uv run --no-sync --with 'cantera==3.1.0' python scripts/validate_native_mechanisms.py \
  --output /tmp/mechanism-comparison.json
```

The optional reference command runs 45 rate-grid cases, nine heated trajectories
and three equilibrium limits. The development note fixes denominators and
tolerances before execution. Final evidence and the first failed endpoint run
are retained beside `NATIVE_MECHANISM_ACCEPTANCE.md`. This does not certify other
mechanisms, platforms, wet-lab outcomes or a release build.

The current source-selection qualification uses actual Cantera 3.2: complete
h2o2 and GRI30 in three file layouts, an explicitly selected hydrogen subset,
mixed source units and a nonreacting gas. All nine units and 12,552 checks pass;
the original 3,520-check gas-thermo/rate block also passes unchanged. These static
comparisons remain distinct from reactor trajectory qualification. The new
file-imported gas is also exercised in a real heat/wait world with exact snapshot
continuation. See the source-selection report under
`workstreams/world_foundation/reports/convergence/34-source/` for retained failures
and the installed-package result.
