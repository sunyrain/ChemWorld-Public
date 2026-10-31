<!-- Generated from https://github.com/sunyrain/ChemWorld/blob/main/docs/native_foundation.md; edit the upstream guide. -->

# Native physical sessions

Native property methods use explicit data adapters and shared sources; EOS and
flash calculations have separate responsibilities. This does not imply drop-in
compatibility with the chemicals 1.5.2 API or empirical accuracy for every material.
External source files remain caller-supplied; production does not import chemicals.

## Design and operate a finite process graph

`FlowsheetGraph` declares sources, units, named product ports and simultaneous
recycle tears. It supports original molecular mixing, splitting, heating,
pressure stages, selective flash/cascade, two-sided heat exchange and shaft
recovery. `ProcessFlowsheetInventorySpec` assigns each boundary to a finite tank;
internal recycle streams own no extra inventory. Tanks require an explicit
pressure, homogeneous phase branch, positive initial component amounts and limits.

This installed example defines two coupled recycles, optimizes an authored feed
cost under a product-rate constraint, and executes the selected process on real
finite stocks. Its ideal-gas A/B parameters and currency coefficients are synthetic
API examples. Supply your own physical model, controls, constraints and costs.

```python
from dataclasses import replace

from chemworld.examples.process_flowsheet import (
    recycle_feed_design, recycle_flowsheet_example,
)
from chemworld.runtime import ChemWorldRuntime, optimize_flowsheet
from chemworld.world import build_process_flowsheet_world

design = optimize_flowsheet(recycle_feed_design())
if not design.success:
    raise RuntimeError(design.optimization.failure_reason)
spec, amounts, temperatures = recycle_flowsheet_example()
spec = replace(spec, graph=design.selected_graph)
world = build_process_flowsheet_world(
    spec, amounts, temperatures, formulas={"A": "Ar", "B": "C3H8"},
)
session = ChemWorldRuntime(composition=world)
try:
    session.reset(seed=33)
    result = session.execute({"operation": "wait", "duration_s": 0.1})
    assert result.info["transaction_status"] == "committed"
    snapshot = session.snapshot().to_dict()
finally:
    session.close()
```

For direct inventory integration, import `initialize_process_flowsheet_inventory`
and `advance_process_flowsheet` from `chemworld.physchem`. The initializer's public
name differs from the submodule name, so importing examples or world builders
first does not turn the callable into a module. Full snapshots can resume through
`ChemWorldRuntime.from_snapshot(snapshot)`; preserve the original declared model.

`TrajectoryDesignSpec` describes periods of original actions and a user-defined
objective. `optimize_trajectory(session, spec)` predicts on private snapshots;
`execute_first_period(session, plan)` commits only the verified first period.
`run_receding_horizon` repeats planning from the actual current state. Planning
retains computational evaluations and failures without charging experimental
stock/time; real actions retain their normal resource charges. These engineering
tools use internal model states explicitly; they do not grant an agent access to
hidden state through its ordinary observation API. Agent rewards remain user code.

Thermal import/export is counted per external port, including opposite duties in
one cascade. Internal exchanger heat and recovered shaft work are not charged
twice. This builder declares subsecond `run_flow` for local operation and `wait`
for the shared clock; existing worlds keep their own declared action ranges.
Source-only uniform drainage preserves intensive state while integrating n/H;
receivers stop at their actual amount, volume and temperature events. Arbitrary
empty tanks, automatic tank phase switching and global optimization guarantees
are outside this declared model.

Development qualification covers 43 graph cases, 42 finite-world cases, 22 steady
design cases (including six expected refusals), and 22 trajectory/control cases.
The installed package also passes all six declared consumers, including actual
fresh-process continuation. Numerical search is bounded and can be expensive:
inspect the reported evaluations and
failure reason, and execute a plan only when its verified result succeeds.

## Inspect and choose property methods

`ComponentPropertyPackage.selection_report(property_id, temperature_K=..., pressure_Pa=...)`
lists every declared candidate, source ID, unit, temperature/pressure domain,
missing input and the selection reason. It inspects declarations without trying
alternative formulas. `with_selection` returns a new package whose choices are
used by ordinary evaluation, caloric integrals, transport, virial gas and VLE.

Three policies are available:

- `PropertySelectionPolicy()` preserves declaration order, preferring a candidate
  whose declared domain contains the requested state.
- `PropertySelectionPolicy("correlation", correlation_id)` chooses that exact curve.
- `PropertySelectionPolicy("source_priority", source_priority=("b", "a"))` permits
  only the listed source IDs and orders candidates within that shortlist. Domain
  compatibility is checked before priority. The list is a caller choice, not a
  claim about scientific quality.

If all permitted candidates are outside their declared ranges, the selected
curve is still governed by the evaluator's `raise`/`warn`/`ignore` policy.
Mathematical failure never selects another curve. A missing pressure remains a
missing input, and a source shortlist with no match raises `PropertySelectionError`
with its complete report.

This self-contained synthetic example changes which vapor-pressure curve the
existing flash calculation consumes:

```python
import json
from dataclasses import replace
from math import log

from chemworld.examples.native_flash_research import synthetic_vle_properties
from chemworld.physchem import PropertySelectionPolicy
from chemworld.physchem.flash_units import VLEPropertyPackage, tp_flash
from chemworld.physchem.property_reports import R_J_PER_MOL_K

model = synthetic_vle_properties()
light = model.packages[0]
original = light.by_property("vapor_pressure")[0]
alternative = replace(
    original, correlation_id="alternative-pressure",
    coefficients={**original.coefficients, "A": log(1e5) + 30000 / R_J_PER_MOL_K / 345},
    metadata={"source": {"source_id": "example-b"}},
)
light = replace(light, correlations=(*light.correlations, alternative)).with_selection({
    "vapor_pressure": PropertySelectionPolicy("correlation", "alternative-pressure"),
})
model = replace(model, packages=(light, model.packages[1]))
restored = VLEPropertyPackage.from_dict(json.loads(json.dumps(model.to_dict())))
result = tp_flash(restored, {"light": .5, "heavy": .5},
                  temperature_K=350, pressure_Pa=1e5)
assert result.vapor_fraction == 0
assert restored.packages[0].selection_report(
    "vapor_pressure", temperature_K=350).selected_id == "alternative-pressure"
```

For source-backed cards, pass `selection_policies` to `PropertyInputCard.combine`
to retain overlapping alternatives, or call `card.with_selection(...)`. Each
curve's declared source must agree with its uncertainty/source envelope. Original
source records and extra provenance survive selection and JSON; a contradictory
segment source is rejected instead of relabelled.

`ConstantDatabase.inspect_value(identity, field_id, methods=...)` reports every
requested method, including unavailable files, missing identities/fields/values,
unknown units, ambiguous records and conflicting values. Derived data readers
share this API. It also retains unevaluated correlations and temperature series;
use `point=` for an explicit series point. `IdentifierDatabase.inspect_search`
likewise preserves all source candidates and distinguishes missing sources from
no match, multiple registry identities and structural-field conflicts. Neither
diagnostic chooses a record, merges identities or downloads external data.

## Supply a reviewed component for a source identity

A CAS alias can hide a different stereoisomer or a superseded registry number.
When the identifier tables have no exact record, pass an explicit `ComponentSpec`
with reviewed identity evidence to the existing property input API. Do not copy
the structure of the first alias search result.

The [reviewed CRC component inputs](https://github.com/sunyrain/ChemWorld-Public/blob/main/examples/fixtures/crc_electro_optical_components.json)
cover 200 identities and 246 original dielectric/refractive-index rows.
They retain the source CAS, available full InChI, cited public records, documented
stereochemical scope, and explicit superseded-CAS relationships. Their default phase is unknown.
These are partial, explicit inputs: they do not modify the identifier database or
resolve the other 39 original missing-identity rows. Those rows remain in the
285-row denominator, with 34 explicit CAS diagnoses under `source_gaps`.
The racemate metadata must
be retained; a connectivity-only InChI alone does not specify a racemate.
Vanadyl tribromide uses the exact name, CAS and VOBr3 formula in INERIS's compound
table. Its optional InChI and InChIKey remain empty because that source provides
neither; this does not infer a molecular graph or resolve unspecified mixtures.

`examples/crc_electro_optical_inputs.py` exposes `inspect_source_identity(payload,
cas, source_unit)` and `reviewed_property_input(database, payload, identity,
method=..., source_point=True)`. Load `payload` with its `load_inputs()` helper.
Source units use the fixture's original zero-based table row (for example
`CRC_RI/3372`). The first function reports reviewed input, source gap, unreviewed
row or unknown CAS; the second uses the existing native card API only for a
reviewed row. `SourceIdentityGapError.diagnostic` retains the reason and source
references. Raw source values remain queryable even when their material cannot
be resolved. No alias fallback or assumed mixture ratio is introduced.
This example's supplied-input policy is separate from the native API: users can
still supply their own reviewed `ComponentSpec` to `database.property_input`.
Functional acceptance with explicit gaps does not claim 285 complete identities.

Forty-one further inputs use EPA SRS's explicit former-to-current CAS relationship
and the current CAS's complete GSRS single-center racemate definition. They retain
the original CRC CAS as the input key and equal enantiomer fractions as material
metadata. Conflicting original candidates and unsuccessful old-CAS searches remain
in provenance. A historical-CAS relationship alone does not resolve a source-name
or structural conflict.
Fenchone's historical CAS is separately linked to a complete two-center GSRS
racemate. Official default and `SRac` conversions of the original MOL verify its
representative structure and racemic encoding. The component retains relative
geometry, equal enantiomer fractions and the original CRC key; PubChem's generic
connectivity record does not replace this material definition.
Nine inputs use explicitly reviewed EPA DSSTox deposits accessed through PubChem:
the deposit links the exact source CAS and material name to its standardized CID.
Unsuccessful CIR lookups and unselected compound candidates remain in the input provenance.

Two further DSSTox deposits supply an explicit 4S enantiomer and a two-center
racemate. The official InChI encoding of each deposited SDF corroborates its
representative structure; only the racemate uses the `SRac` result and equal
enantiomer fractions. Two further single-center racemates bind their exact CAS
through explicit source annotations; their CIR results corroborate connectivity
only, because the resolver reports `name_by_cir`, not `cas_number`.
The CAMEO record supports a fixed 1:1 boron trifluoride/dimethyl ether adduct.
Its disconnected Standard InChI is a formula-unit representation, not an
arbitrary free mixture or a dissociation model. HTTP failures remain recorded.

Three inputs retain geometry from the original deposited SDF where the
standardized compound loses it: trans-p-menthane, trans-1-ethyl-4-methylcyclohexane
and E-azobenzene. Their exact CAS and geometry names are bound by the deposits,
and the official default InChI conversion preserves the full relative-ring or
azo-double-bond layer. The two 1,4-substituted rings are achiral: their `/t`
layers do not imply a selected absolute enantiomer or a 1:1 racemate.
The incomplete standardized compounds remain in provenance, together with
other source candidates and failed queries.

Eight racemates with defined relative configurations use non-standard `InChI=1/`
identifiers with `/s3`, generated from the saved GSRS structures by the official
InChI 1.07.5 executable with `SRac`. Their keys come from the same conversion.
The original standard representative enantiomer remains provenance, together
with every PubChem candidate; it is not used as the bulk material identity.
Relative `/s2` alone denotes a single enantiomer of unspecified absolute
configuration and is not a replacement for the 1:1 racemate definition.
No InChI executable or network service is required to consume the supplied JSON.

Where a CAS query returns several structures, the supplied input retains every
candidate and the reviewed selection. Nicotine also retains the previously
observed NIST stereochemistry disagreement: its explicit S assignment follows
the original L-nicotine label, GSRS and corroborating material identity records.
These decisions do not silently merge identifiers or repair external databases.

The two azoxybenzene isomers retain their full GSRS/ChEBI double-bond geometry;
PubChem's generic candidate does not identify either specific material. Twelve
additional achiral structures use explicitly reviewed NCI CIR exact-CAS records
and matching, CAS-annotated PubChem candidates. All candidate structures remain
in metadata; shared upstream sources are not independent laboratory validation.
The E alkenes and Z-cyclododecene retain exact CAS annotations that explicitly
name their geometry; opposite-geometry and mixture synonyms are not aliases.
For trans-1-methyl-2-propylcyclopentane, GSRS's original structure and Sander's
identity fields support the trans racemate; the conflicting PubChem `cis` name
is retained as provenance and excluded from the material aliases.

Three further Z alkenes use explicitly reviewed Sander identity pages and full
PubChem structures with exact CAS annotations. Sander attributes its structures
to NIST; these are not independent measurements. For 1501-60-6, the complete-key
query resolves the 2-methyl structure, while the original empty CAS query and
NCI's conflicting 3-methyl structure remain provenance. Neither becomes an alias.

Eight further single-stereocenter racemates retain exact CAS annotations naming
the racemate, their reviewed stereogenic atom and 1:1 enantiomer fractions.
Their connectivity-only InChI does not encode that material definition by itself:
the full `ComponentSpec` metadata and provenance must accompany the input.
No absolute enantiomer is selected and no SRac conversion is claimed for this group.

Run the [cis/trans example](https://github.com/sunyrain/ChemWorld-Public/blob/main/examples/crc_electro_optical_inputs.py) with your
directory containing the original external CRC source tables:

```text
uv run --no-sync python examples/crc_electro_optical_inputs.py SOURCE_DIRECTORY
```

It loads the ordinary `ComponentSpec` JSON, creates source-point input cards,
round-trips them, and evaluates the original dielectric values before computing
capacitance for an explicitly supplied example geometry. At the source's 298.2 K,
cis-1,2-dibromoethene gives 7.08 and trans gives 2.88. Their distinct double-bond
stereo layers survive the card JSON; cross-binding their CAS identities is rejected.
The example imports no external property implementation and downloads no data.
It does not extrapolate a single observation into a temperature-dependent model.

## Critical estimates as explicit inputs

`hekayati_raeissi_critical` estimates missing critical properties using molecular
weight and either saturated liquid molar volume at 298.15 K or a supplied critical
volume. Supply at least one of critical temperature and pressure. A supplied
critical volume selects a different fitted equation from the saturation-volume
route; passing an estimated volume back as `Vc` deliberately changes the model.
Known critical values are preserved, including when both temperature and pressure
are supplied. This does not reconcile inconsistent measurements.

Use `estimate_critical_property` to keep sources, units, component identity and
the selected equation with each derived result. Select `output_property` when
more than one field is missing. Estimates remain configuration inputs and never
overwrite measured constants:

```python
from chemworld.data.scientific_inputs import InputValue
from chemworld.physchem import (
    ComponentSpec, CubicEOSSpec, estimate_critical_property,
    eos_component_from_critical_inputs, evaluate_cubic_eos,
)

component = ComponentSpec("toluene", "C7H8", cas_number="108-88-3")
inputs = {
    "MW": InputValue(92.13842, "g/mol", None, "Explicit example input"),
    "V_sat": InputValue(.10686, "L/mol", None, "Saturated liquid at 298.15 K"),
    "Pc": InputValue(41.08, "bar", None, "Explicit example critical pressure"),
}
temperature = estimate_critical_property(
    component, method="Hekayati_Raeissi", parameters=inputs, output_property="Tc",
)
eos_component = eos_component_from_critical_inputs(
    component, Tc=temperature, Pc=inputs["Pc"],
    omega=InputValue(.2, "dimensionless", None, "Explicit example acentric factor"),
)
state = evaluate_cubic_eos(
    CubicEOSSpec("estimated", "peng_robinson", (eos_component,)), {"toluene": 1},
    temperature_K=700, pressure_Pa=1e6,
)
```

The fitted equations and coefficients were read from the public chemicals 1.5.2
implementation attributed to Hekayati and Raeissi (2020, DOI
[10.1002/aic.17004](https://doi.org/10.1002/aic.17004)); the original paper was not
directly verified. The native implementation expands those equations independently
and has no chemicals runtime dependency. Reference agreement and positive fit
outputs do not establish predictive accuracy for an arbitrary material. Missing,
nonpositive and nonfinite inputs are rejected; saturation-volume predictions also
require a critical temperature above 298.15 K. There is no implicit source ranking,
fit selection by compound name, uncertainty estimate or new EOS runtime.

## Derived parameters in finite processes

`critical_correlation_input` binds selected `ConstantValue`, `CriticalEstimate`
and unit-bearing `InputValue` parameters to the existing volume or virial
correlations. Supply the application range and an `InputSource` for the derived
model. Each parameter's original report remains in `metadata["parameter_sources"]`
through the property card and JSON. No measured source field is overwritten.
Source aliases such as ChemSep's `CriticalTemperature` and `CriticalVolume`
use the original database mapping. Their source field names and units remain in
the report while the consumer converts them to SI and checks the component identity.
`estimate_critical_property(..., method="Zc_definition", parameters={"Tc": ..., "Pc": ...,
"Vc": ...})` explicitly identifies the derived critical compressibility.

```python
from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem import critical_correlation_input
from chemworld.physchem.data_conflicts import DataSourceProvenance

model_source = InputSource(DataSourceProvenance(
    "example-derived-volume", 0, "Explicit Rackett application using estimated Tc",
    version="1", license="synthetic example",
), "synthetic")
volume_input = critical_correlation_input(
    component, family="volume", method="Yamada_Gunn",
    parameters={"Tc": temperature, "Pc": inputs["Pc"], "omega": .2},
    temperature_range_K=(300, 400), source=model_source,
)
```

Combine this volume card with explicitly selected heat capacity, viscosity and
conductivity cards using `PropertyInputCard.combine`, then use the existing
`LiquidTransportProperties` and `FiniteTransportSpec`. Source-backed volume fits
can also consume these estimates as their declared supplemental inputs through
`VolumeDatabase.property_input`. Source-owned coefficients cannot be overridden.

For mixtures, `LiquidTransportProperties.volume_mixing_method` selects `Amgat`
(the default), `Rackett_mixture`, `COSTALD_mixture` or
`COSTALD_mixture_compressed`. The optional `volume_mixing_parameters` map is keyed
by the existing component IDs: Rackett needs `Tcs/Pcs/Zrs`; COSTALD needs
`Tcs/Vcs/omegas`. Molecular weights come from those components. Compressed COSTALD
also requires explicit saturation-pressure parameters. Provide
`volume_mixing_note` to identify their source and scope. Each finite transfer
uses the current reservoir temperature, declared pressure and actual composition;
JSON snapshots retain the selection. Amgat uses the current pure volumes and
rejects frozen-volume overrides. Liquid enthalpy is still the separately declared
additive model, so these volume laws do not establish a complete liquid EOS.

Pure gas heating uses `VirialGasThermodynamics` and the original coupled solver.
Its ideal heat-capacity datum can be outside the residual coefficient's domain:
the ideal integral must cover its own reference path, while residual Helmholtz
terms must be valid at every actual gas state. Neither domain is silently widened.
Unlocated constant-volume observations retain an explicit single application
temperature; solid volumes remain solid properties. They cannot supply an
undeclared temperature-dependent liquid process.

## Source-backed transport inputs

The Lennard-Jones, viscosity, conductivity and interface databases expose 19
external source tables containing 4,573 records. Select a source explicitly:
`ViscosityDatabase.property_input`, `ConductivityDatabase.property_input`,
`InterfaceDatabase.property_input` and `LennardJonesDatabase.viscosity_input`
return the original `PropertyInputCard`. Tables are read from a caller-supplied
directory; native execution does not import chemicals or download missing data.
Missing identities and unknown validity intervals are reported, not filled in.

`PropertyInputCard.combine(cards, reference_note=...)` assembles selected inputs
for the same complete component declaration. It retains source and uncertainty
metadata, shares identical declarations and rejects conflicting data or caloric
references. Choose one correlation where the same property domains overlap;
input order never selects a preferred source. A transport source alone does not
provide heat capacity, density or a caloric datum. Supply those explicitly:

```python
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.transport_devices import FiniteTransportSpec, LiquidTransportProperties
from chemworld.world.transport_world import transport_world_spec

def source_backed_liquid_world(viscosity, conductivity, calorics_and_density, bounds):
    card = PropertyInputCard.combine(
        (viscosity, conductivity, calorics_and_density),
        reference_note="Explicit source choices and liquid enthalpy reference",
    )
    if card.reference is None:
        raise ValueError("Supply an explicit liquid enthalpy reference")
    key = card.package.component.identifier
    properties = LiquidTransportProperties(
        (card.package,), temperature_bounds_K=bounds,
        pressure_Pa=card.reference.pressure_Pa,
        caloric_references={key: card.reference},
    )
    return transport_world_spec(
        FiniteTransportSpec(properties),
        feed_mole_fractions={key: 1.0}, receiver_mole_fractions={key: 1.0},
        feed_temperature_K=300, receiver_temperature_K=300,
    )
```

The finite liquid apparatus consumes these curves during actual pump, pipe and
heat-exchanger calculations. Mixtures use declared ideal volumes, log viscosity
and DIPPR9H conductivity. The optional `viscosity_pressure_parameters` mapping
selects Lucas for named components using `critical_temperature_K`,
`critical_pressure_Pa` and `acentric_factor`; each affected package must also
provide its `vapor_pressure` curve. Saturation viscosity and pressure are
evaluated at the current temperature, and the actual reservoir pressure enters
the correction. Missing baselines, temperatures above Tc and pressures below
Psat are rejected. Input JSON and full runtime snapshots retain this declaration.

`ideal_gas_transport_state(packages, amounts, T, P, references)` supplies the same
`FluidState`, molar enthalpy and total volume tuple for gas pipe and heat-transfer
consumers. It uses explicit ideal-gas volume, Wilke viscosity and
Wassiljewa–Herning–Zipperer conductivity; it does not imply a dense EOS or a
compressible-flow solver. Amounts and reference maps use canonical component IDs.
LJ binary diffusivity can feed the existing conservative finite-volume operator.
Interface tension feeds Young–Laplace pressure and the pump's pressure-load
function; fixed-curvature qualification does not describe dynamic wetting.

All 19 sources have process-consumption examples with independent equation
checks. Supplementary properties in those examples are explicitly synthetic.
Material prediction accuracy is a separate question. At large NTU, heat transfer
can reach the stream-capacity limit: changing conductivity need not measurably
change the outlet heat, even though the property is correctly consumed.

## Pure virial coefficients

`virial_correlation` declares seven second-virial methods (`Pitzer_Curl`,
`Abbott`, `Tsonopoulos`, `Tsonopoulos_extended`, `Oconnell_Prausnitz`, `Xiang`,
`Meng`) and two third-virial methods (`Orbey_Vera`, `Liu_Xiang`) for the existing
property package and input card. For example:

```python
from chemworld.physchem import virial_correlation, virial_temperature_report

curve = virial_correlation(
    "Tsonopoulos",
    parameters={"Tc": 425.2, "Pc": 3.8e6, "omega": 0.193},
    temperature_range_K=(300, 900),
    source_note="Explicit example critical parameters; application range chosen by caller",
)
value, first, second, third = virial_temperature_report(curve, temperature_K=510)
```

Inputs use K, Pa and m³/mol; B is m³/mol and C is m⁶/mol². Signed coefficients
are valid. `virial_derivatives` and `virial_fast` return the value through the
third temperature derivative; `virial_vector` and `virial_matrix` apply the same
kernel to explicit parameter sets. They do not generate cross parameters or
perform mixture rules. All parameters remain fixed during differentiation.

`virial_value(..., order=-1)` and `order=-2` provide the documented indefinite
integrals for Pitzer–Curl, Abbott and the two Tsonopoulos variants, with zero
constants in the dimensional-temperature basis (`log(T / 1 K)`). Other methods
do not declare these indefinite integrals. Normal `integrate_correlation`
remains available for definite integrals within the declared application range.

Extended Tsonopoulos requires explicit `a` and `b`, or a `polar` selection with
`species_type` and the dipole in debye where required. The default `documented`
polar convention includes methanol and the zero-dipole alkanol limit; explicitly
select `convention="chemicals_1_5_2"` to reproduce that API's different behavior.
Meng requires explicit `a`, or `polar={"dipole": ..., "haloalkane": ...}`.
Choices and source notes survive JSON. Unknown classes and conflicting choices
are rejected. Xiang/Liu constants were recovered from public calls in their
documented mathematical basis and checked on separate inputs; their original
paper tables have not been directly verified. Development mathematical agreement
does not establish material accuracy or the validity of a truncated gas EOS.

## External density and volume sources

`chemworld.data.volume.VolumeDatabase(data_root)` reads explicitly supplied
Density tables and the existing ChemSep/VDI sources. It uses the same lazy
identity and source database as other property inputs. Call
`db.volumes("64-17-5", method="PERRY_VOLUME")` to inspect candidates, then
`db.property_input("64-17-5", component, method="PERRY_VOLUME")` to obtain a
normal `PropertyInputCard`; its `evaluate_state` consumes liquid `density`
and `molar_volume` with the component's molecular weight.

Select `curve_id` if a source has multiple candidates. Missing critical inputs
must be supplied in `parameters`; unknown temperature bounds require
`temperature_bounds_K`. Source coefficients cannot be overwritten. CRC constant
observations require `application_temperature_K` and `application_note` and
remain single points. VDI tables require `interpolation="linear"`; missing
intervals and extrapolation are rejected. These temperature paths do not imply
an independent pressure equation or caloric data. Source identity and molecular
weight must agree with the component. External scientific datasets are not
bundled or silently discovered through an upstream library.

These development examples use the installed current runtime, without a model
API key or Cantera. They describe a finite synthetic CO oxidation model, not a
general chemistry predictor or a newly qualified release. Python 3.11 or 3.12
is the validation target.

For the current ten cross-domain workflows, shared-clock boundaries and the
separate source/install denominators, start with [native domains](native_domains.md).
Both sequential and keyed detection are supported on the qualified shared-clock
histories. Explicitly declared multi-device queues and idle-system material
replacement use the same runtime, transaction and continuation machinery.
The sections below also describe later scientific extensions; their individual
acceptance scope does not follow automatically from stage-A completion.

After installing the wheel or sdist in an isolated environment:

```sh
python -m chemworld.examples.native_surface_research --bulk --mobile --terminal-assay
python -m chemworld.examples.reuse_native_equipment
```

In a development checkout, prefix Python commands with `uv run --no-sync`.
The first example feeds finite stocks, heats, measures, routes material and ends
the task. `--no-surface` runs the same apparatus as an inert or bulk-only system;
`--bulk` adds the declared bulk reaction. `--mobile` attaches a material-borne
surface. Without it, the stationary surface remains on its vessel when gas moves.

The second example uses the same heater model in inert, bulk, surface and combined
systems. It initializes fresh material in the reactor between two batches and
checks that the retained heater keeps its identity and calibration. It also binds
one compiled physical world to two tasks: direct termination and termination
followed by a required final assay. Their physical response before ending agrees.

## Use the same session through other entry points

```python
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.surface_world import surface_world_spec

spec = surface_world_spec(terminal_assay=True)
session = ChemWorldRuntime(composition=spec, observation_noise_mode="keyed")
session.reset(seed=12)
result = session.execute({"operation": "add_component", "component": "CO-feed",
                          "amount_mol": 2e-5})
reading = session.execute({"operation": "measure", "instrument": "gas_pressure"})
for channel in reading.observation.channels:
    print(channel.channel_id, channel.value, channel.unit, channel.quality)
session.close()
```

`ChemWorldEnv(composition=spec)` provides the Gym `reset`/`step` interface over
the same runtime. `chemworld.eval.runner.run_agent` accepts `composition=spec.to_dict()`
for an Agent. Both entry points retain transaction outcomes, physical duration,
resource accounting and the native observation in `info`.

To execute JSON actions from the CLI, save `spec.to_dict()` as `world.json` and
each action as a separate JSON file, then run:

```sh
chemworld render --composition world.json --seed 12 --noise-mode keyed \
  --actions feed.json pressure.json --format json --trajectory run.jsonl \
  --snapshot continuation.json
chemworld verify --submission run.jsonl
```

The CLI refuses to overwrite an existing trajectory or snapshot. Snapshots and
trajectory provenance contain private simulator state/configuration for evaluator
continuation and replay; they are not Agent observations. JSON can reorder map
keys, so physical replay uses a numerical tolerance (the CLI default is `1e-5`),
while categorical outcomes must agree. Replay requires the original runtime
semantics; current main does not substitute for frozen paper physics.

## Readings, time and reuse

Native channels own physical values and units, uncertainty, quality, axes and
acquisition times. Invalid or unobserved channels are explicit; a valid physical
reading need not fit in `[0, 1]`. The fixed Gym scalar view is a derived
compatibility projection. Use `info["native_observation"]` or session results for
the complete channel record. Calling `observe()` returns cached readings and
does not acquire another sample.

In-situ acquisition advances the world through the same physical integrator.
Offline analysis keeps its acquisition snapshot and accounts for analysis time
separately. A task requiring a final assay first freezes production, then ends
after that assay; it does not advance the frozen production vessel again.

`session.snapshot().save(path)` and
`ChemWorldRuntime.from_snapshot(SessionSnapshot.load(path))` support
private continuation with compatible bindings. `session.reset()` creates a fresh
world. Evaluator-only `reset(options={"scope": "material", "vessel_ids": ["reactor"]})`
replaces that material while retaining attached stationary equipment and its
calibration. Partial initialization retains clocks, budgets and RNG state; it is
not a free Agent operation.
`SessionSnapshot` is also exported from `chemworld.runtime`.

Current private sessions use `chemworld-session-v2`. Their phase identities and
material coordinate order survive sorted JSON, including retained samples;
pending events, apparatus queues, RNG and resource events remain part of the
same continuation. Configuration comparison uses the installed model codecs.
Supply the same `domain_modules` registry when restoring a local extension;
callable names in a saved binding describe installed code and are never imports.

Older v1 sessions require an explicit `SessionSnapshot.migrate_v1(payload)` call.
This converts the encoding only when runtime semantics match and omitted
assembly order is unambiguous. Multiple records without saved order require
the original ordering information. Restoration still checks the complete
execution binding. Older physical-world inputs retain their supported boundary
conversions; this does not authorize reinterpretation of frozen session evidence.

## Model and acceptance boundaries

The gas reaction, five-step surface rates, finite material transfers, shared
internal-energy/heat-capacity balance and heater model are executable physics.
The tests' injected numerical failure, process failure and invalid readout are
interface fixtures, not implemented failure mechanisms. Thermal stopping uses a
real event. Property tables and rate parameters are finite synthetic contracts.
Surface flow, arbitrary thermodynamics and new electrode kinetics remain outside
this foundation slice.

The native acceptance covers fixed native/Gym/CLI/Agent histories, both observation
noise modes, direct/assay endings, replay and retained equipment. Installation
checks build both archives and run the installed modules outside the checkout,
with Cantera absent and model keys removed from the child environment. Reported
platform results apply only to actual runs; this is development acceptance, not a
formal release or a claim of general chemical validity. See
[distribution boundaries](community_release.md) for packaging and frozen releases.

## Properties at the current state

`ComponentPropertyPackage.evaluate_state(...)` evaluates a pure state;
`evaluate_property_state(packages, amounts_mol, ...)` evaluates a composition.
Both use the existing component IDs, declared aliases and CAS identifiers.
Supply `temperature_K`, `pressure_Pa`, `phase` and the requested `quantities`.
The report retains units and validity warnings for each property. For example:

```python
from chemworld.physchem.curated_properties import curated_property_package

report = curated_property_package("7732-18-5").evaluate_state(
    temperature_K=400, pressure_Pa=1000, phase="gas",
    quantities=("cp", "enthalpy", "internal_energy", "density"),
)
cp = report.properties["cp"].quantity().to("J/(mol*K)")
```

Energy values are molar J/mol, heat capacities J/(mol K), volume m³/mol and
mass density kg/m³. The default caloric datum is explicitly sensible H=0 at
298.15 K; it supplies no formation enthalpy or absolute entropy. Supply a
`CaloricReference` when another datum is needed. Entropy and chemical potential
require an entropy datum. The six curated packages currently supply gas Cp and
vapor pressure; their default material phase does not imply liquid Cp or
transport data. Missing quantities raise an error.

Gas states default to the ideal-gas closure. An explicit existing `CubicEOSSpec`
selects PR/SRK volume, fugacity, departures and derivatives on the requested
gas/liquid root. This is a single-phase evaluation; phase equilibrium belongs
to the flash model. Its Cp derivative follows the selected outer root and does
not support root-topology changes. `validity_policy="warn"` or `"ignore"` can
evaluate beyond a correlation range, but this API retains the invalid-domain
record under either policy.

Native reactor and equipment consumers can share
`PropertyPackageThermodynamics` with `SpeciesCaloricModel` data. They obtain
U(n,T), Cv and composition derivatives from the same evaluations at the current
candidate state. The common integrator owns time and heat accounting; material
mixing solves the same U relation for temperature. Accepted states and authored
operating intervals must remain inside the declared caloric domain. This
additive caloric runtime does not claim a nonideal closed-volume EOS balance.
Existing constant-Cv declarations delegate to this evaluation path; the older
liquid volumetric-Cp model and fixed-capacity phase plateaus remain explicitly
selected models. Variable-caloric phase change requires a flash/latent closure.

Run `python -m chemworld.examples.state_property_research` in an installed
environment for a property query and an actual variable-Cv reactor history.
Its temperature dependence is synthetic and retains the original native U
datum; it is an interface example rather than new material data. Production
evaluation imports no reference backend. The optional `physchem-ref` tools are
used only for the separate same-equation and model-bias checks documented in
`workstreams/world_foundation/archive/20261020-native-extension/evidence/NATIVE_PROPERTY_ACCEPTANCE.md`.

## TP/PH flash and finite phase inventories

`VLEPropertyPackage` combines existing property packages, explicit gas enthalpy
references and a declared liquid activity model. `tp_flash(...)` updates K from
the current liquid and vapor compositions; an optional existing gas EOS updates
vapor fugacity and departure enthalpy. Liquid enthalpy is the temperature
derivative of the same Psat/activity free energy, sharing the gas datum.
`ph_flash(..., pressure_Pa=..., enthalpy_J=...)` holds total H fixed and solves T.
Pure saturation uses an enthalpy lever rule. Missing/vanished phases carry zero
inventory. Unreachable H and nonconvergence raise explicitly.

TP returns a state query when no inlet H is supplied; its heat duty is then
unavailable. With explicit `feed_enthalpy_J`, TP reports the heat required to
maintain the specified T. PH reports zero added heat and its actual enthalpy
residual. It does not close the balance by assigning the residual to a heater.

This first slice supports one liquid plus one gas, two/three components and a
convex ideal liquid or certified convex binary Margules liquid. Other activity
models and uncertified/nonconvex liquids require the separate LLE/stability
solver. Critical transitions and three-phase states are outside this model.
The Raoult liquid thermodynamic model neglects liquid volume/Poynting work;
its nominal liquid holdup is a separate geometric filling approximation.
No exact liquid internal-energy/PV-work balance is claimed.

`FlashSystemSpec` declares a finite constant-pressure thermal system. A native
`vapor_liquid_equilibrium` component selects its enthalpy strategy through the
existing physical advance coordinator. Candidate H is integrated once, PH closes
each candidate, and accepted liquid/gas quantities occupy the normal phase
ledger. The normal device, feed, material routing, measurement, transaction,
resource and snapshot services consume that same inventory. Registered liquid
species explicitly declare their carrier representation, preventing a second
legacy solvent-volume inventory from being inferred from geometric volume.
Element and carrier balance checks remain active.

Run `python -m chemworld.examples.native_flash_research` for a PH query followed
by actual finite feeding, cross-phase heating, a timed phase probe, transfer and
termination. `phase_state` is an explicitly noiseless synthetic sensor, not a
claim of experimental phase identification. See
`workstreams/world_foundation/archive/20261020-native-extension/evidence/NATIVE_FLASH_ACCEPTANCE.md` for the fixed 27-state
thermo comparison and development boundaries.

## Molecular multiphase equilibrium and finite thermal exploration

`chemworld.physchem.MolecularMultiphaseSpec` declares one `MolecularCaloricModel`,
named density basins, temperature/pressure bounds and an explicit phase budget.
`solve_molecular_multiphase(spec, amounts_mol, inputs={"T": 300, "P": 600000})`
returns actual phase amounts, densities, caloric totals and stability diagnostics.
The same entry accepts `P/H`, `P/S`, `U/V` and declared mass/volume coordinates.
H, S, U and V are totals for the supplied material; mass-specific inputs use
explicit molar masses. Check `success` and the diagnostics before consuming a
result. Finite composition/density searches do not certify global uniqueness.

For a live environment, map each possible phase to a `MolecularPhaseTarget` in
`MolecularMultiphaseInventorySpec`. Several liquid targets may share a density
basin. `equilibrate_molecular_multiphase_inventory` updates selected material in
the existing `PhaseLedger`, removes vanished phases and preserves unrelated stock
and failed transactions. `molecular_multiphase_stream` reads the selected live
phase. Solver candidates and stored observations are not separate inventories.

`MolecularMultiphaseThermalSpec` declares a fixed pressure or fixed volume.
`chemworld.world.molecular_multiphase_world.build_molecular_multiphase_world`
combines it with explicit initial material, H or U, component formulas and a finite
`ThermalDeviceSpec`. The returned composition goes directly into
`ChemWorldRuntime(composition=composition)` for ordinary `heat`, `wait`,
`measure` with `phase_state`, and JSON snapshot continuation. At fixed pressure
the runtime records PΔV once; at fixed volume it advances U with zero boundary
work. This thermal world supports phase appearance/disappearance within its
declared model and bounds; continuous material feeds/transfers are not declared
by this strategy.

`MolecularPhaseBoundarySpec` provides bubble/dew points, isolated binary azeotropes
and pseudo-arclength envelopes, including turning points. Its local-coexistence
mode is explicitly distinct from a finite stability search. An optional
`MolecularSolidBoundarySpec` maps molecular chemical potentials and atom counts
to the original solid-equilibrium API with explicit element-reference shifts.
It supplies onset tangents and solid-consumer seeds, not a coupled fluid/solid
precipitation inventory.

Development qualification covered explicit PR mixtures and analytic regular-solution controls, original
runtime consumers and the solid interface. It does not qualify every EOS or
every state in a declared rectangular domain.

## Molecular interfaces from current phase inventories

`chemworld.runtime.molecular_interface_from_state(state, inventory_spec, influence, ...)`
queries a named pair of phases in the current `WorldState`. Pass the same
`MolecularMultiphaseInventorySpec` (or `MolecularPhaseEquilibriumSpec`) used by
the environment. The query takes its EOS from that declaration and reads the
current temperature, pressure, amounts and phase volumes. It checks phase
identity, type, density basin and coexistence; a vanished or changed phase is
never replaced with a cached equilibrium result.

For a pure fluid, provide `interior_density_grid`; for a freely varying mixture
profile, provide a planar `DiffusionMesh` and explicit component-density bounds.
`SquareGradientInfluence` declares constant positive-definite coefficients.
`PolynomialSquareGradientInfluence` declares temperature-dependent pure
coefficients and pair interactions; `SingularSquareGradientInfluence` explicitly
declares a deficient-rank gradient factor. Their JSON declarations are decoded
by `molecular_influence_from_dict`. A singular model does not acquire artificial
diagonal gradient energy to make it numerically invertible.

Check `result.interface.success` and its stationarity, curvature and resolution
diagnostics. Optional `area_m2` returns `surface_excess_grand_potential_J = gamma*A`.
This is an excess grand potential, not internal energy or heat; the query does
not advance time or write any ledger. An environment author can expose the
computed response through an explicitly designed tool or instrument. The
query itself is not a timed measurement or a capillary-flow process.

`fit_molecular_influence` fits explicitly selected pure coefficients or pair
interactions using uncertainty-weighted training observations and bounded
parameters. It reports failed forward trials, training sensitivity rank and
condition, active bounds, local covariance and held-out prediction errors.
Physical sensitivity diagnostics require successful recorded finite-difference
evaluations; a failed-trial penalty cannot establish parameter identifiability.
The original optimizer diagnostics and every final prediction remain available
when the physical derivative cannot be verified.
Optimizer success and predictive applicability are separate results. The current
real-data development block uses Rolo et al.'s alkane measurements in saturated
air and explicitly approximates the boundary by hydrocarbon VLE with no
noncondensable air; its errors must be read with that modeling assumption.

Development qualification covered pure, binary, ternary, singular and near-critical profiles, independent
PhasePy comparisons, original inventory consumers, installed-package use and the
complete measured-data fit with held-out predictions and mesh/domain convergence.
Earlier fitting attempts failed. Several fitted parameters
reach their declared bounds; this qualification does not remove that limitation
or establish same-boundary experimental validation for the omitted air phase.

## Strict liquid stability and finite washing

`liquid_tpd_minimum(activity_model, composition, temperature_K=...)` minimizes
the actual tangent-plane distance on a deterministic set of simplex starts.
`equilibrate_liquids(activity_model, amounts_mol, temperature_K=...)` minimizes
constrained mixing Gibbs energy, refines equality of phase chemical potentials,
checks component balances and repeats TPD searches on both resulting phases.
A stable feed returns one liquid. A remaining negative post-split TPD rejects
an unsupported additional phase. Binary/ternary models support at most two
liquids; the finite numerical search is not a proof of exhaustive global
optimization for arbitrary activity functions.

`mixing_chemical_potentials_RT` reports `ln(x_i gamma_i)`, relative to the
common pure-liquid standard and divided by RT. Common pure reference terms
cancel across equal-T liquid phases; these are not fabricated absolute chemical
potentials. Activity coefficients use the existing parameter model, shared with
the property and contact calculations.

`StrictLiquidSystemSpec` declares an isothermal material law, component molar
holdups and finite wash policy. A `liquid_equilibrium` component and matching
`CompositionPhysics.liquid_models` entry enable the normal mix/wash, feed,
clock, measurement, material routing and snapshot services. Actual wash liquid
comes from a finite named supply; discarded material is an explicit molecular
outlet. A miscible wash retains one mixed liquid. Droplet entrainment uses the
same inventory transfer function as the empirical contact model. The phase
probe reports liquid count and second-liquid fraction as synthetic native
channels. Mixing heat and PV work remain explicitly unmodeled in this
isothermal slice; no heat-control operation is implied.

The older distribution-coefficient contact model remains a separate model.
Its `minimum_tpd_like` includes an empirical partition drive and is explicitly
marked `thermodynamic_stability_certified=False`. Strict liquid equilibrium
does not use that drive or fall back to it after a solver failure.

Run `python -m chemworld.examples.native_liquid_research` for finite mixing,
washing, a timed observation, transfer and termination. The example's Margules
parameters and component identities are synthetic. See
`workstreams/world_foundation/archive/20261020-native-extension/evidence/NATIVE_LLE_ACCEPTANCE.md` for common-tangent,
ternary and actual PhasePy validation.

## Conservative diffusion primitive

`chemworld.physchem.finite_volume` provides planar and spherical one-dimensional
Fick operators in SI units. `DiffusionMesh` defines physical volumes and face
areas; `ConservativeDiffusion.evaluate(amounts_mol)` returns molar derivatives
and signed face rates without advancing time. Positive prescribed boundary flux
means outward flux in mol/(m² s). A spherical center requires symmetry.

`operator.couple(existing_inventory_keys, process_id=..., owner_id=...,
boundary_enthalpies_J_mol=...)` supplies a process and material ports to an
existing `CoupledSystem`. Explicit boundary enthalpies are required for open
boundaries. BDF/Radau use the assembled sparse stencil; outward fixed-flux
depletion stops at its actual event. The caller retains inventory ownership.
This is the shared spatial primitive for subsequent electrode/particle models;
these mathematical consumers do not implement a battery or a new PDE runtime.

## Finite dynamic distillation

`DynamicColumnSpec` reuses `ShortcutColumnSpec` for the hardware envelope and
`VLEPropertyPackage` for the thermodynamic law. It has 2–8 equilibrium trays,
a finite reboiler, condenser and product collector. Each compartment advances
actual component amounts and H; task-18 PH closes its phase allocation at each
candidate. Vapor withdrawal and liquid overflow use declared finite holdup laws.
Heat and material streams enter one shared BDF advance. This model excludes
full tray hydraulics, pressure-drop networks and simultaneous reactions.

A `dynamic_distillation` component and matching `CompositionPhysics.column_models`
entry select this strategy. `dynamic_column_world_spec(spec, initial, ...)`
constructs a native apparatus from explicitly supplied properties and initial
inventories. Real `distill`, heat/wait and timed observations use the common
coordinator; dry, flood, temperature, collection and native vessel-volume
boundaries stop at their actual roots. Product history records actual collector
increments over time. `collect_fraction` moves actual liquid n and H into its
separate retained phase; it never resets the column or manufactures recovery.

Distill controls are target temperature, duration and finite `reflux_ratio`.
Native event JSON additionally accepts numeric `full_reflux: 1` (with the usual
required reflux field present); `full_reflux: 0` resumes finite collection.
The unchanged fixed Gym numeric carrier explicitly rejects this additional
field and directs callers to event JSON. The old `distillation` component
remains a labelled `shortcut_estimate`, without stage histories or a fallback
from failed dynamic integration.

PhaseLedger alone owns bulk n. Material and stationary support H occupy separate
typed slots. Replacing material retains apparatus heat and device calibration,
then equilibrates the new fluid plus retained body through PH. Batch reset uses
the declared initial thermal boundary while retaining hardware settings.
Snapshots retain all compartments and controls. The synthetic phase probe
observes the reboiler compartment. Liquid volume remains the geometric holdup
approximation of task 18, with no exact liquid U/PV-work claim.

Run `python -m chemworld.examples.native_distillation_research` for five actual
actions: distill, timed measurement, finite collection, reflux change and stop.
The example needs no optional reference packages. Its Cp, latent heat and vapor
pressure data are explicitly synthetic. The fixed dynamic grid, IDAES MESH
steady comparisons, Fenske limit and time refinement are documented in
`workstreams/world_foundation/archive/20261020-native-extension/evidence/NATIVE_DYNAMIC_DISTILLATION_ACCEPTANCE.md`.

## Finite vacuum concentration

`CoupledConcentrationSpec` uses the same state-dependent VLE properties and PH
closure as flash and distillation. A finite pot and receiving bottle each own
component inventory and H. Actual retained pot vapor supplies the withdrawal
port. The existing equipment capture fraction routes vapor to the bottle;
uncaptured vapor crosses an explicit component/H boundary. Captured vapor
remains bottle inventory until PH and the finite cooling port condense it.
There is no separate instantaneous condensate calculation or latent-heat debit.

A `vacuum_concentration` component and matching
`CompositionPhysics.concentration_models` entry select the strategy.
`concentration_world_spec(spec, initial, ...)` constructs a native apparatus.
`concentrate` enables heating and vapor transport. Heat/wait and timed in-situ
measurements continue the same physical advance. Heater pause leaves the
transport setting unchanged. An operating stop closes transport and heat at its
actual root; subsequent passive measurement can cool the bottle. Duration
exhaustion with an unmet solvent endpoint is a valid result.

Pot, bottle, condenser gas saturation, residual liquid, target recovery,
temperature and vessel volume are finite boundaries. Accepted time, heater
delivery, cooling and outlet integrals settle once. Native material receipts
authenticate the capture split, component withdrawal and transported enthalpy;
reusing an action name does not authorize a new outlet. Numerical failures roll
back current material, H and time. PhaseLedger owns all current bulk material;
typed material and vessel slots distinguish fluid H from stationary heat
capacity. Material replacement retains body heat and calibration, and PH
reconciles the fresh charge without advancing time. Snapshots retain actual
inventories, controls and cumulative boundary receipts.

An explicitly nonvolatile component uses a liquid Cp/H reference at the model
pressure and exactly zero gas allocation. Missing gas properties remain
unavailable. The shared RR report retains a trace liquid fraction independently
when its vapor fraction rounds to one; the undefined beta=1 objective is null
for positive nonvolatile material. This slice supports ideal mixtures with one
or two volatile species and an optional nonvolatile target, at fixed pressure,
without simultaneous reactions. Liquid volume remains a declared geometric
holdup approximation; liquid PV/Poynting work is neglected.

`simulate_vacuum_concentration(CoupledConcentrationRequest(...))` and the native
operation use the same contribution and common compartment solver. Existing
`VacuumConcentrationRequest` inputs explicitly prescribe a fixed evaluated
operating profile. Their differential approximation remains labelled
`declared_fixed_operating_profile`, shares phase-advection and solver utilities,
and is never a fallback for a failed native solve. Its legacy service now charges
actual elapsed time and delivered heat, including early stops.

Run `python -m chemworld.examples.native_concentration_research` for concentrate,
timed measurement, heater pause, wait and termination. The source/installable
example uses synthetic properties and needs no optional reference package. The
fixed nine-control grid, two material fixtures, Rayleigh holdup limit and native
snapshot continuation are recorded in
`workstreams/world_foundation/archive/20261020-native-extension/evidence/NATIVE_CONCENTRATION_ACCEPTANCE.md`.

## Persistent finite sorbent contact

`SorbentKineticsSpec` supplies competitive Langmuir affinities, local LDF rates
and adsorption energies. Fluid concentration and temperature come from the
same candidate used by `CoupledSystem`; bound molecules enter its inventory
once. Temperature-dependent affinity and bulk/bound U offsets must use the
same adsorption energy. Carrier heat capacity joins the common temperature
balance; no separate adsorption heat is charged. Unequal component mobilities
use the documented finite-vacancy uptake constraint to preserve shared sites.
This is a zero-dimensional, single-site-class liquid contact with two competing
components, without axial bed, pressure-drop or intraparticle resolution.

`sorbent_contact` components select `CompositionPhysics.sorbent_models` and
the existing coupled thermal strategy. A fixed carrier owns its loading in an
equipment slot; a mobile carrier moves actual mass, sites, heat capacity and
bound molecules with its material portion. Material replacement retains fixed
loading and apparatus heat, mixing fresh material through the common U model.
Actual dry/heat/wait advances share one clock and solver. Solver failure rolls
back fluid, loading, temperature and physical time; established failed-action
cost/risk penalties still apply.

`CondensedBackgroundSpec` explicitly defines volumetric Cv and U for an L-only
carrier, with condensed PV/Poynting work neglected. It does not invent molecular
carrier or gas data. A native sorbent fixture must declare a finite carrier
source vessel. Existing add_solvent debits that actual stock and transports its
energy; routing carries solute U, carrier U and mobile bound material. The static
wash source has no active thermal process. Source exhaustion rejects the input,
and normal material/batch initialization preserves finite supplies. Explicit
entity replacement and new-world reset remain lifecycle boundaries.

The legacy standalone isothermal drying request now delegates to the same
instantaneous finite-fluid LDF contribution. It retains an explicit finite fresh
bed request and the final spent outlet, without certifying unprovided caloric
data. The legacy laboratory runtime owns a 20 g unused-media kit per vessel;
each actual dry packet debits it and leaves through an authenticated kg entity
receipt. Repeated dry or material reset cannot manufacture fresh media. No
second current spent-bed stock is maintained.

Run `uv run --no-sync python -m chemworld.examples.native_sorbent_research`
for eight existing operations through adsorption, finite wash, heating, timed
measurement and stop. All constitutive data are explicitly synthetic. The fixed
12-condition contact grid, independent equilibrium references and three native
cycles are recorded in
`workstreams/world_foundation/archive/20261020-native-extension/evidence/NATIVE_SORBENT_REGENERATION_ACCEPTANCE.md`.

## Finite transient working electrode

`electrode_world_spec()` creates a native finite planar O + e ↔ R system.
The shared conservative diffusion operator and `CoupledSystem` advance both
species profiles, Butler–Volmer exchange and double-layer charge together.
The Robin boundary includes the first half-cell diffusion resistance; the
outer boundary is closed. Positive current is anodic. Material and signed
charge relations are checked before a candidate is returned for transaction
validation. Electrical work is recorded, without claiming complete cell
energy closure or counterelectrode chemistry.

```python
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.electrode_world import electrode_world_spec

session = ChemWorldRuntime(composition=electrode_world_spec())
session.reset(seed=7)
session.execute({"operation": "configure_electrical_control", "potential_V": 0.1})
session.execute({"operation": "wait", "duration_s": 5})
reading = session.execute({"operation": "measure", "instrument": "electrical_meter"})
print(reading.observation.native_record())
```

Set `ramp_duration_s` on the control action for a linear potential ramp from
the current setpoint. Configuration does not advance time; `wait` and the
meter's one-second in-situ acquisition use the same physical coordinator.
Measurements expose terminal V and total A, without internal concentrations
or interfacial potential. This synthetic meter is noiseless and nonconsuming.

Profiles live once in material phase compartments. Charge and cumulative
electrical work belong to the electrode equipment; material reset preserves
them, while explicit equipment replacement leaves the actual electrolyte
profile intact. Snapshots preserve the control programme, inventories and
charge. Static equipment supplies remain separate from moving liquid; dynamic
or unsupported material state still requires a compatible advance strategy.

The first model uses constant diffusivities, sufficient supporting electrolyte,
isothermal operation and finite series resistance. It excludes migration,
convection, homogeneous reactions, current control and full-cell thermal
physics. Existing fixed diffusion-layer and RC models remain explicit simpler
alternatives sharing the same electrochemical constants and BV factors.

Run `uv run --no-sync python -m chemworld.examples.native_electrode_research`
for three controls, waits and native measurements followed by termination.
Fixed numerical checks and development failures are recorded in
`workstreams/world_foundation/archive/20261020-native-extension/evidence/NATIVE_ELECTRODE_ACCEPTANCE.md`.

## Programmed chromatography and persistent instrument state

`chemworld.world.chromatography_world.chromatography_world_spec("hplc")`
(or `"gc"`) creates a finite synthetic three-analyte world with source bottles,
a receiver, material routing, waste and one persistent chromatograph. It uses the
same runtime, resource ledger, measurement records, snapshots and replay as other
native worlds; no model provider or API key is required.

```python
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.chromatography_world import chromatography_world_spec

session = ChemWorldRuntime(composition=chromatography_world_spec("hplc"))
session.reset(seed=12)
session.execute({"operation": "configure_chromatography",
                 "instrument": "programmed_hplc", "program_index": 1})
# Initially the receiver contains blank solvent. Route a finite source aliquot
# into it to measure a standard or sample (see the complete example below).
result = session.execute({"operation": "measure", "instrument": "programmed_hplc"})
print(result.observation.channel("peak_1_area"))
```

Program 0 is constant (298.15 K, HPLC mobile-phase fraction 0.2); program 1
ramps over 30 s to fraction 0.8 for HPLC or 338.15 K for GC. Program 2 uses
knots at 0, 15 and 30 s, reaching 20% of that change at 15 s. These programmes
reuse the existing retention equations, with explicit programme delay and a
time-integrated elution criterion; the constant case regresses to the existing
isocratic/isothermal method. Curves reuse the Gaussian peak basis.

Each acquisition consumes 0.0002 L, costs 0.05 currency and adds 60 s of offline
analysis time without evolving the host. Actual analyte residue belongs to an
equipment-owned state slot. A blank washes this inventory into the recorded
external waste outlet; it is not a reset. Use count, gain/baseline drift,
programme and calibration persist through material replacement and snapshot
restore. Full reset restores the declared initial instrument.

The public record contains a signal curve, fitted peak areas and concentration
estimates with units and quality, plus the declared peak templates and response
calibration needed to reproduce the inverse fit without simulator state. This
finite instrument assumes known peak identities/shapes; it does not identify an
unknown retention mechanism. `calibrate_chromatography` records the last
acquired public peak areas with declared `standard_1_M`, `standard_2_M` and
`standard_3_M`; at least three standard acquisitions fit the existing response
calibration. It does not inspect hidden concentrations, detector gain or baseline.
Saturation, overlap and incomplete elution invalidate estimates, not completed
sampling. Re-reading cached observations consumes no sample, time or random draw.

Run the complete standard–sample–blank sequence twice for both instruments and
all three programmes, including JSONL replay and independent residue/waste checks:

```sh
uv run --no-sync python -m chemworld.examples.programmed_chromatography --output /tmp/chromatography-example
```

Choose a new output directory; existing results are never overwritten. The same
module runs from an installed wheel. This is a finite development example, not
a fitted real column: three synthetic analytes, symmetric peaks, lumped residue,
fixed offline duration and a non-evolving host. Continuous mobile-phase usage,
column spatial transport and instrument thermal utilities are not modelled, so
no energy-closure or general gradient-column claim is made.

## Native single-particle lithium battery

`battery_world_spec()` installs a finite battery with one spherical particle
population per electrode. Each population uses the same conservative diffusion
operator as the planar electrode. An internal lithium transfer couples the two
surfaces to the applied current; voltage follows the synthetic OCV curves and
symmetric Butler–Volmer overpotentials. Current, lithium and delivered electrical
work advance in the common solver, with actual surface and voltage cutoff events.

```python
from chemworld.physchem.battery_models import BatterySPMSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.battery_world import battery_world_spec

spec = BatterySPMSpec()
session = ChemWorldRuntime(composition=battery_world_spec(spec))
session.reset(seed=4)
session.execute({"operation": "configure_electrical_control",
                 "current_A": spec.current_at_c_rate(0.5)})
session.execute({"operation": "wait", "duration_s": 600})
reading = session.execute({"operation": "measure", "instrument": "battery_meter"})
```

The same electrical-control action accepts exactly one supported quantity:
`current_A` or `potential_V` for this battery, `potential_V` for the finite working electrode.
Both share scalar ramp interpolation and the native meter contract; a source
rejects modes and limits it does not support. Positive battery current means
discharge. C-rate is defined from the initial usable lithium capacity, not
recomputed from subsequent measured capacity. The native meter exposes only
terminal V and A. Its one-second acquisition advances the same battery state.

Lithium profiles, signed charge and work belong to the battery equipment and
are saved only once. Material reset retains them; replacing the actual battery
requires explicit new device state. A surface cutoff preserves the real elapsed
time and inventory. Reconfiguring to zero current allows subsequent relaxation.
The shared ledger uses positive electrical work into the system, so delivered
battery work is recorded with the opposite sign.

The default parameters are synthetic and use uniform unresolved electrolyte. The default
isothermal option remains available; electrolyte transport remains unresolved,
and the optional finite SEI extension is described below. Run
`uv run --no-sync python -m chemworld.examples.native_battery_research` for a
discharge/rest/recharge history. PyBaMM is used only by the optional reference
block; the ordinary installation runs this example without it. Numerical scope,
all development failures and fixed comparisons are recorded in
`workstreams/world_foundation/archive/20261020-native-extension/evidence/NATIVE_BATTERY_SPM_ACCEPTANCE.md`.

### Released Chen2020 parameters in the original SPM

`battery_parameter_set("Chen2020", version="PyBaMM-25.4.2")` reads a bundled,
immutable catalog of all 94 released entries: 86 numerical inputs, six named
constitutive functions and two metadata entries. Every field retains its source,
unit and explicit consumed or retained-not-consumed classification. Production
does not import PyBaMM or execute external parameter callbacks.

```python
from chemworld.physchem import battery_parameter_set

parameters = battery_parameter_set()
cell = parameters.build_spm(maximum_current_A=10.0, radial_cells=64)
session = ChemWorldRuntime(composition=battery_world_spec(cell))
session.reset(seed=4)
```

This projection is a single-material, fixed-temperature, fixed-electrolyte SPM
without heat or side reactions. It derives particle volumes, reaction areas,
initial lithium and exchange-current coefficients from the source values.
The separately sourced OCV endpoint convention applies its clipped coordinate
to both the material expression and rational barrier. The declared cell
envelope occupies 0.0242 L; active volumes still determine lithium capacity.
Snapshots verify the parameter binding against the actual physical declaration.
Other material sets and concentration-dependent transport consumers remain open.

`parameters.current_at_nominal_c_rate()` uses the source's nominal 5 Ah.
The existing `cell.current_at_c_rate()` keeps its initial usable stoichiometric
capacity convention; neither is a measurement of voltage-window capacity.
The caller declares the source current limit explicitly: the released default
5 A current is not an equipment limit. Queue, meter, signed work and reset
semantics remain those of the original battery runtime. Full dynamic source
qualification is recorded separately from catalog and functional checks.

### Cycles and a consistent synthetic thermal model

Pass `BatterySPMSpec(thermal=BatteryThermalSpec(...))` to evolve temperature.
The cell's chemical energy is integrated from the same linear OCV functions,
and the common solver balances terminal electrical work, chemical storage,
heater input and ambient loss. Kinetic, ohmic and diffusion dissipation are
independently derived and checked; they are not added again as reaction heat.
The OCV curves have no explicit temperature dependence, so reversible entropy
heat is zero. Positive OCV slopes are excluded from this convex thermal model.
`series_resistance_ohm` optionally adds actual voltage drop and Joule heating.

`battery_world_spec(spec, thermal_device=ThermalDeviceSpec(...))` attaches the
existing heater or isothermal bath. `heat`, `wait` and thermal pause/resume
use that same device; `stirring_speed_rpm` is optional on `heat`. A finite-power
heater changes actual temperature gradually. An ideal bath must start at its
setpoint and records its actual compensation heat.

Use `queue_electrical_stage` with exactly one of `current_A`/`potential_V`, a
finite `stage_duration_s`, and optionally one native terminal condition:
`stop_voltage_below_V`, `stop_voltage_above_V`, or `stop_abs_current_A`.
Zero-current stages are rests. Stages append to the source-owned queue;
`wait` advances the common controller, whose interval loop also serves the
existing thermal/gas programmes. No battery-specific scheduler is installed.
Completed queues switch to zero current. Explicit `configure_electrical_control`
clears the queue and installs a manual step/ramp. Voltage control solves for
the current from the actual profile; an unreachable voltage remains at the
source current limit, with the actual voltage and limit status recorded.

Each stage transition records the terminal reading and actual elapsed time.
Physical cutoff events retain the reached state; numerical failures roll back
the complete candidate action, including preceding candidate stages, while
retaining diagnostics. Snapshots preserve the queue, cursor, temperature and
inventories exactly. Material initialization is an evaluator boundary: it
retains battery SOC/profiles and uses the declared initial temperature, recorded
in its lifecycle receipt. Continuous cycling uses no such initialization.

Run `uv run --no-sync python -m chemworld.examples.native_battery_cycles` for two
CC/CV/rest/charge cycles. The nine thermal comparisons against optional PyBaMM
and two native cycles are documented in
`workstreams/world_foundation/archive/20261020-native-extension/evidence/NATIVE_BATTERY_THERMAL_ACCEPTANCE.md`.

### Finite SEI growth and retained degradation

Add `sei=SEISpec()` to `BatterySPMSpec` to enable solvent-diffusion-limited SEI.
The explicit total reaction is `2 Li + EC -> Li2CO3 + C2H4`, corresponding to
two lithium ions and two electrons at the negative interface. EC, carbonate,
retained ethylene and both active-lithium profiles live once in the cell's
equipment slot. Film thickness and resistance are derived from carbonate stock;
there is no independent thickness inventory or per-cycle capacity decrement.

The side current is `-2 F r`. Negative intercalation carries `I + 2 F r`, while
the positive electrode carries I. Both the particle boundary and negative BV
relation use that main current. Film resistance changes measured terminal
voltage. Its dissipation and reaction heat use the same chemical state function
as the thermal model. The irreversible side process stops at its electrochemical
driving-force boundary; high-rate or unfavorable conditions are not silently
extrapolated into an endergonic growth regime.

The initial film is explicitly present in the initial elemental inventory. A
closed cell consumes its actual finite EC; no hidden solvent reservoir is used.
The analytic maintained-solvent check adds a declared external feed instead.
Species formulas make Li/C/H/O checks meaningful at the runtime transaction.
Material initialization preserves the whole cell's chemical history; replacing
the cell uses a new explicit equipment inventory. Snapshot and Agent replay
retain film, solvent, gas, active lithium, electrical controls and temperature.

The source queue bounds pending work at 32 stages. Completed stages remain in
operation history and are removed from the pending list when new work is added;
`stage_index` remains absolute and `stage_offset` locates the retained list.
Thus repeated finite programmes do not exhaust a lifetime queue limit.

Run `uv run --no-sync python -m chemworld.examples.native_battery_degradation`
for ten continuous native cycles. The synthetic model includes finite solvent,
film resistance and the two-electron accounting; it excludes plating, cracking,
electrolyte transport and gas-pressure mechanics. Fixed analytic, nine-condition
PyBaMM and ten-cycle checks are in
`workstreams/world_foundation/archive/20261020-native-extension/evidence/NATIVE_BATTERY_SEI_ACCEPTANCE.md`. Windowed discharge
charge and lost-lithium equivalent charge are reported separately: neither is
used as a direct substitute for the other.

### Finite pump, pipe and heat-exchanger transport

`transport_world_spec()` installs a real finite feed and receiver connected by
a bounded pump, pipe and optional Ergun bed resistance. `set_flow_rate` sets a
request in mL/min without advancing time; `run_flow` or `wait` advances both
inventories together. Zero request stops pumping. `run_flow` may set the utility
temperature through `target_temperature_K`; otherwise the installed value is used.

The pump solves the greatest flow permitted by the requested rate, pressure
head and shaft-power limits. `PumpSpec` and `pump_operating_point` accept the
connected device's monotone pressure-load law for reuse by a membrane. The
example reaches about 12.21 mL/min for a 20 mL/min request and stops at the finite
pickup heel without charging the unexecuted time tail.

The common finite-inventory solver advances component amounts and enthalpy.
Liquid properties use native packages at the candidate T/x; the exchanger uses
the actual mass rate after pump heating. Hydraulic work heats the fluid, shaft
inefficiency is explicit external loss, and utility heat uses the existing
Nu/U/NTU functions. Source and receiver share pressure and elevation; the
pipe/exchanger has negligible holdup and the utility is a declared constant
temperature boundary, not an untracked coolant inventory.

Material slots own fluid enthalpy; vessel slots own container heat. The pump
owns its specification and control. Default batch initialization preserves the
finite feed and its local age. Explicit receiver material replacement retains
the actual container heat and recloses temperature without advancing time.
Source exhaustion, full receiver and outlet property limits stop the process;
numerical failure rolls back the candidate with a diagnostic.

`flow_meter` reports mL/min, Pa, W, K and L through native channels. Native, Gym,
CLI and Agent interfaces share execution, accounting and exact replay. Run
`uv run --no-sync python -m chemworld.examples.native_liquid_transport`.
The bounded synthetic model excludes reactions, recirculation, cavitation,
two-phase flow and an empirical industrial pump curve. Existing spatial PFR
experiments retain their declared geometry model. Validation and optional
fluids/ht comparisons are in
`workstreams/world_foundation/archive/20261020-native-extension/evidence/NATIVE_TRANSPORT_ACCEPTANCE.md`.

## Finite crystal growth and dissolution

`crystal_thermal` components install `CompositionPhysics.crystal_models` beside
an explicit thermal model. Existing seed, cool_crystallize, heat, wait, filter
and resuspend operations advance the same finite slurry at its actual
temperature. A requested setpoint supplies bounded heating or cooling; it does
not allocate dissolved molecules in advance. PhaseLedger owns current bulk
molecules. Cohort constituent rows decompose the solid into particle counts,
diameters, seed provenance and occluded impurity; they are not additional stock.

The declared van't Hoff curve and finite radial growth/recession law share
liquid/solid caloric references. One rate/energy assembler and ODE backend
advance material, temperature and utility integrals. Geometry coordinates have
units m and mol/mol, reconstructing actual molecular stocks for each candidate.
Zero-diameter roots remove hosts and release their actual occluded material.
Finite impurity capture tapers only below the serialized inventory resolution
(default 1e-8 of total impurity); actual remaining stock and numerical
corrections are retained and checked against unchanged independent balances.

Nucleation uses explicit temporal birth quadrature, normally 30 s: actual
molecules enter nucleus-size reservoirs and join growing cohorts at interval
ends. This finite-resolution approximation has a separate interval-refinement
check. There is no aggregation, breakage, polymorph or axial population model.
Authored condensed Cv and U require negligible condensed PV work; missing
phase calorics remain unavailable. Simultaneous bulk reaction and crystal
thermal processes are outside this slice.

The native ideal filter retains all particles, solid target, seed and occluded
impurity, and 8% wet mother liquor. Its 480 s advances the same active thermal
control. Resuspension stores historical filtrate at its own temperature and
mixes declared incoming carrier through U, without instant dissolution. Existing
resource cards limit fresh carrier and seed. Material replacement retains
apparatus heat; snapshots retain cohorts, controls and resource continuation.
Current container transfers require equal temperatures when mixing occupied
native crystal compartments. Numerical failure rolls back physical state and
time, retaining the established failed-action penalties.

The legacy laboratory slurry also uses the finite radial contribution, with its
explicit condensed enthalpy boundary. Its carrier Cp is not relabelled Cv. The
legacy 96%/92% mechanical retention approximation remains separate and does not
certify a complete native U balance. Quench legacy bulk reactions before slurry
thermal holds. The standalone prescribed-temperature crystallization provider
remains an explicit reference approximation, not a failed-native-solver fallback.

Run `uv run --no-sync python -m chemworld.examples.native_crystal_research`
for eight operations and 901 s of actual time using synthetic properties and
no optional reference library. Fixed contact, birth-refinement and two-cycle
coverage and all attempted outcomes are recorded in
`workstreams/world_foundation/archive/20261020-native-extension/evidence/NATIVE_CRYSTAL_THERMAL_ACCEPTANCE.md` and its JSON.

## Finite carbonate water chemistry

`aqueous_world_spec` installs a declared `aqueous_mineral` component with actual
water, H+/OH-, CO2*, bicarbonate/carbonate, Ca, Na, Cl and finite pure calcite.
Molality uses actual remaining water kg; geometric liquid volume is a separate
declared holdup. Element, carbon, charge, water dissociation and carbonate
mass action close together. Finite calcite may appear or deplete; no balancing
ion or mineral reservoir is supplied after the solve. Derived speciation is
checked against PhaseLedger before a read and is not another material stock.

The same closure serves `add_component` from finite neutral solution stocks,
`wait`, aqueous-only `sample` and the non-destructive `aqueous_ph` electrode.
Sampling retains the separate mineral phase. The existing pH meter model uses
hydrogen activity and the session RNG; the older simplified acid/base model
remains explicit for historical tasks. Both observation noise schedules support
exact snapshots and resource continuation. Invalid inputs and numerical failure
preserve physical inventories and time under the existing transaction rules.

This synthetic system is restricted to 298.15 K, low pressure and ionic strength
<=.1 mol/kg. Constants are pKw=14, pKa1=6.35, pKa2=10.33, log10 Ksp=-8.48,
Davies A=.509/b=.3, neutral gamma=1 and water activity=1. Actual water still
participates in the reaction element balance. These equilibrium constants do
not provide caloric data, temperature dependence or mineral kinetics. Mineral
advection and thermal mixing are outside this slice.

Run `uv run --no-sync python -m chemworld.examples.native_aqueous_research`
without optional libraries. `scripts/check_native_aqueous.py --reaktoro-prefix
<compatible-prefix>` runs the fixed 33-condition block, including 32 comparisons
against Reaktoro 2.13.0 with the same custom standard states and activities,
one domain rejection and four analytical limits. The reference lives outside
the default runtime dependency set. Its CLI flushes results before bypassing
an observed Windows callback-finalization failure; calculation failures retain
nonzero exit status. Coverage and all outcomes are in
`workstreams/world_foundation/archive/20261020-native-extension/evidence/NATIVE_AQUEOUS_MINERAL_ACCEPTANCE.md` and its JSON.

## Finite calcite kinetics and material surface ownership

`mineral_world_spec` installs a `mineral_kinetics` owner beside the same bounded
carbonate chemistry. A single shared BDF solve advances finite calcite while
every candidate closes fast aqueous reactions at the current solid amount.
The shrinking-area law is dn/dt=-A k(1-Omega), A=A_ref(n/n_ref)^(2/3).
PhaseLedger owns molecules; a phase-bound material slot owns reference area
and reference mol, without registering that reference as another atom stock.

The native aqueous and calcite ports move their own phases. A solid aliquot
carries proportional reference geometry; aqueous transfer/sampling retains
the solid and its surface. Equipment stays in place. The first model supports
one evolving mineral family per compartment. Different origins or different
size histories require separate compartments. Same-progress portions of one
lot can split and recombine without changing their reference area or stock.

At exhaustion, actual dissolved atoms remain and the exhausted surface is
removed. Seedless supersaturation has zero area and does not nucleate. A
declared zero rate/surface retains its stock. Chemical exhaustion or a numerical
saturation tolerance stops reaction while the supported isothermal hold fills
the remaining physical wait; these times are recorded separately. Finite acid
input changes the same water state before subsequent kinetic time. The pH
electrode uses the committed speciation after its actual acquisition interval.

The temperature, activity and caloric limits of task 32 still apply. This is
neither a reactive-transport database nor a distribution of independently
evolving crystal sizes. Run `uv run --no-sync python -m
chemworld.examples.native_mineral_research` for ten native actions, phase
transfer and a 1402 s shared-clock history without an upstream dependency.

The optional `scripts/check_native_minerals.py` compares twelve fixed
trajectories/120 states and nine early-time limits. Its independent reference
uses Reaktoro 2.13.0 constrained Gibbs chemistry and DOP853 integration; it does
not call the native algebraic solver/ODE backend or treat an upstream implicit
finite time step as an exact continuous trajectory. Coverage, all failures and
results are in `workstreams/world_foundation/archive/20261020-native-extension/evidence/NATIVE_MINERAL_KINETICS_ACCEPTANCE.md`
and its JSON summary.

## Finite dilute reverse osmosis

`membrane_world_spec()` installs a finite, well-mixed feed chamber, a persistent
membrane/hydraulic drive, and a finite permeate collector. The initial example
contains 1 L of water plus dilute NaCl. Native physical-world capacities come
from the declared vessel hardware; the legacy template size range is unchanged.

`set_flow_rate` requests a swept water flow in mL/min. `run_flow` and `wait`
advance the same common finite-inventory solver. The task-24 pump function
limits actual flow by required membrane pressure, pipe resistance, maximum
head and shaft power. Pressure is derived from the coupled solution-diffusion
and concentration-polarization equations, not from requested rejection. The
remaining feed is the concentrate: there is no duplicate concentrate stock.

The native `membrane_meter` exposes water flux in m/s, membrane pressure in Pa,
shaft power in W, feed/outlet/collected salinity in mol/m^3 and collected L.
Instantaneous permeate salinity differs from the collector's accumulated
composition. `collected_fraction` is collected water divided by all currently
retained water; after explicit material replacement it is not cumulative
recovery against all historical feed. Task-specific recovery uses the actual
input/output history. Zero-flow outlet salinity reports the model's right limit.

With finite salt permeability, zero-drive permeate concentration tends to the
bulk concentration. A perfectly rejecting membrane instead has a finite
osmotic threshold. These are distinct limits. Reverse drive is unsupported;
finite pickup, collector capacity, no forward pump drive and concentration
domain limits retain the reached state/time. Numerical failure rolls back the
candidate. Batch initialization preserves both actual liquid inventories and
their local ages. Explicit material replacement records its exchange while
preserving membrane identity and the other vessel.

This first model is isothermal at 298.15 K with ideal dilute osmotic pressure,
water density 1000 kg/m^3 and negligible salt partial volume. Bulk/interface
concentrations stay within 0–100 mol/m^3. Hydraulic displacement pressurizes a
dead-end chamber with negligible compression storage and membrane/pipe holdup.
No recirculation, compressible transient or collector backflow is implied.
Fouling is an explicit option described below. Ideal mixing enthalpy is zero; the boundary removes actual pump work
as heat. This scope does not predict solution heats or high-pressure calorics.

Run `uv run --no-sync python -m chemworld.examples.native_membrane` for two
successive half-hour intervals and native readings. Independent WaterTAP local
SD/CP equation comparisons use the same imposed ideal boundary properties;
they do not claim a full WaterTAP flowsheet or its default property package.
Coverage and all attempts are in
`workstreams/world_foundation/archive/20261020-native-extension/evidence/NATIVE_MEMBRANE_ACCEPTANCE.md`.

## Persistent membrane fouling and finite rinsing

Pass `MembraneSpec(fouling=MembraneFoulingSpec())` to `membrane_world_spec` to
include an inert SiO2 pollutant, a finite pure-water wash tank and a finite
waste tank. Feed pollutant moves into an equipment-owned film; the same film
inventory sets resistance in the same RO flux/pump solve. Desorption returns
material to feed. No second membrane solver, clock or mass copy is introduced.

`set_flow_rate` accepts optional `flow_path="process"` or `"rinse"`. Omitting
the path retains its current selection. The clean-only membrane, finite
transport path and old PFR reject a rinse path they do not implement. Switching
the ideal valve consumes no time or liquid; `wait` or `run_flow` performs the
actual operation. The rinse bypass isolates feed, carries wash over the
membrane surface and sends all detached pollutant into waste. It has zero
holdup and no membrane permeation. Water use and pump work follow actual flow;
zero flow does not clean. Empty wash or full waste stops at the reached time.

Batch boundaries and replacing feed preserve the membrane and bound film.
Replacing a membrane uses the common equipment lifecycle and records the
removed bound inventory. This evaluator initialization is not free agent
cleaning. The public meter adds wash/waste volumes, rinse flow and waste
pollutant mass; it does not expose hidden fouling parameters or bound inventory.

This is a synthetic dilute suspension model: SiO2 has zero partial volume and
adds only hydraulic resistance, with no osmotic, reaction or caloric term.
The 298.15 K boundary removes shaft work. The rinse route is not a model of
chemical cleaning, backwash or pump/valve transients. Clean limits retain the
task-34 WaterTAP comparison; fouling uses independent mass balances, analytic
desorption and time-step refinement.

Run `uv run --no-sync python -m chemworld.examples.native_membrane_fouling` for
three process/rinse cycles using the same finite tanks. The nine fixed
development cycle units and all attempted outcomes are in
`workstreams/world_foundation/archive/20261020-native-extension/evidence/NATIVE_MEMBRANE_FOULING_ACCEPTANCE.md` and its JSON.

## Finite binary solid equilibrium

`solid_world_spec()` installs a finite A/B specimen, the existing thermal
device and a synthetic phase meter. `SolidSolutionSpec` declares one or two
substitutional phases with ideal mixing and L0/L1 Redlich–Kister terms. The
solver checks common tangents against every stationary point of every declared
phase. A nominal phase can split into two composition domains; a grid only
initializes the tangent solve. The existing phase ledger stores each domain's
actual A/B inventory, with no separate mass copy in equipment settings.

`heat` and `wait` use the common finite-compartment enthalpy integrator. At each
candidate enthalpy, stable phase equilibrium determines temperature and phase
composition. The existing finite-power heater/cooler supplies the actual heat;
time and utility use are settled once. Material and batch replacement retain
the specimen holder's sensible heat and equilibrate it with the fresh material;
vessel replacement initializes the holder too. These are recorded evaluator
initializations, not free experimental actions.

The meter reads committed phase fractions/compositions and temperature in K.
Native/Gym/CLI/Agent sessions and independent-process JSON continuation share
the same implementation. The module declares its own 290–850 K validity range;
the historical liquid action range does not silently cap a solid at 430 K.
Authored limits and device/vessel limits still narrow the permitted request.

Run `uv run --no-sync python -m chemworld.examples.native_solid_solution` to
heat a demixed solid and then cool it back into two composition domains.
This is a synthetic binary, constant-pressure, instantaneous local-equilibrium
model with common constant Cp. Al/Cu formulas provide elemental counters only;
it is not a fitted Al–Cu alloy, a nucleation/growth model, a magnetic or
multisublattice description, or a runtime TDB interpreter. The optional
pycalphad reference uses the same synthetic energies and explicitly aligns its
rounded gas constant with the native constant. No upstream solver is required
by the runtime. The fixed 33-unit development block and all attempts are in
`workstreams/world_foundation/archive/20261020-native-extension/evidence/NATIVE_SOLID_SOLUTION_ACCEPTANCE.md` and its JSON.

## Selected native TDB/CEF material

`TdbDatabase` accepts declarative TDB text with an explicit `InputSource`.
`CefPhaseModel` evaluates the selected phase's sublattices, vacancies, charged
constituents, ordering, magnetic and caloric contributions without an upstream
runtime dependency. `CefMaterialSpec` adds an authored temperature interval,
fixed pressure, holder heat capacity and mechanical atom-molar volume.
`calphad_world_spec(model, element_amounts_mol=..., temperature_K=...)` installs
that material on the existing solid/thermal strategy and finite-power device.

Inventory keys are real TDB elements and amounts are mol of nonvacant atoms.
Charged constituents and vacancies belong to the committed phase's site
fractions; they do not create another material inventory. The model is stored
once in the versioned `calphad_material` physical declaration. Its equipment
record only references that binding. `calphad_phase_meter` reads committed
temperature, element amounts and site fractions. It does not synthesize a
diffraction spectrum or pretend that real elements are the old A/B counters.

`heat`/`wait` integrate the same finite-inventory total enthalpy. Internal
constitution and relaxed heat capacity come from the common constrained root;
temperature comes from the common caloric root. Constant element amounts are
eliminated from numerical unknowns in this heat-only process, and every
contribution must have zero matter rate. Holder heat is retained on material
replacement; snapshot continuation, history replay, failure settlement and
thermal controls retain their existing contracts.

This consumer holds one explicitly selected solid or liquid phase at a local
internal minimum. It does not select a globally stable multiphase assemblage.
Line compounds may have valid thermal response but underdetermined individual
element potentials; no arbitrary multiplier is reported as a determined
chemical potential. Nonpositive effective heat capacity, failed constraints
and unsupported derivative boundaries reject the closure. Gas EOS and general
phase selection require their separately declared models. The mechanical
volume is authored geometry, not a pressure derivative inferred from TDB.

`ChemSageDatabase.from_file(path, source=...)` accepts original declarative
ChemSage DAT input on that same Gibbs/material interface. The immutable input
retains source provenance, phase occurrences, suppression flags, original
stoichiometry, all intervals and model metadata. Repeated labels use
`NAME#OCCURRENCE1`, etc.; `phase_records` exposes the original labels and records.
G and Cp descriptions compile to the common caloric evaluator, including
reference H/S, latent heat, extended powers and magnetic terms. QKTO uses the
declared chemical groups and directional Kohler/Toop exponents. Auxiliary
electron balances constrain CEF internal constitution without adding atoms.

Source suppression requires explicit `activate_suppressed=True`. Nonzero
excess pressure coefficients require an explicit `excess_pressure_unit="Pa"`
or `"bar"`. Select `MQMQAPhaseModel` for SUBG/SUBQ quadruplet phases. It consumes
source coordination numbers and their mixed/reciprocal recurrences, reference
pair energies, both entropy models, and directional G/Q/R interactions with
chemical groups and ternary terms. Quadruplet fractions are internal
constitution; actual atom amounts remain in the existing material inventory.
It shares Gibbs responses, constrained constitution and `CefMaterialSpec` with
CEF. Original pair atoms and quadruplet charge must be consistent; the model
rejects contradictory input and preserves its original declarations.
Legacy annotations with no defined energy effect remain visible in diagnostics.
Nonzero MQMQA source auxiliary balances need a separately declared consumer.
The CEF consumer rejects MQMQA input. Unsupported real-gas, aqueous, volume and group
override physics are also rejected explicitly. JSON stores original input
and declared options, then recompiles through known native codecs.

The complete seven-file ChemSage qualification covers 114 original phase
declarations: 111 valid phases pass 666 response points; three inconsistent
source phases are explicitly rejected, leaving 18 original numerical points
unexecuted. A separate 96-input, 576-point synthetic block covers additional-ion
chemical groups on both ionic lattices. These are software development
qualifications, not a claim of measured accuracy for every imported material.
The fixed coverage, independent reference and preserved failures are documented
in `workstreams/world_foundation/archive/20261023-completed/convergence/41/NATIVE_CONV41D3_EXPERIMENT.md`.

## Two-site surface interactions

Pass `MultisiteSurfaceSpec` to the existing `surface_world_spec(surface_spec=...)`.
Both site classes belong to the same carrier, with separate finite capacities,
CO/O inventories and CO exchange between classes. The existing surface process
factory, thermal coordinator, gas instruments and material-routing operations
perform the experiment. There is no second catalyst service or time loop.

The new mechanism uses reversible adsorption/oxidation and a common transition
state for each forward/reverse pair. Standard Gibbs potentials and caloric
energy share the declared U/Cv table and reference entropy. A symmetric
`interaction_matrix_J_mol` contributes G_ex=U_ex=1/2 n.epsilon.n/N_ref.
The same energy and its gradient enter the heat balance; it is not an extra
external reaction-heat source. State-slot declarations reference existing
adsorbate/site amounts, so carrier splitting, merging and material transfer
retain extensive energy. A conflicting authored caloric table is rejected.

The historical five-step mechanism remains the default and retains its exact
rates. It contains irreversible oxidation. Selecting that legacy specification
restores the original model; removing B from an explicitly reversible model
does not silently turn its A reactions into the old irreversible kinetics.
The optional old bulk reaction can still contribute through the common solver;
the reversible-pair validation concerns the declared new surface channels.

Run `uv run --no-sync python -m chemworld.examples.native_multisite_surface`.
The example uses finite gas sources and the existing in-situ detector latency.
It has 120 s of requested heating/waiting and 10 s of acquisition, counted once.
The fixed 57-unit development comparison uses CatMAP's actual local interaction
and rate-constant kernels with independently solved steady equations; it does
not exercise descriptor fitting or CatMAP's mapper. All attempts remain in
`workstreams/world_foundation/archive/20261020-native-extension/evidence/NATIVE_MULTISITE_RESULTS.json`. This is a finite
synthetic mean-field model, with no spatial kMC or real-catalyst prediction.

## Empirical circuits and stationary model analysis

`uv run --no-sync python -m chemworld.examples.native_thevenin --output /tmp/circuit.jsonl`
runs the authored RC/two-temperature example through the original laboratory clock,
control queue, native voltage/current readout, transaction records and replay. An
existing trajectory is never overwritten. These are synthetic sourced parameters,
not an identified real battery or a substitute for the electrochemical models.

Add `--analysis` to explicitly enable a stationary small-signal model analysis
before the example begins charging. The fixed public 13-point grid spans
1e-4–1e2 Hz. This computes the local response of the same residual, not a physical
sinusoidal acquisition: it advances no physical time, consumes no sample and costs
one declared analysis use. It returns only real/imaginary impedance in `Ohm`.
Nonstationary polarization/temperatures and nonzero-current or voltage control
are rejected; state is not quietly relaxed. Parameter uncertainty is not propagated.

The pure read-only API is `chemworld.physchem.thevenin_impedance(spec, frequencies_Hz,
state=state)`. It retains the actual cell and jig temperature responses. Its
derivative modes, per-frequency failures and linear residuals are explicit;
it is not a nonlinear stability or real-material accuracy certificate.

Add `--controls` to exercise algebraic power, a resistive load, differential
power and genuine CCCV through the same operations and replay. It can be combined
with `--diffusion` and `--analysis`. `TheveninControlSpec` declares sourced current
memory, scales and response times; `TheveninControlProgram` declares the named
mode and native-unit target. Power requires an explicit current-root interval.
The original `configure_electrical_control` and `queue_electrical_stage` accept
`control_mode`, `power_W`, `resistance_Ohm`, `current_limit_A` and
`branch_lower_A`/`branch_upper_A` only on explicitly capable circuit equipment.
CCCV uses a negative charging-current limit and a persistent internal current
state. Switching to CC/CV pauses that memory; returning to dynamic control resumes
it. Snapshots and material resets preserve it. Differential resistance retains
the published source equation, including its possible instability; zero-current
singularities and current limits stop or reject explicitly. Arbitrary executable
callbacks and undocumented control defaults are not accepted.

Add `--diffusion` to the example to enable a sourced, authored 16-cell planar SOC
diffusion model. `--diffusion --analysis` also analyzes its initial uniform state.
`TheveninDiffusionSpec` declares the diffusion time constant in seconds, normalized
cell faces on 0–1, and the explicit `surface-ocv-passive-pr5869` convention. The
original mean SOC and N−1 independent deviations form one persistent equipment
state; there is no second charge inventory. The same profile changes voltage,
heat, finite-source voltage control, surface stopping events and impedance. A
material reset preserves the profile, and a full session reset restores the
declared uniform initial state. Nonuniform profiles are not stationary analysis
points merely because the applied current is zero.

The voltage and heat equations include the explicitly attributed corrections in
[PyBaMM PR 5869](https://github.com/pybamm-team/PyBaMM/pull/5869). The unmodified
25.4.2 reference remains recorded as a version comparison, not a target for the
corrected physics. Instantaneous diffusion heat can be negative after reversal;
it is not clipped. The model declares empirical heat and charge balances, not a
complete chemical-energy function. Spatial resolution must be checked for the
requested transient: the fixed coarse-grid surface-cutoff qualification did not
meet its original convergence criterion. Independent refinement and all failures
are recorded with the task's development qualification, rather than hidden by a
changed acceptance tolerance. Other control modes and the rest of task 45 remain
separate implementation and qualification work.

## Cross-domain examples

The installed [native domain workflows](native_domains.md) exercise shared
equipment, finite process inventories, delayed stochastic readouts, and replay.
The [capability table](native_capabilities.md) is generated from the installed
module declarations; supported composition domains remain explicitly bounded.

## Parameter and measurement inputs

Run `uv run --no-sync python -m chemworld.examples.scientific_inputs` for a complete
import example: an existing water property package, six synthetic cooling-curve
records in two experiment groups, and two heat-loss parameters applied through
the native model declaration. The example emits JSON with sources, uncertainties,
train/validation counts and public pressure readings. It performs no fitting.

`chemworld.data.scientific_inputs.ScientificTable.from_json(text)` reads either
parameter or measurement tables. Columns declare a quantity, unit, role, finite
valid range and reference state. Each cell contains `value`, `unit`, standard
`uncertainty`, `uncertainty_note` and `quality`. A source has an explicit citation,
version, licence declaration and `synthetic`, `measured` or `literature` label.
Unknown uncertainty stays `None`; it must have an explanatory note. Celsius
values receive the temperature offset, while uncertainty uses the scale only.

Each row supplies `row_id`, `source_id`, `group_id`, `sample_id`, `split` and all
declared cells. `sample_id` identifies one acquisition, including all its channels;
repeat acquisitions need separate IDs. Measurement splits are `train`,
`validation`, `test` or `unused`. All rows in one experiment group must share a
split. Parameter rows use `parameters`. Missing or invalid observations remain
in the table with their quality flags; conditions and parameters must be valid.
The importer rejects duplicate identities, unknown columns, incompatible units
and valid values outside their declared range.

`table.to_csv()` writes a wide CSV with the row identifiers and five cell fields
per quantity, for example `pressure.value` and `pressure.uncertainty`.
`ScientificTable.from_csv(text, metadata)` reads it using the same contract;
`metadata` is `table.to_dict()` with the `rows` entry removed. No split is inferred
or randomized. `table.summary()` reports complete counts, and
`table.rows_for("validation")` selects the declared holdout.

`chemworld.physchem.property_inputs.PropertyInputCard` imports the existing
component/correlation format with per-correlation source and uncertainty records.
Its `evaluate_state` method uses the existing property evaluator. Energy and
entropy requests require an explicit imported caloric reference. The curated
example preserves unknown upstream table redistribution terms and unquantified
uncertainty; successful parsing does not establish a licence or calibration.

`chemworld.world.parameter_inputs.bind_parameter_row(physics, table, row_id, targets)`
applies one parameter row to new model configuration. The application supplies
`ModelParameterTarget` objects independently of the input, defining the installed
model/version, binding, numeric field path and quantity contract. The function
returns new `physics` and an import `receipt`; pass that configuration through
the usual composition/session constructor. Model codecs validate coupled changes
together. Measurement tables cannot be bound as parameters, and imports never
write observations or inventory into a running world.

### Finite native calibration

`chemworld.data.calibration.NativeCalibration` fits a finite parameter whitelist
by executing fixed histories in fresh native sessions. Supply a `ScientificTable`,
`CalibrationParameter` declarations and one `NativeCalibrationExperiment` per
data group. Each `CalibrationReadout` binds an observation column to an actual
measurement step and public channel, with an optional sampling-time column.
Parameter paths may address existing continuous model configuration fields;
network structure, task targets and live inventory cannot be fitted.

Only training rows enter uncertainty-weighted, bounded least squares. Successful
fits execute the validation groups separately and report errors in original
units, weighted errors, active bounds and local Jacobian rank. Non-valid input
observations remain in the exclusion counts; valid observations require positive
declared standard uncertainties. Failed forward histories retain actual readings,
operation outcomes, time, cost and sampling. They stop the fit instead of becoming
artificial residual penalties. The forward limit counts actual group executions,
including finite-difference evaluations.

Run the fixed synthetic thermal, SPM and confounded thermal examples with:

```powershell
uv run --no-sync python -m chemworld.examples.native_calibration --output $env:TEMP/chemworld-calibration.json
```

The thermal example uses an explicitly ideal synthetic pressure gauge and fixed
data perturbations. The SPM example fits one diffusion coefficient from public
terminal voltage. Joint heat-loss/heat-capacity fitting demonstrates that a good
curve fit can still leave parameters unidentifiable. These examples establish a
calibration workflow; empirical material validation requires separately sourced
data and a fixed validation design.

### DIPPR correlations and caloric requests

`PropertyCorrelation` accepts `dippr100`, `101`, `102`, `104`, `105`, `106`,
`107`, `114`, `115`, `116`, `127` (each with the `dippr` prefix), plus
`dippr100_reciprocal`, `dippr105_reciprocal`, and `dippr106_reciprocal`.
Declare a supported `property_id`, output unit, absolute-K input, coefficients
and a material validity interval. `property_equation_contracts()` exposes the
required/optional parameters, property dimensions, equation source and domains.
Material parameters still need their own source and uncertainty metadata.

The existing `evaluate_correlation` and property package use these declarations.
`correlation_temperature_derivative` returns the analytic first derivative;
`integrate_correlation` returns a definite integral or, with
`over_temperature=True`, its integral divided by T. Its numerical error estimate
and quadrature evaluation count describe the adaptive integration only, excluding
domain checks. Failed quadrature and intervals crossing poles are rejected.

`caloric_correlation_change`, `evaluate_caloric_correlation` and the existing
phase-path enthalpy reports consume these same equations. Output-unit conversion
scales the result while retaining characteristic temperatures and exponents.
Gas Cv-to-entropy conversion uses the declared ideal-gas reference. Other phases
still need an explicit closure before H and U can be interchanged.

The fixed development block compares 142 cases with chemicals 1.5.2; production
evaluation does not import chemicals. This covers formula behavior, not a new
material database. Polynomial inventory admission stays unchanged pending the
later process-integration task; the full chemicals expansion remains in progress.

The same 14 DIPPR expressions now support temperature derivatives through order
three and named coefficient/mixed partials through total order three:

```python
from chemworld.physchem import (
    PropertyCorrelation, dippr_temperature_response, dippr_parameter_jacobian,
    dippr_mixed_derivative, dippr106_coefficients_from_point,
)

curve = PropertyCorrelation(
    "example-cp", "dippr107", {"A": 25, "B": 30, "C": 800, "D": 10, "E": 400},
    {"temperature": "K"}, "J/(mol*K)", "ideal_gas_heat_capacity",
    {"temperature": (200, 500)}, "Synthetic example; not a material fit",
)
response = dippr_temperature_response(curve, temperature_K=300, order=3)
jacobian = dippr_parameter_jacobian(curve, [250, 300, 400], parameters=("A", "B", "C", "D", "E"))
mixed = dippr_mixed_derivative(curve, temperature_K=300, coordinates=("temperature", "C"))
```

Response entry `n` uses the declared output unit divided by `K**n`;
`response.to("kJ/(mol*K)")` converts all entries consistently. Jacobian columns
use the named source-coefficient coordinates, including critical temperature
when explicitly selected. To fit an omitted optional coefficient, first add it
explicitly to the declaration. These are analytic chain rules, not finite
differences. The existing application-range policy and mathematical domains
remain in force; no extension through a critical singularity is inferred.

`dippr106_coefficients_from_point` matches value/first derivative (A/B), or also
second derivative (A/B/C), with D=E=0. Inputs must share one output-unit basis.
The ABC system is locally rank deficient at `T/Tc = 1-exp(-2)` and is rejected;
recovering coefficients at one point is not a fitted temperature interval.
The additional fixed block covers 84 states and 2,136 comparisons against public
Jacobian/recovery functions and an independent 80-digit formula implementation.
This completes the DIPPR mathematical increment. Its fitting consumer and the
remaining G2 boundaries are described below.

`chemworld.data.correlation_calibration.CorrelationCalibration` now fits an
existing declaration to a measurement `ScientificTable`, sharing the bounded
TRF optimizer, weighted metrics and local rank diagnostics with
`NativeCalibration`. Its analytic mode consumes the DIPPR, vapor-pressure,
volume and Yaws-viscosity coefficient Jacobians. For other scalar declarations,
`derivative_mode="3-point"` must be selected explicitly; numerical differences
are labelled in the report and are not presented as analytic derivatives.

```python
from chemworld.data.correlation_calibration import CorrelationCalibration, CorrelationFitParameter

# observations: a ScientificTable with temperature/value columns, declared
# standard uncertainties and preassigned train/validation experiment groups.
fit = CorrelationCalibration(
    observations, curve,
    (CorrelationFitParameter("A", (10, 50), 25),
     CorrelationFitParameter("B", (10, 60), 30)),
    fitted_correlation_id="my-fitted-cp",
)
report = fit.fit(max_nfev=100)
if report["status"] == "fitted":
    fitted_curve = PropertyCorrelation.from_dict(report["fitted_correlation"])
```

Parameter bounds and initial values use the formula's original coefficient
coordinates; the report binds them to the equation and input/output units.
Observation values and standard uncertainties are converted together. Log
parameter normalization contributes its own chain-rule factor to the Jacobian.
Only training rows enter optimization or sensitivity calculations; validation
is evaluated once after convergence, with no model selection. Test/unused rows
are not evaluated. Invalid observations remain explicitly excluded, failed
forward evaluations retain their attempted rows, and rank deficiency is reported
even when predictions fit well. A failed or unconverged fit exports no curve.

The fitted curve has a new requested identity and a new `fit:` source. Original
coefficients/source metadata, measurement provenance, residuals and rank are
retained, so a regression is not mislabelled as the original published fit.
Declared application bounds remain bounds, not a claim of material validation.
The fitted declaration can enter existing property packages, JSON and caloric
consumers without adding a runtime or a second property database.

The fixed synthetic qualification fits every explicit coefficient of all 14
DIPPR forms, in two parameter sets each: 28/28 fits pass 2,524 checks. Among
these, 23 are rank deficient under the declared local threshold; this is retained
as a limit of the observation design. This qualifies fitting behavior, not unique
parameter recovery or empirical chemical accuracy. The six other public fitting
Jacobian families (Antoine, extended TRC Antoine, both Wagner forms, Yaws pressure
and Yaws viscosity) also pass full-column regression: 12/12 fits and 948 checks,
with four locally rank-deficient cases retained. Critical constants and log bases
remain fixed where the public fitting API defines them as fixed inputs.

The derivative and inverse review also binds the existing pressure, pure/mixed
virial and Henry numerical evidence. Direct PPDS9/Yaws checks cover six states
and 36 public/high-precision comparisons through third temperature order.
Lastovka H/S inversion uses the existing `temperature_from_caloric_change` with
an explicitly bounded Cp curve; a positive multiplicative heat-capacity factor
is represented by dividing the specified caloric change by that factor. Its
74 direct public-reference states pass 218 comparisons, including cyclic and
non-cyclic choices, different reference temperatures and negative/zero/positive
changes. The full G2 mathematical review maps 159 relevant public interfaces;
it does not establish the unresolved component identities or the final G3
whole-source/process qualification. Scientific utility and signed-virial
reference discrepancies remain explicit rather than being counted as agreement.

### Read-only battery impedance (development)

The existing isothermal SPM, SPMe and DFN expose local rest-point analysis through
`chemworld.physchem.battery_impedance.battery_impedance`. It consumes their actual
inventory rates and electrical constraints, not an independently fitted circuit:

```python
from chemworld.physchem.battery_models import BatterySPMSpec
from chemworld.physchem.battery_impedance import battery_impedance

cell = BatterySPMSpec()
state = cell.initial_state()
spectrum = battery_impedance(cell, [1e-4, 1e-3, 1e-2], state=state)
report = spectrum.to_dict()  # per-frequency real/imaginary ohm and failures
```

This does not advance time, spend experimental resources, expose a new agent
instrument, or modify a snapshot. Positive current means discharge; passive
impedance is `-delta_V/delta_I`, with `exp(+i*2*pi*f*t)` phasors. Frequency must be
strictly positive. A nonstationary concentration profile or a stopping boundary
is rejected rather than silently relaxed. Thermal and side-reaction combinations
are not yet supported. The present quasi-static BV models have no double-layer
capacitance, so their spectra must not be interpreted as including its RC arc.

The shared `residual_frequency_response` in `physchem.residual_frequency` also
supports general declared residuals, non-diagonal mass matrices and algebraic
feedthrough. It uses the existing derivative modes and sparse linear solver,
retaining singular-frequency failures rather than substituting a least-squares
answer. Physical input/output units and derivative provenance remain in the
report. This is local response, not a nonlinear stability certificate.

The first independent development block compares six SPM cases against continuous
spherical diffusion, including four radial resolutions and a flat-OCV limit.
The subsequent SPMe/DFN block checks terminal impedance and the complex mean salt
concentration response in each of the three electrolyte regions against released
PyBaMM 25.4.2 equations, independently differentiated with CasADi and solved with
SciPy. Ten cases, each at 13 frequencies, pass the fixed comparison and native /
reference refinement bounds. The original uniform-radial reference was insufficiently
resolved for five SPMe cases (5/10 overall); that result is retained separately.
Changing only the reference radial mesh to boundary refinement gives 10/10 without
changing native physics or error bounds. This is a finite, synthetic, constant-property
domain, not validation for arbitrary materials, nonstationary points or wet experiments.
Other model families in NATIVE-CONV-45 remain outstanding. The fixed inputs, limits and
all attempts are linked from `NATIVE_CONV45_IMPEDANCE_EXPERIMENT.md` in the workstream.

### Sourced nonlinear electrode OCV

`physchem.battery_materials.ElectrodeOCV` represents a bounded polynomial plus
exponential and tanh terms. It stores the material, source, model compatibility and
stoichiometry domain. Use `from_parameter_row` to bind explicitly named columns from
the existing `ScientificTable`; amplitudes use V and scales/shifts use `dimensionless`.
The original parameter values, units and uncertainty notes remain in its JSON.

```python
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import BatterySPMSpec

# table is a validated ScientificTable of kind "parameters".
ocv = ElectrodeOCV.from_parameter_row(
    table, "electrode-row", material_id="my-electrode",
    polynomial=("constant_V", "linear_V"),
    tanh_terms=(("transition_V", "transition_scale", "transition_center"),),
)
spec = BatterySPMSpec(negative_ocv=ocv)
```

An explicit curve replaces that electrode's linear OCV fields. The actual SPM,
SPMe and DFN voltage laws consume it, and DFN uses its analytic slope in the coupled
Jacobian. Chemical potential and integrated electrode energy derive from the same
expression. Public curve evaluation rejects a point outside its declared domain;
solver trials may explicitly evaluate its analytic continuation while the battery's
physical stopping surfaces still govern accepted states.

The fixed eight-unit development block compares two Chen2020 OCV relationships,
three continuous discharge/rest/charge trajectories and three local spectra against
released PyBaMM 25.4.2 functions and models: 8/8 passed. Geometry and transport remain
synthetic in that block. It is not a complete LGM50 parameterization. Nonlinear OCV
with thermal or side-reaction models is currently rejected; the full material,
transport, thermal and other model-family scope of CONV-45 remains open. Inputs and
all outcomes are in `workstreams/world_foundation/NATIVE_CONV45_OCV_EXPERIMENT.md` and
`NATIVE_CONV45_OCV_RESULTS.json`.

### Equilibrium electrode SOH (development)

`physchem.battery_soh` solves the voltage-window geometry using those same OCV
functions. `ElectrodeSOHSpec` accepts known electrode capacities and either
`known="cyclable_lithium"` (active electrode Li charge) or `known="cell_capacity"`.
`HalfCellSOHSpec` is explicitly a positive working electrode versus Li/Li+ at 0 V;
this is an equilibrium analysis, not yet a native half-cell dynamical process.
Capacities are C; `ScientificTable` parameter bindings can explicitly convert Ah
or mAh while retaining original units, uncertainty and source. No estimator is
added: native public-observation calibration remains the existing data calibration
workflow, and local implicit sensitivities do not prove parameter identifiability.

```python
from chemworld.physchem import soh_from_battery, battery_spec_at_soc
from chemworld.physchem.battery_models import BatterySPMSpec

cell = BatterySPMSpec(minimum_voltage_V=3.38, maximum_voltage_V=3.92)
state = cell.initial_state()
soh = soh_from_battery(cell, state)  # active Li only, not electrolyte salt or locked SEI
if soh.status == "unique":
    solution = soh.branch()
    energy_J = soh.theoretical_energy_j()  # reversible OCV integral, not loaded work
    fresh_cell = battery_spec_at_soc(cell, soh, .5)  # a NEW initial configuration
```

This does not advance time, spend resources, relax a live concentration profile,
or reset an existing device. SPM, SPMe and DFN read their actual electrode bins.
`at_soc()` and `soc_at_voltage()` use the selected equilibrium branch. Complete
multiple results require an explicit branch index; incomplete isolation retains
unresolved boxes and cannot be treated as unique. Known capacity at a declared
electrode face is handled without widening the domain. Impossible inventory,
no voltage-window solution, and numerical/rank uncertainty have distinct statuses.
The shared bounded interval/Krawczyk search uses the existing point-root solver;
outward floating-point enclosures are not a formal exact-real proof. Cutoff roots
must delimit a complete physical path, not merely solve two endpoint equations.

Offline example: `uv run --no-sync python examples/native_battery_soh.py`.
The fixed 12-unit development block compares released PyBaMM 25.4.2 SOH equations,
hand algebra and independent polynomial roots, including multiple/infeasible/
nonidentifiable cases. Inputs and all attempts are in
`workstreams/world_foundation/NATIVE_CONV45_SOH_EXPERIMENT.md` and the matching
`NATIVE_CONV45_SOH_RESULTS*.json`. MSMR-specific SOH, ageing inference, wet-cell
accuracy and the remaining task-45 model families are not covered by this block.

### Many-particle electrodes (development; scientific qualification pending)

`BatteryMPMSpec` is an explicit ideal-uniform-electrolyte MPM. Each electrode
retains its total active volume and material coefficients in `ParticleSpec`.
Optional `ParticleSizeDistribution` declarations allocate that volume across
positive radii. Every size has its own actual radial inventory, diffusion and
interfacial current; currents are solved together at the shared electrode
potentials, never assigned in advance from the volume fractions. With no size
distribution, the declared particle radius gives the monodisperse SPM limit.

```python
from chemworld.physchem import BatteryMPMSpec, ParticleSizeDistribution, mpm_fields
from chemworld.world.battery_world import battery_world_spec

# source is the common InputSource for this distribution.
sizes = ParticleSizeDistribution((5e-6, 15e-6), (.5, .5), source)
cell = BatteryMPMSpec(negative_distribution=sizes, positive_distribution=sizes)
fields = mpm_fields(cell, cell.initial_state(), current_A=.2)
world = battery_world_spec(cell)
```

Direct weights are normalized volume fractions. `from_weights(..., basis=...)`
explicitly converts and normalizes positive relative number, area or volume
weights. `from_parameter_table` retains original common source rows, units and
uncertainty; `um` is an explicit length conversion, not an inferred unit.
`lognormal_size_distribution` discretizes a stated truncated density in the
specified input basis, preserving median, log-standard-deviation, bounds and
quadrature count. It does not recover unobserved distribution tails.

All sizes use the existing battery equipment slot, source controls, shared
clock, native measurements, snapshot and replay. Surface readouts are area
averages; bulk stoichiometry is volume weighted. Stops check every size, not
only an average. Rest can redistribute Li between sizes at zero terminal
current. Electrode chemical energy, dissipation, SOH totals and read-only local
impedance consume the same bins. A sourced nonlinear OCV must explicitly declare
`MPM` compatibility; stored older material compatibility lists are not widened.

Offline workflow: `uv run --no-sync python -m chemworld.examples.battery_mpm --output /tmp/mpm.jsonl`.
Functional checks include true CC/CV/source limits,
analytic derivatives, conservation, energy balance, both measurement noise modes
and cross-process continuation. Independent released-model comparisons and
subsequent resolution blocks retain their separate denominators and failures in
`workstreams/world_foundation/NATIVE_CONVERGENCE_ACCEPTANCE.md`.
Do not infer arbitrary-mesh accuracy, measured-cell accuracy, composite electrodes
or thermal/ageing capability from these functional checks.

### Directional OCP and electrode memory

Isothermal SPM and MPM cells can declare `negative_ocp` and `positive_ocp` using
`ElectrodeOCPSpec`. Explicit lithiation/delithiation `ElectrodeOCV` branches and a
source-bearing parameter table select `current_sigmoid` or `axen`. The sigmoid
uses effective collector area in m² and an inverse-current-density scale in m²/A.
Axen uses an initial dimensionless memory and separate positive decay parameters.
The existing `single`/absent declaration retains the original equilibrium curve.

Each Axen electrode owns one persistent `h` coordinate, shared by all its MPM
sizes and driven by its total current. Rest holds `h` while particle profiles may
relax; current reversal changes it continuously. Material reset preserves the
installed cell's history, while replacing the cell uses the new declaration.
Snapshots and exact replay retain both inventories and memory. Meter acquisition
still advances its declared duration, so it does not secretly pause the cell.

The authored offline example exercises discharge, rest, reversal and replay:

```sh
uv run --no-sync python -m chemworld.examples.battery_hysteresis --output /tmp/hysteresis-spm.jsonl
uv run --no-sync python -m chemworld.examples.battery_hysteresis --many-particle --output /tmp/hysteresis-mpm.jsonl
uv run --no-sync python -m chemworld.examples.battery_hysteresis --mode current_sigmoid --output /tmp/directional-spm.jsonl
```

These are synthetic branches, not calibrated materials. Operational OCP is
distinct from a single-valued equilibrium free-energy curve. Thermal/SEI coupling,
equilibrium SOH for these branches and full hysteresis energy accounting are not
provided. The original chemical-energy helper evaluates only the declared
single-valued reference; it does not include the memory state's energy, and
directional-OCP dissipation requests are rejected. Axen ordinary rest EIS is
rejected because its two directional current
derivatives generally differ; smooth sigmoid analysis uses the same physical
residual. Wycisk/MSMR and SPMe/DFN directional OCP remain separate model extensions.

### Finite lithium-metal / positive-working half cell

`BatteryHalfCellSPMSpec` declares a real finite lithium-metal reservoir and one
positive working particle. It has no negative intercalation particle. The metal's
area, thickness, conductivity, molar volume, exchange-current density, temperature
and salt condition come from an explicit `ScientificTable` row in
`LithiumMetalCounterElectrodeSpec`; the working electrode declares a sourced
`ElectrodeOCV`. This first scope is isothermal, uniform-electrolyte SPM.

Positive current removes lithium from the metal and inserts it into the working
particle. Both inventories, passed charge and delivered electrical work live in
the existing equipment-owned state. Their shared solver enforces total lithium
and both Faraday balances. The measured voltage includes the working OCV, both
Butler–Volmer overpotentials and metal ohmic loss. Finite current or voltage
programmes, staged controls, meter acquisition, snapshots and exact replay use the
existing battery runtime. A material reset cannot replenish installed metal.

Metal depletion stops discharge at its actual event time. The depleted state is
valid: further positive current stops immediately; explicit rest or negative
current permits relaxation or replating. Metal geometry and interface area stay
fixed, including during replating. This is a fixed-interface approximation, not a
moving-boundary or morphology model. Working-particle surface and voltage limits
also remain real stopping events.

```sh
uv run --no-sync python -m chemworld.examples.battery_half_cell --output /tmp/half-cell.jsonl
```

The example uses authored synthetic inputs. Half-cell thermal/SEI coupling,
full chemical-energy/dissipation accounting and automatic full-cell SOH
conversion are explicitly unsupported. The older explicitly declared
`HalfCellSOHSpec` equilibrium calculation remains separate; it is not silently
substituted for finite-metal dynamics. Functional tests and independent released
PyBaMM comparisons have distinct scopes and denominators in the native
convergence reports. The reference adapter's finite-metal inventory and depletion
event are an explicit Faraday extension of PyBaMM's positive-working SPM, not a
claim that the upstream model already has a finite-metal state.

Stationary half-cell impedance is available through the existing read-only
`chemworld.physchem.battery_impedance` API. It uses the original metal and working
particle residual, both interface kinetics and the fixed metal resistance.
Every radial state must be stationary, metal inventory strictly positive, and
working surface and voltage strictly inside their allowed ranges. A worked
particle still relaxing after current shutoff is rejected; analysis never waits
or modifies inventories. The original time-domain depleted-metal recovery
behavior is unchanged.

```sh
uv run --no-sync python -m chemworld.examples.battery_half_cell_impedance --output /tmp/half-cell-eis.json
```

The output retains full complex state responses, signed charge/work and both
inventory responses. With positive discharge and `exp(+iωt)`,
`iωF δn_metal = −δI`, `iωF δn_working = δI`, and `iω δW = V_rest δI`.
Fixed geometry does not create an extra metal-capacity impedance pole; finite
metal still constrains the operating domain. These derivatives do not guarantee
that a finite oscillation at an arbitrary low frequency avoids depletion. The
analysis consumes no experimental time or resources and is not a new instrument.
Independent frequency comparisons and the earlier half-cell time-domain grid
qualification gaps remain separate.

The fixed 80/160-cell frequency block completed all 50 routes and 680 frequencies,
but only 2/10 complete units met every precision criterion. Surface extrapolation
differences caused 32 failed comparisons, including 10 impedance comparisons;
the slow-diffusion coarse-grid surface discrepancy reached 5.70%. Conservation,
full state checks and read-only behavior passed. The eight fine native spectra
met the continuous-sphere comparison, which does not replace the failed whole
block qualification. Mesh and frequency accuracy remain explicit limitations.

### Dynamic-salt DFN lithium-metal half cell

`BatteryHalfCellDFNSpec` adds a separate two-domain half-cell declaration:
separator and positive working electrode, with one radial particle per working
thickness cell. It has no artificial negative porous electrode. Finite metal
lithium, spatial electrolyte salt, particle lithium and signed charge/work use
the same equipment slots, shared clock, queue and snapshot codec as other native
batteries. `half_cell_dfn_fields()` returns local reaction currents, salt and
potential fields; the agent-facing meter still returns only terminal V and A.

Positive current strips metal and inserts working-electrode lithium. The real
left salt flux balances the working reaction source. Its reconstructed boundary
concentration enters metal Butler–Volmer kinetics and the liquid potential.
Metal geometry is fixed while its inventory changes. Empty metal can rest or
recover under reverse current; actual stripping stops at depletion.

CC accepts an explicitly sourced nonnegative metal salt exponent. Finite CV
requires an analytic passive-branch certificate over all surface trials, a
positive zero-current salt reconstruction and last-cell bulk fractions in
`[0,1]`. Unproved or nonunique constitutive branches are rejected. This does not
clip local reverse currents or repair inventories. Heat, side reactions,
impedance, automatic SOH and complete chemical-energy closure are not supplied
by this family. Independent released-model and mesh qualification remains
separate from its functional and runtime tests.

### Independent material phases in a composite electrode

`BatteryCompositeSPMSpec` declares two negative active-material phases and one
positive phase. Each `ActiveMaterialPhaseSpec` has its own particle radius,
diffusivity, active volume, capacity, exchange-current scale, initial lithium
fraction, sourced single-valued `ElectrodeOCV` and parameter source. These are
material identities, not particle-size bins or a fitted average electrode.

The composite and existing MPM use one particle-electrode residual kernel. It
solves a common electrode potential and individual Butler–Volmer reaction
currents, together with conservative spherical diffusion. The two negative
currents sum to the external current; they are not assigned by fixed volume
fractions. At zero external current, unequal material potentials can drive
opposing internal currents and transfer lithium between the negative phases.
The positive inventory and external charge then stay constant. This process
continues through the existing runtime's `wait`, measurement and queued stages.

Every phase inventory remains equipment-owned through snapshots and material
resets. The shared clock, electrical work and Faraday balances are unchanged.
Replacing a compatible cell introduces the replacement's explicit inventories.
Current/voltage control, individual material-surface stopping events and terminal
voltage cutoffs use the same phase-resolved electrical solution. A monotonicity
certificate for each OCV defines the supported current-sharing branch; an
unavailable physical root is rejected instead of clipping a phase current to the
external source limit.

```sh
uv run --no-sync python -m chemworld.examples.battery_composite --output /tmp/composite.jsonl
```

The example uses authored synthetic material parameters, with an initially
unequilibrated pair of negative phases. The first scope is isothermal, fixed salt,
one particle population per material and single-valued monotone OCV. It does not
claim the graphite tables or directional silicon curves of
`Chen2020_composite`. Thermal/SEI coupling, phase-specific size distributions,
directional hysteresis, composite DFN, half-cell combinations, SOH and
chemical-energy/dissipation closure are explicitly outside this scope. Independent
comparisons use the actual released PyBaMM multi-phase SPM; functional tests and
fixed-grid scientific qualification are recorded separately in the convergence
reports.

The subsequent composite frequency API accepts explicitly stationary operating
points. Each particle must be uniform and both negative materials must have the
same OCV; equal lithium fractions alone do not guarantee this. It evaluates the
original algebraic closure at the supplied inventories, then checks every
inventory equation. Zero external current with opposing internal phase currents
is rejected, as is a nonuniform particle that would continue diffusing. The API
never advances time or modifies a caller's lithium fractions to find equilibrium.

```sh
uv run --no-sync python -m chemworld.examples.battery_composite_impedance --output /tmp/composite-eis.json
```

This separate example explicitly declares negative fractions .6/.6, giving
.13 V for both authored curves. The original time-domain example retains its
unequal potentials and internal redistribution. The output includes terminal
impedance, complete complex state responses and all three phase-current
responses with their identities. Currents divide according to the coupled
frequency response, rather than fixed area fractions; each phase obeys
`iωF δn = −δq`. The shared `BatteryImpedance` and `FrequencyResponse` reports keep
failed frequencies and linear-solver diagnostics. This remains a read-only
physical analysis, not a new agent-facing instrument or measured material fit.

The independent continuous-sphere comparison is limited to the predeclared
frequency/grid block. Finite-volume surface reconstruction contributes a
half-cell surface term at very high frequency, so a fixed finite mesh is not
claimed to reproduce the continuous infinite-frequency intercept. Ordinary
EIS for hysteresis, side reactions or nonstationary states, composite SOH/SOC
initialization and complete chemical-energy closure remain unsupported.

### Spatial composite electrodes with dynamic electrolyte

`BatteryCompositeDFNSpec` resolves two negative material phases and one positive
phase at every electrode thickness cell. Each phase reuses the original sourced
`ActiveMaterialPhaseSpec` with its own radius, diffusivity, capacity, exchange
scale and single-valued OCV explicitly supporting `DFN`. Its active volume is the
whole electrode phase volume; the model divides it among the thickness cells.
The sum of phase volumes must fit the non-pore electrode volume. Zero binder is
allowed; extra active volume is rejected.

```sh
uv run --no-sync python -m chemworld.examples.battery_dfn_composite --output /tmp/composite-dfn.jsonl
```

The example uses synthetic inputs, three CC/rest/charge stages, actual meter time
and exact native replay. `composite_dfn_fields` returns each local phase reaction
current, the summed cell current, radial-surface and mean fractions, electrolyte
concentration, solid/liquid potentials and face currents. Material identities and
all local inventories survive JSON continuation. `CompositeDFNReadout` summarizes
the three phases while retaining the port voltage/current.

All phases share the local electrical potentials. Their independent BV currents
sum into the original DFN charge and salt equations. Rest can transfer lithium
between phases with zero terminal charge; fixed phase-current weights cannot
represent that behavior. A single coupled residual advances particles, salt and
signed terminal charge/work, using the existing finite-source CC/CV controls,
phase/salt/voltage stopping events, session queue and equipment ownership.

This first combination is isothermal with constant transport and single-valued
OCV. Heat, SEI/plating, hysteresis, particle-size distributions, MSMR, half-cell,
SOH and impedance combinations are unsupported. Signed terminal work is retained;
it does not define a complete chemical-energy closure. Independent qualification
uses the actual modular PyBaMM DFN with two/one phases and single OCP, and is
reported separately from functional tests. This scope does not reproduce the
silicon directional-OCP default of `BasicDFNComposite` or calibrated cell data.

The fixed development comparison completed all 14 units and 72 routes, including
60 histories and 12 refusals, with no execution or assessment exception. Strict
qualification remains open: 2/14 units and 929/964 checks passed. The failures
include actual source salt conservation, finite-grid responses and cutoff
endpoints; successful functional/runtime tests do not establish those accuracy
bounds. The complete original trajectories and all failures remain archived.

### Multiple-reaction MSMR particles and stationary impedance

`BatteryMSMRSPMSpec` uses one lithium inventory per radial shell. Its sourced
`MSMRMaterialSpec` rows describe one or two overlapping reactions in each
electrode, with a shared potential and overpotential. They are not independently
conserved material phases. The original runtime supports CC, finite-source CV,
queued stages, measurements and exact JSON continuation at fixed temperature
and electrolyte concentration.

```sh
uv run --no-sync python -m chemworld.examples.battery_msmr --output /tmp/msmr.jsonl
uv run --no-sync python -m chemworld.examples.battery_msmr_impedance --output /tmp/msmr-eis.json
```

The read-only `chemworld.physchem.battery_impedance` API differentiates that same
residual, retaining every shell, signed charge/work and the electrical closure.
It requires a stationary operating point with strictly interior shell amounts,
surface occupancy and voltage. It never relaxes a worked particle or changes
the experiment's clock, queue or future. The example exports the complete
complex response, sourced reaction identities and both total inventory
responses; it does not invent separate inventories for individual reactions.

The potential-driven transport law gives chemical diffusivity
`Dchem = -D F x(1-x)/(RT dx/dU)` at uniform rest. A single unit-capacity reaction
with width 1 gives D, while width 2 gives 2D. The original linear extrapolation
of surface potential is retained. Reaction current shares follow their exchange
currents, whereas occupancy shares follow the derivatives of their occupancy
curves; these shares generally differ. Only each electrode's total lithium
satisfies its own Faraday balance.

These examples use authored synthetic parameters. The fixed 10-unit development
frequency block passed all 50 routes, 680 frequency points and 1336 checks,
using 80/160 radial cells and 1e-5–1e-2 Hz. Its actual PyBaMM reference retains
all original states; the distinct electrolyte reference concentration is mapped
explicitly, with each authored/source parameter retained. This bounded result
is separate from functional compatibility and the original 12-unit time-domain
result; it does not qualify arbitrary materials or frequencies. Thermal coupling, side reactions,
MSMR-specific SOH and complete chemical-energy closure remain unsupported.

### Average-electrode Newman–Tobias model

`BatteryNewmanTobiasSpec` keeps one finite lithium inventory per electrode while
solving the spatial solid/liquid potentials and local Butler–Volmer currents.
`AveragedElectrodeSpec` declares radius, active volume, maximum concentration and
exchange-current scale; radius determines reaction area, not a radial mesh.
`PorousElectrodeGeometrySpec` declares the three layer thicknesses, area,
porosities, thickness mesh and conductivities. Solid conductivities are effective
values; the liquid value is bulk conductivity with the declared porosity factors.
The charge integration kernel is shared with the existing DFN model.

Positive current removes lithium from the negative electrode and inserts it into
the positive electrode. The original equipment state contains those two amounts,
passed charge and terminal work. Fixed electrolyte concentration is a declared
approximation, not an additional mutable salt inventory. Local current density,
solid/liquid potentials and face currents returned by `newman_tobias_fields` are
derived from that same state. Spatially uniform concentration does not imply
spatially uniform reaction current.

```sh
uv run --no-sync python -m chemworld.examples.battery_newman_tobias --output /tmp/newman-tobias.jsonl
```

The authored example performs charge/discharge, rest, measurement and exact replay
through the existing battery runtime. CC, finite-source CV, queued stages, shared
laboratory time and JSON continuation use their original interfaces. Material
reset preserves installed lithium; device replacement installs the newly declared
inventory once. Stoichiometry and voltage cutoffs stop the actual advance, and
reversing current can move inward from a stoichiometry limit.

This finite model is isothermal with constant electrolyte concentration and fixed
geometry/transport. It has no radial diffusion, dynamic salt, side reaction or
thermal closure. Supplied `ElectrodeOCV` expressions must explicitly include
`NewmanTobias` in `supported_models`; existing material declarations are not
silently broadened. Chemical-energy, full dissipation and automatic SOH requests
remain explicitly unsupported. The example uses synthetic inputs, not a calibrated
commercial cell. Functional checks and released-source scientific qualification
have separate denominators in the native convergence records.

Stationary impedance uses the existing read-only
`chemworld.physchem.battery_impedance` API and the original mean-inventory and
porous electrical residual. Both lithium stocks and terminal voltage must lie
strictly inside their event bounds. Unlike a spatial concentration model, these
mean inventories have no concentration gradients to relax after resting a worked
cell. The analysis neither advances a session nor alters its queue, noise, charge
or work ledger.

```sh
uv run --no-sync python -m chemworld.examples.battery_newman_tobias_impedance --output /tmp/newman-tobias-eis.json
```

The example exports every complex state response, each thickness-cell reaction
current, the two lithium-stock responses, signed charge and signed terminal work.
At uniform rest the finite electrical network is resistive; the OCV derivatives
and conserved mean inventories supply an inverse-frequency contribution. Local
reaction-current and face-current responses are frequency independent. This model
does not imply a diffusion impedance or double-layer relaxation. Independent
continuous-network electrical-loss checks do not define a full chemical or thermal
energy model. The previous time-domain qualification and its unresolved CV mesh
difference remain separate.

The first fixed stationary-NT comparison block solved all 45 routes and 455
frequency points. Strict qualification passed 3/8 units. After correcting an evaluator summation-order
bug, a complete unchanged rerun remained 3/8: source current/Faraday invariants and
one high-conductivity initialization current exceeded the original bounds. All
45 complete route objects, except elapsed time, matched the first run exactly.
Both complete records and all failed checks are retained. Functional availability
does not close that qualification gap.

### Closed-volume-compatible lead-acid LOQS

`BatteryLeadAcidSpec` adds a finite isothermal lead-acid model on the original
battery runtime. Six equipment-owned inventories represent negative Pb/PbSO4,
positive PbO2/PbSO4, and electrolyte H2SO4/H2O. Discharge consumes actual Pb,
PbO2 and acid; charging reverses that reaction. Pore volume, acid concentration,
water concentration and Bode molality OCPs follow those same stocks. Terminal
charge/work, ordinary CC and finite-current CV, the existing meter, queues,
shared time and JSON continuation use the original session interfaces.

```python
from chemworld.physchem import BatteryLeadAcidSpec, lead_acid_readout
from chemworld.world.battery_world import battery_world_spec
from chemworld.runtime import ChemWorldRuntime

cell = BatteryLeadAcidSpec()
world = battery_world_spec(cell)
runtime = ChemWorldRuntime(composition=world)
runtime.reset(seed=4)
runtime.execute({"operation": "configure_electrical_control", "current_A": 0.5})
runtime.execute({"operation": "wait", "duration_s": 3.0})
measurement = runtime.execute({"operation": "measure", "instrument": "battery_meter"})
runtime.close()
```

The public instrument exposes terminal potential/current. `LeadAcidReadout`
provides explicit model analysis of concentration, porosity and electrode
response; it does not invent lithium surface fractions or laboratory sensors.
Run an eleven-action discharge/rest/partial-recharge example with exact replay:

```sh
uv run --no-sync python -m chemworld.examples.battery_lead_acid --output /tmp/lead-acid-new.jsonl
```

Defaults are an explicitly synthetic, volume-compatible closed material,
**not the default Sulzer material set or measured-cell qualification**. Changing
geometry or material requires all six initial amounts explicitly. Initial acid
and water must fill the real pores, and reaction molar volumes must preserve
closed saturation. Unsupported combinations are rejected; water is not added
to repair volume. Initial zero sulfate permits discharge, while finite reactant,
acid/water, pore and voltage boundaries stop the actual trajectory. Inward
reversal may resume it. Old lithium device/slot codecs remain unchanged; the
lead-acid cell uses its own chemical slot type in the same runtime. Equipment
replacement continues to require the same equipment type.

This finite leading-order model has no spatial transport loss, double layer,
hydrolysis, thermal/chemical-energy closure, side reactions, EIS or automatic
SOH. Its independent source comparison and any unresolved cases are recorded
in `workstreams/world_foundation/NATIVE_CONV45_LOQS_EXPERIMENT.md`; functional
availability alone is not a claim of complete scientific qualification.

### Separate electrode OCV and resistance

`BatterySplitOCVRSpec` represents two finite electrode Li inventories, separate
capacity/OCV declarations, and a series resistance. It has no radial particles,
BV kinetics or electrolyte concentration state. Discharge moves Li from the
negative electrode to the positive electrode; the same two amounts determine
both OCVs and terminal voltage. Capacities are in coulombs, resistance in ohms,
and initial mean fractions and each electrode's lower/upper limits are explicit.

```python
from chemworld.physchem import BatterySplitOCVRSpec, split_ocvr_readout
from chemworld.world.battery_world import battery_world_spec
from chemworld.runtime import ChemWorldRuntime

cell = BatterySplitOCVRSpec()
runtime = ChemWorldRuntime(composition=battery_world_spec(cell))
runtime.reset(seed=4)
runtime.execute({"operation": "configure_electrical_control", "current_A": 0.5})
runtime.execute({"operation": "wait", "duration_s": 3.0})
measurement = runtime.execute({"operation": "measure", "instrument": "battery_meter"})
runtime.close()
```

Run the eleven-action example with exact replay:

```sh
uv run --no-sync python -m chemworld.examples.battery_split_ocvr --output /tmp/split-ocvr-new.jsonl
```

The model uses the existing generic electrical equipment declaration and Li
inventory slot, meter, queue, shared clock and initialization semantics. Its
readout reports mean electrode fractions and ohmic drop, without inventing
surface fractions. A positive resistance supports finite-current CV. With
zero resistance, CC remains defined, but CV is rejected because current is
nonunique at the requested OCV and otherwise cannot attain that voltage.

Ordinary `battery_impedance` uses this model's same residual at an interior
state; its response represents the OCV inventory derivative and resistance,
not an RC or diffusion model. `soh_from_battery` and `battery_spec_at_soc` reuse
the original equilibrium root/branch analysis with the declared capacities,
actual total Li and physical fraction domains. A requested voltage window
must be reachable; an absent or ambiguous branch is not a valid SOC estimate.
These analyses do not advance or rewrite the real cell state.

Defaults are explicitly synthetic curves and capacities. Thermal coupling,
complete chemical-energy closure, side reactions and particle transport are
unsupported. The actual released SplitOCVR reference contains only two mean
fractions and charge under prescribed current; native CV and terminal work
are checked against separate analytical references. Numerical qualification
and all retained differences are recorded in
`workstreams/world_foundation/NATIVE_CONV45_SPLIT_OCVR_EXPERIMENT.md`.

### Finite SPM lithium plating and stripping

An optional `LithiumPlatingSpec` on `BatterySPMSpec` adds two actual metal-Li
inventories to the original cell equipment slot. A sourced parameter table
declares the kinetic rate (`m/s`), transfer coefficient, fixed electrolyte
concentration (`mol/m^3`), and negative electrode geometric volume (`m^3`).
The reversible mode transfers Li between the active negative particle and
plated metal; the irreversible mode deposits into dead Li and has no stripping
term. Initial plated and dead amounts are explicit, independent inputs.

The side current and intercalation current share one negative electrode
potential and sum to the terminal current. The two original radial diffusion
problems, circuit constraint and side-current equation use the same residual
solver and analytical Jacobian. At zero terminal current, reversible metal can
reintercalate while external charge and terminal work remain zero. Neither
metal stock is a post-step SOC correction, and side current is not bounded by
the external source's current limit.

```sh
uv run --no-sync python -m chemworld.examples.battery_plating --output /tmp/plating-new.jsonl
```

This eleven-action synthetic example charges, rests, discharges and verifies
exact replay through the original runtime. CC, finite-current CV, the existing
I/V meter, staged control, shared clocks and JSON continuation use the ordinary
battery interfaces. Measurement time advances the cell. Material reset preserves
the equipment-owned stocks; replacing the cell initializes new stocks once.
The occupied volume is the fixed negative geometric volume plus positive active
volume, not a complete cell geometry or dynamic porosity model.

The SPM declaration is isothermal with fixed electrolyte and certified
nonincreasing, single-valued electrode OCVs. SEI combinations, partial
reversibility, thermal coupling, pore evolution,
cracking and complete chemical-energy closure are not defined by this model.
Existing EIS, SOH and SOC analyses reject a plating-enabled cell rather than
silently omit its metal inventory. The default example is not a calibrated
commercial-cell degradation model. Actual released-source comparison and all
numerical differences are recorded in
`workstreams/world_foundation/NATIVE_CONV46_PLATING_EXPERIMENT.md`.

The same `LithiumPlatingSpec` can be declared on `BatteryMPMSpec` for a
single-material particle distribution in each electrode. Each size bin retains
its actual radial Li profile and its own reaction current at the shared
electrode potential. There is still only one negative plated/dead Li pair.
The side reaction uses the sum of actual particle surface areas; a nominal
representative radius does not replace that sum. The particle distributions
retain their sourced number/area/volume basis and explicit volume weights.

```sh
uv run --no-sync python -m chemworld.examples.battery_mpm_plating --output /tmp/mpm-plating-new.jsonl
```

`plating_fields` exposes per-bin reaction currents and profiles alongside the
shared metal response. `battery_readout` reports capacity-weighted mean
fractions and area-weighted surface fractions and overpotentials. These
averages do not replace individual bin cutoff checks. `battery_particle_flux`
returns the total molar flux for the requested electrode, not one bin's flux.
All original equipment and electrical actions apply to this model; replacing
a cell can install a different distribution without preserving the old cell's
metal stocks. Thermal, SEI, directional OCP, multi-material composites and
porosity evolution remain outside this combination, and EIS/SOH/SOC stay
explicitly unavailable. Continuous-distribution source comparisons and their
finite-grid limits are documented separately in
`workstreams/world_foundation/NATIVE_CONV46_MPM_PLATING_EXPERIMENT.md`.

### SPM metal deactivation at an explicit constant rate

`LithiumPlatingSpec(mode="partially reversible", ...)` adds an explicit,
sourced `gamma_s_inv` column in `1/s` to the original four plating parameters.
Its separate schema version is required for that five-column declaration;
the original reversible and irreversible declarations remain unchanged.
The interfacial current still follows reversible plating/stripping kinetics.
Independently, the declared constant rate transfers actual plated Li to dead Li:
`d(plated)/dt = -q_strip/F - gamma*plated` and
`d(dead)/dt = gamma*plated`. This transfer creates no additional terminal current
or charge receipt. Both inventories remain in the original equipment slot.

With `gamma=0` the response reduces to reversible plating. With `k=0`, seeded
metal still deactivates exponentially; turning off electrochemical exchange
does not silently turn off the declared loss process. Zero plated Li has zero
deactivation rate and is an admissible initial state. Replacing material retains
the cell history; replacing equipment installs the explicitly declared new cell.

```sh
uv run --no-sync python -m chemworld.examples.battery_partial_plating --output /tmp/partial-plating-new.jsonl
```

The example uses synthetic parameters, including an explicit nonzero initial
dead-Li stock. This initialization extension is distinct from the actual
PyBaMM source comparison, whose initial dead concentration is zero. The
constant-rate source function is defined at zero SEI thickness; no fictitious
film or inverse-thickness default is inserted. This version supports SPM only:
the actual reference version's MPM partial-plating model fails during domain
construction. Existing MPM reversible/irreversible modes remain available.
SEI coupling, thermal physics, porosity, cracking, EIS/SOH/SOC and complete
chemical energy remain outside this mode. A synthetic constant loss rate does
not establish a material mechanism or a calibrated battery lifetime.

### Axen memory coupled to reversible SPM plating

An isothermal `BatterySPMSpec` can combine the existing Axen electrode OCP
declarations with `LithiumPlatingSpec(mode="reversible", ...)`. Each electrode
may use its single OCP or an explicitly sourced Axen branch pair. The original
particle amounts, metal stocks and electrode memory coordinates are retained
in the same equipment slot.

The memory follows the actual main intercalation current. With stripping
current positive, the negative electrode uses `q_main = I - q_strip`; the
positive electrode uses `-I`. Consequently, at zero terminal current, internal
stripping can continue to change negative-electrode memory while terminal
charge and electrical work stay zero. The evolving branch mixture feeds the
same Butler–Volmer closure and finite-voltage controller. Current-sigmoid OCP
has a different source definition and is not silently reinterpreted here.

```sh
uv run --no-sync python -m chemworld.examples.battery_axen_plating --output /tmp/axen-plating-new.jsonl
```

The example reuses the synthetic OCP and plating declarations and executes
the original charge/rest/discharge workflow with exact action replay. Meter
acquisition time, queued stages, shared time, material reset and equipment
replacement all act on the same persistent cell. The two supported noise modes
remain available via `--noise`.

This combination requires certified nonincreasing branches and the existing
isothermal finite-source domain. MPM with directional OCP and plating,
current-sigmoid with plating, Axen with partial/irreversible plating, SEI,
thermal physics, ordinary stationary EIS/SOH/SOC and complete chemical energy
remain outside this combination. Operational memory does not itself define
a thermodynamic hysteresis-energy law or a calibrated degradation mechanism.

### Historical temperature calibration

`temperature_convert(value_K, current, desired)` supports ITS-90 and the complete
tabulated ITS-68, ITS-48, ITS-27 and ITS-76 ranges. Historical IPTS/EPT aliases are
accepted explicitly. Each historical scale uses one cubic calibration and the
shared bounded root solver for its inverse; domains use actual mapped endpoints.
`temperature_scale_conversion` also returns the derivative of that same mapping.
The Kelvin/Celsius unit system and world temperature semantics remain unchanged.

```python
from chemworld.foundation.units import Quantity
from chemworld.physchem import temperature_quantity, water_refractive_index_iapws

calibrated = temperature_quantity(Quantity(26.85, "degC"), "IPTS-68", "ITS-90")
index = water_refractive_index_iapws(calibrated.value, 997)
```

`temperature_input` preserves original input and scale provenance in the existing
InputValue note. `temperature_observation` retains the raw scalar channel and adds
a derived channel in the original Observation, with sample/timestamps and native
JSON roundtrip. Measurement uncertainty uses first-order derivative propagation;
unknown calibration uncertainty stays unknown. Invalid numeric readings and curves
requiring per-point uncertainty are explicitly rejected by this scalar adapter.

`its90_68_difference` is a separate approximate polynomial method. Its default
`published_overlap` fills the old documentation's low-temperature gap with the
published middle polynomial. Explicit `convention="chemicals_1_5_2"` reproduces
the older API's zero correction in that gap and its junction choices. Such zeros
are compatibility behavior, not evidence that the scales coincide. Neither method
asserts thermodynamic-temperature accuracy or silently modifies caloric references.
Full node/interval, cross-function and source-inventory development evidence is
recorded in the foundation acceptance document. The original 285-row primary-CAS
gap remains the qualification denominator: 221 rows now have reviewed component
mappings, while 64 rows across 54 CAS numbers remain unresolved. Earlier source
failures and conflicting structures remain visible.

### Composition, flow and partial-molar utilities

The native `composition_math` and `fluid_response` modules cover the 63 scientific
public functions in `chemicals.utils` 1.5.2, through shared mathematical operations.
They require no reference package at runtime. `MixtureSpec` uses the same fraction
conversion and normalization functions; pure virial gas reports use the same
density, compressibility and heat-capacity identities.

```python
from chemworld.physchem import (
    fraction_basis, convert_component_flows, solve_stream_composition,
    partial_molar_properties, composition_gradient_to_moles,
)

molecular_weights = (18., 44., 70.)  # g/mol
mass_fractions = fraction_basis((.2, .3, .5), molecular_weights)
mass_flows = convert_component_flows(
    (2., 3., 5.), source="molar", target="mass",
    molecular_weights=molecular_weights,
)  # kg/s from mol/s
flows, zs, ws = solve_stream_composition(
    (2., None, None), (None, .3, None), (None, None, 350/518),
    molecular_weights,
)
partial = partial_molar_properties(13., (-5., 3., 9.), zs)
dn = composition_gradient_to_moles((-5., 3., 9.), zs, total_moles=10.)
```

Reference-liquid volume conversion requires the component reference molar volumes
in m³/mol; it does not calculate the actual volume of a mixed stream. A mixed flow
specification supplies exactly one molar flow, mole fraction or mass fraction per
component, including a positive absolute flow scale. Merging streams retains
component identities. Unknown partial-flow streams contribute zero only when
explicitly supplied as `None`. Zero total has no defined composition.

`weighted_property` supplies linear, logarithmic and real power means. The power
mean has the geometric limit at exponent zero; non-linear means require positive
properties. `regularized_fractions` is explicit preprocessing of zeros, and is
never applied to an inventory automatically. Partial-molar conversions distinguish
unconstrained fractions, the n−1 simplex and mole numbers at a stated total amount;
the extensive Hessian includes the required first-derivative contributions.

`fluid_response` supplies density/volume and mass/molar property conversion,
specific gravity/API/Baume scales, Watson characterization, parachor, molecular
radius of gyration, vapor mass quality and thermodynamic response identities.
All use SI except molecular weight in g/mol and explicitly named gravity scales.
`acoustic_velocity` returns m/s when molecular weight is provided, otherwise
m·sqrt(kg/mol)/s. Pressure/volume derivatives must refer to the same state;
derivative-based phase classification is not a phase-equilibrium calculation.

The fixed public comparison retains two discrepancies at power exponent 1.5:
the native result follows the documented equation, verified independently with
60-digit arithmetic (26/26 power-mean inputs); chemicals 1.5.2 agrees on 24/26.
The complete public comparison remains 203/205, not an all-pass claim. Six
independent partial-molar states also verify amount scaling, Euler and Gibbs–Duhem
identities. Input parsing and generic shape/copy/hash/JIT helpers are outside
this scientific reproduction scope.

### Pure-fluid Helmholtz properties

`chemworld.data.load_helmholtz_fluid` imports an explicitly supplied public
coefficient mapping and source. A multi-EOS source requires `eos_index`.
`fluid.state(temperature_K=..., molar_density_mol_m3=...)` computes pressure,
H/U/S, heat capacities, sound speed and response derivatives from the same
native energy function; `fluid.alpha_derivative(...)` exposes mixed derivatives
through order three. No CoolProp installation is used by this production path.

`load_helmholtz_phase_model` also imports critical values and ancillary guesses.
Its `saturation(T)` solves equal pressure and chemical potential; `solve_tp(...)`
selects an explicit or automatic homogeneous branch. A coexistence T/P pair
does not determine a unique density. `solve_homogeneous_state(inputs,
initial_state=...)` supports pairs drawn from T/P/rho/H/U/S with an explicit
starting state and reports local closure, not a two-phase flash.

The existing `ComponentPropertyPackage.evaluate_state(...,
helmholtz_model=phase_model)` returns normal unit-bearing property reports for
the same component identity. EOS caloric offsets are preserved; do not also
supply an unrelated `reference` or gas model. Branch diagnostics, source M/R
and source identity remain in the report. Uniform-state checks establish local
stability; they do not guarantee global phase equilibrium. Pseudo-pure phase
closure and unprovided transport laws remain explicit unsupported requests.

Explicit viscosity, conductivity or liquid surface-tension correlations in the
same component package can accompany these EOS properties. They retain their
own source, phase, selection policy and validity range; the EOS supplies no
implicit transport law.

`HelmholtzIsochoricThermodynamics` connects the same EOS to `CoupledSystem`
for one homogeneous pure-fluid inventory at a declared fixed volume. Heating
and material exchange use nU, nCv and the density-dependent inventory energy
derivative; port enthalpy uses the same fluid reference. Both original ODE and
residual strategies apply. Empty inventories, changing volumes, foreign fluids
and duplicate reaction heat are rejected. Numerical tolerances must resolve
the original energy balance; the adapter does not weaken that balance.

Install this thermodynamics object in the existing
`ThermalSystemSpec(..., pressure_law="helmholtz")` for native `heat`, `wait`,
pause/resume control and snapshots. The original vessel inventory owns n/T/V;
pressure and pressure-stop events use the same EOS. Material replacement mixes
the new fluid energy with retained support energy. This world mode accepts one
pure-fluid inventory and no concurrent reaction or undeclared material port.

Choose `HelmholtzEquilibriumIsochoricThermodynamics` explicitly for liquid–gas
equilibrium, supplying the phase model and the same fixed-volume inventory.
Its heat capacity includes the latent-energy response along constant density;
it is not an average of endpoint Cv values. A native world uses the aggregate
`phase="fluid"` declaration: n/T/V remain the sole inventory, and equilibrium
phase amounts are derived from the EOS. The original thermal actions, pressure
events, material replacement and snapshots consume this declaration. Pseudo-pure
glide, solids and changing volumes are outside this model. Independent tests cover
heating and cooling across the coexistence boundary with both numerical forms.

`solve_two_phase_state(model, inputs)` closes independent pairs drawn from
T/P/rho/H/U/S/Q on pure-fluid liquid–gas coexistence. T/P alone cannot determine
quality; pairs without T or P require an explicit temperature guess.
`flash_at_pressure(model, pressure_Pa=..., coordinate="H", value=...)` selects
the two-phase or stable homogeneous branch; U, S and rho are also supported.
Quality is vapor mole fraction, which equals mass fraction for one pure fluid.
Volume mixes by phase volume, while H/U/S mix by amount. These APIs supply no
unqualified two-phase Cp, sound speed or transport average.

`HelmholtzPressureStage` models an adiabatic throttle, compressor or expander
using the same H/S reference. Its `evaluate(HelmholtzStream(state, rate))`
returns outlet state and shaft power; compression and expansion apply their
declared isentropic efficiency to a PS reference state. A resolved two-phase
outlet can be separated into conserving liquid and gas streams. A stream's
`contribution(port_id)` supplies the existing material port's molar flow and
enthalpy to `CoupledSystem`. Shaft work is already in outlet enthalpy and must
not be added to the receiving tank again. Rates are explicit reservoir inputs;
the stage supplies no pressure-driven flow law.

The implementation has exercised all 159 source EOS parameter sets for alpha
and its derivatives, and the full 136-fluid default-EOS catalogue for phase and
state-pair behavior. These are distinct coverage claims. The fixed phase block
retains 21 exceptions: 20 targets outside the source's valid physical domain and
one low-temperature propylene-glycol disagreement with the reference saturation
solver. Separate source-domain and equal-pressure/equal-chemical-potential
checks are recorded; those checks do not relabel the original exceptions as passes.
Ancillary correlations provide initial guesses only. Extrapolating a guess is
reported and never extends the EOS temperature or pressure validity range.

### Liquid valves and differential-pressure meters

`LiquidControlValveSpec` declares installed metric Kv, geometry, recovery and
style factors. `size_liquid_control_valve` performs IEC engineering selection;
`liquid_valve_pressure_drop` and `liquid_valve_flow` evaluate the installed
single-liquid relation. Nominal conservative sizing is not an exact inverse.
Low-Re flow with attached reducers remains unsupported.

`LiquidOrificeSpec` supports ISO 5167 corner, flange and D-D/2 taps, or an
explicit calibrated coefficient with a declared Reynolds interval. Its pressure
report separates measured differential pressure from permanent pressure loss.
Mass-flow, bore and pressure-endpoint inverses share the same equation. Lab-scale
geometry requires an explicit calibration; it does not inherit industrial ISO
validity. All inputs are SI except explicitly named metric Kv in m³/h.

Set `TransportPathSpec.control_valve` and `.orifice_meter` to use these devices
in the existing finite pump/pipe/receiver world. The installed valve requires a
pure liquid package, an explicit vapor-pressure curve and
`component.metadata['critical_pressure_Pa']`. Current properties determine the
operating point. The original amount/enthalpy inventories, energy accounting,
shared clock and snapshot restoration are retained. A closed valve blocks
transfer and shaft work; a single-liquid path rejects flashing conditions.

`LiquidTransportProperties(activity_model=...)` can include the same declared
activity model in liquid enthalpy, heat capacity and stream heat transfer.
Nonideal mixing then changes the receiver temperature through its existing
enthalpy inventory. The model and provenance survive ordinary world snapshots.
This retains a homogeneous single-liquid assumption; liquid-liquid phase
splitting and excess volume are separate models.

### Isothermal gas-pipe engineering

`IsothermalGasState` declares constant temperature, specific gas constant and Z.
`isothermal_gas_pipe(...)` computes mass flow with density changing as
P/(Z R T). `solve_isothermal_gas_pipe(...)` solves one missing mass flow,
inlet pressure, outlet pressure, diameter or length using the same equation.
Pressure inverses update inlet density. Darcy friction and Z are explicit
constants, and the result reports the critical outlet pressure and capacity.
Equal pressures give zero flow; outlets below the choking boundary are rejected.
These steady engineering calculations do not create a dynamic gas inventory.

Constant-gamma engineering helpers also cover isothermal and isentropic work,
compression/expansion efficiency, outlet temperature, polytropic exponent,
stagnation states and critical conditions. `ideal_gas_nozzle(...)` reports
subcritical or choked flow for an ideal converging nozzle. Work is positive
into the gas, with separate compression and expansion efficiency conventions.

`EmpiricalGasPipelineSpec` selects IGT, Muller, Fritzsche, Panhandle A/B,
Weymouth, Oliphant or Spitzglass high/low and records its applicability.
`solve_empirical_gas_pipeline(...)` solves Q/P1/P2/D/L; Q is volume flow at
the declared reference T/P, not local pipe volume flow. Pressure-squared models
require absolute pressure. Oliphant has no efficiency or Z correction and
rejects non-unity inputs instead of silently ignoring them.

### Uniform channels and free-discharge weirs

`manning_velocity`, `chezy_velocity` and their coefficient conversions use SI
roughness in s/m^(1/3) and Chezy coefficients in m^0.5/s.
`uniform_channel_flow` uses explicit area, wetted perimeter and energy slope;
it is a steady relation. `manning_roughness_table` imports a source's nested
minimum/normal/maximum values into the original `ScientificTable`, and
`select_manning_roughness` selects an exact category path. These statistics
are not standard uncertainties, and external roughness tables are not bundled.

`weir_flow` covers Shen V notches and the declared rectangular Kindsvater–Carter,
Ackers and Rehbock relations with geometric domain checks. Both historical SIA
expressions require `reference_only=True` and return explicit validity warnings;
the full-width expression is dimensionally inconsistent and its source warns
against use. Reproducing either expression does not qualify it as a flow meter.
Submerged flow, water jumps and unprovided installation corrections are outside
these free-discharge formulas.

### Gas control valves

`GasControlValveSpec` declares installed Kv, geometry, recovery, style and
pressure-ratio factors. `size_gas_control_valve` selects nominal Kv;
`gas_valve_flow` evaluates a fixed installed valve and returns reference volume
flow, mass flow and actual inlet volume flow separately. Reference volume uses
273.15 K and 101325 Pa. Low-Re nominal selection is conservative and is not an
exact inverse of installed flow.

The default `expansion_basis="installed"` uses the installed xTP in the expansion
factor, including the choked plateau. `gas_valve_outlet_pressure` solves the
unique subcritical branch and returns the highest choked outlet at capacity.
`bare_valve_reference` explicitly reproduces fluids 1.3.1's fixed bare-xT
expansion convention; pressure inversion requires the installed convention.
Low-Re flow with attached reducers requires an additional declared correction
and is rejected. These engineering functions do not introduce a dynamic gas
inventory into the single-liquid transport world.

`ValveOpeningCurve.apply(full_open_spec, opening)` sets the original gas or
liquid valve's effective Kv and closed state. Linear opening is exact; quick
and equal-percentage curves require an explicit sourced `TabulatedScalar`.
The curve and source survive JSON roundtrip. Scaling always starts from the
full-open rating. Coefficient conversion supports Kv/Cv/Av, and the cavitation
and choking helpers retain explicit pressure/recovery conventions.

`liquid_control_valve_noise` and `gas_control_valve_noise` consume declared
`ValveAcousticGeometry` and operating properties. They report 33 frequency bands,
total A-weighted level, peak frequency, sound power and flow regime. The gas
model includes five regimes and the outlet-expander contribution. Its default
outlet density includes the stated T1/T2 ratio at fixed Z; supply actual outlet
density explicitly when the property model requires it. These are engineering
acoustic predictions, not a spatial sound-field simulation.

### Pneumatic conveying thresholds

`saltation_velocity` selects Rizk, Schade, Geldart-Ling, three Matsumoto variants
or Weber using current gas `FluidState`, solid mass flow and declared geometry.
Particle density and terminal velocity are explicit where required. Results
include velocity, gas volume/mass flow, the selected branch and source warnings.
Matsumoto 1974/1975 retain the reference's unresolved high-prediction warning.
These are empirical transport thresholds, not evolving particle trajectories or
a model of plugging and recovery.

### Straight, helical and plate friction

`straight_friction_correlation` exposes 34 named equations and explicit Darcy or
Fanning factors. `pipe_friction_factor` adds a declared laminar transition;
`available_straight_friction_methods` filters known source ranges. Colebrook and
Clamond represent the same physical equation here and share the existing root
solver. `crane_fully_turbulent_friction` uses its nominal-steel effective inputs.

`helical_friction_factor` supports 11 helical models, six transition correlations,
Dean number and model-specific curvature/roughness limits. `plate_friction_factor`
provides Kumar, Martin 1999/VDI and Muley-Manglik with explicit chevron geometry.
Published fit discontinuities and undefined source ranges remain visible;
formula evaluation outside recommended ranges requires an explicit choice.

### Screens and grills

`filter_loss_coefficient` covers round wire screens, four open-mesh patterns,
square screens and square/round grills. `filter_pressure_drop` uses current
`FluidState` density and viscosity, a declared Reynolds length and the upstream
approach velocity. Long-grill friction requires thickness, passage hydraulic
diameter and Darcy factor together; the extra term starts above 50 diameters.

`FilterCurve` wraps an existing sourced `TabulatedScalar`, with linear or quadratic
interpolation and explicit `raise`/`clip` endpoint behavior. Blevins curve arrays
are supplied by the caller. Recommended open fractions and turbulent mesh flow
are checked. This is an engineering pressure calculation; it does not simulate
particle capture or progressive filter fouling.

### Packed-bed pressure laws

`PackedBedSpec` selects one of 14 pressure correlations with explicit length,
particle size, voidage, sphericity and, where consumed, tube diameter.
`packed_bed_pressure` accepts the existing `FluidState`, reports particle and
modified Reynolds numbers, and returns source-range notes. `range_policy="raise"`
requires all available range qualifications. Ergun reuses the original kernel.
`packed_bed_voidage` supplies the general, spherical and cylindrical
Benyahia-O'Neil estimates within their declared geometry ranges.

The existing `TransportPathSpec` accepts `bed_method`, `bed_sphericity` and
`bed_tube_diameter_m`. They feed its current-property pressure loss and pump
working point while retaining the same material and enthalpy inventories.
Guo-Sun requires a real tube diameter; negative predicted pressure loss is
rejected. The cylindrical voidage law does not accept an unused sphericity.

### Gas/liquid holdup and mixture viscosity

`TwoPhaseProperties` declares both phase densities and any required viscosities,
surface tension or pressure. `two_phase_void_fraction` exposes 29 named models;
`two_phase_mixture_viscosity` exposes six. Mass quality and volumetric void fraction
are separate inputs: at equal mass, a low-density vapor can occupy most of the
volume. Density and its void-fraction inverse use the same declared phase values.

Availability functions describe input completeness. Known recommended ranges are
checked by default; an explicit formula-analysis option can disable those range
checks. Nonphysical void fractions still raise with the unmodified prediction.
Pure phases use exact zero/one endpoints. These engineering correlations do not
create a dynamic two-phase inventory or establish validity for every flow regime.

`two_phase_pressure_drop` evaluates 24 named pressure relationships with explicit
properties, geometry and quality. Its report distinguishes friction-only loss
from Beggs-Brill's coupled friction, holdup and gravity result. The primitive
Beggs acceleration correction is an explicit option, off by default. Bankoff
requires `reference_only=True` because its source warns against engineering use.
Known recommended ranges are checked by default; an explicit formula-analysis
request retains warnings. Model inputs are not a general validity certificate.

`colebrook_friction_factor` uses the shared root solver. The separate engineering
`pipe_friction_factor` uses an explicit transition Reynolds number, default 2040;
the existing transport auto/Haaland convention retains its original behavior.

Signed gravity and acceleration APIs operate on explicit phase endpoints or
local derivatives. Expansion at constant density produces pressure recovery;
the upstream reference's opposite area-term sign is recorded as a discrepancy.
These functions preserve mass-quality versus void-fraction bases.

The Mandhane-Gregory-Aziz and Taitel-Dukler flow-map APIs return SI superficial
velocities and the selected regime. Taitel-Dukler consumes three caller-supplied
`LogRegimeBoundary` curves through existing sourced tables. Chart extrapolation
is off by default. The selected Colebrook convention differs from the original
paper's Blasius convention and is reported in the result.

### Relief sizing and flare noise

`size_gas_relief_valve`, `size_liquid_relief_valve` and
`size_steam_relief_valve` solve API-style area or inverse mass flow with explicit
pressure, discharge and correction factors. Steam service must be declared dry
saturated or superheated; these engineering formulas do not classify phase.
`api526_select_orifice` selects the next of 14 effective standard areas.

API520 correction tables are explicit, source-bearing `TabulatedScalar` inputs;
standard table data are not bundled. Table domains are strict unless a supported
call explicitly requests endpoint clipping, and standard model limits still apply.
The liquid viscosity correction retains the declared two-step sizing convention.
`api521_flare_noise` applies distance attenuation; `vdi3732_flare_noise` uses
tonnes/hour converted from kg/s by 3.6. The latter intentionally differs from
fluids 1.3.1's factor of 360. These are sizing/noise relations, not a dynamic
two-phase relief model.

### Activity thermodynamics and liquid property consumption

The existing `ActivityModelSpec` supports ideal, Margules, Wilson, NRTL,
UNIQUAC, regular solution, Flory–Huggins, Hansen and six UNIFAC variants through a shared
excess Gibbs-energy contract. Request excess H, S, Cp,
analytic temperature/composition derivatives and partial molar quantities with
`activity_thermodynamic_report(model, composition, temperature_K=...)` from
`chemworld.physchem`. `activity_infinite_dilution(...)` evaluates exact zero
solute concentration in explicitly supplied remaining solvent proportions.
The results are immutable and provide `to_dict()` for JSON output.

`regular_solution_model(...)` compiles explicit reference molar volumes,
solubility parameters and directed interaction corrections. Without a correction
request it records explicit zero corrections; incomplete supplied pairs fail.
`hansen_activity_model(...)` consumes source-bearing `HansenParameters` and
unit-bearing reference volumes, with an explicit alpha. These models hold
reference volumes and cohesive parameters constant in temperature. The sum of
opposite regular-solution corrections is identifiable; fitting cannot identify
the two directions separately. Source data remain in the model's provenance.

Pass `activity_model=model` to the existing `evaluate_property_state(...)`
liquid interface to combine these responses with declared pure-liquid caloric
references. Supported requests include `enthalpy`, `cp`, `entropy`,
`chemical_potential` and `activity_coefficient`. Component IDs must match the
property packages. An activity-only request needs no unrequested heat-capacity
data; entropy and chemical potentials require explicit entropy references.
The model supplies neither a volume law nor a constant-volume energy closure.

`chemworld.data.ActivityCalibration` fits explicit model coefficients through
the existing bounded calibration driver. Its ScientificTable conditions contain
temperature and all component fractions; observations may contain ln-gamma,
GE, HE, SE or CpE. Training and holdout groups remain separate, and fitting
retains the actual evaluation ledger. These functions do not imply a complete
interaction-parameter database, group-contribution coverage or measured material
accuracy. Asymmetric Margules results now follow an integrable three-suffix
equation; the former componentwise expression was thermodynamically inconsistent.

`unifac_activity_model(...)` accepts explicit molecular subgroup counts,
source-bearing subgroup parameters and every directed unlike-main-group
interaction. Versions 0–5 select original/LLE, Dortmund/NIST, PSRK, VTPR,
Lyngby/Larsen and KT. The returned `ActivityModelSpec` uses the same caloric,
derivative, dilution, calibration and liquid-property interfaces above.
Group topology is discrete; only continuous interaction coefficients are fit.

The `chemworld.data` readers `read_unifac_subgroups`,
`read_unifac_interactions`, `iter_ddbst_text` and `iter_ddbst_sqlite` consume
explicit local source files. `select_unifac_interactions` requires an explicit
choice for overlapping revisions and retains superseded records.
`unifac_model_from_assignments` maps declared native component identities to
source assignments; automatic DDBST numbering applies to versions 0/1/2 only.
Missing interactions and invalid assignments fail explicitly. Source temperature
ranges remain provenance; they are not automatically enforced as model domains.
The source tables and the reference package are not runtime dependencies.

### Cubic EOS alpha and Bell2018 parameters

`EOSComponentSpec` retains its default Soave alpha and can explicitly select
`alpha_model="twu91"` with the `(L, M, N)` coefficient tuple and source provenance.
`cubic_alpha_response` reports alpha and its first three analytic temperature
derivatives. The original PR/SRK state and departure-property calculations consume
the selected alpha directly.

`chemworld.data.twu_eos_component_from_record` consumes an existing `BELL_TWU`
phase-change record, preserving its InChIKey, Tc/Pc, coefficients and source.
An optional native component must match that identity. Bell supplies no acentric
factor; the legacy zero field is unused by this explicit Twu path. Full source
consumption establishes equation compatibility, not transferability of fitted
coefficients between EOS families or measured predictive accuracy.

### Predictive PR78 interactions

`PPR78Spec` binds explicit molecular groups to a selected original or extended
interaction table and its provenance. Missing or asymmetric pairs are errors.
`ppr78_interaction_response` supplies k and its first three temperature derivatives.
Use `CubicEOSSpec(pure_parameterization="pr78", predictive_interaction=...)`
to consume those predictions in the original EOS roots, fugacities and departure
properties. Temperature-dependent k contributes to mixture da/dT and hence H/S.

This path uses exact PR78 constants and its high-acentric-factor branch with
Soave alpha. It rejects SRK, Twu and simultaneous constant interactions; existing
legacy PR/SRK defaults remain available. Explicit group assignment is required.
Full source consumption verifies equations, not material prediction accuracy;
combinations with negative mixture attraction are outside the admitted EOS domain.

### Excess-Gibbs cubic mixing

`GibbsExcessMixingSpec` connects the existing activity models to PR/SRK through
Huron-Vidal, exact MHV, MHV1, MHV2 or Wong-Sandler. Set it as `CubicEOSSpec.ge_mixing`.
MHV1/2 require explicit q coefficients; Wong-Sandler requires every unlike binary
pair. Quadratic and predictive interactions cannot be silently combined with GE
mixing. Exact MHV requires a differentiable zero-pressure liquid branch.

The original EOS roots remain the state solver. Fugacity, enthalpy and entropy
differentiate one residual Helmholtz potential, including Wong-Sandler's
temperature-dependent covolume. Thus selecting a new mixing rule changes both
phase behavior and energy consistently. The existing activity families and all
six UNIFAC variants can use this interface; model choice and material calibration
remain explicit.

### Unified homogeneous phase properties

`PhasePropertyModel` freezes existing component packages, phase, references,
activity/EOS choices and explicit transport declarations. For one component,
use `package.phase_model(phase=..., reference=...)`; a pure Helmholtz model may
be supplied with its own caloric reference. `capabilities()` shows structurally
available quantities. `evaluate(amounts, temperature_K=..., pressure_Pa=...,
quantities=...)` reads caller-owned amounts and returns an immutable
`PhaseEvaluation` with units, sources, validity and composition.

`PhaseTransportSpec` selects one of 12 existing mixture transport methods.
Pure curves are evaluated at current conditions, with explicit method parameters;
LBC consumes current volume, and Diguilio-Teja consumes pure surface tension at
each boiling temperature. Fixed-composition correlations require composition
binding. Missing sources or closures stay unavailable. This interface owns no
inventory, clock or phase-split solver, and preserves existing JSON declarations.

### Properties from declared molecular groups

`GroupContributionTable` stores a selected source's coefficients, including
explicit missing values. `joback_estimate` produces all eleven Joback outputs;
`structural_group_estimate` handles Fedors critical volume, first/second-order
Wilson–Jasperson critical properties and Bondi volume, surface, R and Q.
Inputs are explicit molecular groups, atom counts and structural features.
These APIs do not perform SMILES fragmentation.

`GroupEstimateReport.critical_estimate(...)` feeds the existing critical-property
and cubic-EOS inputs. `joback_property_correlations(...)` compiles Cp and viscosity
into the original property catalogue for an authored temperature interval.
Missing increments remain unavailable; a partial area or arbitrary replacement
critical temperature is not returned as a complete estimate. Predicted formation
energy does not silently establish an absolute caloric reference, and source
coverage does not establish predictive accuracy for every material.

### Formal SBML models on the native world

Given a provenance-bearing `SBMLModel`, declare its numerical profile and the
quantities that an instrument may observe:

```python
from chemworld.physchem.sbml_system import SBMLObservable
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.sbml_world import sbml_world

world = sbml_world(model, maximum_step_s=0.02,
                   observables=(SBMLObservable("amount", "A"),))
runtime = ChemWorldRuntime(composition=world, observation_noise_mode="keyed")
runtime.reset(seed=4)
runtime.execute({"operation": "wait", "duration_s": 0.25})
runtime.execute({"operation": "measure", "instrument": "formal_quantity_meter"})
reading = runtime.observe()  # declared mol channel with detector uncertainty
saved = runtime.snapshot().to_dict()  # private continuation; keep outside agent payloads
restored = ChemWorldRuntime.from_snapshot(saved)
```

The bounded Level 3 Version 2 profile supports explicit 3D compartments, source
units, kinetic laws, initial/assignment/rate rules and transverse inequality
events with delays, cancellation, captured assignments and execution priorities.
Unsupported semantics are rejected. Events require an authored finite maximum
step; arbitrarily fast crossings and tangencies have no general guarantee.

One existing equipment entity owns all live amounts, compartment volumes,
parameters and pending events. The existing shared laboratory clock owns time.
`numerical_form="residual"` selects the common residual solver. Source quantities
become mol, L and seconds at explicit boundaries; model formulas retain their
source basis. A concentration rate rule includes the volume chain term, and an
event concentration assignment uses the volume immediately before execution.

The workspace has no physical chemical stock or liquid heat capacity. Formal
coordinates do not establish elemental, charge, energy or carrier-mass closure,
and heat/material sampling operations are unavailable. The gauge is instantaneous
and consumes no sample; its allowed amount/concentration/volume channels carry
mol, mol/L or L with declared Gaussian uncertainty. Public state omits evolving
coordinates and queues. Material preparation preserves the formal entity; a full
experiment reset initializes it again. A failed event cascade rolls back the
complete advance. Full private snapshots preserve pending events and detector
continuation across processes.

The fixed native-world block passed all 36 runtime contracts and 770 independent
closed-form comparisons. Full SDK parity remains explicitly false: the 18-model
COPASI block has three retained differences, and RoadRunner has one source-model
initialization difference. Official SBML volume/event cases and all raw failed
attempts are retained in the foundation evidence records.

`SEDMLProtocol` compiles bounded SED-ML L1V4 model changes, uniform time courses,
finite parameter scans and ordered or nested tasks into existing native worlds.
`run_sedml_experiment` executes their native waits, measurements and declared
parameter controls, returning reports with source units, sample coordinates and
repeat axes. `SEDMLProtocol.from_archive` accepts a validated `CombineArchive`;
archive import retains the original XML and resolves local model references.
It executes no archive code and fetches no external models. Unsupported algorithms,
targets or experiment structures are rejected before creating native sessions.

The twelve-case development block passed 910 independent analytic comparisons
and 910 exact archive round-trip report comparisons. Complete SDK interoperability
remains unqualified: retained differences include species readout bases, computed
model changes, nested clocks and SDK failures on functional ranges or unequal
subtasks. Final native model snapshots preserve those sessions. For the complete
finite experiment, `SEDMLExperimentSession` in `chemworld.eval.sedml_continuation`
also preserves the task stack, collected observations and future reset identities.
Call `advance(max_courses=1)` to pause after a complete time course, serialize
`to_dict()`, then restore with `from_dict()` and continue. This checkpoint is private:
it contains native model state and must not be exposed as an agent observation.
Failed continuations remain terminal and retain the actual operation prefix.
The fixed twelve-protocol process block and explicit pending/terminal cases pass
complete private JSON continuation checks under both detector noise modes.
The complete four-entry block passes 304 successful and 32 expected-failure histories,
all 336 replays, and 84 native process continuations across ODE/residual forms and both
detector noise modes. The original failed recorder attempt remains preserved.
Neutral optional CPU declarations preserve real saved v2 checkpoints, with the
four-entry block, whole protocols and installed wheel requalified after the fix.
Non-default and unknown execution contracts remain strict.
Retained independent SDK disagreements remain separate from continuation correctness;
exact continuation does not establish agreement with an external simulator.

## Multifluid mixtures, local criticality and fitting

`MultifluidModel` combines explicitly supplied pure Helmholtz equations, oriented
binary reducing parameters and departure terms. Import a complete declaration
with `chemworld.data.read_multifluid_model(path, source=...)`, or select named
components from a caller-provided `MultifluidSourceCatalog`. Missing pairs are
errors. The gas constant convention, component order, original source constants
and fitted parameters remain explicit. Production does not download parameters
or depend on teqp/CoolProp.

The model's `caloric_model()` returns the existing `MolecularCaloricModel` with
source-derived ideal references. Use it with `MolecularIsochoricThermodynamics`,
`MolecularStream` and `CoupledSystem` to advance real inventories and account for
heat and incoming enthalpy. Model selection does not create another inventory.
These homogeneous consumers require a locally stable state. General phase
selection is a separate calculation with explicitly declared density basins.

For an already imported methane/ethane declaration ordered in that sequence:

```python
from chemworld.physchem import (
    molecular_composition_response, solve_molecular_critical,
    trace_molecular_critical,
)

response = molecular_composition_response(model, 450.0, (250.0, 250.0))
critical = solve_molecular_critical(
    model, (0.5, 0.5), initial_temperature_K=260.0,
    initial_density_mol_m3=9000.0,
    temperature_bounds_K=(180.0, 350.0),
    density_bounds_mol_m3=(3000.0, 20000.0),
)
if critical.success:
    print(critical.response["temperature_K"], critical.response["pressure_Pa"])
else:
    print(critical.failure_reason, critical.diagnostics)

curve = trace_molecular_critical(
    model, ((0.4, 0.6), (0.5, 0.5), (0.6, 0.4)),
    initial_temperature_K=260.0, initial_density_mol_m3=9000.0,
    temperature_bounds_K=(180.0, 350.0),
    density_bounds_mol_m3=(3000.0, 20000.0),
)
```

Composition responses include ideal mixing, the full chemical-potential density
Hessian, fixed-temperature/pressure and fixed-temperature/density composition
responses, pressure derivatives and a susceptibility only when invertible and
locally stable. Exact zero components select an active composition face; they
do not request absent-component insertion derivatives. Critical roots require
one soft mode, zero cubic coefficient, positive transverse curvature and a
positive fourth coefficient after transverse relaxation. Singular heat capacity
is not replaced by a finite placeholder. Source pressure limits still apply.
The curve follows the supplied composition path and stops at an unresolved
station; it does not claim to continue through every fold or prove global phase
stability. Keep the returned failure and completed/requested station counts.

`MultifluidCalibration` fits declared continuous coefficients using the shared
`ScientificTable` measurement schema, uncertainties and train/validation/test
splits. Conditions default to `temperature`, `molar_density` and `x:<component>`;
observations map to state fields, `mu:<component>` or `lnphi:<component>`.
Parameters address an exact oriented binary pair, a departure coefficient or a
pure residual amplitude. For example, with a prepared measurement `table`:

```python
from chemworld.data import MultifluidCalibration, MultifluidFitParameter
from chemworld.physchem import MultifluidModel

pair = model.binary_pairs[0]
parameter = MultifluidFitParameter(
    "binary", pair.component_ids, "gamma_T",
    bounds=(0.9 * pair.gamma_T, 1.1 * pair.gamma_T), initial=pair.gamma_T,
)
report = MultifluidCalibration(
    table, model, (parameter,), fitted_source="My declared training dataset",
).fit(max_nfev=100)
if report["status"] == "fitted":
    fitted = MultifluidModel.from_dict(report["fitted_model"])
    caloric = fitted.caloric_model()
```

Inspect held-out errors, parameter bounds and the local identifiability report
before using a fit. Fitting does not change source validity limits or add missing
terms. Synthetic parameter recovery demonstrates the fitting interface, not
empirical accuracy for a material. The fitted result uses the original model
codec and carries a new source description.

Two explicit model combinations share these consumers:

- `MultifluidAssociationModel` adds reciprocal cross-component association using
  the existing masked SAFT site solver. Same-component bonds must be disabled so
  the already fitted pure-fluid EOS is preserved.
- `MultifluidActivityModel` uses the Jaeger activity departure with declared
  covolumes, reference density parameter and residual-activity convention.
  `wilson_volume_residual` removes the Wilson volume contribution explicitly;
  `declared_residual` means the caller supplied the residual activity model.

Both require `combination_policy="explicit_joint_parameterization"` and a source
description. They preserve pure-fluid limits and the common gas constant, but
do not inherit empirical qualification of the original mixture parameters.
`caloric_model()` and the common codec carry the complete coupled potential into
the original heat/material consumers. Validation uses the final coupled pressure.
