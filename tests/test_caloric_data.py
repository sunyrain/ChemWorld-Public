"""External caloric sources share native provenance, units and input contracts."""

import json
from dataclasses import replace

import pytest

from chemworld.data.caloric import CaloricDatabase
from chemworld.data.constants import (
    AmbiguousConstantIdentityError,
    ConstantSourceUnavailableError,
    ConstantValueUnavailableError,
    NonScalarConstantError,
    UnknownConstantMethodError,
)


def write_source(directory, source, content):
    path = directory / source.relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_caloric_catalog_laziness_and_explicit_missing_source(tmp_path):
    database = CaloricDatabase(tmp_path)
    assert len(database.catalog) == 21
    assert database.loaded_sources == ()
    heat_sources = [s for s in database.catalog if s.relative_path.startswith("Heat Capacity/")]
    assert len(heat_sources) == 16
    with pytest.raises(UnknownConstantMethodError):
        database.records("guessed-source")
    with pytest.raises(ConstantSourceUnavailableError):
        database.records("POLING_CP")
    assert database.loaded_sources == ()


def test_caloric_scalar_phase_reference_missing_and_ambiguity(tmp_path):
    database = CaloricDatabase(tmp_path)
    write_source(
        tmp_path,
        database.source("ATCT_GAS"),
        "CAS\tChemical\tHfg_0K\tHfg\tuncertainty\n"
        "50-00-0\tFormaldehyde\t-105330\t-109160\t110\n"
        "50-00-0\tAlternate\t-105330\t\t110\n",
    )
    rows = database.records("ATCT_GAS")
    assert len(rows) == 2 and database.loaded_sources == ("ATCT_GAS",)
    assert database.records("ATCT_GAS") is rows
    with pytest.raises(AmbiguousConstantIdentityError):
        database.record("50-00-0", method="ATCT_GAS")
    value = database.value("Formaldehyde", "Hfg", method="ATCT_GAS")
    assert value.quantity("kJ/mol").value == pytest.approx(-109.16)
    assert value.input_value("kJ/mol").value == pytest.approx(-109.16)
    assert value.record.source.provenance.source_id == "chemicals-caloric:ATCT_GAS"
    assert "0 K" in rows[0].descriptor("Hfg_0K").description
    assert "298.15 K" in value.descriptor.description
    missing = database.value("ATCT_GAS:2", "Hfg", method="ATCT_GAS")
    assert missing.quality == "missing"
    with pytest.raises(ConstantValueUnavailableError):
        missing.quantity()
    assert len(database.candidates("50-00-0", "Hfg", methods=("ATCT_GAS",))) == 2


def test_nested_caloric_sources_keep_all_payloads_and_do_not_become_scalars(tmp_path):
    database = CaloricDatabase(tmp_path)
    payload = {"7732-18-5": [[0, 100, 200], [0, 20, None]], "unidentified": [[], []]}
    write_source(tmp_path, database.source("JANAF_GAS_CP"), json.dumps(payload))
    rows = database.records("JANAF_GAS_CP")
    assert [row.source_key for row in rows] == list(payload)
    assert rows[0].raw("payload") == payload["7732-18-5"]
    with pytest.raises(NonScalarConstantError):
        rows[0].value("payload")
    assert rows[1].cas_number is None


def test_piecewise_property_input_roundtrip_uses_existing_envelope():
    from chemworld.examples.scientific_inputs import curated_input
    from chemworld.physchem import PiecewiseHeatCapacity
    from chemworld.physchem.property_inputs import PropertyInputCard

    original = curated_input()
    first = next(
        c for c in original.package.correlations if c.property_id == "ideal_gas_heat_capacity"
    )
    lower, upper = first.validity_ranges["temperature"]
    middle = (lower + upper) / 2
    curve = PiecewiseHeatCapacity(
        first.correlation_id,
        (
            replace(first, validity_ranges={"temperature": (lower, middle)}),
            replace(
                first, correlation_id="second", validity_ranges={"temperature": (middle, upper)}
            ),
        ),
        "Synthetic imported same-phase ranges",
    )
    selected_uncertainty = {first.correlation_id: original.uncertainties[first.correlation_id]}
    card = replace(
        original,
        package=replace(original.package, correlations=(curve,)),
        uncertainties=selected_uncertainty,
    )
    restored = PropertyInputCard.from_json(json.dumps(card.to_dict()))
    assert restored.to_dict() == card.to_dict()
    request = {
        "temperature_K": 400,
        "pressure_Pa": 1000,
        "phase": "gas",
        "quantities": ("cp", "enthalpy"),
    }
    assert restored.evaluate_state(**request) == card.evaluate_state(**request)
    payload = card.to_dict()
    payload["package"]["correlations"][0]["segments"][1]["coefficients"]["a"] = True
    with pytest.raises(ValueError):
        PropertyInputCard.from_dict(payload)


def test_source_shomate_ranges_really_feed_piecewise_enthalpy(tmp_path):
    from chemworld.physchem.heat_capacity import PiecewiseHeatCapacity
    from chemworld.physchem.property_equations import integrate_correlation

    database = CaloricDatabase(tmp_path)
    write_source(
        tmp_path,
        database.source("WEBBOOK_SHOMATE"),
        json.dumps(
            {
                "7732-18-5": [
                    None,
                    None,
                    [[200, 400, 30, 0, 0, 0, 0], [400, 600, 40, 0, 0, 0, 0]],
                ]
            }
        ),
    )
    curve = database.heat_capacity("7732-18-5", method="WEBBOOK_SHOMATE", phase="gas")
    assert isinstance(curve, PiecewiseHeatCapacity)
    integral = integrate_correlation(curve, initial_temperature_K=300, final_temperature_K=500)
    assert integral.value == pytest.approx(7000)
    selected = database.heat_capacity(
        "7732-18-5", method="WEBBOOK_SHOMATE", phase="gas", temperature_bounds_K=(250, 450)
    )
    assert selected.validity_ranges == {"temperature": (250, 450)}
    with pytest.raises(ValueError, match="coverage"):
        database.heat_capacity(
            "7732-18-5", method="WEBBOOK_SHOMATE", phase="gas", temperature_bounds_K=(100, 500)
        )


def test_tabulated_cp_requires_interpolation_and_preserves_missing_points(tmp_path):
    from chemworld.physchem.property_equations import evaluate_correlation

    database = CaloricDatabase(tmp_path)
    write_source(
        tmp_path,
        database.source("JANAF_GAS_CP"),
        json.dumps(
            {
                "7732-18-5": [[0, 100, 200, 300, 400], [0, 20, 30, None, 50]],
            }
        ),
    )
    data = database.heat_capacities("7732-18-5", method="JANAF_GAS_CP")[0]
    assert data.temperatures_K[0] == 0 and data.capacities_J_mol_K[3] is None
    with pytest.raises(ValueError, match="positive temperature"):
        data.build(interpolation="linear")
    with pytest.raises(ValueError, match="explicit interpolation"):
        data.build(temperature_bounds_K=(100, 200))
    with pytest.raises(ValueError, match="missing capacity"):
        data.build(interpolation="linear", temperature_bounds_K=(100, 400))
    curve = data.build(interpolation="linear", temperature_bounds_K=(100, 200))
    assert evaluate_correlation(curve, temperature_K=150).value == pytest.approx(25)


def test_perry_source_phases_keep_polynomial_calorie_conversion(tmp_path):
    from chemworld.physchem.property_equations import evaluate_correlation

    database = CaloricDatabase(tmp_path)
    phases = {
        key: {
            "Phase": key,
            "Subphase": None,
            "Const": 7,
            "Lin": 0.01,
            "Quadinv": 0,
            "Quad": 0,
            "Tmin": 200,
            "Tmax": 600,
        }
        for key in ("c", "l", "g")
    }
    write_source(tmp_path, database.source("PERRY_151_JSON"), json.dumps({"test": phases}))
    for phase in ("solid", "liquid", "gas"):
        curve = database.heat_capacity("test", method="PERRY_151_JSON", phase=phase)
        assert evaluate_correlation(curve, temperature_K=300).value == pytest.approx(41.84)
        assert curve.metadata["phase"] == phase


def test_repeated_table_temperature_keeps_distinct_one_sided_limits(tmp_path):
    from chemworld.physchem.property_equations import evaluate_correlation

    database = CaloricDatabase(tmp_path)
    write_source(
        tmp_path,
        database.source("JANAF_SOLID_CP"),
        json.dumps(
            {
                "synthetic": [[100, 200, 200, 300], [10, 20, 40, 50]],
            }
        ),
    )
    data = database.heat_capacities("synthetic", method="JANAF_SOLID_CP")[0]
    with pytest.raises(ValueError, match="one side"):
        data.build(interpolation="linear")
    left = data.build(temperature_bounds_K=(100, 200), interpolation="linear")
    right = data.build(temperature_bounds_K=(200, 300), interpolation="linear")
    assert evaluate_correlation(left, temperature_K=200).value == 20
    assert evaluate_correlation(right, temperature_K=200).value == 40
    assert data.temperatures_K == (100, 200, 200, 300)


def test_vibrational_source_never_guesses_shape_or_empirical_range(tmp_path):
    from chemworld.physchem.property_equations import evaluate_correlation

    database = CaloricDatabase(tmp_path)
    write_source(tmp_path, database.source("PSI4_ADJUSTED_THETA"), json.dumps({"test": [3000]}))
    data = database.heat_capacities("test", method="PSI4_ADJUSTED_THETA")[0]
    with pytest.raises(ValueError, match="supply exactly"):
        data.build(temperature_bounds_K=(100, 500))
    with pytest.raises(ValueError, match="temperature range"):
        data.build(parameters={"linear": 1})
    curve = data.build(parameters={"linear": 1}, temperature_bounds_K=(100, 500))
    assert evaluate_correlation(curve, temperature_K=100).value == pytest.approx(
        3.5 * 8.31446261815324
    )


def test_chemsep_equations_units_and_explicit_missing_range(tmp_path):
    from chemworld.physchem.property_equations import evaluate_correlation

    database = CaloricDatabase(tmp_path)
    write_source(
        tmp_path,
        database.source("CHEMSEP_XML"),
        """<compounds><compound>
    <CAS value="7732-18-5"/><IdealGasHeatCapacityCp units="J/kmol/K">
    <eqno value="16"/><A value="20000"/><B value="0"/><C value="0"/>
    <D value="0"/><E value="0"/><Tmin value="200"/><Tmax value="600"/>
    </IdealGasHeatCapacityCp><SolidHeatCapacityCp units="J/kmol/K">
    <eqno value="1"/><A value="10000"/><Tmin value="200"/><Tmax value="0"/>
    </SolidHeatCapacityCp></compound></compounds>""",
    )
    data = database.heat_capacities("7732-18-5", method="CHEMSEP_XML")
    assert database.constants.loaded_sources == ("CHEMSEP_XML",)
    gas = next(c for c in data if c.phase == "gas").build()
    assert gas.equation_id == "chemsep16"
    assert evaluate_correlation(gas, temperature_K=300).to("J/(mol*K)").value == 20.001
    solid = next(c for c in data if c.phase == "solid")
    assert solid.source_bounds_K == (200, None) and solid.notes
    with pytest.raises(ValueError, match="temperature range"):
        solid.build()
    declared = solid.build(temperature_bounds_K=(200, 400))
    assert declared.metadata["temperature_range_basis"] == "caller-supplied"
    assert evaluate_correlation(declared, temperature_K=300).to("J/(mol*K)").value == 10


def test_chemsep_capacity_bounds_reject_an_interior_negative_pocket():
    from chemworld.physchem.property_equations import evaluate_correlation, integrate_correlation
    from chemworld.physchem.specs import PropertyCorrelation

    curve = PropertyCorrelation(
        "interior",
        "chemsep16",
        {"A": -0.5, "B": 0, "C": 0, "D": -2, "E": 1},
        {"temperature": "K"},
        "J/(mol*K)",
        "liquid_heat_capacity",
        {"temperature": (0.1, 3)},
    )
    assert all(evaluate_correlation(curve, temperature_K=T).value > 0 for T in (0.1, 3))
    with pytest.raises(ValueError, match="nonpositive"):
        integrate_correlation(curve, initial_temperature_K=0.1, final_temperature_K=3)


def test_full_source_table_keeps_zero_kelvin_and_missing_cp_in_native_input(tmp_path):
    from chemworld.data.scientific_inputs import QuantityColumn, ScientificTable

    database = CaloricDatabase(tmp_path)
    write_source(
        tmp_path,
        database.source("JANAF_SOLID_CP"),
        json.dumps(
            {
                "test": [[0, 100, 200, 300], [0, None, 30, "invalid"]],
            }
        ),
    )
    curve = database.heat_capacities("test", method="JANAF_SOLID_CP")[0]
    table = curve.scientific_table(
        temperature_column=QuantityColumn("T", "K", "condition", (0, 300), "Source temperature"),
        capacity_column=QuantityColumn("Cp", "J/(mol*K)", "observation", (0, 100), "Solid Cp"),
        dataset_id="table",
        group_id="original-source",
        split="unused",
    )
    assert len(table.rows) == 4
    assert table.rows[0].values["T"].value == 0
    assert table.rows[1].values["Cp"].quality == "missing"
    assert table.rows[3].values["Cp"].quality == "invalid"
    assert ScientificTable.from_json(json.dumps(table.to_dict())).to_dict() == table.to_dict()


def test_source_formation_datum_and_curve_feed_existing_native_property_input(tmp_path):
    from math import log

    from chemworld.physchem.property_inputs import PropertyInputCard
    from chemworld.physchem.specs import ComponentSpec
    from chemworld.physchem.thermochemistry import CorrelationSpeciesThermo

    database = CaloricDatabase(tmp_path)
    write_source(
        tmp_path,
        database.source("CRC_THERMODYNAMIC"),
        "CAS\tChemical\tHfg\tS0g\tHfl\n7732-18-5\twater\t-10000\t50\t-20000\n",
    )
    write_source(
        tmp_path,
        database.source("WEBBOOK_SHOMATE"),
        json.dumps(
            {
                "7732-18-5": [
                    None,
                    None,
                    [[200, 400, 30, 0, 0, 0, 0], [400, 600, 40, 0, 0, 0, 0]],
                ]
            }
        ),
    )
    reference = database.formation_reference(
        "7732-18-5",
        phase="gas",
        enthalpy_method="CRC_THERMODYNAMIC",
        entropy_method="CRC_THERMODYNAMIC",
        pressure_Pa=100000,
    )
    assert reference.energy_J_mol == -10000 and reference.entropy_J_mol_K == 50
    card = database.property_input(
        "7732-18-5",
        ComponentSpec("water", "H2O"),
        phase="gas",
        method="WEBBOOK_SHOMATE",
        reference=reference,
        reference_note="Authored synthetic source values",
    )
    restored = PropertyInputCard.from_json(json.dumps(card.to_dict()))
    thermo = CorrelationSpeciesThermo("water", restored.package.correlations[0], restored.reference)
    state = thermo.evaluate(500)
    assert state.enthalpy_J_mol == pytest.approx(-2944.5)
    assert state.entropy_J_mol_K == pytest.approx(50 + 30 * log(400 / 298.15) + 40 * log(500 / 400))
    assert (
        restored.evaluate_state(
            temperature_K=500, pressure_Pa=100000, phase="gas", quantities=("enthalpy",)
        )
        .properties["enthalpy"]
        .value
        == state.enthalpy_J_mol
    )


def test_imported_source_curve_drives_native_heat_wait_and_snapshot(tmp_path):
    from chemworld.physchem.property_packages import (
        PropertyPackageThermodynamics,
        SpeciesCaloricModel,
    )
    from chemworld.physchem.property_reports import R_J_PER_MOL_K, CaloricReference
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.surface_world import surface_world_spec

    database = CaloricDatabase(tmp_path)
    write_source(
        tmp_path,
        database.source("WEBBOOK_SHOMATE"),
        json.dumps(
            {
                "synthetic": [
                    None,
                    None,
                    [[200, 400, 30, 0, 0, 0, 0], [400, 800, 40, 0, 0, 0, 0]],
                ]
            }
        ),
    )
    curve = database.heat_capacity("synthetic", method="WEBBOOK_SHOMATE", phase="gas")
    world = surface_world_spec(include_surface=False)
    thermal = world.physics.models_of("thermal")["reactor"]
    thermo = PropertyPackageThermodynamics(
        {
            key: SpeciesCaloricModel(
                curve,
                CaloricReference(
                    phase="gas",
                    temperature_K=350,
                    energy_J_mol=value.u_ref_J_mol + R_J_PER_MOL_K * 350,
                ),
            )
            for key, value in thermal.thermodynamics.species.items()
        },
        20,
        350,
    )
    models = {key: dict(value) for key, value in world.physics.models.items()}
    models["thermal"]["reactor"] = replace(thermal, thermodynamics=thermo)
    runtime = ChemWorldRuntime(
        composition=replace(world, physics=replace(world.physics, models=models))
    )
    restored = None
    try:
        runtime.reset(seed=23)
        result = runtime.execute(
            {
                "operation": "heat",
                "target_temperature_K": 400,
                "duration_s": 2,
                "stirring_speed_rpm": 600,
            }
        )
        assert result.info["transaction_status"] == "committed", result.info
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        for session in (runtime, restored):
            assert (
                session.execute({"operation": "wait", "duration_s": 2}).info["transaction_status"]
                == "committed"
            )
        assert runtime._state.temperature_K > 350
        assert runtime.snapshot().to_dict() == restored.snapshot().to_dict()
    finally:
        runtime.close()
        if restored is not None:
            restored.close()
