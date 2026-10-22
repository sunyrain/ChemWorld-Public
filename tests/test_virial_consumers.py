import json
from dataclasses import replace

import pytest

from chemworld.data.constants import ConstantDatabase
from chemworld.data.volume import VolumeDatabase
from chemworld.physchem import (
    ComponentPropertyPackage,
    ComponentSpec,
    PropertyCorrelation,
    VirialGasThermodynamics,
    evaluate_virial_gas_state,
    virial_gas_batch_system,
    virial_gas_molar_volume_report,
    virial_temperature_report,
)
from chemworld.physchem.coupled_system import (
    CoupledProcess,
    CoupledState,
    HeatFlux,
    ProcessContribution,
)
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.property_packages import SpeciesCaloricModel
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.solver_backend import ODESolverPolicy


def thermal_model():
    return SpeciesCaloricModel(
        PropertyCorrelation(
            correlation_id="declared-cv",
            property_id="isochoric_heat_capacity",
            equation_id="cp_polynomial",
            coefficients={"a": 25.0},
            input_units={"temperature": "K"},
            output_unit="J/(mol*K)",
            validity_ranges={"temperature": (200, 800)},
            source_note="synthetic ideal heat capacity",
        ),
        CaloricReference(
            phase="gas",
            temperature_K=300,
            pressure_Pa=1e5,
            energy_basis="internal_energy",
            entropy_J_mol_K=100,
            source_note="synthetic ideal gas reference",
        ),
    )


@pytest.fixture
def crc_card(tmp_path):
    db = VolumeDatabase(tmp_path)
    path = tmp_path / db.source("CRC_VIRIAL").relative_path
    path.parent.mkdir(parents=True)
    path.write_text(
        "CAS\tChemical\ta1\ta2\ta3\ta4\ta5\n74-82-8\tmethane\t-42\t-123\t-20\t0\t0\n",
        encoding="utf-8",
    )
    return db.property_input(
        "74-82-8",
        ComponentSpec("methane", "CH4", cas_number="74-82-8"),
        method="CRC_VIRIAL",
        temperature_bounds_K=(200, 800),
    )


def test_crc_card_actual_state_json_and_legacy_root_converge(crc_card):
    card = PropertyInputCard.from_json(json.dumps(crc_card.to_dict()))
    report = card.evaluate_state(
        temperature_K=400,
        pressure_Pa=1e5,
        phase="gas",
        quantities=("molar_volume", "density", "dV_dT_P"),
        gas_model="virial",
    )
    (curve,) = card.package.correlations
    b = virial_temperature_report(curve, temperature_K=400)
    legacy = virial_gas_molar_volume_report(
        temperature_K=400, pressure_Pa=1e5, second_virial_m3_mol=b[0].value
    )
    assert legacy.molar_volume_m3_mol == pytest.approx(report.properties["molar_volume"].value)
    assert report.model_details["root"]["candidates"]
    assert legacy.to_dict()["gas_root"]["candidates"]
    assert len(b) == 4
    assert json.loads(json.dumps(report.to_dict()))["valid"]


def test_sourced_constants_use_one_cache_and_preserve_missing(tmp_path):
    db = ConstantDatabase(tmp_path)
    path = tmp_path / db.source("HEOS").relative_path
    path.parent.mkdir(parents=True)
    path.write_text(
        "CAS\tname\tTc\tPc\tVc\tomega\n"
        "74-82-8\tmethane\t190.564\t4599200\t0.00009863\t0.01142\n"
        "74-84-0\tethane\t305.3\t4872000\t\t0.0995\n",
        encoding="utf-8",
    )
    comp = ComponentSpec("methane", "CH4", cas_number="74-82-8")
    card = db.virial_input(
        "74-82-8", comp, method="Liu_Xiang", constant_source="HEOS", temperature_range_K=(250, 800)
    )
    restored = PropertyInputCard.from_json(json.dumps(card.to_dict()))
    assert restored.package.correlations[0].coefficients["Tc"] == 190.564
    assert set(restored.package.correlations[0].metadata["parameter_sources"]) == {
        "Tc",
        "Pc",
        "Vc",
        "omega",
    }
    assert db.loaded_sources == ("HEOS",)
    with pytest.raises(ValueError):
        db.virial_input(
            "74-84-0",
            ComponentSpec("ethane", "C2H6", cas_number="74-84-0"),
            method="Liu_Xiang",
            constant_source="HEOS",
            temperature_range_K=(250, 800),
        )
    with pytest.raises(ValueError, match="override"):
        db.virial_input(
            "74-82-8",
            comp,
            method="Abbott",
            constant_source="HEOS",
            temperature_range_K=(250, 800),
            parameters={"Tc": 200},
        )
    with pytest.raises(ValueError, match="identity"):
        db.virial_input(
            "74-84-0", comp, method="Abbott", constant_source="HEOS", temperature_range_K=(250, 800)
        )


def test_nonideal_potential_energy_and_heat_capacity_are_consistent(crc_card):
    ideal = thermal_model()
    package = ComponentPropertyPackage(
        crc_card.package.component, (*crc_card.package.correlations, ideal.heat_capacity)
    )
    model = VirialGasThermodynamics(package, ideal)
    state = CoupledState({"methane": 2.0}, 400, 10)
    delta = 0.001
    derivative = (
        model.internal_energy_j(replace(state, temperature_K=400 + delta))
        - model.internal_energy_j(replace(state, temperature_K=400 - delta))
    ) / (2 * delta)
    assert model.heat_capacity_j_k(state) == pytest.approx(derivative, rel=1e-8)
    p = model.pressure_pa(state)
    report = package.evaluate_state(
        temperature_K=400,
        pressure_Pa=p,
        phase="gas",
        quantities=("enthalpy", "internal_energy", "cv", "cp", "entropy", "molar_volume"),
        reference=ideal.reference,
        gas_model="virial",
    )
    q = {key: val.value for key, val in report.properties.items()}
    assert q["enthalpy"] - q["internal_energy"] == pytest.approx(p * q["molar_volume"])
    assert q["internal_energy"] * 2 == pytest.approx(model.internal_energy_j(state))
    assert q["cv"] * 2 == pytest.approx(model.heat_capacity_j_k(state))
    assert q["cp"] > q["cv"]


def test_common_finite_heat_clock_balance_resume_and_json(crc_card):
    model = VirialGasThermodynamics(crc_card.package, thermal_model())
    clone = VirialGasThermodynamics.from_dict(json.loads(json.dumps(model.to_dict())))
    policy = ODESolverPolicy(
        solver_id="virial-development", method="DOP853", rtol=1e-10, atol=1e-12
    )
    system = virial_gas_batch_system(model, heat_flux=lambda _: HeatFlux(heater_W=5), policy=policy)
    state = CoupledState({"methane": 2}, 300, 10)
    full = system.advance(state, 60)
    half = system.advance(state, 30)
    resumed = virial_gas_batch_system(
        clone, heat_flux=lambda _: HeatFlux(heater_W=5), policy=policy
    ).advance(half.final_state, 30)
    assert full.final_state.temperature_K == pytest.approx(
        resumed.final_state.temperature_K, abs=2e-6
    )
    assert full.final_state.time_s == resumed.final_state.time_s == 60
    assert dict(full.final_state.amounts_mol) == {"methane": 2}
    assert full.final_state.boundary_heat_J == pytest.approx(300, abs=1e-7)
    assert abs(full.energy_balance_residual_J) <= 1e-4


@pytest.mark.parametrize(
    "case",
    [
        "missing_cv",
        "missing_reference",
        "wrong_phase",
        "out_of_range",
        "unknown_range",
        "missing_coefficient",
        "duplicate_B",
        "flow",
    ],
)
def test_missing_physics_never_falls_back_to_ideal(crc_card, case):
    package, ideal = crc_card.package, thermal_model()
    (curve,) = package.correlations
    with pytest.raises(ValueError):
        if case == "missing_cv":
            package.evaluate_state(
                temperature_K=300,
                pressure_Pa=1e5,
                phase="gas",
                quantities=("cv",),
                reference=ideal.reference,
                gas_model="virial",
            )
        elif case == "missing_reference":
            replace(package, correlations=(curve, ideal.heat_capacity)).evaluate_state(
                temperature_K=300,
                pressure_Pa=1e5,
                phase="gas",
                quantities=("cv",),
                gas_model="virial",
            )
        elif case == "wrong_phase":
            VirialGasThermodynamics(
                package,
                SpeciesCaloricModel(ideal.heat_capacity, replace(ideal.reference, phase="liquid")),
            )
        elif case == "out_of_range":
            evaluate_virial_gas_state(
                package, temperature_K=900, pressure_Pa=1e5, quantities=("density",)
            )
        elif case in {"unknown_range", "missing_coefficient", "duplicate_B"}:
            if case == "unknown_range":
                curves = (replace(curve, validity_ranges={}),)
            elif case == "missing_coefficient":
                curves = (replace(curve, coefficients={"a1": 0}),)
            else:
                curves = (curve, replace(curve, correlation_id="other"))
            evaluate_virial_gas_state(
                replace(package, correlations=curves),
                temperature_K=300,
                pressure_Pa=1e5,
                quantities=("density",),
            )
        else:
            system = virial_gas_batch_system(VirialGasThermodynamics(package, ideal))
            system = replace(
                system,
                processes=(
                    CoupledProcess(
                        "flow", lambda _: ProcessContribution(amount_rates_mol_s={"methane": 0.01})
                    ),
                ),
            )
            system.advance(CoupledState({"methane": 2}, 300, 10), 1)
