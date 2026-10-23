import json

import pytest

from chemworld.physchem import PhasePropertyModel
from test_helmholtz_property_consumers import fixtures
from test_phase_evaluation import packages, references


def test_pure_package_phase_entry_retains_caloric_reference_and_source_snapshot():
    package = packages()[0]
    reference = references()["A"]
    model = package.phase_model(phase="gas", reference=reference)
    restored = PhasePropertyModel.from_dict(json.loads(json.dumps(model.to_dict())))
    quantities = ("cp", "enthalpy", "internal_energy", "entropy", "molar_volume")
    result = restored.evaluate({"alias-A": 3}, temperature_K=340,
                                pressure_Pa=2e5, quantities=quantities)
    direct = package.evaluate_state(phase="gas", reference=reference, temperature_K=340,
                                     pressure_Pa=2e5, quantities=quantities)
    for quantity in quantities:
        assert result.value(quantity) == direct.properties[quantity].value
    assert result.report["component_amounts_mol"] == {"A": 3}
    assert result.value("enthalpy")-result.value("internal_energy") == pytest.approx(
        2e5*result.value("molar_volume"))


def test_pure_helmholtz_package_entry_retains_existing_eos_reference():
    package, helmholtz = fixtures()
    model = package.phase_model(phase="gas", helmholtz_model=helmholtz)
    args = {"temperature_K": 360, "pressure_Pa": 2e5, "quantities": ("enthalpy", "cp")}
    snapshot = model.evaluate({package.component.identifier: 2}, **args)
    direct = package.evaluate_state(phase="gas", helmholtz_model=helmholtz, **args)
    assert snapshot.value("enthalpy") == direct.properties["enthalpy"].value
    assert snapshot.value("cp") == direct.properties["cp"].value
