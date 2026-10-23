"""New alpha models preserve old component payloads and real EOS consumption."""

import json
from dataclasses import replace

import pytest

from chemworld.data.phase_change import PhaseChangeRecord
from chemworld.data.twu_eos import twu_eos_component_from_record
from chemworld.physchem.cubic_alpha import cubic_alpha_response
from chemworld.physchem.eos import (
    CubicEOSSpec,
    EOSComponentSpec,
    cubic_pure_parameters,
    evaluate_cubic_eos,
)
from chemworld.physchem.specs import ComponentSpec


def record():
    names = ("Tc", "Pc", "c0", "c1", "c2", "name", "inchikey")
    return PhaseChangeRecord(
        "BELL_TWU",
        1,
        "CSCPPACGZOOCGX-UHFFFAOYSA-N",
        None,
        names,
        names,
        (508.1, 4700000.0, 0.25, 0.8, 2.0, "acetone", "CSCPPACGZOOCGX-UHFFFAOYSA-N"),
    )


def test_twu_source_components_roundtrip_and_cubic_consumer():
    component = ComponentSpec("acetone", "C3H6O", inchi_key="CSCPPACGZOOCGX-UHFFFAOYSA-N")
    pure = twu_eos_component_from_record(record(), component=component)
    spec = CubicEOSSpec(
        "twu-mix", "peng_robinson", (pure, EOSComponentSpec("co2", 304.2, 7.38e6, 0.225))
    )
    restored = CubicEOSSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored.to_dict() == spec.to_dict()
    result = evaluate_cubic_eos(
        restored, {"acetone": 0.4, "co2": 0.6}, temperature_K=400, pressure_Pa=2e5
    )
    assert result.molar_volume_m3_mol > 0
    assert result.fugacity_coefficients["acetone"] > 0
    original = replace(pure, alpha_model="soave", alpha_coefficients=())
    assert (
        cubic_pure_parameters(original, model="peng_robinson", temperature_K=400).a_alpha
        != cubic_pure_parameters(pure, model="peng_robinson", temperature_K=400).a_alpha
    )
    response = cubic_alpha_response(
        temperature_K=508.1,
        critical_temperature_K=508.1,
        model="twu91",
        coefficients=(0.25, 0.8, 2.0),
    )
    assert response.alpha == pytest.approx(1.0)
    assert response.d_alpha_dT == pytest.approx((2 * (0.8 - 1) - 0.25 * 0.8 * 2) / 508.1)
    assert pure.alpha_provenance["parameters"]["c0"]["unit"] == "dimensionless"


def test_default_soave_payload_stays_compatible_and_twu_is_explicit():
    component = EOSComponentSpec("A", 400, 4e6, 0.2)
    assert set(component.to_dict()) == {
        "component_id",
        "critical_temperature_K",
        "critical_pressure_Pa",
        "acentric_factor",
    }
    for coefficients in ((), (0.2, 0.3), (0.2, 0.3, float("nan"))):
        with pytest.raises(ValueError):
            replace(component, alpha_model="twu91", alpha_coefficients=coefficients)
    with pytest.raises(ValueError):
        replace(component, alpha_coefficients=(0.2, 0.3, 0.4))
    with pytest.raises(ValueError, match="does not match"):
        twu_eos_component_from_record(record(), component=ComponentSpec("water", "H2O"))


@pytest.mark.parametrize("T", [0, -1, float("nan")])
def test_invalid_temperature_is_rejected(T):
    with pytest.raises(ValueError):
        cubic_alpha_response(
            temperature_K=T, critical_temperature_K=400, model="twu91", coefficients=(0.2, 0.8, 2.0)
        )
