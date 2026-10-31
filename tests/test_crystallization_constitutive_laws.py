"""Product behavior retained from the retired study's mixed tests."""

import pytest

from chemworld.physchem.crystallization_units import CrystallizationKineticsSpec, _occlude_impurity
from chemworld.world.scenario import DefaultScenarioGenerator, get_scenario


def _kinetics(law_id: str) -> CrystallizationKineticsSpec:
    return CrystallizationKineticsSpec(
        model_id="test",
        primary_nucleation_coefficient_per_L_s=1.0,
        primary_nucleation_exponent=2.0,
        growth_coefficient_m_s=1.0e-8,
        growth_exponent=1.0,
        crystal_density_kg_m3=1200.0,
        target_molecular_weight_kg_mol=0.18,
        nucleus_diameter_m=8.0e-6,
        impurity_occlusion_mol_per_mol=0.02,
        supersaturation_occlusion_factor=0.5,
        impurity_occlusion_law_id=law_id,
        impurity_surface_half_saturation_mol_L=0.01,
        provenance_id="test",
    )


def test_structural_occlusion_laws_are_functionally_distinct_not_scalar() -> None:
    parent = _kinetics("linear_supersaturation_transfer_v1")
    child = _kinetics("surface_saturation_occlusion_v1")

    def ratio(impurity: float, supersaturation: float) -> float:
        parent_value = _occlude_impurity(
            0.001,
            relative_supersaturation=supersaturation,
            dissolved_impurity=impurity,
            solvent_volume_L=0.05,
            kinetics=parent,
        )
        child_value = _occlude_impurity(
            0.001,
            relative_supersaturation=supersaturation,
            dissolved_impurity=impurity,
            solvent_volume_L=0.05,
            kinetics=child,
        )
        return child_value / parent_value

    assert ratio(0.001, 0.2) != pytest.approx(ratio(0.004, 0.2))
    assert ratio(0.004, 0.2) != pytest.approx(ratio(0.004, 1.0))


def test_occlusion_axis_switches_executable_private_law_selector() -> None:
    scenario = get_scenario("reaction-to-crystallization")
    generator = DefaultScenarioGenerator()
    parent = generator.generate(scenario, 401)
    child = generator.generate(
        scenario,
        401,
        (
            {
                "axis_id": "crystallization.impurity-occlusion-law",
                "mode": "extrapolation",
                "severity": 1.0,
            },
        ),
    )
    assert "crystallization_impurity_occlusion_law_id" not in parent.initial_state.metadata
    assert (
        child.initial_state.metadata["crystallization_impurity_occlusion_law_id"]
        == "surface_saturation_occlusion_v1"
    )
    assert parent.parameters.world_id != child.parameters.world_id


@pytest.mark.parametrize(
    ("mode", "severity"),
    (("interpolation", 1.0), ("extrapolation", 0.5), ("composition", 1.0)),
)
def test_occlusion_law_axis_rejects_nonbinary_interventions(mode: str, severity: float) -> None:
    scenario = get_scenario("reaction-to-crystallization")
    with pytest.raises(ValueError, match=r"does not support|frozen binary axis"):
        DefaultScenarioGenerator().generate(
            scenario,
            401,
            (
                {
                    "axis_id": "crystallization.impurity-occlusion-law",
                    "mode": mode,
                    "severity": severity,
                },
            ),
        )


def test_scalar_null_axis_sets_only_constant_capacity_multiplier() -> None:
    scenario = get_scenario("reaction-to-crystallization")
    child = DefaultScenarioGenerator().generate(
        scenario,
        401,
        (
            {
                "axis_id": "crystallization.impurity-occlusion-capacity",
                "mode": "extrapolation",
                "severity": 0.6,
            },
        ),
    )
    assert child.initial_state.metadata[
        "crystallization_impurity_occlusion_capacity_multiplier"
    ] == pytest.approx(4.0)
    assert "crystallization_impurity_occlusion_law_id" not in child.initial_state.metadata


@pytest.mark.parametrize("severity", (0.25, 0.5, 0.999))
def test_mechanism_occlusion_law_path_rejects_nonbinary_severity(
    severity: float,
) -> None:
    scenario = get_scenario("reaction-to-crystallization")
    intervention = {
        "kind": "mechanism_family",
        "mode": "constitutive_law_family",
        "severity": severity,
        "constitutive_law_change": {
            "transform_id": "crystallization_occlusion_response_stress_v1",
            "occlusion_law_selector_at_full_severity": 2.0,
        },
    }
    with pytest.raises(ValueError, match="requires severity 1"):
        DefaultScenarioGenerator().generate(scenario, 401, (intervention,))
