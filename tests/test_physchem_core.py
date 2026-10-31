from __future__ import annotations

import pytest

from chemworld.foundation import convert_value
from chemworld.physchem import (
    ComponentConflictPolicy,
    ComponentFieldCandidate,
    ComponentProvenance,
    ComponentSpec,
    ComponentUncertainty,
    MixtureSpec,
    PropertyCorrelation,
    component_alias_index,
    element_matrix,
    hill_formula,
    mass_fractions_from_mole_fractions,
    mole_fractions_from_mass_fractions,
    molecular_weight,
    parse_formula,
    property_equation_contracts,
    resolve_component_field_conflict,
    resolve_component_identifier,
    supported_property_equations,
)


def test_formula_parser_and_element_matrix() -> None:
    water = parse_formula("H2O")
    ethanol = parse_formula("C2H6O")
    carbon_dioxide = parse_formula("CO2")
    calcium_hydroxide = parse_formula("Ca(OH)2")

    assert water == {"H": 2.0, "O": 1.0}
    assert ethanol == {"C": 2.0, "H": 6.0, "O": 1.0}
    assert carbon_dioxide == {"C": 1.0, "O": 2.0}
    assert calcium_hydroxide == {"Ca": 1.0, "O": 2.0, "H": 2.0}
    assert hill_formula(ethanol) == "C2H6O"

    matrix, elements = element_matrix([water, ethanol, carbon_dioxide])
    assert elements == ("H", "C", "O")
    assert matrix == ((2.0, 0.0, 1.0), (6.0, 2.0, 1.0), (0.0, 1.0, 2.0))


def test_component_spec_is_json_friendly_and_validated() -> None:
    component = ComponentSpec(
        identifier="ethanol",
        formula="C2H6O",
        cas_number="64-17-5",
        default_phase="liquid",
        safety_tags=("flammable", "volatile"),
        allowed_property_correlations=("antoine", "liquid_density"),
    )

    payload = component.to_dict()
    assert payload["identifier"] == "ethanol"
    assert payload["cas_number"] == "64-17-5"
    assert payload["composition"] == {"C": 2.0, "H": 6.0, "O": 1.0}
    assert payload["hill_formula"] == "C2H6O"
    assert payload["units"] == {"molecular_weight_g_mol": "g/mol"}
    assert component.molecular_weight_g_mol == pytest.approx(46.069)
    assert ComponentSpec.from_dict(payload).to_dict() == payload

    with pytest.raises(ValueError, match="Unknown element"):
        ComponentSpec(identifier="bad", formula="Xx2")
    with pytest.raises(ValueError, match="Unsupported default phase"):
        ComponentSpec(identifier="bad_phase", formula="H2O", default_phase="plasma")
    with pytest.raises(ValueError, match="inconsistent with formula"):
        ComponentSpec(identifier="bad_mw", formula="H2O", molecular_weight_g_mol=99.0)
    with pytest.raises(ValueError, match="cannot contain"):
        ComponentSpec(identifier="bad id", formula="H2O")
    with pytest.raises(ValueError, match="must be unique"):
        ComponentSpec(identifier="dup_tags", formula="H2O", safety_tags=("safe", "safe"))
    with pytest.raises(ValueError, match="checksum"):
        ComponentSpec(identifier="bad_cas", formula="H2O", cas_number="7732-18-4")


def test_component_provenance_and_uncertainty_round_trip() -> None:
    component = ComponentSpec(
        identifier="water",
        formula="H2O",
        default_phase="liquid",
        aliases=("dihydrogen_oxide",),
        provenance=(
            ComponentProvenance(
                source_id="chemicals:Psat_data_Perrys2_8",
                source_name="chemicals",
                source_table="Psat_data_Perrys2_8",
                source_key="7732-18-5",
                source_path="reference_repos/chemicals/chemicals/vapor_pressure.py",
                notes=("identifier and CASRN checked in curated registry",),
            ),
        ),
        uncertainty=(
            ComponentUncertainty(
                field_id="molecular_weight_g_mol",
                unit="g/mol",
                relative_uncertainty=0.02,
                coverage="schema_consistency_tolerance",
            ),
        ),
    )

    payload = component.to_dict()
    assert payload["provenance"][0]["source_name"] == "chemicals"
    assert payload["uncertainty"][0]["field_id"] == "molecular_weight_g_mol"
    assert ComponentSpec.from_dict(payload).to_dict() == payload

    with pytest.raises(ValueError, match="relative_uncertainty"):
        ComponentUncertainty(field_id="bad", relative_uncertainty=-0.1)


def test_component_alias_index_rejects_registry_conflicts() -> None:
    water = ComponentSpec(
        identifier="water",
        formula="H2O",
        cas_number="7732-18-5",
        default_phase="liquid",
        aliases=("dihydrogen oxide",),
    )
    ethanol = ComponentSpec(
        identifier="ethanol",
        formula="C2H6O",
        default_phase="liquid",
        aliases=("ethyl_alcohol",),
    )

    index = component_alias_index((water, ethanol))
    assert index["dihydrogen_oxide"] == "water"
    assert index["7732-18-5"] == "water"
    assert resolve_component_identifier((water, ethanol), "7732185") == "water"
    assert resolve_component_identifier((water, ethanol), "ethyl alcohol") == "ethanol"

    conflicting = ComponentSpec(
        identifier="conflicting_water",
        formula="H2O",
        cas_number="7732-18-5",
        default_phase="liquid",
    )
    with pytest.raises(ValueError, match="alias conflict"):
        component_alias_index((water, conflicting))

    with pytest.raises(KeyError, match="unknown component"):
        resolve_component_identifier((water, ethanol), "acetone")


def test_component_field_conflict_policy_is_auditable() -> None:
    candidates = (
        ComponentFieldCandidate(
            field_id="normal_boiling_point_K",
            value=373.124,
            source_id="chemicals",
        ),
        ComponentFieldCandidate(
            field_id="normal_boiling_point_K",
            value=373.125,
            source_id="thermo",
        ),
    )
    policy = ComponentConflictPolicy(
        mode="warn",
        source_priority=("thermo", "chemicals"),
        field_atol={"normal_boiling_point_K": 0.01},
    )
    resolution = resolve_component_field_conflict(
        "normal_boiling_point_K",
        candidates,
        policy,
    )

    assert resolution.status == "consistent"
    assert resolution.resolved_source_id == "thermo"
    assert resolution.to_dict()["candidates"][0]["source_id"] == "thermo"
    assert ComponentConflictPolicy.from_dict(policy.to_dict()).to_dict() == policy.to_dict()
    assert ComponentFieldCandidate.from_dict(candidates[0].to_dict()).to_dict() == (
        candidates[0].to_dict()
    )

    warning = resolve_component_field_conflict(
        "normal_boiling_point_K",
        candidates,
        ComponentConflictPolicy(
            mode="warn",
            source_priority=("thermo", "chemicals"),
            field_atol={"normal_boiling_point_K": 1e-6},
        ),
    )
    assert warning.status == "conflict_warning"
    assert "selected" in warning.message

    with pytest.raises(ValueError, match="candidates conflict"):
        resolve_component_field_conflict(
            "normal_boiling_point_K",
            candidates,
            ComponentConflictPolicy(mode="raise", source_priority=("thermo",)),
        )


def test_mole_and_mass_fraction_conversions_are_reversible() -> None:
    water = ComponentSpec(identifier="water", formula="H2O", default_phase="liquid")
    ethanol = ComponentSpec(identifier="ethanol", formula="C2H6O", default_phase="liquid")
    components = (water, ethanol)

    zs = (0.25, 0.75)
    ws = mass_fractions_from_mole_fractions(components, zs)
    recovered_zs = mole_fractions_from_mass_fractions(components, ws)

    assert sum(ws) == pytest.approx(1.0)
    assert recovered_zs == pytest.approx(zs)

    mixture = MixtureSpec.from_mole_fractions(
        components,
        zs,
        phase_label="liquid",
        temperature_K=298.15,
        pressure_Pa=101325.0,
    )
    assert mixture.mass_fractions == pytest.approx(ws)
    assert mixture.average_molecular_weight_g_mol == pytest.approx(
        0.25 * molecular_weight(water.composition)
        + 0.75 * molecular_weight(ethanol.composition)
    )
    assert mixture.to_dict()["units"]["pressure_Pa"] == "Pa"
    assert MixtureSpec.from_dict(mixture.to_dict()).to_dict() == mixture.to_dict()

    with pytest.raises(ValueError, match="component_ids must be unique"):
        MixtureSpec(
            component_ids=("water", "water"),
            molecular_weights_g_mol=(18.015, 18.015),
            mole_fractions=(0.5, 0.5),
            mass_fractions=(0.5, 0.5),
            phase_label="liquid",
            temperature_K=298.15,
            pressure_Pa=101325.0,
        )
    with pytest.raises(ValueError, match="not compatible"):
        MixtureSpec.from_mole_fractions(
            (water, ComponentSpec(identifier="co2", formula="CO2", default_phase="gas")),
            (0.5, 0.5),
            phase_label="liquid",
            temperature_K=298.15,
            pressure_Pa=101325.0,
        )


def test_property_correlation_units_fail_before_kernel_use() -> None:
    correlation = PropertyCorrelation(
        correlation_id="water_cp_poly",
        equation_id="cp_polynomial",
        coefficients={"a": 75.3, "b": 0.0},
        input_units={"temperature": "K"},
        output_unit="J/(mol*K)",
        validity_ranges={"temperature": (273.15, 373.15)},
        source_note="benchmark placeholder correlation",
        metadata={"source_quality": "audit"},
    )
    assert correlation.to_dict()["output_unit"] == "J/(mol*K)"
    assert PropertyCorrelation.from_dict(correlation.to_dict()).to_dict() == correlation.to_dict()
    assert correlation.model_card()["output_dimension"] == "molar_heat_capacity"

    with pytest.raises(ValueError, match="Unsupported unit"):
        PropertyCorrelation(
            correlation_id="bad",
            equation_id="cp_polynomial",
            coefficients={"a": 1.0},
            input_units={"temperature": "rankine"},
            output_unit="J/(mol*K)",
        )
    with pytest.raises(ValueError, match="Invalid validity range"):
        PropertyCorrelation(
            correlation_id="bad_range",
            equation_id="cp_polynomial",
            coefficients={"a": 1.0},
            input_units={"temperature": "K"},
            output_unit="J/(mol*K)",
            validity_ranges={"temperature": (400.0, 300.0)},
        )
    with pytest.raises(ValueError, match="Unsupported property equation_id"):
        PropertyCorrelation(
            correlation_id="bad_equation",
            equation_id="bad",
            coefficients={"a": 1.0},
            input_units={"temperature": "K"},
            output_unit="J/mol",
        )
    with pytest.raises(ValueError, match="Missing coefficients"):
        PropertyCorrelation(
            correlation_id="missing_coeff",
            equation_id="antoine",
            coefficients={"A": 1.0},
            input_units={"temperature": "K"},
            output_unit="Pa",
        )
    with pytest.raises(ValueError, match="output_unit"):
        PropertyCorrelation(
            correlation_id="bad_dimension",
            equation_id="antoine",
            coefficients={"A": 1.0, "B": 1.0, "C": 1.0},
            input_units={"temperature": "K"},
            output_unit="J/mol",
        )


def test_property_equation_contracts_are_public_and_json_friendly() -> None:
    equations = supported_property_equations()
    contracts = property_equation_contracts()

    assert "antoine" in equations
    assert "sutherland_gas_viscosity" in equations
    assert "linear_thermal_conductivity" in equations
    assert set(equations) == set(contracts)
    assert contracts["antoine"] == {
        "required_coefficients": ["A", "B", "C"],
        "input_dimensions": {"temperature": "temperature"},
        "output_dimension": "pressure",
    }
    assert contracts["cp_polynomial"]["output_dimension"] == "molar_heat_capacity"


def test_extended_units_cover_physchem_dimensions() -> None:
    assert convert_value(1.0, "kW", "W") == 1000.0
    assert convert_value(1.0, "kJ/mol", "J/mol") == 1000.0
    assert convert_value(1.0, "g/mL", "kg/m^3") == 1000.0
    assert convert_value(1.0, "mPa*s", "Pa*s") == 0.001
    assert convert_value(1.0, "mW/(m*K)", "W/(m*K)") == 0.001
    assert convert_value(1.0, "cm^2/s", "m^2/s") == 0.0001
