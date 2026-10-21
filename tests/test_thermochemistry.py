from __future__ import annotations

import math

import pytest

from chemworld.physchem import (
    R_J_PER_MOL_K,
    NASA7SpeciesThermo,
    NASA7TemperatureSegment,
    equilibrium_constant_from_delta_g,
    reaction_thermochemistry,
    thermochemistry_model_cards,
    validate_model_card,
)


def _constant_cp_species(
    species_id: str,
    *,
    a5: float = 0.0,
    a6: float = 0.0,
) -> NASA7SpeciesThermo:
    return NASA7SpeciesThermo(
        species_id=species_id,
        segments=(
            NASA7TemperatureSegment(
                min_temperature_K=200.0,
                max_temperature_K=2000.0,
                coefficients=(3.5, 0.0, 0.0, 0.0, 0.0, a5, a6),
                label=f"{species_id}:constant-cp",
            ),
        ),
    )


def test_nasa7_constant_cp_identities() -> None:
    species = _constant_cp_species("A", a5=-1200.0, a6=2.0)
    temperature = 500.0
    state = species.evaluate(temperature)
    assert state.cp_over_R == pytest.approx(3.5)
    assert state.cp_J_mol_K == pytest.approx(3.5 * R_J_PER_MOL_K)
    assert state.h_over_RT == pytest.approx(3.5 - 1200.0 / temperature)
    assert state.enthalpy_J_mol == pytest.approx(R_J_PER_MOL_K * (3.5 * temperature - 1200.0))
    assert state.s_over_R == pytest.approx(3.5 * math.log(temperature) + 2.0)
    assert state.gibbs_J_mol == pytest.approx(
        state.enthalpy_J_mol - temperature * state.entropy_J_mol_K
    )


def test_cantera_style_nasa7_yaml_node_parses_segments() -> None:
    thermo = {
        "model": "NASA7",
        "temperature-ranges": [200.0, 1000.0, 6000.0],
        "data": [
            [2.5, 0.0, 0.0, 0.0, 0.0, -745.375, 4.37967491],
            [2.5, 0.0, 0.0, 0.0, 0.0, -745.375, 4.37967491],
        ],
        "note": "Cantera argon example shape",
    }
    species = NASA7SpeciesThermo.from_cantera_yaml_thermo(
        species_id="AR",
        thermo=thermo,
        composition={"Ar": 1},
    )
    low = species.evaluate(300.0)
    high = species.evaluate(3000.0)
    assert low.cp_over_R == pytest.approx(2.5)
    assert high.cp_over_R == pytest.approx(2.5)
    assert species.to_dict()["composition"] == {"Ar": 1.0}
    assert species.continuity_report()["passed"]


def test_reaction_thermochemistry_and_equilibrium_constant_from_species_gibbs() -> None:
    temperature = 600.0
    delta_h_target = -42_000.0
    a5_shift = delta_h_target / R_J_PER_MOL_K
    species = {
        "A": _constant_cp_species("A"),
        "B": _constant_cp_species("B", a5=a5_shift),
    }
    result = reaction_thermochemistry(
        reaction_id="A_to_B",
        stoichiometry={"A": -1.0, "B": 1.0},
        species_thermo=species,
        temperature_K=temperature,
    )
    assert result.delta_h_J_mol == pytest.approx(delta_h_target)
    assert result.delta_s_J_mol_K == pytest.approx(0.0)
    assert result.delta_g_J_mol == pytest.approx(delta_h_target)
    assert result.equilibrium_constant == pytest.approx(
        math.exp(-delta_h_target / (R_J_PER_MOL_K * temperature))
    )
    assert equilibrium_constant_from_delta_g(
        delta_g_J_mol=result.delta_g_J_mol,
        temperature_K=temperature,
    ) == pytest.approx(result.equilibrium_constant)


def test_nasa7_validation_and_missing_species_fail_fast() -> None:
    with pytest.raises(ValueError, match="exactly seven"):
        NASA7TemperatureSegment(
            min_temperature_K=200.0,
            max_temperature_K=1000.0,
            coefficients=(1.0, 2.0, 3.0),  # type: ignore[arg-type]
        )
    species = _constant_cp_species("A")
    with pytest.raises(ValueError, match="no NASA7 segment"):
        species.evaluate(100.0)
    with pytest.raises(KeyError, match="missing"):
        reaction_thermochemistry(
            reaction_id="bad",
            stoichiometry={"A": -1.0, "B": 1.0},
            species_thermo={"A": species},
            temperature_K=300.0,
        )


def test_nasa7_continuity_report_flags_gaps_and_jumps() -> None:
    continuous = NASA7SpeciesThermo(
        species_id="A",
        segments=(
            NASA7TemperatureSegment(200.0, 1000.0, (3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
            NASA7TemperatureSegment(1000.0, 2000.0, (4.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
        ),
    )
    assert not continuous.continuity_report(relative_tolerance=1.0e-4)["passed"]

    with_gap = NASA7SpeciesThermo(
        species_id="B",
        segments=(
            NASA7TemperatureSegment(200.0, 900.0, (3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
            NASA7TemperatureSegment(1000.0, 2000.0, (3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
        ),
    )
    report = with_gap.continuity_report()
    assert not report["passed"]
    assert report["checks"][0]["reason"] == "temperature_gap"


def test_thermochemistry_model_card_is_valid() -> None:
    cards = thermochemistry_model_cards()
    assert [card.model_id for card in cards] == [
        "nasa7_species_reaction_thermochemistry_v1", "native_caloric_data_formation_stoichiometry",
    ]
    for card in cards:
        assert validate_model_card(card) == []
    assert any(
        evidence.evidence_id == "nasa7-detailed-balance-reaction-network-tests"
        for evidence in cards[0].validation_evidence
    )
