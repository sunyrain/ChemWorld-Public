"""Finite-grid harmonic state counts, RRKM authoring and original material ownership."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.rrkm_density_states import harmonic_state_counts
from chemworld.physchem.rrkm_grain_binding import (
    RRKMGrainSpec,
    compile_rrkm_grains,
    rrkm_canonical_report,
    rrkm_grain_network,
    rrkm_microcanonical_report,
)

loader = importlib.util.spec_from_file_location(
    "rrkm38_cases", Path(__file__).parent / "reference/rrkm38_cases.py"
)
cases = importlib.util.module_from_spec(loader)
loader.loader.exec_module(cases)


def test_integer_counting_uses_degenerate_modes_as_distinct_oscillators():
    model = RRKMGrainSpec.from_dict(cases.count_card("two-mode"))
    result = harmonic_state_counts(model.well, model.grid)
    assert result["counts"][:7] == (1, 1, 2, 2, 3, 3, 4)
    twin = replace(model.well, vibrational_quanta_J_mol=(1000.0, 1000.0))
    assert harmonic_state_counts(twin, model.grid)["counts"][:7] == (1, 2, 3, 4, 5, 6, 7)


def test_zero_point_basis_is_explicit_and_added_exactly_once():
    model = RRKMGrainSpec.from_dict(cases.count_card("degeneracy-zpe"))
    corrected = replace(
        model.well,
        reference_energy_J_mol=model.well.ground_energy_J_mol,
        energy_basis="zero-point-corrected",
    )
    first = harmonic_state_counts(model.well, model.grid)
    second = harmonic_state_counts(corrected, model.grid)
    assert first["ground_energy_J_mol"] == second["ground_energy_J_mol"] == 0.0
    assert first["zero_point_added_J_mol"] == 3000.0
    assert second["zero_point_added_J_mol"] == 0.0
    assert first["counts"] == second["counts"]


def test_quantized_quanta_and_barrier_report_original_signed_errors():
    model = RRKMGrainSpec.from_dict(cases.count_card("nearest"))
    result = rrkm_microcanonical_report(model)
    assert [q["error_J_mol"] for q in result["well"]["quantization"]] == [-100.0, 200.0, 100.0]
    assert result["channels"]["to-P"]["barrier"]["error_J_mol"] == -200.0
    with pytest.raises(ValueError, match="quantization"):
        rrkm_microcanonical_report(
            replace(model, grid=replace(model.grid, maximum_quantization_error_J_mol=50.0))
        )


def test_sparse_bins_never_create_fake_species_or_dense_canonical_claim():
    model = RRKMGrainSpec.from_dict(cases.count_card("sparse"))
    micro = rrkm_microcanonical_report(model)
    grains = compile_rrkm_grains(model)
    assert micro["well"]["counts"][1] == 0
    assert micro["channels"]["to-P"]["rate_s"][1] is None
    assert "A:grain:1" not in {g.species_id for g in grains.wells[0].grains}
    canonical = rrkm_canonical_report(model, 600.0)
    assert not canonical["infinite_grid_formula_applicable"]
    assert canonical["channels"]["to-P"]["infinite_grid_average_s"] is None
    assert canonical["channels"]["to-P"]["finite_grid_average_s"] > 0


def test_path_and_electronic_degeneracy_have_distinct_normalization():
    model = RRKMGrainSpec.from_dict(cases.count_card("two-mode"))
    first = rrkm_microcanonical_report(model)["channels"]["to-P"]["rate_s"]
    channel = replace(model.channels[0], path_degeneracy=3)
    tripled = rrkm_microcanonical_report(replace(model, channels=(channel,)))["channels"]["to-P"][
        "rate_s"
    ]
    assert tripled == pytest.approx([3 * r for r in first])
    doubled = replace(model, well=replace(model.well, electronic_degeneracy=2))
    assert rrkm_microcanonical_report(doubled)["channels"]["to-P"]["rate_s"] == pytest.approx(
        [r / 2 for r in first]
    )


def test_molar_energy_to_rate_conversion_uses_planck_times_avogadro():
    model = RRKMGrainSpec.from_dict(cases.count_card("one-mode"))
    report = rrkm_microcanonical_report(model)
    assert report["channels"]["to-P"]["rate_s"][3] == 0.0
    assert report["channels"]["to-P"]["rate_s"][4] == pytest.approx(
        1000.0 / (6.62607015e-34 * 6.02214076e23), rel=1e-14
    )


def test_absent_accessible_ts_is_zero_rate_without_zero_rate_channels():
    model = RRKMGrainSpec.from_dict(cases.count_card("empty-ts"))
    grains = compile_rrkm_grains(model)
    assert grains.channels == ()
    assert grains.sinks[0].species_id == "P"
    assert rrkm_canonical_report(model, 600.0)["channels"]["to-P"]["finite_grid_average_s"] == 0.0


@pytest.mark.parametrize("name", cases.COUNT_CASES)
def test_declarations_and_compiled_network_roundtrip_exactly(name):
    model = RRKMGrainSpec.from_dict(cases.count_card(name))
    assert RRKMGrainSpec.from_dict(json.loads(json.dumps(model.to_dict()))) == model
    network = rrkm_grain_network(model)
    restored = ReactionNetworkSpec.from_dict(json.loads(json.dumps(network.to_dict())))
    assert restored.to_dict() == network.to_dict()
    assert network.metadata["rrkm_declaration"] == model.to_dict()
    assert model.well.state_id not in network.species_ids
    assert network.check_element_balance()


def test_original_grain_thermochemistry_uses_same_ground_and_state_entropy():
    model = RRKMGrainSpec.from_dict(cases.count_card("two-mode"))
    network = rrkm_grain_network(model)
    first = network.imported_gas_thermochemistry["A:grain:0"].evaluate(600.0)
    third = network.imported_gas_thermochemistry["A:grain:2"].evaluate(600.0)
    assert third.enthalpy_J_mol - first.enthalpy_J_mol == pytest.approx(2000.0)
    assert third.entropy_J_mol_K - first.entropy_J_mol_K == pytest.approx(
        cases.R * __import__("math").log(2)
    )
    assert first.cp_J_mol_K == third.cp_J_mol_K == 30.0


def test_canonical_cutoff_fraction_does_not_claim_missing_tail_is_zero():
    model = RRKMGrainSpec.from_dict(cases.base())
    result = rrkm_canonical_report(model, 600.0)
    assert 0 < result["well_partition"]["retained_partition_fraction"] < 1
    assert result["infinite_grid_formula_applicable"]
    assert result["dense_grid_prefactor_ratio_to_continuous_tst"] > 1


@pytest.mark.parametrize("kind", ("imaginary", "baseline", "source", "mode_count"))
def test_authoring_refuses_silent_missing_or_double_counted_inputs(kind):
    data = cases.base()
    if kind == "imaginary":
        data["channels"][0]["imaginary_frequency_cm_inverse"] = 0.0
    elif kind == "baseline":
        data["baseline"]["excluded_internal_modes"] = ["all NASA modes"]
    elif kind == "source":
        data["well"]["source"]["license"] = ""
    else:
        data["channels"][0]["transition_state"]["vibrational_quanta_J_mol"].append(1000.0)
    with pytest.raises(ValueError):
        RRKMGrainSpec.from_dict(data)
