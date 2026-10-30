"""Actual one/two-phase flow, source domains and independent finite heat balance."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.examples.molecular_flash_separator import molecular_flash_separator_example
from chemworld.physchem.molecular_flash_separator import evaluate_molecular_flash_separator
from chemworld.physchem.molecular_flash_separator_inventory import (
    MolecularFlashSeparatorInventorySpec,
    advance_molecular_flash_separator,
    molecular_flash_separator_inventory,
)
from reference.molecular_flash_separator31_reference import finite, steady


@pytest.mark.parametrize("z", [0.4, 0.6, 0.9])
@pytest.mark.parametrize("hot", [False, True])
def test_original_flash_actual_phase_labels_and_independent_properties(z, hot):
    spec, amounts, temperatures, inlet = molecular_flash_separator_example(z=z, hot_feed=hot)
    source = spec.at_ph(0, amounts[0], sum(amounts[0].values()) * inlet.state.enthalpy_J_mol)
    assert source.temperature_K == pytest.approx(temperatures[0], abs=2e-7)
    result = evaluate_molecular_flash_separator(spec.separator, inlet)
    expected = steady(300.0, z, hot)
    assert result.success, result.failure_reason
    assert result.diagnostics["outlet_phase_labels"] == expected["labels"]
    assert len(result.outlets) == len(expected["phases"])
    for actual, ref in zip(result.outlets, expected["phases"], strict=True):
        assert actual.molar_rate_mol_s == pytest.approx(ref["rate"], abs=1e-8)
        assert actual.state.mole_fractions == pytest.approx(ref["x"], abs=2e-5)
        assert actual.state.molar_density_mol_m3 == pytest.approx(ref["rho"], rel=3e-5)
        assert actual.state.enthalpy_J_mol == pytest.approx(ref["h"], abs=0.02)
    assert result.heat_duty_W == pytest.approx(expected["Q"], abs=5e-5)
    assert max(map(abs, result.diagnostics["component_residuals_mol_s"].values())) < 1e-10
    json.dumps(result.to_dict())


@pytest.mark.parametrize("z", [0.4, 0.6, 0.9])
def test_finite_n_h_original_u_and_absent_phase_invariance(z):
    spec, amounts, temperatures, _ = molecular_flash_separator_example(z=z)
    initial = molecular_flash_separator_inventory(spec, amounts, temperatures)
    result, diagnostic = advance_molecular_flash_separator(spec, initial, 0.5)
    expected = finite([list(n.values()) for n in amounts], 300.0, z, False, 0.5)
    np.testing.assert_allclose(
        [list(n.values()) for n in result.final_state.amounts_mol], expected["amounts"], atol=2e-9
    )
    assert result.final_state.enthalpies_J == pytest.approx(expected["enthalpies"], abs=2e-4)
    assert diagnostic["pressure_work_J"] == pytest.approx(expected["pressure_work"], abs=2e-5)
    assert diagnostic["internal_energy_residual_J"] == pytest.approx(0.0, abs=2e-5)
    for i, (n, h, ref) in enumerate(
        zip(
            result.final_state.amounts_mol,
            result.final_state.enthalpies_J,
            expected["final"],
            strict=True,
        )
    ):
        closed = spec.at_ph(i, n, h)
        assert closed.temperature_K == pytest.approx(ref["T"], abs=2e-4)
        assert sum(n.values()) / closed.molar_density_mol_m3 == pytest.approx(ref["V"], rel=3e-6)
    for i in set(range(3)) - set(diagnostic["participating_tank_indices"]):
        assert result.final_state.amounts_mol[i] == initial.amounts_mol[i]
        assert result.final_state.enthalpies_J[i] == initial.enthalpies_J[i]


def test_power_budget_failure_has_no_executable_or_dummy_outlet():
    spec, _, _, inlet = molecular_flash_separator_example()
    limited = replace(
        spec.separator,
        device=replace(spec.separator.device, maximum_power_W=1e-6, maximum_cooling_power_W=1e-6),
    )
    result = evaluate_molecular_flash_separator(limited, inlet)
    assert not result.success and not result.outlets
    assert result.failure_reason == "insufficient_separator_heating_or_cooling_power"
    assert result.diagnostics["outlet_phase_labels"] == []
    exhausted = evaluate_molecular_flash_separator(replace(spec.separator, max_iterations=1), inlet)
    assert not exhausted.success and not exhausted.outlets
    assert exhausted.diagnostics["max_iterations"] == 1


def test_source_phase_and_codec_are_explicit():
    spec, _, _, inlet = molecular_flash_separator_example(hot_feed=True)
    payload = json.loads(json.dumps(spec.to_dict()))
    assert (
        json.loads(json.dumps(MolecularFlashSeparatorInventorySpec.from_dict(payload).to_dict()))
        == payload
    )
    result = evaluate_molecular_flash_separator(replace(spec.separator, feed_phase="liquid"), inlet)
    assert not result.success and not result.outlets
    with pytest.raises(ValueError, match="explicit phases"):
        replace(spec.separator, feed_phase="automatic")
