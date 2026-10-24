"""Original molecular cascade science, finite-state projection and rejection."""

import json
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("thermo", reason="optional independent reference package")

from chemworld.examples.molecular_cascade import molecular_cascade_example
from chemworld.physchem.molecular_cascade import MolecularCascadeSpec, solve_molecular_cascade
from chemworld.physchem.molecular_cascade_inventory import (
    MolecularCascadeInventorySpec,
    advance_molecular_cascade,
    molecular_cascade_inventory,
)
from chemworld.physchem.residual_system import RootPolicy
from reference.molecular_cascade32_reference import finite, steady


@pytest.mark.parametrize(
    "temperatures", [(300.0,), (300.0, 302.0), (300.0,) * 4, tuple(np.linspace(300, 302, 6))]
)
def test_original_stage_equations_against_independent_thermo(temperatures):
    spec, _, _, feeds = molecular_cascade_example(temperatures_K=temperatures)
    result = solve_molecular_cascade(spec.cascade, *feeds)
    expected = steady(temperatures, 0.75)
    assert result.success, result.failure_reason
    assert expected["success"]
    for actual, target in zip(result.diagnostics["stages"], expected["stages"], strict=True):
        for k, side in enumerate(("vapor", "liquid")):
            assert actual[side]["molar_rate_mol_s"] == pytest.approx(target["rates"][k], abs=1e-8)
            assert actual[side]["state"]["mole_fractions"] == pytest.approx(
                target["compositions"][k], abs=2e-5
            )
            assert actual[side]["state"]["molar_density_mol_m3"] == pytest.approx(
                target["densities"][k], rel=3e-5
            )
            assert actual[side]["state"]["enthalpy_J_mol"] == pytest.approx(
                target["enthalpies"][k], abs=0.02
            )
        assert actual["heat_duty_W"] == pytest.approx(target["Q"], abs=5e-5)
    assert result.diagnostics["component_residual_mol_s"] < 1e-10
    assert len(result.outlets) == 2
    assert result.shaft_work_W == 0


def test_four_tank_n_h_u_pdv_match_independent_endpoint():
    spec, amounts, temperatures, _ = molecular_cascade_example()
    initial = molecular_cascade_inventory(spec, amounts, temperatures)
    result, diagnostic = advance_molecular_cascade(spec, initial, 0.5)
    expected = finite([list(n.values()) for n in amounts], (300.0,) * 4, 0.75, 0.5)
    np.testing.assert_allclose(
        [list(n.values()) for n in result.final_state.amounts_mol], expected["amounts"], atol=2e-9
    )
    assert result.final_state.enthalpies_J == pytest.approx(expected["enthalpies"], abs=2e-4)
    assert result.enthalpy_residual_J == pytest.approx(0, abs=2e-5)
    assert diagnostic["internal_energy_residual_J"] == pytest.approx(0, abs=2e-5)
    assert diagnostic["shaft_work_J"] == 0
    assert diagnostic["pressure_work_J"] != 0
    assert initial.amounts_mol == amounts
    internal_energies = []
    for inventory, reference in (
        (initial, expected["initial"]),
        (result.final_state, expected["final"]),
    ):
        energy = 0.0
        for i, (n, h, target) in enumerate(
            zip(inventory.amounts_mol, inventory.enthalpies_J, reference, strict=True)
        ):
            phase = spec.at_ph(i, n, h)
            total = sum(n.values())
            assert total / phase.molar_density_mol_m3 == pytest.approx(target["V"], rel=3e-6)
            assert phase.temperature_K == pytest.approx(target["T"], abs=2e-4)
            energy += total * phase.internal_energy_J_mol
        internal_energies.append(energy)
    assert internal_energies[1] - internal_energies[0] == pytest.approx(
        expected["Q"] - diagnostic["pressure_work_J"], abs=2e-5
    )


@pytest.mark.parametrize("boundary", ["source", "receiver", "volume"])
def test_earliest_finite_event(boundary):
    spec, amounts, temperatures, _ = molecular_cascade_example()
    ref = steady((300.0,) * 4, 0.75)
    if boundary == "source":
        amounts = ({k: v * (0.00101 / 0.02) for k, v in amounts[0].items()}, *amounts[1:])
        prefix = "source_reserve:"
    elif boundary == "receiver":
        spec = replace(spec, receiver_capacities_mol=(0.02 + ref["stages"][0]["rates"][0], 0.1))
        prefix = "receiver_capacity:"
    else:
        target = finite([list(n.values()) for n in amounts], (300.0,) * 4, 0.75, 1.0)
        spec = replace(spec, tank_capacities_L=(1.0, 1.0, target["final"][2]["V"] * 1000, 1.0))
        prefix = "tank_volume:"
    result, _ = advance_molecular_cascade(
        spec, molecular_cascade_inventory(spec, amounts, temperatures), 2.0
    )
    assert result.actual_duration_s == pytest.approx(1, abs=2e-6)
    assert result.stopping_event.startswith(prefix)


def test_finite_power_root_and_infeasible_flow_fail_without_outlets():
    spec, _, _, feeds = molecular_cascade_example(1)
    low = replace(
        spec.cascade,
        devices=tuple(
            replace(d, maximum_power_W=1e-6, maximum_cooling_power_W=1e-6)
            for d in spec.cascade.devices
        ),
    )
    denied = solve_molecular_cascade(low, *feeds)
    assert not denied.success and not denied.outlets
    assert denied.failure_reason == "insufficient_stage_heating_or_cooling_power"
    limited = solve_molecular_cascade(
        replace(spec.cascade, root_policy=RootPolicy(1e-9, 1)), *feeds
    )
    assert not limited.success and not limited.outlets
    assert limited.diagnostics["root"]["iterations"] <= 1
    infeasible, _, _, feeds2 = molecular_cascade_example(1, flow_ratio=1.25)
    rejected = solve_molecular_cascade(infeasible.cascade, *feeds2)
    assert not rejected.success and not rejected.outlets
    assert not steady((300.0,), 1.25)["success"]


def test_codec_and_declared_domain_are_not_silently_relaxed():
    spec, _, _, _ = molecular_cascade_example()
    data = json.loads(json.dumps(spec.to_dict()))
    restored = MolecularCascadeInventorySpec.from_dict(data)
    assert json.loads(json.dumps(restored.to_dict())) == data
    with pytest.raises(ValueError, match="unknown"):
        MolecularCascadeSpec.from_dict({"model_type": "shortcut"})
    with pytest.raises(ValueError, match=r"1\.\.6"):
        replace(spec.cascade, devices=spec.cascade.devices * 2)
    with pytest.raises(ValueError, match="four finite owners"):
        replace(spec, vessel_ids=("reactor",) * 4)
