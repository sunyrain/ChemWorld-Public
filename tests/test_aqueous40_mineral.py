"""Pure-mineral appearance/depletion with original finite inventory ownership."""

import json
from dataclasses import replace

import pytest

from chemworld.foundation.state import WorldState
from chemworld.foundation.state_ledgers import PhaseLedger, VesselLedger
from chemworld.physchem.aqueous_database import AqueousReaction
from chemworld.physchem.aqueous_mineral_equilibrium import (
    AqueousMineralEquilibriumSpec,
    solve_aqueous_mineral_equilibrium,
)
from chemworld.physchem.residual_system import RootPolicy
from chemworld.runtime.aqueous_mineral_equilibrium import equilibrate_aqueous_mineral_inventory
from reference.aqueous40_mineral_cases import cases, inventory, owned_inventory


def project(case, spec=None):
    spec, ledger, vessel, solids = owned_inventory(case, spec)
    return equilibrate_aqueous_mineral_inventory(
        ledger,
        spec,
        "water",
        solids,
        vessel=vessel,
        temperature_K=case["temperature_K"],
        pressure_Pa=case["pressure_Pa"],
    )


@pytest.mark.parametrize("case", cases(), ids=lambda case: case["id"])
def test_declared_chemistry_closes_with_real_heat_and_invariants(case):
    result = project(case)
    assert result.success, result.equilibrium.to_dict()
    diagnostics = result.equilibrium.diagnostics
    assert max(map(abs, diagnostics["element_residuals_mol"].values())) < 1e-9
    assert abs(diagnostics["charge_residual_mol"]) < 1e-10
    assert max(map(abs, diagnostics["invariant_residuals_mol"])) < 1e-9
    assert (
        result.final_properties.internal_energy_J - result.initial_properties.internal_energy_J
        == pytest.approx(result.heat_duty_J - result.boundary_work_J, abs=1e-8)
    )
    assert result.ledger.phases["other"].species_amounts_mol == {"unrelated": 1.0}
    assert json.loads(json.dumps(result.to_dict(), allow_nan=False))["success"]


def test_si_sign_and_actual_appearance_exhaustion():
    undersaturated, supersaturated, _, exhausted = [project(c) for c in cases()[:4]]
    assert undersaturated.equilibrium.saturation_indices["NaCl(s)"] < 0
    assert undersaturated.equilibrium.amounts_mol["NaCl(s)"] == 0
    assert supersaturated.equilibrium.amounts_mol["NaCl(s)"] == pytest.approx(0.04, abs=1e-9)
    assert abs(supersaturated.equilibrium.saturation_indices["NaCl(s)"]) < 1e-8
    assert exhausted.equilibrium.amounts_mol["NaCl(s)"] == 0
    assert exhausted.equilibrium.saturation_indices["NaCl(s)"] < 0


def test_competition_and_omitted_reaction_invariant():
    both, competing = (project(c).equilibrium.amounts_mol for c in cases()[5:7])
    assert both["NaCl(s)"] > 0 and both["KCl(s)"] > 0
    assert competing["NaCl(s)"] > 0 and competing["KCl(s)"] == 0
    omitted = project(cases()[8]).equilibrium.amounts_mol
    assert omitted["NaCl(aq)"] == pytest.approx(0.003, abs=1e-12)


def test_original_phase_identity_and_json_continuation():
    case = cases()[5]
    spec, ledger, vessel, solids = owned_inventory(case)
    original = ledger.to_dict()
    restored = AqueousMineralEquilibriumSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    first = equilibrate_aqueous_mineral_inventory(
        ledger,
        restored,
        "water",
        solids,
        vessel=vessel,
        temperature_K=case["temperature_K"],
        pressure_Pa=case["pressure_Pa"],
    )
    assert first.success
    assert ledger.to_dict() == original
    assert set(first.ledger.phases) == set(ledger.phases)
    local = PhaseLedger({key: p for key, p in first.ledger.phases.items() if p.vessel_id == "cell"})
    other = first.ledger.phases["other"]
    inactive = WorldState(
        other.species_amounts_mol,
        other.volume_L,
        298.15,
        1e5,
        "solid",
        "other-cell",
        phases=PhaseLedger({"other": other}),
    )
    world = WorldState(
        local.total_amounts_mol(),
        sum(p.volume_L for p in local.phases.values()),
        case["temperature_K"],
        case["pressure_Pa"],
        "aqueous",
        "cell",
        phases=local,
        vessels=VesselLedger({"cell": vessel}),
        inactive_vessels={"other-cell": inactive},
    )
    restored_world = WorldState.from_dict(json.loads(json.dumps(world.to_dict())))
    clone = restored_world.phases
    assert restored_world.ledger.time_s == world.ledger.time_s
    assert restored_world.inactive_vessels["other-cell"].to_dict() == inactive.to_dict()
    second = equilibrate_aqueous_mineral_inventory(
        clone,
        restored,
        "water",
        solids,
        vessel=vessel,
        temperature_K=case["temperature_K"],
        pressure_Pa=case["pressure_Pa"],
    )
    assert second.success
    assert abs(second.heat_duty_J) < 1e-6 and abs(second.boundary_work_J) < 1e-9
    assert second.ledger.phases["water"].metadata["user_tag"] == "keep"


def test_failed_root_returns_unchanged_ledger():
    case = cases()[2]
    spec, ledger, vessel, solids = owned_inventory(case)
    result = equilibrate_aqueous_mineral_inventory(
        ledger,
        replace(spec, root_policy=RootPolicy(maximum_iterations=1)),
        "water",
        solids,
        vessel=vessel,
        temperature_K=case["temperature_K"],
        pressure_Pa=case["pressure_Pa"],
    )
    assert not result.success and result.ledger is ledger
    assert result.heat_duty_J is None and result.equilibrium.amounts_mol is None


def test_valid_local_candidate_does_not_hide_unresolved_other_phase_sets():
    case = cases()[0]
    spec, _, _, _ = owned_inventory(case)
    result = project(case, replace(spec, root_policy=RootPolicy(maximum_iterations=1)))
    assert result.success
    diagnostic = result.equilibrium.diagnostics
    assert diagnostic["phase_sets_enumerated"]
    assert not diagnostic["phase_search_complete"]
    assert not diagnostic["global_stability_claimed"]
    assert diagnostic["unresolved_phase_sets"] == [
        {"present_minerals": ["NaCl(s)"], "reason": "iteration_limit"}
    ]
    assert result.equilibrium.amounts_mol["NaCl(s)"] == 0
    assert result.equilibrium.saturation_indices["NaCl(s)"] < 0


def test_identical_mineral_coexistence_marks_marginal_curvature_unresolved():
    case = {**cases()[1], "na": 0.03}
    spec, _, _, _ = owned_inventory(case)
    original = next(s for s in spec.database.species if s.phase == "mineral")
    duplicate_id = "second-NaCl(s)"
    duplicate = replace(
        original,
        species_id=duplicate_id,
        standard_state=replace(
            original.standard_state,
            thermo=replace(original.standard_state.thermo, species_id=duplicate_id),
        ),
    )
    reaction = AqueousReaction(
        "dissolve-second", {duplicate_id: -1, "Na+": 1, "Cl-": 1}, original.source_id
    )
    spec = replace(
        spec,
        database=replace(
            spec.database,
            species=(*spec.database.species, duplicate),
            reactions=(*spec.database.reactions, reaction),
        ),
        reaction_ids=(*spec.reaction_ids, reaction.reaction_id),
    )
    result = project(case, spec)
    assert result.success
    attempt = next(
        p for p in result.equilibrium.diagnostics["phase_sets"] if len(p["present_minerals"]) == 2
    )
    assert attempt["failure"] == "unresolved_marginal_local_curvature"
    assert attempt["unresolved"]
    assert not result.equilibrium.diagnostics["phase_search_complete"]


def test_postsolve_property_failure_is_unresolved(monkeypatch):
    from chemworld.physchem import aqueous_mineral_equilibrium as module

    case = cases()[0]
    spec, _, _, _ = owned_inventory(case)
    original = module.aqueous_mineral_properties
    calls = 0

    def fail_after_initial(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ValueError("postsolve_property_unavailable")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "aqueous_mineral_properties", fail_after_initial)
    result = solve_aqueous_mineral_equilibrium(
        spec,
        inventory(case, spec),
        temperature_K=case["temperature_K"],
        pressure_Pa=case["pressure_Pa"],
    )
    assert not result.success
    assert {"present_minerals": [], "reason": "postsolve_property_unavailable"} in (
        result.diagnostics["unresolved_phase_sets"]
    )
    assert not result.diagnostics["phase_search_complete"]


def test_owner_volume_and_capacity_rejections():
    case = cases()[2]
    spec, ledger, vessel, solids = owned_inventory(case)
    kwargs = {"temperature_K": case["temperature_K"], "pressure_Pa": case["pressure_Pa"]}
    with pytest.raises(ValueError, match="owned"):
        equilibrate_aqueous_mineral_inventory(
            ledger, spec, "water", solids, vessel=replace(vessel, vessel_id="wrong"), **kwargs
        )
    solid = next(iter(solids.values()))
    invalid = PhaseLedger({**ledger.phases, solid: replace(ledger.phases[solid], volume_L=0.1)})
    with pytest.raises(ValueError, match="mineral volume"):
        equilibrate_aqueous_mineral_inventory(
            invalid, spec, "water", solids, vessel=vessel, **kwargs
        )
    initial_volume = sum(
        p.volume_L for p in ledger.phases.values() if p.vessel_id == vessel.vessel_id
    )
    result = equilibrate_aqueous_mineral_inventory(
        ledger,
        spec,
        "water",
        solids,
        vessel=replace(vessel, max_volume_L=initial_volume + 1e-12),
        **kwargs,
    )
    assert result.equilibrium.success and not result.success and result.ledger is ledger
    assert "capacity" in result.failure_reason
