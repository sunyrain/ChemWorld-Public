"""General phase selection must change real stock without losing ownership."""

import json
from dataclasses import replace

import pytest

from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.foundation.state_serialization import _phases
from chemworld.physchem.molecular_multiphase import MolecularMultiphaseSpec
from chemworld.runtime.molecular_multiphase_inventory import (
    MolecularMultiphaseInventorySpec,
    equilibrate_molecular_multiphase_inventory,
    molecular_multiphase_inventory_state,
    molecular_multiphase_stream,
)
from chemworld.runtime.molecular_phase_inventory import MolecularPhaseTarget
from reference.molecular28_caloric_cases import caloric_spec
from test_molecular_multiphase import regular_spec


def ternary():
    return MolecularMultiphaseInventorySpec(
        regular_spec(), tuple(MolecularPhaseTarget(f"liquid-{i}", "fluid", "liquid", i == 0)
                              for i in range(3)),
    )


def stock():
    return PhaseLedger({
        "feed": PhaseRecord("feed", "vessel", "liquid", 1, {"a": 1., "b": 1., "c": 1.}),
        "other": PhaseRecord("other", "other-vessel", "liquid", 3, {"a": 2.}),
        "solid": PhaseRecord("solid", "vessel", "solid", .01, {"inert": 1.}),
    })


def test_three_phase_ledger_live_stream_and_absent_phase_after_collapse():
    spec, original = ternary(), stock()
    result = equilibrate_molecular_multiphase_inventory(
        original, spec, vessel_id="vessel", source_phase_ids=("feed",),
        inputs={"T": 300., "P": 1e5},
    )
    assert result.success, result.to_dict()
    assert len(result.installed_phase_ids) == 3
    assert result.ledger.phases["other"] == original.phases["other"]
    assert result.ledger.phases["solid"] == original.phases["solid"]
    assert result.ledger.total_amounts_mol() == pytest.approx(original.total_amounts_mol())
    current = _phases(json.loads(json.dumps(result.ledger.to_dict())))
    actual = molecular_multiphase_inventory_state(
        current, spec, vessel_id="vessel", source_phase_ids=result.installed_phase_ids,
        temperature_K=300., pressure_Pa=1e5,
    )
    assert actual.enthalpy_J == pytest.approx(result.flash.enthalpy_J, abs=1e-9)
    stream = molecular_multiphase_stream(
        current, spec, vessel_id="vessel", temperature_K=300., molar_rate_mol_s=.01,
    )
    assert stream.state.mole_fractions == pytest.approx(
        result.flash.phases[0].state.mole_fractions)
    collapsed = equilibrate_molecular_multiphase_inventory(
        current, spec, vessel_id="vessel", source_phase_ids=result.installed_phase_ids,
        inputs={"T": 650., "P": 1e5}, seed=actual.seed,
    )
    assert collapsed.success, collapsed.to_dict()
    assert len(collapsed.installed_phase_ids) == 1
    assert "liquid-1" not in collapsed.ledger.phases
    assert "liquid-2" not in collapsed.ledger.phases
    with pytest.raises(ValueError, match="missing"):
        molecular_multiphase_stream(collapsed.ledger, spec, vessel_id="vessel",
            temperature_K=650., molar_rate_mol_s=.01, phase_id="liquid-2")


def test_phase_target_exhaustion_and_failed_search_never_commit():
    original, spec = stock(), ternary()
    for bad in (replace(spec, phase_targets=spec.phase_targets[:2]),
                replace(spec, equilibrium=replace(spec.equilibrium, max_phases=2))):
        result = equilibrate_molecular_multiphase_inventory(
            original, bad, vessel_id="vessel", source_phase_ids=("feed",),
            inputs={"T": 300., "P": 1e5},
        )
        assert not result.success
        assert result.ledger is original
        assert result.failure_reason


def test_source_ownership_and_target_collision_are_checked_before_solve():
    original, spec = stock(), ternary()
    for sources in (("feed", "other"), ("feed", "feed"), ("missing",)):
        with pytest.raises(ValueError):
            equilibrate_molecular_multiphase_inventory(original, spec, vessel_id="vessel",
                source_phase_ids=sources, inputs={"T": 300., "P": 1e5})
    collision = replace(spec, phase_targets=(
        MolecularPhaseTarget("solid", "fluid", "liquid"), *spec.phase_targets[1:]))
    with pytest.raises(ValueError, match="overwrite"):
        equilibrate_molecular_multiphase_inventory(original, collision, vessel_id="vessel",
            source_phase_ids=("feed",), inputs={"T": 300., "P": 1e5})


def test_actual_cubic_codec_uv_stock_roundtrip():
    old = caloric_spec("PR")
    spec = MolecularMultiphaseInventorySpec(
        MolecularMultiphaseSpec(old.model, old.basins, (280., 330.), (4e5, 8e5),
            phase_kind_by_basin={"vapor": "gas", "liquid": "liquid"}),
        (MolecularPhaseTarget("vapor", "vapor", "gas", True),
         MolecularPhaseTarget("liquid", "liquid", "liquid")),
    )
    recovered = MolecularMultiphaseInventorySpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert recovered == spec
    original = PhaseLedger({"feed": PhaseRecord("feed", "vessel", "liquid", .01,
        {"propane": .006, "butane": .004})})
    first = equilibrate_molecular_multiphase_inventory(
        original, recovered, vessel_id="vessel", source_phase_ids=("feed",),
        inputs={"T": 300., "P": 6e5},
    )
    assert first.success, first.to_dict()
    actual = molecular_multiphase_inventory_state(first.ledger, recovered, vessel_id="vessel",
        source_phase_ids=first.installed_phase_ids, temperature_K=300., pressure_Pa=6e5)
    second = equilibrate_molecular_multiphase_inventory(
        first.ledger, recovered, vessel_id="vessel", source_phase_ids=first.installed_phase_ids,
        inputs={"U": actual.internal_energy_J, "V": actual.volume_m3}, seed=actual.seed,
    )
    assert second.success, second.to_dict()
    assert second.flash.temperature_K == pytest.approx(300., abs=2e-5)
    assert second.ledger.total_amounts_mol() == pytest.approx(original.total_amounts_mol())
    json.dumps(second.to_dict(), allow_nan=False)
