"""Interface endpoints are read from real native inventories, without mutation."""

import json
from dataclasses import replace

import pytest

from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.runtime.molecular_interface import molecular_interface_from_ledger
from reference.run_molecular_interface import analytic_pair


def stock(binary=False):
    model, states, influence, gamma = analytic_pair(binary)
    phases = {key: PhaseRecord(key, "vessel", "liquid", 1000/state.molar_density_mol_m3,
        dict(zip(model.component_ids, state.mole_fractions, strict=True)),
        metadata={"carrier_representation": "species_inventory"})
        for key, state in zip(("left", "right"), states, strict=True)}
    phases["other"] = PhaseRecord("other", "other-vessel", "solid", .01, {"inert": 2})
    return PhaseLedger(phases), model, influence, gamma


def test_actual_phase_pair_read_only_area_and_stale_cache_rejection():
    ledger, model, influence, gamma = stock()
    before = ledger.to_dict()
    result = molecular_interface_from_ledger(ledger, model, influence, vessel_id="vessel",
        left_phase_id="left", right_phase_id="right", temperature_K=300, area_m2=2e-4,
        interior_density_grid=(1400., 3000., 4600.))
    assert result.interface.success
    assert result.surface_excess_grand_potential_J == pytest.approx(gamma*2e-4, rel=2e-8)
    assert ledger.to_dict() == before
    json.dumps(result.to_dict(), allow_nan=False)
    changed = PhaseLedger({**ledger.phases, "left": replace(ledger.phases["left"], volume_L=1.01)})
    with pytest.raises(ValueError, match="coexist"):
        molecular_interface_from_ledger(changed, model, influence, vessel_id="vessel",
            left_phase_id="left", right_phase_id="right", temperature_K=300,
            interior_density_grid=(1400., 3000., 4600.))
    assert ledger.to_dict() == before


def test_binary_inventory_endpoints_and_failed_profile_leave_stock_unchanged():
    ledger, model, influence, _ = stock(True)
    before = ledger.to_dict()
    result = molecular_interface_from_ledger(ledger, model, influence, vessel_id="vessel",
        left_phase_id="left", right_phase_id="right", temperature_K=300,
        mesh=DiffusionMesh.planar(40e-9, 64), component_density_bounds=((100., 8000.),)*2)
    assert result.interface.success, result.to_dict()
    assert result.surface_excess_grand_potential_J is None
    coarse = molecular_interface_from_ledger(ledger, model, influence, vessel_id="vessel",
        left_phase_id="left", right_phase_id="right", temperature_K=300, area_m2=1,
        mesh=DiffusionMesh.planar(40e-9, 32), component_density_bounds=((100., 8000.),)*2)
    assert not coarse.interface.success
    assert coarse.interface.failure_reason == "profile_underresolved_first_integral"
    assert coarse.surface_excess_grand_potential_J is None
    failure = molecular_interface_from_ledger(ledger, model, influence, vessel_id="vessel",
        left_phase_id="left", right_phase_id="right", temperature_K=300, area_m2=1,
        mesh=DiffusionMesh.planar(40e-9, 32), component_density_bounds=((100., 8000.),)*2,
        max_iterations=1)
    assert not failure.interface.success and failure.surface_excess_grand_potential_J is None
    assert ledger.to_dict() == before


def test_phase_identity_ownership_and_area_require_explicit_valid_values():
    ledger, model, influence, _ = stock()
    for kwargs, match in (({"right_phase_id": "left"}, "distinct"),
                          ({"vessel_id": "other-vessel"}, "ownership"),
                          ({"area_m2": -1}, "area")):
        options = {"vessel_id": "vessel", "left_phase_id": "left", "right_phase_id": "right",
                   "temperature_K": 300, "interior_density_grid": (1400., 3000., 4600.)} | kwargs
        with pytest.raises(ValueError, match=match):
            molecular_interface_from_ledger(ledger, model, influence, **options)
