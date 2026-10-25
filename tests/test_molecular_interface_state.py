"""Interface queries consume live phase identity, vessel conditions and one EOS."""

import json
from dataclasses import replace

import pytest

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.physchem.molecular_flash import MolecularDensityBasin
from chemworld.physchem.molecular_interface import (
    PolynomialSquareGradientInfluence,
    molecular_influence_from_dict,
)
from chemworld.runtime import molecular_interface_from_state
from chemworld.runtime.molecular_phase_inventory import (
    MolecularPhaseEquilibriumSpec,
    MolecularPhaseTarget,
)
from reference.run_molecular_interface import analytic_pair


def declared_state():
    model, endpoints, influence, gamma = analytic_pair()
    phases = {
        key: PhaseRecord(
            key, "vessel", "liquid", 1000 / endpoint.molar_density_mol_m3,
            dict(zip(model.component_ids, endpoint.mole_fractions, strict=True)),
            metadata={"carrier_representation": "species_inventory"},
        )
        for key, endpoint in zip(("left", "right"), endpoints, strict=True)
    }
    ledger = PhaseLedger(phases)
    state = WorldState(
        ledger.total_amounts_mol(), sum(p.volume_L for p in phases.values()),
        endpoints[0].temperature_K, endpoints[0].pressure_Pa, "liquid", "vessel", phases=ledger,
    )
    declaration = MolecularPhaseEquilibriumSpec(
        model, (MolecularDensityBasin("liquid", (100.0, 8000.0), 3000.0),),
        tuple(MolecularPhaseTarget(k, "liquid", "liquid") for k in phases),
    )
    return state, declaration, influence, gamma


def query(state, declaration, influence, **options):
    return molecular_interface_from_state(
        state, declaration, influence, left_phase_id="left", right_phase_id="right",
        interior_density_grid=(1400.0, 3000.0, 4600.0), **options,
    )


def test_current_world_query_preserves_all_ledgers_and_json():
    state, declaration, influence, gamma = declared_state()
    before = state.to_dict()
    result = query(state, declaration, influence, area_m2=0.02)
    assert result.interface.success
    assert result.surface_excess_grand_potential_J == pytest.approx(0.02 * gamma, rel=2e-8)
    assert state.to_dict() == before
    restored = WorldState.from_dict(json.loads(json.dumps(before, allow_nan=False)))
    assert query(restored, declaration, influence, area_m2=0.02).to_dict() == result.to_dict()
    # Extensivity of bulk stock does not change an intensive interface.
    doubled = state.replace(phases=PhaseLedger({
        k: replace(p, volume_L=2 * p.volume_L,
                   species_amounts_mol={key: 2 * n for key, n in p.species_amounts_mol.items()})
        for k, p in state.phases.phases.items()
    }))
    assert query(doubled, declaration, influence).interface == result.interface


def test_current_pressure_and_actual_model_are_checked_before_query():
    state, declaration, influence, _ = declared_state()
    before = state.to_dict()
    with pytest.raises(ValueError, match="pressure"):
        query(state.replace(pressure_Pa=state.pressure_Pa * 1.1), declaration, influence)
    with pytest.raises(ValueError, match="pressure"):
        changed = replace(declaration, residual_model=replace(
            declaration.residual_model, left=(1200.0,),
        ))
        query(state, changed, influence)
    assert state.to_dict() == before


@pytest.mark.parametrize("change,match", [
    ({"phase_type": "gas"}, "phase type"),
    ({"volume_L": 0.0}, "occupied phases"),
    ({"volume_L": 0.01}, "density"),
    ({"vessel_id": "other"}, "ownership"),
    ({"phase_id": "renamed"}, "ownership"),
    ({"species_amounts_mol": {"a": True}}, "inventory"),
])
def test_live_phase_replacement_cannot_reuse_stale_endpoint(change, match):
    state, declaration, influence, _ = declared_state()
    changed = state.replace(phases=PhaseLedger({
        **state.phases.phases, "left": replace(state.phases.phases["left"], **change),
    }))
    before = changed.to_dict()
    with pytest.raises(ValueError, match=match):
        query(changed, declaration, influence)
    assert changed.to_dict() == before


def test_missing_or_undeclared_phase_is_not_recreated():
    state, declaration, influence, _ = declared_state()
    empty = state.replace(phases=PhaseLedger({"left": state.phases.phases["left"]}))
    with pytest.raises(ValueError, match="present, explicitly declared"):
        query(empty, declaration, influence)
    single = replace(declaration, phase_targets=declaration.phase_targets[:1])
    with pytest.raises(ValueError, match="present, explicitly declared"):
        query(state, single, influence)


def test_temperature_dependent_influence_consumes_live_temperature_and_codec():
    state, declaration, _, gamma = declared_state()
    influence = PolynomialSquareGradientInfluence(
        declaration.residual_model.component_ids, ((1e-18, 5e-19),), ((0.0,),),
        (250.0, 350.0), "explicit analytic test influence", 300.0, 100.0,
    )
    restored = molecular_influence_from_dict(json.loads(json.dumps(influence.to_dict())))
    initial = query(state, declaration, restored)
    warmer = state.replace(temperature_K=320.0)
    before = warmer.to_dict()
    after = query(warmer, declaration, restored)
    assert initial.interface.success and after.interface.success
    assert initial.interface.surface_tension_N_m == pytest.approx(gamma, rel=2e-8)
    assert after.interface.surface_tension_N_m == pytest.approx(gamma * 1.1**.5, rel=2e-8)
    assert warmer.to_dict() == before
    with pytest.raises(ValueError, match="temperature"):
        query(state.replace(temperature_K=360.0), declaration, restored)
