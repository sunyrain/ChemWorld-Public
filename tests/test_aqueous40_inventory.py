"""Actual aqueous phase ownership, Gibbs energy balance and rollback."""

import json
from dataclasses import replace

import pytest

from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord, VesselRecord
from chemworld.physchem.aqueous_phase_properties import aqueous_phase_properties
from chemworld.physchem.aqueous_reaction_equilibrium import AqueousEquilibriumSpec
from chemworld.physchem.residual_system import RootPolicy
from chemworld.runtime.aqueous_phase_equilibrium import equilibrate_aqueous_phase_inventory
from reference.aqueous40_fixtures import feed, model


def setup():
    spec = model(network="water")
    species = []
    for s in spec.database.species:
        if s.species_id == "H+":
            thermo = s.standard_state.thermo
            thermo = replace(thermo, reference=replace(
                thermo.reference, energy_J_mol=1000.0,
                entropy_J_mol_K=thermo.reference.entropy_J_mol_K + 1000.0 / 298.15,
            ))
            s = replace(s, standard_state=replace(
                s.standard_state, thermo=thermo,
                molar_volume_m3_mol=s.standard_state.molar_volume_m3_mol + 1e-6,
            ))
        species.append(s)
    spec = replace(spec, database=replace(spec.database, species=tuple(species)))
    amounts = feed(spec)
    state = aqueous_phase_properties(spec.database, amounts, temperature_K=298.15, pressure_Pa=1e5)
    selected = PhaseRecord("aqueous", "tank", "aqueous", state.volume_m3 * 1000,
                           amounts, selected=True, metadata={"carrier_representation":
                           "species_inventory", "user_tag": "retain"})
    other = PhaseRecord("other", "other_tank", "solid", 0.2, {"inert": 1.0})
    vessel = VesselRecord("tank", "TP-cell", 2.0, 350.0, 1e6,
                          temperature_K=298.15, pressure_Pa=1e5)
    return spec, PhaseLedger({"aqueous": selected, "other": other}), vessel


def project(spec, ledger, vessel):
    return equilibrate_aqueous_phase_inventory(
        ledger, spec, "aqueous", vessel=vessel, temperature_K=298.15, pressure_Pa=1e5,
    )


def test_real_inventory_heat_work_and_recovery():
    spec, ledger, vessel = setup()
    original = ledger.to_dict()
    restored = AqueousEquilibriumSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    result = project(restored, ledger, vessel)
    assert result.success, result.to_dict()
    assert ledger.to_dict() == original
    assert result.ledger.phases["other"].to_dict() == original["other"]
    phase = result.ledger.phases["aqueous"]
    assert phase.selected and phase.vessel_id == "tank" and phase.metadata["user_tag"] == "retain"
    extent = phase.species_amounts_mol["H+"]
    assert result.heat_duty_J == pytest.approx(1000.0 * extent, rel=1e-9)
    assert result.boundary_work_J == pytest.approx(0.1 * extent, rel=1e-9)
    assert result.final_properties.internal_energy_J - result.initial_properties.internal_energy_J \
        == pytest.approx(result.heat_duty_J - result.boundary_work_J, abs=1e-9)
    repeated = project(restored, result.ledger, vessel)
    assert repeated.success and abs(repeated.heat_duty_J) < 1e-6
    assert json.loads(json.dumps(result.to_dict(), allow_nan=False))["success"]


def test_failed_solve_retains_original_phase_ledger():
    spec, ledger, vessel = setup()
    result = project(replace(spec, root_policy=RootPolicy(maximum_iterations=1)), ledger, vessel)
    assert not result.success and result.ledger is ledger
    assert result.heat_duty_J is None and result.final_properties is None


def test_capacity_failure_cannot_install_converged_candidate():
    spec, ledger, vessel = setup()
    result = project(spec, ledger, replace(vessel, max_volume_L=ledger.phases["aqueous"].volume_L))
    assert result.equilibrium.success and not result.success
    assert result.ledger is ledger and "capacity" in result.failure_reason


def test_owner_temperature_pressure_and_initial_volume_domains():
    spec, ledger, vessel = setup()
    with pytest.raises(ValueError, match="declared vessel"):
        project(spec, ledger, replace(vessel, vessel_id="other_tank"))
    for limited in (replace(vessel, max_temperature_K=290), replace(vessel, max_pressure_Pa=9e4)):
        with pytest.raises(ValueError, match="limits"):
            project(spec, ledger, limited)
    with pytest.raises(ValueError, match="current temperature and pressure"):
        project(spec, ledger, replace(vessel, temperature_K=305))
    bad = PhaseLedger({**ledger.phases, "aqueous": replace(ledger.phases["aqueous"], volume_L=1.0)})
    with pytest.raises(ValueError, match="initial aqueous phase volume"):
        project(spec, bad, vessel)
